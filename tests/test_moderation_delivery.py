import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test_moderation_state import ModerationTestCase


class ModerationDeliveryTests(ModerationTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.log_channel = Mock(spec=self.module.discord.TextChannel)
        self.log_channel.id = self.module.LOG_CHANNEL_ID
        self.log_channel.send = AsyncMock(side_effect=lambda _: self.log_post())
        self.available = True
        self.enterContext(patch.object(
            self.bot, "get_channel",
            side_effect=lambda channel_id: (
                (self.log_channel if self.available else None)
                if channel_id == self.log_channel.id else self.channel
            ),
        ))
        self.enterContext(patch.object(self.module, "Path", return_value=SimpleNamespace(exists=lambda: True)))
        classifier = SimpleNamespace(
            feature_names=["discusses_ellie"], predict=lambda features: ["flag"] * len(features),
        )
        self.enterContext(patch.object(self.module, "load_classifier", return_value=classifier))
        self.extract = self.enterContext(patch.object(self.module, "get_candidate_features", return_value=[]))
        self.next_post_id = 1000

    def log_post(self):
        self.next_post_id += 1
        return SimpleNamespace(id=self.next_post_id, add_reaction=AsyncMock())

    def http_error(self, status=503):
        return self.module.discord.HTTPException(
            SimpleNamespace(status=status, reason="synthetic failure"), "synthetic failure",
        )

    async def seed(self, count=1):
        await self.add_messages(1, count)
        self.extract.return_value = [
            {
                "discord_message_id": index, "relative_message_index": index,
                "target_user_id": 999, "target_username": "Bob",
                "features": {"discusses_ellie": 0},
            }
            for index in range(1, count + 1)
        ]

    def records(self):
        with self.sessions() as session:
            flags = session.query(self.module.FlaggedMessage).order_by(self.module.FlaggedMessage.message_id).all()
            return (
                [(row.message_id, row.was_acted_upon, row.waiver_filtered) for row in flags],
                session.query(self.module.LogChannelRatingPost).count(),
                session.query(self.module.MessageFeatures).count(),
            )

    async def test_partial_send_failure_preserves_completed_and_pending_records_then_retries(self):
        await self.seed(count=3)
        first_post = self.log_post()

        async def send(_content):
            # The current pending flag and its features must exist before sending.
            flags, mappings, features = self.records()
            self.assertEqual(len(flags), 3)
            self.assertEqual(features, 3)
            if self.log_channel.send.await_count == 1:
                return first_post
            self.assertEqual(mappings, 1)
            self.assertTrue(flags[0][1])
            raise self.http_error()

        self.log_channel.send.side_effect = send
        result = await self.bot.run_moderation_now(self.channel)
        self.assertFalse(result.success)
        self.assertEqual(result.flagged_new_count, 3)
        self.assertEqual(self.records(), ([(1, True, False), (2, False, False), (3, False, False)], 1, 3))
        self.assertEqual(self.state.messages_since_check, 3)
        self.assertEqual(
            [call.args[0] for call in first_post.add_reaction.await_args_list],
            [f"{digit}\ufe0f\u20e3" for digit in range(1, 6)],
        )

        # Retry delivery from its stored payload after history eviction and no LLM candidates.
        self.bot.message_store._channel_messages.clear()
        self.extract.return_value = []
        self.log_channel.send.side_effect = lambda _: self.log_post()
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.assertEqual(self.records(), ([(1, True, False), (2, True, False), (3, True, False)], 3, 3))
        self.assertEqual(self.log_channel.send.await_count, 4)
        self.assertIn("test message", self.log_channel.send.call_args.args[0])
        self.assertIn("/456/123/3", self.log_channel.send.call_args.args[0])
        self.assertEqual(self.state.messages_since_check, 0)
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.assertEqual(self.log_channel.send.await_count, 4)

    async def test_missing_log_channel_then_recovery_retries_existing_flag(self):
        await self.seed()
        self.available = False
        message = self.bot.message_store.get_message_by_id(1, self.channel.id)
        self.assertFalse((await self.bot.run_moderation_now(self.channel)).success)
        message.add_reaction.assert_awaited_once()
        self.assertEqual(self.records(), ([(1, False, False)], 0, 1))
        self.log_channel.send.assert_not_awaited()
        self.available = True
        self.extract.return_value = []
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.assertEqual(self.records(), ([(1, True, False)], 1, 1))
        self.log_channel.send.assert_awaited_once()

    async def test_explicit_pending_flag_without_mapping_still_retries_delivery(self):
        from datetime import datetime, timezone
        with self.sessions() as session:
            session.add(self.module.FlaggedMessage(
                message_id=1, channel_id=self.channel.id, guild_id=self.channel.guild.id,
                author_id=789, author_username="Alice", content="persisted flag",
                timestamp=datetime.now(timezone.utc), was_acted_upon=False, waiver_filtered=False,
                pending_log_delivery=True,
            ))
            session.commit()
        self.available = False
        self.assertFalse((await self.bot.run_moderation_now(self.channel)).success)
        self.assertEqual(self.records(), ([(1, False, False)], 0, 0))
        self.available = True
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.assertEqual(self.records(), ([(1, True, False)], 1, 0))
        self.log_channel.send.assert_awaited_once()

    async def test_repeated_candidate_with_successful_mapping_does_not_duplicate(self):
        await self.seed()
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        result = await self.bot.run_moderation_now(self.channel)
        self.assertTrue(result.success)
        self.assertEqual(result.flagged_existing_count, 1)
        self.assertEqual(result.flagged_new_count, 0)
        self.assertEqual(self.records(), ([(1, True, False)], 1, 1))
        self.log_channel.send.assert_awaited_once()
        self.bot.message_store.get_message_by_id(1, self.channel.id).add_reaction.assert_awaited_once()

    async def test_failed_original_reaction_does_not_block_log_delivery(self):
        await self.seed()
        message = self.bot.message_store.get_message_by_id(1, self.channel.id)
        message.add_reaction.side_effect = self.http_error(status=403)
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.assertEqual(self.records(), ([(1, True, False)], 1, 1))
        self.log_channel.send.assert_awaited_once()

    async def test_mapping_is_committed_before_rating_reaction_cancellation(self):
        await self.seed()
        post = self.log_post()

        async def add_rating_reaction(_emoji):
            self.assertEqual(self.records(), ([(1, True, False)], 1, 1))
            raise asyncio.CancelledError()

        post.add_reaction.side_effect = add_rating_reaction
        self.log_channel.send.side_effect = None
        self.log_channel.send.return_value = post
        with self.assertRaises(asyncio.CancelledError):
            await self.bot.run_moderation_now(self.channel)
        self.assertEqual(self.records(), ([(1, True, False)], 1, 1))
        self.assertEqual(self.state.messages_since_check, 1)
        self.extract.return_value = []
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.log_channel.send.assert_awaited_once()

    async def test_waiver_records_are_saved_without_any_delivery_or_retry(self):
        await self.seed()
        self.channel.guild.roles = [SimpleNamespace(
            name=self.module.WAIVER_ROLE_NAME, members=[SimpleNamespace(id=999)],
        )]
        self.enterContext(patch.object(self.module, "SAVE_WAIVER_FILTERED_FLAGS", True))
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.assertEqual(self.records(), ([(1, False, True)], 0, 1))
        self.log_channel.send.assert_not_awaited()
        self.bot.message_store.get_message_by_id(1, self.channel.id).add_reaction.assert_not_awaited()

    async def test_waiver_records_are_filtered_when_saving_is_disabled(self):
        await self.seed()
        self.channel.guild.roles = [SimpleNamespace(
            name=self.module.WAIVER_ROLE_NAME, members=[SimpleNamespace(id=999)],
        )]
        self.enterContext(patch.object(self.module, "SAVE_WAIVER_FILTERED_FLAGS", False))
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.assertEqual(self.records(), ([], 0, 0))
        self.log_channel.send.assert_not_awaited()
