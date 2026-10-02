"""Offline integration coverage for snapshot ownership and pending delivery."""

import asyncio
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import test_moderation_state as state_fixtures
import test_moderation_delivery as delivery_fixtures


class ReviewRound1RuntimeTests(state_fixtures.ModerationTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.log_channel = Mock(spec=self.module.discord.TextChannel)
        self.log_channel.id = self.module.LOG_CHANNEL_ID
        self.log_channel.send = AsyncMock(side_effect=lambda _: self.log_post())
        self.enterContext(patch.object(
            self.bot, "get_channel",
            side_effect=lambda channel_id: self.log_channel if channel_id == self.log_channel.id else self.channel,
        ))
        self.enterContext(patch.object(self.module, "Path", return_value=SimpleNamespace(exists=lambda: True)))
        classifier = SimpleNamespace(feature_names=["discusses_ellie"], predict=lambda features: ["flag"] * len(features))
        self.enterContext(patch.object(self.module, "load_classifier", return_value=classifier))
        self.extract = self.enterContext(patch.object(self.module, "get_candidate_features", return_value=[]))
        self.next_post_id = 1000

    async def add_messages(self, first, last):
        await super().add_messages(first, last)
        for stored in self.bot.message_store.get_whole_history(self.channel.id):
            stored.mentions = []
            stored.guild = self.channel.guild

    log_post = delivery_fixtures.ModerationDeliveryTests.log_post

    def stored_flag(self, message_id, **kwargs):
        return self.module.FlaggedMessage(
            message_id=message_id, channel_id=self.channel.id, guild_id=self.channel.guild.id,
            author_id=789, author_username="Alice", content="stored flag",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), **kwargs,
        )

    async def test_pending_scheduler_stops_on_unavailable_channel_and_can_restart(self):
        self.state.pending_log_delivery = True
        with patch.object(self.bot, "get_channel", side_effect=[None, AssertionError("Scheduler spun")]) as lookup:
            self.module.ExcelsiorBot._ensure_scheduler_task(self.bot, self.channel)
            scheduler = self.state.task
            self.tasks.append(scheduler)
            await asyncio.wait_for(scheduler, 2)
            lookup.assert_called_once_with(self.channel.id)
            self.assertTrue(self.state.pending_log_delivery)
        self.module.ExcelsiorBot._ensure_scheduler_task(self.bot, self.channel)
        restarted = self.state.task
        self.tasks.append(restarted)
        self.assertIsNot(restarted, scheduler)
        await asyncio.sleep(0)
        self.assertFalse(restarted.done())
        self.assertFalse(self.state.pending_log_delivery)

    async def test_unavailable_scheduler_preserves_an_active_manual_batch(self):
        await self.add_messages(1, 30)
        entered, release = asyncio.Event(), asyncio.Event()

        async def extract(*args, **kwargs):
            entered.set()
            await release.wait()
            return []

        self.extract.side_effect = extract
        manual = self.start(self.bot.run_moderation_now(self.channel))
        await asyncio.wait_for(entered.wait(), 2)
        with patch.object(self.bot, "get_channel", return_value=None):
            scheduler = self.start(self.bot._moderation_scheduler(self.channel.id))
            await asyncio.wait_for(scheduler, 2)
        self.assertEqual(self.state.messages_since_check, 30)
        release.set()
        self.assertTrue((await asyncio.wait_for(manual, 2)).success)
        self.assertEqual(self.state.messages_since_check, 0)
        await self.add_messages(31, 60)
        self.assertEqual(self.state.messages_since_check, 30)
        self.assertTrue(self.bot._should_moderate(self.state))

    async def test_pending_send_freezes_first_30_and_leaves_next_30_eligible(self):
        import llms
        with self.sessions() as session:
            session.add(self.stored_flag(100, was_acted_upon=False, pending_log_delivery=True))
            session.commit()
        await self.add_messages(1, 30)
        entered, release = asyncio.Event(), asyncio.Event()
        first_post = self.log_post()

        async def send(_content):
            if self.log_channel.send.await_count == 1:
                entered.set()
                await release.wait()
                return first_post
            return self.log_post()

        self.log_channel.send.side_effect = send
        histories = []

        async def formatted_extract(**kwargs):
            # The formatter and resolver run on the real llms.get_candidate_features.
            histories.append(kwargs["formatted_message_history"])
            return [
                {"message_id": index, "target_username": "Bob", "features": {"discusses_ellie": 0}}
                for index in range(kwargs["ignore_first_message_count"] + 1, len(kwargs["formatted_message_history"].splitlines()) + 1)
            ]

        async def extract(store, channel_id, **kwargs):
            # Keep the real extraction/resolution boundary, mocking only provider work.
            candidates = await llms.get_candidate_features(store, channel_id, **kwargs)
            eligible.append([row["discord_message_id"] for row in candidates])
            return candidates

        eligible = []
        with patch.object(llms, "extract_features_from_formatted_history", side_effect=formatted_extract):
            self.extract.side_effect = extract
            first = self.start(self.bot.run_moderation_now(self.channel))
            await asyncio.wait_for(entered.wait(), 2)
            await self.add_messages(31, 60)
            release.set()
            self.assertTrue((await asyncio.wait_for(first, 2)).success)
            self.assertEqual(eligible[0], list(range(1, 31)))
            self.assertEqual(self.state.last_checked_message_id, 30)
            self.assertEqual(self.state.messages_since_check, 30)
            self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.assertEqual(eligible[1], list(range(31, 61)))
        self.assertEqual(self.state.last_checked_message_id, 60)
        self.assertEqual(self.state.messages_since_check, 0)
        self.assertEqual(len(histories), 2)
        with self.sessions() as session:
            self.assertEqual(session.query(self.module.LogChannelRatingPost).count(), 61)
            self.assertFalse(any(row.pending_log_delivery for row in session.query(self.module.FlaggedMessage)))

    async def test_imported_no_flag_historical_flag_and_waiver_never_replay(self):
        import bootstrapping
        imported = []
        for message_id, category in ((1, "no-flag"), (2, "unconstructive")):
            imported.append(bootstrapping.RatedMessage(
                message_id=message_id, channel_id=self.channel.id, guild_id=self.channel.guild.id,
                author_id=789, author_name="Alice", content="historical message",
                timestamp="2026-01-01T00:00:00Z", flagged_at="2026-01-01T00:00:00Z",
                jump_url="", category=category, rater_user_id=1, rating_id=str(message_id),
            ))
        lane = Path(__file__).resolve().parents[1] / "workspace" / "review-fixes" / "sole-review"
        lane.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=lane) as directory:
            raw_flags = Path(directory) / "historical_flags.json"
            raw_flags.write_text(json.dumps([{
                "message_id": 5, "author_id": 789, "channel_id": self.channel.id,
                "guild_id": self.channel.guild.id, "content": "historical unrated flag",
                "timestamp": "2026-01-01T00:00:00Z",
            }]), encoding="utf-8")
            with patch.object(bootstrapping, "init_db"), patch.object(bootstrapping, "get_session", self.sessions), patch.object(
                bootstrapping, "FLAGGED_MESSAGES_PATH", raw_flags
            ):
                self.assertEqual(bootstrapping.load_flagged_messages_into_db(), 1)
                bootstrapping.save_to_database(imported)
        with self.sessions() as session:
            session.add(self.stored_flag(3, waiver_filtered=True, was_acted_upon=False, pending_log_delivery=True))
            session.add(self.stored_flag(4, was_acted_upon=False, pending_log_delivery=True))
            session.commit()
        # Even a provider that rediscovers historical flags must not replay them.
        await self.add_messages(1, 2)
        self.extract.return_value = [
            {"discord_message_id": index, "relative_message_index": index,
             "target_user_id": 999, "features": {"discusses_ellie": 0}}
            for index in (1, 2)
        ]
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.log_channel.send.assert_awaited_once()
        self.assertIn("/4", self.log_channel.send.call_args.args[0])
        self.assertTrue((await self.bot.run_moderation_now(self.channel)).success)
        self.log_channel.send.assert_awaited_once()

    async def test_startup_recovers_persisted_delivery_without_chatter(self):
        with self.sessions() as session:
            session.add(self.stored_flag(100, was_acted_upon=False, pending_log_delivery=True))
            session.commit()
        finished = asyncio.Event()

        post = self.log_post()
        post.add_reaction.side_effect = lambda _: finished.set()
        self.log_channel.send.side_effect = None
        self.log_channel.send.return_value = post
        self.bot.message_store._channel_messages.clear()
        with patch.object(self.module.ExcelsiorBot, "guilds", new=property(lambda _: [])):
            await self.bot.initialize_moderation_tasks()
        self.assertTrue(self.state.pending_log_delivery)
        self.start(self.bot._moderation_scheduler(self.channel.id))
        await asyncio.wait_for(finished.wait(), 2)
        self.log_channel.send.assert_awaited_once()
        self.assertFalse(self.state.pending_log_delivery)
        self.assertEqual(self.state.messages_since_check, 0)
        self.extract.assert_not_awaited()
