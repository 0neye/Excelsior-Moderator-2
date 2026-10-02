"""Regression tests for bootstrap extra-candidate handling and client cleanup."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import discord
import numpy as np
import bootstrapping
from bootstrapping import RatedMessage, extract_features, fetch_discord_context


def rated_message(message_id, category):
    context = [
        {
            "id": candidate_id,
            "author_id": candidate_id,
            "author_username": f"user-{candidate_id}",
            "content": f"message-{candidate_id}",
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        for candidate_id in (101, 102, 103)
    ]
    return RatedMessage(
        message_id=message_id,
        channel_id=1,
        guild_id=1,
        author_id=message_id,
        author_name=f"user-{message_id}",
        content=f"message-{message_id}",
        timestamp="2026-01-01T00:00:00+00:00",
        flagged_at="2026-01-01T00:00:00+00:00",
        jump_url="",
        category=category,
        rater_user_id=1,
        rating_id=str(message_id),
        context_messages=context,
    )


class ExtraCandidateTests(unittest.IsolatedAsyncioTestCase):
    async def test_known_rated_messages_are_not_reintroduced_as_na(self):
        async def extract_stub(**_kwargs):
            return [
                {"message_id": "101", "features": {"tone_harshness_score": 1.0}},
                {"message_id": "102", "features": {"tone_harshness_score": 2.0}},
                {"message_id": "103", "features": {"tone_harshness_score": 3.0}},
            ]

        messages = [
            rated_message(101, "unconstructive"),
            rated_message(102, "unsolicited"),
        ]
        with patch("bootstrapping.ensure_user_stats_ready"), patch(
            "bootstrapping.extract_features_from_formatted_history", side_effect=extract_stub
        ):
            extracted, run_id = await extract_features(
                messages,
                model="synthetic",
                known_rated_message_ids={message.message_id for message in messages},
                max_concurrent=1,
                auto_save=False,
                include_extra_candidates_as_na=True,
            )

        self.assertIsNone(run_id)
        by_id = {message.message_id: message for message in extracted}
        self.assertEqual(set(by_id), {101, 102, 103})
        self.assertEqual(by_id[101].category, "unconstructive")
        self.assertEqual(by_id[102].category, "unsolicited")
        self.assertEqual(by_id[103].category, "NA")
        self.assertEqual(len(by_id[103].features), 2)

    async def test_full_pipeline_keeps_failed_context_message_out_of_extra_na_samples(self):
        messages = [rated_message(101, "unconstructive"), rated_message(102, "unconstructive")]
        cached = SimpleNamespace(context_messages=messages[0].context_messages, context_message_ids=[101, 102, 103])

        class CacheSession(FakeSession):
            def query(self, _model):
                return SimpleNamespace(filter_by=lambda message_id: SimpleNamespace(
                    first=lambda: cached if message_id == 101 else None,
                ))

        class UnavailableChannel(FakeTextChannel):
            async def history(self, **kwargs):
                raise discord.HTTPException(SimpleNamespace(status=503, reason="synthetic failure"), "synthetic")
                yield

        client = FakeClient(UnavailableChannel("read_error"))
        extraction = AsyncMock(return_value=[
            {"message_id": str(message_id), "features": {"tone_harshness_score": 1.0}}
            for message_id in (101, 102, 103)
        ])
        for target, replacement in (
            ("state", bootstrapping.BootstrapState()),
            ("get_session", lambda: CacheSession()),
            ("DISCORD_BOT_TOKEN", "synthetic-token"),
        ):
            self.enterContext(patch.object(bootstrapping, target, replacement))
        self.enterContext(patch.object(bootstrapping.discord, "Client", return_value=client))
        self.enterContext(patch.object(bootstrapping.discord, "TextChannel", FakeTextChannel))
        self.enterContext(patch.object(bootstrapping, "load_rating_data", return_value=messages))
        for target in ("init_db", "save_to_database", "ensure_user_stats_ready", "train_model", "evaluate_model"):
            self.enterContext(patch.object(bootstrapping, target))
        self.enterContext(patch.object(bootstrapping, "save_features_to_db", return_value=1))
        self.enterContext(patch.object(bootstrapping, "extract_features_from_formatted_history", extraction))
        prepare = self.enterContext(patch.object(bootstrapping, "prepare_training_data", return_value=(
            np.zeros((1, 1)), np.zeros((1, 1)), np.array(["flag"]), np.array(["no-flag"]),
        )))

        await bootstrapping.run_full_pipeline(model="synthetic", include_extra_candidates_as_na=True)

        prepared = prepare.call_args.args[0]
        self.assertEqual([(message.message_id, message.category) for message in prepared], [
            (101, "unconstructive"), (103, "NA"),
        ])
        self.assertEqual([message.message_id for message in bootstrapping.state.messages_with_context], [101])
        self.assertEqual(client.closed, 1)


class FakeQuery:
    def filter_by(self, **_kwargs):
        return self

    def first(self):
        return None


class FakeSession:
    def query(self, _model):
        return FakeQuery()

    def close(self):
        pass


class FakeTextChannel:
    name = "general"

    def __init__(self, mode, read_started=None):
        self.mode = mode
        self.read_started = read_started

    async def history(self, **_kwargs):
        if self.mode == "read_error":
            raise RuntimeError("history failed")
        if self.mode == "cancel":
            self.read_started.set()
            await asyncio.Event().wait()
        if False:
            yield None

    async def fetch_message(self, message_id):
        return SimpleNamespace(
            id=message_id,
            content="message",
            author=SimpleNamespace(id=1, display_name="User", name="user"),
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            edited_at=None,
            reference=None,
            attachments=[],
            reactions=[],
        )


class FakeThread:
    pass


class FakeClient:
    def __init__(self, channel):
        self.channel = channel
        self.user = "fake"
        self.guilds = []
        self.closed = 0
        self.ready_handler = None

    def event(self, callback):
        self.ready_handler = callback
        return callback

    def get_channel(self, _channel_id):
        return self.channel

    async def fetch_channel(self, _channel_id):
        return self.channel

    async def start(self, _token):
        await self.ready_handler()

    async def close(self):
        self.closed += 1


class ContextClientLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def _fetch(self, mode, read_started=None):
        client = FakeClient(FakeTextChannel(mode, read_started))
        patches = [
            patch("bootstrapping.init_db"),
            patch("bootstrapping.get_session", return_value=FakeSession()),
            patch("bootstrapping.discord.Client", return_value=client),
            patch("bootstrapping.discord.TextChannel", FakeTextChannel),
            patch("bootstrapping.discord.Thread", FakeThread),
            patch("bootstrapping.DISCORD_BOT_TOKEN", "test-token"),
        ]
        if mode != "cancel":
            patches.append(patch("bootstrapping.asyncio.sleep", new=AsyncMock()))
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            if mode != "cancel":
                with patches[6]:
                    result = await fetch_discord_context([rated_message(101, "unconstructive")])
            else:
                task = asyncio.create_task(fetch_discord_context([rated_message(101, "unconstructive")]))
                await asyncio.wait_for(read_started.wait(), timeout=1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                result = None
        return client, result

    async def test_context_fetch_closes_client_after_success(self):
        client, result = await self._fetch("success")

        self.assertEqual(client.closed, 1)
        self.assertEqual([message.message_id for message in result], [101])

    async def test_context_fetch_closes_client_after_read_error(self):
        client, result = await self._fetch("read_error")

        self.assertEqual(client.closed, 1)
        self.assertEqual(result, [])

    async def test_context_fetch_closes_client_after_cancellation(self):
        read_started = asyncio.Event()
        client, result = await self._fetch("cancel", read_started)

        self.assertEqual(client.closed, 1)
        self.assertIsNone(result)
