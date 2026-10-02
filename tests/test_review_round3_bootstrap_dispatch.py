"""Offline ownership regressions using installed py-cord's separate dispatcher."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import discord
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config
with patch.object(config, "DB_FILE", ":memory:"):
    import bootstrapping
    import db_config
    import user_stats
from database import UserStats


def rated_message():
    return bootstrapping.RatedMessage(
        message_id=101, channel_id=1, guild_id=1, author_id=1, author_name="synthetic",
        content="synthetic", timestamp="2026-01-01T00:00:00Z", flagged_at="2026-01-01T00:00:00Z",
        jump_url="", category="unconstructive", rater_user_id=1, rating_id="synthetic",
    )


class Channel:
    name = "synthetic"
    threads = []

    def __init__(self, blocked=False, status=None):
        self.read_started = asyncio.Event()
        self.history_release = asyncio.Event()
        if not blocked:
            self.history_release.set()
        self.status = status
        self.history_calls = 0
        self.fetch_calls = 0

    async def history(self, **kwargs):
        self.history_calls += 1
        self.read_started.set()
        await self.history_release.wait()
        if self.status is not None:
            response = SimpleNamespace(status=self.status, reason="synthetic failure")
            error = {403: discord.Forbidden, 404: discord.NotFound}.get(self.status, discord.HTTPException)
            raise error(response, {"message": "synthetic failure"})
        # Stats history contains one message; context history has no surrounding messages.
        if kwargs.get("before", 0) is None:
            yield self.message(101)

    def message(self, message_id):
        return SimpleNamespace(
            id=message_id, content="synthetic",
            author=SimpleNamespace(id=1, name="synthetic", display_name="synthetic"),
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), edited_at=None,
            reference=None, attachments=[], reactions=[],
        )

    async def fetch_message(self, message_id):
        self.fetch_calls += 1
        return self.message(message_id)

    async def archived_threads(self, **kwargs):
        if False:
            yield None


class DispatchClient:
    _run_event = discord.Client._run_event
    user = "synthetic client"

    def __init__(self, channel, block_close=False, handler_error=False):
        self.channel = channel
        self.guilds = [SimpleNamespace(get_channel=self.get_channel)]
        self.handler_error = handler_error
        self.closed = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        if not block_close:
            self.close_release.set()
        self.close_calls = 0
        self.dispatch_tasks = []
        self.on_error = AsyncMock()

    def get_channel(self, channel_id):
        if self.handler_error:
            raise RuntimeError("synthetic ready-handler failure")
        return self.channel

    def event(self, handler):
        self.handler = handler
        return handler

    def dispatch_ready(self):
        task = asyncio.create_task(self._run_event(self.handler, "on_ready"))
        self.dispatch_tasks.append(task)
        return task

    async def start(self, token):
        self.dispatch_ready()
        # The SDK connection loop does not await the separate ready handler.
        await self.closed.wait()

    async def close(self):
        self.close_calls += 1
        self.close_started.set()
        await self.close_release.wait()
        self.closed.set()


class ReviewRound3BootstrapDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.addCleanup(self.engine.dispose)
        db_config.Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, autoflush=False)
        self.enterContext(patch.object(db_config, "engine", self.engine))
        self.enterContext(patch.object(user_stats, "engine", self.engine))
        self.enterContext(patch.object(user_stats, "get_session", self.sessions))
        self.enterContext(patch.object(bootstrapping, "get_session", self.sessions))
        self.enterContext(patch.object(bootstrapping, "init_db"))
        self.enterContext(patch.object(user_stats, "DISCORD_BOT_TOKEN", "synthetic-token"))
        self.enterContext(patch.object(bootstrapping, "DISCORD_BOT_TOKEN", "synthetic-token"))
        self.enterContext(patch.object(discord, "TextChannel", Channel))
        self.tasks = []
        self.clients = []
        self.addAsyncCleanup(self.finish_tasks)

    async def finish_tasks(self):
        for client in self.clients:
            client.channel.history_release.set()
            client.close_release.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await asyncio.gather(*(task for client in self.clients for task in client.dispatch_tasks), return_exceptions=True)

    def start(self, mode, client, rated=None):
        self.clients.append(client)
        self.enterContext(patch.object(discord, "Client", return_value=client))
        call = (bootstrapping.fetch_discord_context([rated or rated_message()]) if mode == "context"
                else user_stats.bootstrap_user_stats(channel_ids=[1]))
        task = asyncio.create_task(call)
        self.tasks.append(task)
        return task

    def assert_closed(self, client):
        self.assertEqual(client.close_calls, 1)
        self.assertTrue(client.closed.is_set())
        self.assertTrue(all(task.done() for task in client.dispatch_tasks))
        client.on_error.assert_not_awaited()

    async def test_context_cancellation_joins_dispatch_before_return_and_stops_mutation(self):
        channel = Channel(blocked=True)
        client = DispatchClient(channel)
        rated = rated_message()
        task = self.start("context", client, rated)
        await asyncio.wait_for(channel.read_started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assert_closed(client)
        self.assertEqual(channel.fetch_calls, 0)
        self.assertEqual(rated.context_messages, [])
        channel.history_release.set()
        # Even a late SDK dispatch must not restart collection after return.
        await asyncio.wait_for(client.dispatch_ready(), 2)
        self.assertEqual(channel.history_calls, 1)
        self.assertEqual(channel.fetch_calls, 0)
        self.assertEqual(rated.context_messages, [])

    async def test_reconnect_does_not_repeat_context_or_stats(self):
        for mode in ("context", "stats"):
            with self.subTest(mode=mode):
                channel = Channel(blocked=True)
                client = DispatchClient(channel)
                task = self.start(mode, client)
                await asyncio.wait_for(channel.read_started.wait(), 2)
                await asyncio.wait_for(client.dispatch_ready(), 2)
                channel.history_release.set()
                result = await asyncio.wait_for(task, 2)
                self.assert_closed(client)
                if mode == "context":
                    self.assertEqual([item.message_id for item in result], [101])
                    self.assertEqual(result[0].context_message_ids, [101])
                    self.assertEqual(channel.fetch_calls, 1)
                    self.assertEqual(channel.history_calls, 2)
                else:
                    self.assertEqual(result["messages_processed"], 1)
                    self.assertEqual(channel.history_calls, 1)
                    with self.sessions() as session:
                        self.assertEqual(session.query(UserStats).one().message_count, 1)

    async def test_cancellation_during_close_and_repeated_cancel_finish_cleanup(self):
        for mode in ("context", "stats"):
            with self.subTest(mode=mode):
                client = DispatchClient(Channel(), block_close=True)
                task = self.start(mode, client)
                await asyncio.wait_for(client.close_started.wait(), 2)
                task.cancel()
                # Cleanup must join the dispatched event, then keep owning close.
                await asyncio.wait_for(client.dispatch_tasks[0], 2)
                self.assertFalse(task.done())
                task.cancel()
                # Drive a duplicate event to yield while the second cancellation lands.
                await asyncio.wait_for(client.dispatch_ready(), 2)
                self.assertFalse(task.done())
                self.assertFalse(client.closed.is_set())
                client.close_release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
                self.assert_closed(client)

    async def test_sdk_swallowed_ready_failure_surfaces_after_close(self):
        for mode in ("context", "stats"):
            with self.subTest(mode=mode):
                client = DispatchClient(Channel(), handler_error=True)
                with self.assertRaisesRegex(RuntimeError, "synthetic ready-handler failure"):
                    await asyncio.wait_for(self.start(mode, client), 2)
                self.assert_closed(client)

    async def test_context_supported_http_errors_skip_message_and_close(self):
        for status in (403, 404, 503):
            with self.subTest(status=status):
                channel = Channel(status=status)
                client = DispatchClient(channel)
                result = await asyncio.wait_for(self.start("context", client), 2)
                self.assertEqual(result, [])
                self.assertEqual(channel.history_calls, 1)
                self.assertEqual(channel.fetch_calls, 0)
                self.assert_closed(client)
