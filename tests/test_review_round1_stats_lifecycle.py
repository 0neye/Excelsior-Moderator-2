"""Exercise stats bootstrap through installed py-cord's swallowing dispatcher."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import discord
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import config
with patch.object(config, "DB_FILE", ":memory:"):
    import db_config
    import user_stats
    from database import UserStats


def http_error(status):
    response = SimpleNamespace(status=status, reason="synthetic failure")
    error = {403: discord.Forbidden, 404: discord.NotFound}.get(status, discord.HTTPException)
    return error(response, {"message": "synthetic failure"})


class StatsChannel:
    name = "synthetic"
    threads = []

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0
        self.entered = asyncio.Event()

    async def history(self, **kwargs):
        self.calls += 1
        self.entered.set()
        if self.outcome == "blocked":
            await asyncio.Event().wait()
        if self.outcome == "cancelled":
            raise asyncio.CancelledError()
        if isinstance(self.outcome, int):
            raise http_error(self.outcome)
        for message_id in (2, 1):
            yield SimpleNamespace(id=message_id, content="synthetic", author=SimpleNamespace(id=message_id, name=f"user-{message_id}"))

    async def archived_threads(self, **kwargs):
        if False:
            yield None


class DispatchClient:
    _run_event = discord.Client._run_event

    def __init__(self, channel):
        self.channel = channel
        self.guilds = [SimpleNamespace(get_channel=lambda _: channel)]
        self.user = "synthetic client"
        self.on_error = AsyncMock()
        self.closed = asyncio.Event()
        self.close_count = 0
        self.dispatch_task = None

    def event(self, handler):
        self.handler = handler
        return handler

    async def start(self, token):
        self.dispatch_task = asyncio.create_task(self._run_event(self.handler, "on_ready"))
        # Model the actual connect loop: a swallowed event error does not end start.
        await self.closed.wait()
        await self.dispatch_task

    async def close(self):
        self.close_count += 1
        self.closed.set()


class TrackingSession(Session):
    was_closed = False

    def close(self):
        self.was_closed = True
        super().close()


class ReviewRound1StatsLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.addCleanup(self.engine.dispose)
        db_config.Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, class_=TrackingSession, autoflush=False)
        self.created_sessions = []
        self.enterContext(patch.object(db_config, "engine", self.engine))
        self.enterContext(patch.object(user_stats, "engine", self.engine))
        self.enterContext(patch.object(user_stats, "DISCORD_BOT_TOKEN", "synthetic-token"))
        self.enterContext(patch.object(user_stats, "get_session", self.session))
        self.enterContext(patch.object(user_stats.discord, "TextChannel", StatsChannel))

    def session(self):
        session = self.sessions()
        self.created_sessions.append(session)
        return session

    async def run_bootstrap(self, outcome, **kwargs):
        channel = StatsChannel(outcome)
        client = DispatchClient(channel)
        with patch.object(user_stats.discord, "Client", return_value=client):
            try:
                result = await asyncio.wait_for(user_stats.bootstrap_user_stats(channel_ids=[1], **kwargs), 2)
                return client, result
            finally:
                self.assertEqual(client.close_count, 1)
                self.assertTrue(client.dispatch_task.done())
                client.on_error.assert_not_awaited()
                self.assertTrue(all(session.was_closed for session in self.created_sessions))

    async def test_permanent_history_errors_propagate_after_close(self):
        for status, error_type in ((403, discord.Forbidden), (404, discord.NotFound)):
            with self.subTest(status=status), self.assertRaises(error_type):
                await self.run_bootstrap(status)

    async def test_transient_history_error_is_bounded_and_propagates_after_close(self):
        channel = StatsChannel(503)
        client = DispatchClient(channel)
        with patch.object(user_stats.discord, "Client", return_value=client), patch.object(
            user_stats.asyncio, "sleep", new=AsyncMock()
        ) as sleep:
            with self.assertRaises(discord.HTTPException):
                await asyncio.wait_for(user_stats.bootstrap_user_stats(channel_ids=[1]), 2)
        self.assertEqual(channel.calls, user_stats.MAX_HISTORY_RETRIES + 1)
        self.assertEqual(sleep.await_count, user_stats.MAX_HISTORY_RETRIES)
        self.assertEqual(client.close_count, 1)
        self.assertTrue(client.dispatch_task.done())

    async def test_ready_cancellation_is_reported_after_dispatch_swallows_it(self):
        with self.assertRaises(asyncio.CancelledError):
            await self.run_bootstrap("cancelled")

    async def test_caller_cancellation_closes_and_joins_separate_ready_task(self):
        channel = StatsChannel("blocked")
        client = DispatchClient(channel)
        with patch.object(user_stats.discord, "Client", return_value=client):
            task = asyncio.create_task(user_stats.bootstrap_user_stats(channel_ids=[1]))
            try:
                await asyncio.wait_for(channel.entered.wait(), 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(client.close_count, 1)
        self.assertTrue(client.dispatch_task.done())

    async def test_success_persists_stats_closes_session_and_returns_counts(self):
        client, summary = await self.run_bootstrap("success")
        self.assertEqual(summary, {"channels_processed": 1, "messages_processed": 2, "windows_processed": 1})
        self.assertEqual(len(self.created_sessions), 1)
        with self.sessions() as session:
            self.assertEqual([row.message_count for row in session.query(UserStats).order_by(UserStats.user_id)], [1, 1])

    async def test_processing_failure_closes_session_and_propagates(self):
        def fail_commit():
            raise RuntimeError("synthetic commit failure")

        original_factory = self.sessions
        def failing_factory():
            session = original_factory()
            session.commit = fail_commit
            return session

        self.sessions = failing_factory
        with self.assertRaisesRegex(RuntimeError, "synthetic commit failure"):
            await self.run_bootstrap("success")
        self.assertEqual(len(self.created_sessions), 1)
        self.assertTrue(self.created_sessions[0].was_closed)
