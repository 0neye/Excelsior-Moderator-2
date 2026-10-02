"""Regression tests for transactional user statistics and history collection."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import discord
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import UserStats
from db_config import Base
from user_stats import (
    DEFAULT_WINDOW_SIZE,
    MAX_HISTORY_RETRIES,
    _collect_channel_messages,
    _get_or_create_user_stats,
    process_messages_for_stats,
)


class _Response:
    status = 503
    reason = "Service Unavailable"


def _http_error(status: int) -> discord.HTTPException:
    response = _Response()
    response.status = status
    if status == 403:
        return discord.Forbidden(response, {"message": "Forbidden"})
    if status == 404:
        return discord.NotFound(response, {"message": "Not Found"})
    return discord.HTTPException(response, {"message": "Temporary failure"})


class _HistoryChannel:
    def __init__(self, history):
        self._history = history
        self.calls = []

    async def history(self, *, limit, oldest_first, before=None):
        self.calls.append((limit, oldest_first, before))
        async for item in self._history(limit, before):
            if isinstance(item, Exception):
                raise item
            yield item


class UserStatsTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False)

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_default_rolling_window_commits_sixty_distinct_authors(self):
        session = self.Session()
        messages = [
            SimpleNamespace(
                author=SimpleNamespace(id=user_id, name=f"user-{user_id}", display_name=None),
                content="message",
            )
            for user_id in range(60)
        ]

        process_messages_for_stats(messages, session, window_size=DEFAULT_WINDOW_SIZE)
        session.commit()

        self.assertEqual(session.query(UserStats).count(), 60)
        session.close()

    def test_insert_race_preserves_outer_transaction_work(self):
        self.engine.dispose()
        with tempfile.TemporaryDirectory() as directory:
            engine = create_engine(f"sqlite:///{Path(directory) / 'stats.db'}")
            Base.metadata.create_all(engine)
            session_factory = sessionmaker(bind=engine, autoflush=False)
            session = session_factory()
            session.add(UserStats(user_id=999, username="outer", message_count=7, character_count=0))
            original_begin_nested = session.begin_nested

            def raced_savepoint():
                other_session = session_factory()
                try:
                    other_session.add(
                        UserStats(user_id=1, username="winner", message_count=0, character_count=0)
                    )
                    other_session.commit()
                finally:
                    other_session.close()
                return original_begin_nested()

            with patch.object(session, "begin_nested", side_effect=raced_savepoint):
                stats = _get_or_create_user_stats(session, 1, "loser")
            session.commit()

            self.assertEqual(stats.username, "winner")
            self.assertEqual(session.get(UserStats, 1).user_id, 1)
            self.assertEqual(session.query(UserStats).filter_by(user_id=999).one().message_count, 7)
            session.close()
            Base.metadata.drop_all(engine)
            engine.dispose()

    def test_forbidden_and_not_found_history_fail_without_retries(self):
        for status in (403, 404):
            async def failing_history(limit, before):
                yield _http_error(status)

            channel = _HistoryChannel(failing_history)
            with self.assertRaises((discord.Forbidden, discord.NotFound)):
                asyncio.run(_collect_channel_messages(channel, limit=4))
            self.assertEqual(len(channel.calls), 1)

    def test_transient_history_retries_are_bounded(self):
        async def failing_history(limit, before):
            yield _http_error(503)

        channel = _HistoryChannel(failing_history)
        with patch("user_stats.asyncio.sleep", new=AsyncMock()) as sleep:
            with self.assertRaises(discord.HTTPException):
                asyncio.run(_collect_channel_messages(channel, limit=4))

        self.assertEqual(len(channel.calls), MAX_HISTORY_RETRIES + 1)
        self.assertEqual(sleep.await_count, MAX_HISTORY_RETRIES)

    def test_partial_history_retry_resumes_before_oldest_fetched_message(self):
        messages = [SimpleNamespace(id=message_id) for message_id in (4, 3, 2, 1)]

        async def partial_history(limit, before):
            if before is None:
                yield messages[0]
                yield messages[1]
                yield _http_error(503)
                return
            self.assertEqual(before.id, 3)
            yield messages[2]
            yield messages[3]

        channel = _HistoryChannel(partial_history)
        with patch("user_stats.asyncio.sleep", new=AsyncMock()):
            fetched = asyncio.run(_collect_channel_messages(channel, limit=4))

        self.assertEqual([message.id for message in fetched], [1, 2, 3, 4])
        self.assertEqual(len(channel.calls), 2)


if __name__ == "__main__":
    unittest.main()
