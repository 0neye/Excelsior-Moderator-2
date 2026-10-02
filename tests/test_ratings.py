"""Regression coverage for rating upserts and metadata writes."""

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from cogs import rating as rating_module
from cogs.rating import Rating, RatingView, _record_completed_rating
from database import FlaggedMessageRating, LogChannelRatingPost, RatingCategory
from db_config import Base


class _Response:
    async def edit_message(self, **kwargs):
        return None

    async def send_message(self, **kwargs):
        return None


class _Message:
    def __init__(self, on_edit=None):
        self.on_edit = on_edit

    async def edit(self, **kwargs):
        if self.on_edit:
            self.on_edit()


class _Channel:
    guild = None

    def __init__(self, on_fetch=None, on_edit=None, on_send=None):
        self.on_fetch = on_fetch
        self.on_edit = on_edit
        self.on_send = on_send
        self.next_id = 100

    async def fetch_message(self, message_id):
        if self.on_fetch:
            self.on_fetch()
        return _Message(self.on_edit)

    async def send(self, content):
        if self.on_send:
            self.on_send()
        self.next_id += 1
        return SimpleNamespace(id=self.next_id)


class _Bot:
    def __init__(self, session_factory, channel=None):
        self.get_db_session = session_factory
        self.user = SimpleNamespace(id=999)
        self._channel = channel

    def get_channel(self, channel_id):
        return self._channel


class RatingTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False)

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def test_rating_upsert_enforces_one_vote_and_preserves_update_timestamps(self):
        first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        update_time = first_time + timedelta(minutes=1)
        with patch.object(rating_module, "datetime") as clock:
            clock.now.side_effect = [first_time, update_time, update_time]
            first_session = self.Session()
            first = _record_completed_rating(first_session, 1, 101, RatingCategory.NO_FLAG)
            first_session.close()
            update_session = self.Session()
            updated = _record_completed_rating(
                update_session, 1, 101, RatingCategory.UNCONSTRUCTIVE
            )
            update_session.close()
            other_session = self.Session()
            other_rater = _record_completed_rating(
                other_session, 2, 101, RatingCategory.UNSOLICITED
            )
            other_session.close()

        session = self.Session()
        ratings = session.query(FlaggedMessageRating).order_by(FlaggedMessageRating.rater_user_id).all()
        self.assertEqual([rating.rater_user_id for rating in ratings], [1, 2])
        self.assertEqual(ratings[0].category, RatingCategory.UNCONSTRUCTIVE)
        self.assertEqual(ratings[0].started_at, first_time.replace(tzinfo=None))
        self.assertEqual(ratings[0].completed_at, update_time.replace(tzinfo=None))
        self.assertEqual((first, updated, other_rater), (True, False, True))
        constraints = inspect(self.engine).get_unique_constraints("flagged_message_ratings")
        self.assertTrue(any(set(item["column_names"]) == {"rater_user_id", "flagged_message_id"} for item in constraints))
        session.close()

    def test_simultaneous_views_and_reaction_change_keep_one_vote_and_gate_counter(self):
        async def exercise():
            retrain_checks = AsyncMock()
            interaction = SimpleNamespace(user=SimpleNamespace(id=1), response=_Response())
            first_view = RatingView(101, 1, "details", self.Session, AsyncMock(), retrain_checks)
            second_view = RatingView(101, 1, "details", self.Session, AsyncMock(), retrain_checks)
            await asyncio.gather(
                first_view._handle_rating(interaction, RatingCategory.NO_FLAG),
                second_view._handle_rating(interaction, RatingCategory.UNCONSTRUCTIVE),
            )

            session = self.Session()
            session.add(LogChannelRatingPost(bot_message_id=77, flagged_message_id=101))
            session.commit()
            session.close()

            cog = Rating(_Bot(self.Session))
            cog.update_leaderboard = AsyncMock()
            cog._check_and_trigger_retrain = AsyncMock()
            payload = SimpleNamespace(
                emoji=next(iter(rating_module.RATING_EMOJIS)), user_id=1, message_id=77
            )
            self.assertTrue(await cog.handle_log_channel_reaction(payload))
            return retrain_checks, cog._check_and_trigger_retrain

        view_checks, reaction_checks = asyncio.run(exercise())
        session = self.Session()
        ratings = session.query(FlaggedMessageRating).all()
        self.assertEqual(len(ratings), 1)
        self.assertEqual(ratings[0].category, RatingCategory.NO_FLAG)
        self.assertEqual(
            [call.kwargs["is_new_rating"] for call in view_checks.await_args_list],
            [True, False],
        )
        self.assertEqual(reaction_checks.await_args.kwargs["is_new_rating"], False)
        session.close()

    def test_init_rating_channel_keeps_counter_changes_during_fetch_edit_and_send(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "metadata.json"
            metadata_path.write_text(
                json.dumps({"instructions_message_id": 1, "new_ratings_since_retrain": 0}),
                encoding="utf-8",
            )
            with patch.object(rating_module, "RATING_METADATA_FILE", metadata_path):
                channel = _Channel(
                    on_fetch=rating_module._increment_ratings_counter,
                    on_edit=rating_module._increment_ratings_counter,
                    on_send=rating_module._increment_ratings_counter,
                )
                cog = Rating(_Bot(self.Session, channel))
                cog._generate_scoreboard = AsyncMock(return_value="score")
                asyncio.run(cog.init_rating_channel())
                metadata = rating_module._load_rating_metadata()

        self.assertEqual(metadata["new_ratings_since_retrain"], 3)
        self.assertEqual(metadata["last_scoreboard"], "score")
        self.assertIn("leaderboard_message_id", metadata)

    def test_leaderboard_update_keeps_counter_reset_during_fetch_and_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "leaderboard_message_id": 1,
                        "last_scoreboard": "old",
                        "new_ratings_since_retrain": 0,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(rating_module, "RATING_METADATA_FILE", metadata_path):
                channel = _Channel(
                    on_fetch=rating_module._increment_ratings_counter,
                    on_edit=rating_module._reset_ratings_counter,
                )
                cog = Rating(_Bot(self.Session, channel))
                cog._generate_scoreboard = AsyncMock(return_value="new")
                asyncio.run(cog.update_leaderboard())
                metadata = rating_module._load_rating_metadata()

        self.assertEqual(metadata["new_ratings_since_retrain"], 0)
        self.assertEqual(metadata["last_scoreboard"], "new")


if __name__ == "__main__":
    unittest.main()
