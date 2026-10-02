"""Regression tests for bootstrap labels and completed-vote reloads."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bootstrapping import RatedMessage, load_features_from_db, prepare_training_data
from database import (
    FeatureExtractionRun,
    FlaggedMessage,
    FlaggedMessageRating,
    MessageFeatures,
    RatingCategory,
)
from db_config import Base


class BootstrapLabelTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.patches = [
            patch("bootstrapping.init_db"),
            patch("bootstrapping.get_session", side_effect=self.session_factory),
        ]
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self.patches):
            active_patch.stop()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _message(self, message_id, category):
        return RatedMessage(
            message_id=message_id,
            channel_id=1,
            guild_id=1,
            author_id=message_id,
            author_name="author",
            content="message",
            timestamp="2026-01-01T00:00:00+00:00",
            flagged_at="2026-01-01T00:00:00+00:00",
            jump_url="",
            category=category,
            rater_user_id=message_id,
            rating_id=str(message_id),
            features=[{"tone_harshness_score": float(message_id)}],
        )

    def test_bootstrap_training_always_uses_binary_labels(self):
        messages = [
            self._message(1, "unsolicited"),
            self._message(2, "unconstructive"),
            self._message(3, "NA"),
            self._message(4, "no-flag"),
            self._message(5, "ambiguous"),
            self._message(6, "unsolicited"),
            self._message(7, "unconstructive"),
            self._message(8, "NA"),
            self._message(9, "no-flag"),
            self._message(10, "ambiguous"),
        ]

        _, _, train_labels, test_labels = prepare_training_data(
            messages,
            active_feature_names=["tone_harshness_score"],
            refresh_stats=False,
        )

        self.assertEqual(set(train_labels) | set(test_labels), {"flag", "no-flag"})

    def test_reload_uses_all_completed_votes_and_skips_abandoned_rows(self):
        session = self.session_factory()
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        message = FlaggedMessage(
            message_id=100,
            author_id=1,
            content="message",
            channel_id=1,
            guild_id=1,
            timestamp=timestamp,
        )
        run = FeatureExtractionRun(provider="test", created_at=timestamp)
        session.add_all([message, run])
        session.flush()
        run_id = run.id
        session.add(MessageFeatures(extraction_run_id=run_id, message_id=100, run_index=0, features={"tone_harshness_score": 1.0}))
        session.add_all(
            [
                FlaggedMessageRating(
                    rating_id="abandoned",
                    flagged_message_id=100,
                    rater_user_id=1,
                    category=None,
                ),
                FlaggedMessageRating(
                    rating_id="completed-flag",
                    flagged_message_id=100,
                    rater_user_id=2,
                    category=RatingCategory.UNCONSTRUCTIVE,
                    completed_at=timestamp,
                ),
                FlaggedMessageRating(
                    rating_id="completed-no-flag",
                    flagged_message_id=100,
                    rater_user_id=3,
                    category=RatingCategory.NO_FLAG,
                    completed_at=timestamp,
                ),
            ]
        )
        session.commit()
        session.close()

        loaded = load_features_from_db(run_id)

        self.assertEqual([(item.rating_id, item.category) for item in loaded], [
            ("completed-flag", "unconstructive"),
            ("completed-no-flag", "no-flag"),
        ])

    def test_training_split_keeps_every_message_on_one_side(self):
        messages = []
        for message_id in range(1, 11):
            category = "unsolicited" if message_id % 2 else "no-flag"
            for rater_id in (1, 2):
                message = self._message(message_id, category)
                message.rater_user_id = rater_id
                message.rating_id = f"{message_id}-{rater_id}"
                message.features = [
                    {"tone_harshness_score": float(message_id)},
                    {"tone_harshness_score": float(message_id) + 0.1},
                ]
                messages.append(message)

        train_features, test_features, train_labels, test_labels = prepare_training_data(
            messages,
            active_feature_names=["tone_harshness_score"],
            refresh_stats=False,
            test_size=0.3,
        )

        train_ids = {int(row[0]) for row in train_features}
        test_ids = {int(row[0]) for row in test_features}
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(set(train_labels), {"flag", "no-flag"})
        self.assertEqual(set(test_labels), {"flag", "no-flag"})

    def test_training_split_rejects_a_single_message(self):
        with self.assertRaisesRegex(ValueError, "two distinct message IDs"):
            prepare_training_data(
                [self._message(1, "unsolicited")],
                active_feature_names=["tone_harshness_score"],
                refresh_stats=False,
            )


if __name__ == "__main__":
    unittest.main()
