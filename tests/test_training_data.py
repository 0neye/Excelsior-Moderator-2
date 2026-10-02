"""Regression tests for binary labels and per-message feature-run selection."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import FeatureExtractionRun, FlaggedMessage, MessageFeatures
from db_config import Base
from training import load_all_features_from_db, prepare_training_data_simple


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.patches = [
            patch("training.init_db"),
            patch("training.get_session", side_effect=self.session_factory),
        ]
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self.patches):
            active_patch.stop()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _message(self, message_id):
        return FlaggedMessage(
            message_id=message_id,
            author_id=message_id,
            content="message",
            channel_id=1,
            guild_id=1,
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

    def test_loads_newest_batch_run_for_each_message_and_keeps_runtime_priority(self):
        session = self.session_factory()
        session.add_all([self._message(message_id) for message_id in (101, 102, 103)])
        full_run = FeatureExtractionRun(
            provider="test",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        subset_run = FeatureExtractionRun(
            provider="test",
            created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )
        session.add_all([full_run, subset_run])
        session.flush()
        session.add_all(
            [
                MessageFeatures(extraction_run_id=full_run.id, message_id=101, run_index=0, features={"v": 10}),
                MessageFeatures(extraction_run_id=full_run.id, message_id=101, run_index=1, features={"v": 11}),
                MessageFeatures(extraction_run_id=full_run.id, message_id=102, run_index=0, features={"v": 20}),
                MessageFeatures(extraction_run_id=subset_run.id, message_id=103, run_index=0, features={"v": 30}),
                MessageFeatures(extraction_run_id=subset_run.id, message_id=103, run_index=1, features={"v": 31}),
                MessageFeatures(message_id=101, run_index=0, features={"v": 99}),
            ]
        )
        session.commit()
        session.close()

        features = load_all_features_from_db()

        self.assertEqual(features, {101: [{"v": 99}], 102: [{"v": 20}], 103: [{"v": 30}, {"v": 31}]})

    def test_continuous_training_always_uses_binary_labels(self):
        ratings = [
            {"message_id": 1, "category": "unsolicited"},
            {"message_id": 2, "category": "unconstructive"},
            {"message_id": 3, "category": "NA"},
            {"message_id": 4, "category": "no-flag"},
            {"message_id": 5, "category": "ambiguous"},
        ]
        features = {message_id: [{"tone": float(message_id)}] for message_id in range(1, 6)}

        _, labels, _ = prepare_training_data_simple(ratings, features, feature_names=["tone"])

        self.assertEqual(labels.tolist(), ["flag", "flag", "no-flag", "no-flag", "no-flag"])


if __name__ == "__main__":
    unittest.main()
