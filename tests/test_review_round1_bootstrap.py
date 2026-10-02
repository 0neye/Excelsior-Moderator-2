"""Offline bootstrap vote/migration and real binary-model publication regressions."""

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import config
with patch.object(config, "DB_FILE", ":memory:"):
    import bootstrapping
    import db_config
    from database import FlaggedMessage, FlaggedMessageRating, FeatureExtractionRun, MessageFeatures, RatingCategory
from ml import load_classifier

LANE = Path(__file__).resolve().parents[1] / "workspace" / "review-fixes" / "sole-review"


def message(message_id=101, rater=1, rating_id="old", category="no-flag", **kwargs):
    return bootstrapping.RatedMessage(
        message_id=message_id, channel_id=1, guild_id=1, author_id=1,
        author_name="User", content="synthetic", timestamp="2026-01-01T00:00:00Z",
        flagged_at="2026-01-01T00:00:00Z", jump_url="", category=category,
        rater_user_id=rater, rating_id=rating_id, **kwargs,
    )


class ReviewRound1VoteTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.addCleanup(self.engine.dispose)
        db_config.Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, autoflush=False)
        self.enterContext(patch.object(db_config, "engine", self.engine))
        self.enterContext(patch.object(bootstrapping, "get_session", self.sessions))
        LANE.mkdir(parents=True, exist_ok=True)
        self.directory = self.enterContext(tempfile.TemporaryDirectory(dir=LANE))

    def votes(self):
        with self.sessions() as session:
            return [(r.rater_user_id, r.rating_id, r.category.value, r.started_at, r.completed_at)
                    for r in session.query(FlaggedMessageRating).order_by(FlaggedMessageRating.rater_user_id)]

    def test_load_save_duplicate_uuids_preserve_different_raters_and_repeat_idempotently(self):
        original = message(rating_started_at="2026-01-02T01:00:00Z", rating_completed_at="2026-01-02T02:00:00Z")
        latest = replace(original, rating_id="new", category="unconstructive", rating_completed_at="2026-01-03T02:00:00Z")
        other = replace(original, rater_user_id=2, rating_id="other")
        abandoned = replace(latest, rating_id="abandoned", category=None)
        path = Path(self.directory) / "ratings.json"
        path.write_text(json.dumps({
            "flagged_messages": {"101": {
                "channel_id": 1, "guild_id": 1, "author_id": 1, "author_name": "User",
                "content": "synthetic", "timestamp": original.timestamp, "flagged_at": original.flagged_at, "jump_url": "",
            }},
            "ratings": {item.rating_id: {
                "flagged_message_id": "101", "rater_user_id": item.rater_user_id, "rating_id": item.rating_id,
                "category": item.category, "started_at": item.rating_started_at, "completed_at": item.rating_completed_at,
            } for item in (latest, other, original, abandoned)},
        }), encoding="utf-8")
        with patch.object(bootstrapping, "RATING_LOG_PATH", path), patch.object(
            bootstrapping, "FLAGGED_MESSAGES_PATH", Path(self.directory) / "missing.json"
        ):
            loaded = bootstrapping.load_rating_data()
        self.assertEqual({item.rating_id for item in loaded}, {"new", "other"})
        # Also exercise save's boundary directly with unnormalized source duplicates.
        bootstrapping.save_to_database([original, other, latest, abandoned])
        expected = self.votes()
        for _ in range(3):
            bootstrapping.save_to_database(loaded)
        self.assertEqual(self.votes(), expected)
        self.assertEqual(expected[0][1:3], ("new", "unconstructive"))
        self.assertEqual(expected[0][4], datetime(2026, 1, 3, 2))
        with self.sessions() as session:
            self.assertEqual(session.query(FlaggedMessage).count(), 1)
            self.assertFalse(session.query(FlaggedMessage).one().pending_log_delivery)

    def test_stale_import_never_overwrites_newer_live_vote_and_fallback_is_stable(self):
        initial = message()
        bootstrapping.save_to_database([initial])
        first = self.votes()
        bootstrapping.save_to_database([initial])
        self.assertEqual(self.votes(), first)
        self.assertEqual(first[0][4], datetime(2026, 1, 1))
        with self.sessions() as session:
            vote = session.query(FlaggedMessageRating).one()
            vote.rating_id = "live"
            vote.category = RatingCategory.UNSOLICITED
            vote.started_at = datetime(2026, 2, 1)
            vote.completed_at = datetime(2026, 2, 2)
            session.commit()
        bootstrapping.save_to_database([initial])
        self.assertEqual(self.votes()[0][1:3], ("live", "unsolicited"))
        newer = replace(initial, rating_id="newer-source", category="unconstructive",
                        rating_started_at="2026-03-01T01:00:00+01:00",
                        rating_completed_at="2026-03-01T02:00:00+01:00")
        bootstrapping.save_to_database([newer])
        self.assertEqual(self.votes()[0][1:3], ("newer-source", "unconstructive"))
        self.assertEqual(self.votes()[0][4], datetime(2026, 3, 1, 1))

    def test_legacy_category_completion_is_repaired_before_dedup_reload_and_reimport(self):
        bootstrapping.save_to_database([message()])
        with self.engine.begin() as connection:
            # Build the legacy schema without its pair uniqueness constraint.
            connection.execute(text("ALTER TABLE flagged_message_ratings RENAME TO saved_ratings"))
            connection.execute(text(
                "CREATE TABLE flagged_message_ratings (id INTEGER PRIMARY KEY, rating_id VARCHAR UNIQUE NOT NULL, "
                "flagged_message_id BIGINT NOT NULL, rater_user_id BIGINT NOT NULL, category VARCHAR, "
                "target_user_id BIGINT, target_display_name VARCHAR, target_username VARCHAR, "
                "started_at DATETIME NOT NULL, completed_at DATETIME)"
            ))
            connection.execute(text(
                "INSERT INTO flagged_message_ratings (id, rating_id, flagged_message_id, rater_user_id, category, started_at, completed_at) VALUES "
                "(1, 'legacy-old', 101, 1, 'NO_FLAG', '2026-01-01 00:00:00', '2026-01-01 00:00:00'), "
                "(2, 'legacy-completed', 101, 1, 'UNCONSTRUCTIVE', '2026-01-02 00:00:00', NULL), "
                "(3, 'abandoned', 101, 1, NULL, '2026-01-03 00:00:00', NULL), "
                "(4, 'other-rater', 101, 2, 'UNSOLICITED', '2026-01-02 00:00:00', NULL), "
                "(5, 'other-abandoned', 101, 3, NULL, '2026-01-04 00:00:00', NULL)"
            ))
            connection.execute(text("DROP TABLE saved_ratings"))
        db_config._ensure_backwards_compatible_schema()
        with self.sessions() as session:
            run = FeatureExtractionRun(provider="synthetic")
            session.add(run)
            session.flush()
            run_id = run.id
            session.add(MessageFeatures(extraction_run_id=run_id, message_id=101, features={"discusses_ellie": 0}))
            session.commit()
        loaded = bootstrapping.load_features_from_db(run_id)
        self.assertEqual({item.rating_id for item in loaded}, {"legacy-completed", "other-rater"})
        with self.sessions() as session:
            session.query(FlaggedMessageRating).filter_by(rating_id="other-rater").update({"completed_at": None})
            session.commit()
        # This repair must also run when the unique index already exists.
        db_config._ensure_backwards_compatible_schema()
        bootstrapping.save_to_database([message(rating_id="legacy-old"), *loaded])
        with self.sessions() as session:
            self.assertEqual(session.query(FlaggedMessageRating).count(), 3)
            self.assertIsNone(session.query(FlaggedMessageRating).filter_by(rater_user_id=3).one().completed_at)
        self.assertEqual(len(bootstrapping.load_features_from_db(run_id)), 2)

    def test_delivery_migration_recovers_only_runtime_evidence(self):
        bootstrapping.save_to_database([message(message_id=i, rating_id=str(i)) for i in range(1, 6)])
        with self.sessions() as session:
            run = FeatureExtractionRun(provider="synthetic")
            session.add(run)
            session.flush()
            session.add_all([
                MessageFeatures(message_id=2, extraction_run_id=run.id, features={}),
                MessageFeatures(message_id=3, features={}),
                MessageFeatures(message_id=4, features={}),
                MessageFeatures(message_id=5, features={}),
            ])
            session.query(FlaggedMessage).filter_by(message_id=4).update({"waiver_filtered": True})
            from database import LogChannelRatingPost
            session.add(LogChannelRatingPost(bot_message_id=99, flagged_message_id=5))
            session.commit()
        with self.engine.begin() as connection:
            connection.execute(text("ALTER TABLE flagged_messages DROP COLUMN pending_log_delivery"))
        for _ in range(2):
            db_config._ensure_backwards_compatible_schema()
        with self.sessions() as session:
            self.assertEqual([row.message_id for row in session.query(FlaggedMessage).filter_by(pending_log_delivery=True)], [3])


class ReviewRound1ModelTests(unittest.TestCase):
    def test_real_binary_fit_save_load_and_rejected_fit_preserve_prior_model(self):
        LANE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=LANE) as directory, patch.object(bootstrapping, "MODEL_SAVE_DIR", Path(directory)):
            X = np.repeat([[0.0], [1.0]], 60, axis=0)
            labels = np.array(["no-flag"] * 60 + ["flag"] * 60)
            model = bootstrapping.train_model(X, labels, feature_names=["tone_harshness_score"])
            path = Path(directory) / "lightgbm_model.joblib"
            prior_bytes = path.read_bytes()
            loaded = load_classifier(path)
            self.assertEqual(list(loaded.predict(np.array([[0.0], [1.0]]))), ["no-flag", "flag"])
            for invalid in (np.array(["flag"] * len(X)), np.array(["no-flag"] * len(X)), np.array(["unsolicited"] * len(X)), np.array([])):
                with self.assertRaisesRegex(ValueError, "require both flag and no-flag"):
                    model.fit(X, invalid)
                self.assertEqual(list(model.predict(np.array([[0.0], [1.0]]))), ["no-flag", "flag"])
            with self.assertRaisesRegex(ValueError, "require both flag and no-flag"):
                bootstrapping.train_model(X, np.array(["no-flag"] * len(X)), feature_names=["tone_harshness_score"])
            self.assertEqual(path.read_bytes(), prior_bytes)

    def test_rare_positive_default_group_split_fails_without_replacing_model(self):
        examples = [message(message_id=i, rating_id=str(i), category="unconstructive" if i == 2 else "no-flag",
                            features=[{"tone_harshness_score": float(i)}] * 3) for i in range(1, 11)]
        LANE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=LANE) as directory, patch.object(bootstrapping, "MODEL_SAVE_DIR", Path(directory)):
            path = Path(directory) / "lightgbm_model.joblib"
            path.write_bytes(b"prior synthetic artifact")
            with self.assertRaisesRegex(ValueError, "Grouped training split requires both"):
                X, _, labels, _ = bootstrapping.prepare_training_data(
                    examples, active_feature_names=["tone_harshness_score"], refresh_stats=False,
                )
                bootstrapping.train_model(X, labels, feature_names=["tone_harshness_score"])
            self.assertEqual(path.read_bytes(), b"prior synthetic artifact")
