"""Regression coverage for the rating uniqueness migration."""

import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

import db_config


class RatingMigrationTests(unittest.TestCase):
    def test_migration_keeps_latest_completed_vote_and_adds_unique_index(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE flagged_messages (id INTEGER PRIMARY KEY)"))
            connection.execute(
                text(
                    "CREATE TABLE flagged_message_ratings ("
                    "id INTEGER PRIMARY KEY, rating_id VARCHAR NOT NULL, "
                    "flagged_message_id INTEGER NOT NULL, rater_user_id INTEGER NOT NULL, "
                    "category VARCHAR, started_at DATETIME NOT NULL, completed_at DATETIME)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO flagged_message_ratings "
                    "(id, rating_id, flagged_message_id, rater_user_id, category, started_at, completed_at) VALUES "
                    "(1, 'completed-old', 101, 1, 'NO_FLAG', '2026-01-01', '2026-01-01'), "
                    "(2, 'abandoned', 101, 1, NULL, '2026-01-03', NULL), "
                    "(3, 'completed-new', 101, 1, 'UNCONSTRUCTIVE', '2026-01-02', '2026-01-02'), "
                    "(4, 'other-rater', 101, 2, 'NO_FLAG', '2026-01-01', '2026-01-01'), "
                    "(5, 'abandoned-old', 102, 1, NULL, '2026-01-01', NULL), "
                    "(6, 'abandoned-new', 102, 1, NULL, '2026-01-02', NULL)"
                )
            )

        with patch.object(db_config, "engine", engine):
            db_config._ensure_backwards_compatible_schema()
            db_config._ensure_backwards_compatible_schema()

        with engine.begin() as connection:
            rows = connection.execute(
                text("SELECT id FROM flagged_message_ratings ORDER BY id")
            ).scalars().all()
            self.assertEqual(rows, [3, 4, 6])
            with self.assertRaises(IntegrityError):
                connection.execute(
                    text(
                        "INSERT INTO flagged_message_ratings "
                        "(id, rating_id, flagged_message_id, rater_user_id, started_at) "
                        "VALUES (7, 'duplicate', 101, 1, '2026-01-03')"
                    )
                )
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
