"""SQLAlchemy database configuration and session management."""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from config import DB_FILE

# Create SQLite engine using the database file from config
engine = create_engine(f"sqlite:///{DB_FILE}", echo=False)

# Session factory for creating database sessions
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Base class for all ORM models
Base = declarative_base()


def get_session():
    """
    Create and return a new database session.
    
    Returns:
        Session: A new SQLAlchemy session instance.
    """
    return SessionLocal()


def init_db():
    """
    Initialize the database by creating all tables defined in the models.
    Should be called once at application startup.
    """
    # Import models to ensure they're registered with Base before creating tables
    import database  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _ensure_backwards_compatible_schema()


def _ensure_backwards_compatible_schema() -> None:
    """
    Apply additive schema migrations required for older SQLite databases.

    This function only adds missing columns so existing deployments keep working
    without dropping or rewriting existing data.
    """
    db_inspector = inspect(engine)
    existing_tables = set(db_inspector.get_table_names())
    if "flagged_messages" not in existing_tables:
        return

    flagged_columns = {
        column_definition["name"]
        for column_definition in db_inspector.get_columns("flagged_messages")
    }

    migration_statements: list[str] = []

    # Default to "acted upon" for historical rows that predate waiver-aware behavior
    if "was_acted_upon" not in flagged_columns:
        migration_statements.append(
            "ALTER TABLE flagged_messages ADD COLUMN was_acted_upon BOOLEAN NOT NULL DEFAULT 1"
        )

    # Historical rows were not waiver-filtered because this marker did not exist yet
    if "waiver_filtered" not in flagged_columns:
        migration_statements.append(
            "ALTER TABLE flagged_messages ADD COLUMN waiver_filtered BOOLEAN NOT NULL DEFAULT 0"
        )

    # Execute each additive migration in order inside a single transaction scope
    if migration_statements:
        with engine.begin() as connection:
            for migration_statement in migration_statements:
                connection.execute(text(migration_statement))

    if "flagged_message_ratings" not in existing_tables:
        return

    uniqueness_columns = {"rater_user_id", "flagged_message_id"}
    has_rating_uniqueness = any(
        set(constraint["column_names"] or []) == uniqueness_columns
        for constraint in db_inspector.get_unique_constraints("flagged_message_ratings")
    ) or any(
        index["unique"] and set(index["column_names"] or []) == uniqueness_columns
        for index in db_inspector.get_indexes("flagged_message_ratings")
    )
    if has_rating_uniqueness:
        return

    with engine.begin() as connection:
        duplicates = connection.execute(
            text(
                "SELECT rater_user_id, flagged_message_id "
                "FROM flagged_message_ratings "
                "GROUP BY rater_user_id, flagged_message_id HAVING COUNT(*) > 1"
            )
        ).mappings()
        for pair in duplicates:
            rows = connection.execute(
                text(
                    "SELECT id FROM flagged_message_ratings "
                    "WHERE rater_user_id = :rater_user_id "
                    "AND flagged_message_id = :flagged_message_id "
                    "ORDER BY CASE WHEN completed_at IS NULL THEN 1 ELSE 0 END, "
                    "completed_at DESC, started_at DESC, id DESC"
                ),
                pair,
            ).scalars()
            for duplicate_id in list(rows)[1:]:
                connection.execute(
                    text("DELETE FROM flagged_message_ratings WHERE id = :id"),
                    {"id": duplicate_id},
                )
        connection.execute(
            text(
                "CREATE UNIQUE INDEX uq_flagged_message_rating_rater_message "
                "ON flagged_message_ratings (rater_user_id, flagged_message_id)"
            )
        )

