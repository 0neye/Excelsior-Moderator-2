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

    if not migration_statements:
        return

    # Execute each additive migration in order inside a single transaction scope
    with engine.begin() as connection:
        for migration_statement in migration_statements:
            connection.execute(text(migration_statement))

