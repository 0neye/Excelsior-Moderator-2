"""SQLAlchemy database configuration and session management."""

from sqlalchemy import create_engine
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

