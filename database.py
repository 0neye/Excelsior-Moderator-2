"""SQLAlchemy ORM models for the Discord moderation bot."""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Enum as SQLEnum,
    UniqueConstraint,
)
import enum

from db_config import Base


class RatingCategory(enum.Enum):
    """Categories for human ratings of flagged messages."""
    NO_FLAG = "no-flag"
    UNSOLICITED = "unsolicited"
    UNCONSTRUCTIVE = "unconstructive"
    AMBIGUOUS = "ambiguous"
    NA = "NA"


class FlaggedMessage(Base):
    """
    Stores messages that have been flagged by the moderation system.
    
    Includes author info, message content, context message IDs used in the
    flagging decision, and optional target user attribution.
    """
    __tablename__ = "flagged_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(BigInteger, unique=True, nullable=False, index=True)
    
    # Author information
    author_id = Column(BigInteger, nullable=False, index=True)
    author_display_name = Column(String(100), nullable=True)
    author_username = Column(String(50), nullable=True)
    
    # Message details
    content = Column(Text, nullable=False)
    channel_id = Column(BigInteger, nullable=False, index=True)
    guild_id = Column(BigInteger, nullable=False, index=True)
    
    # List of message IDs used as context when deciding to flag
    context_message_ids = Column(JSON, nullable=True, default=list)
    
    # Full context messages data (list of dicts with id, content, author info, etc.)
    # Each dict contains: id, content, author_id, author_name, author_username,
    # timestamp, edited_at, reference_id, attachments, reactions
    context_messages = Column(JSON, nullable=True, default=list)
    
    # Timestamps
    timestamp = Column(DateTime, nullable=False)  # Original message timestamp
    flagged_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    
    # Target user attribution (who the message was directed at, if applicable)
    target_user_id = Column(BigInteger, nullable=True)
    target_display_name = Column(String(100), nullable=True)
    target_username = Column(String(50), nullable=True)


class FlaggedMessageRating(Base):
    """
    Stores human ratings/reviews of flagged messages.
    
    Raters categorize flagged messages and can attribute a target user. Used for continuous training.
    """
    __tablename__ = "flagged_message_ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rating_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Link to the flagged message
    flagged_message_id = Column(BigInteger, ForeignKey("flagged_messages.message_id"), nullable=False, index=True)
    
    # Rater info
    rater_user_id = Column(BigInteger, nullable=False, index=True)
    
    # Rating category
    category = Column(SQLEnum(RatingCategory), nullable=True)
    
    # Target attribution (rater can specify who the flagged message was targeting)
    target_user_id = Column(BigInteger, nullable=True)
    target_display_name = Column(String(100), nullable=True)
    target_username = Column(String(50), nullable=True)
    
    # Timestamps
    started_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)


class LogChannelRatingPost(Base):
    """
    Maps bot messages in the log channel to flagged messages for reaction-based rating.
    
    When a message is flagged, the bot posts to the log channel with emoji reactions
    (1-5) for moderators to rate. This table tracks which bot message corresponds
    to which flagged message.
    """
    __tablename__ = "log_channel_rating_posts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_message_id = Column(BigInteger, unique=True, nullable=False, index=True)
    flagged_message_id = Column(
        BigInteger,
        ForeignKey("flagged_messages.message_id"),
        nullable=False,
        index=True
    )
    posted_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class DictionaryEntry(Base):
    """
    Stores dictionary entries for terms with short and long descriptions.
    Used for game-specific terminology, abbreviations, etc.
    """
    __tablename__ = "dictionary_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    term = Column(String(100), unique=True, nullable=False, index=True)
    short_description = Column(String(255), nullable=False)
    long_description = Column(Text, nullable=True)


class UserStats(Base):
    """
    Tracks user statistics including time in server and message activity.
    """
    __tablename__ = "user_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, unique=True, nullable=False, index=True)
    username = Column(String(50), nullable=False)
    display_name = Column(String(100), nullable=True)
    
    # When the user joined the server
    join_timestamp = Column(DateTime, nullable=True)
    
    # Activity metrics
    message_count = Column(Integer, nullable=False, default=0)
    character_count = Column(Integer, nullable=False, default=0)


class UserCoOccurrence(Base):
    """
    Stores co-occurrence counts for user pairs seen within rolling conversations.
    """
    __tablename__ = "user_co_occurrences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_a_id = Column(BigInteger, nullable=False, index=True)
    user_b_id = Column(BigInteger, nullable=False, index=True)
    co_occurrence_count = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint(
            "user_a_id",
            "user_b_id",
            name="uq_user_co_occurrence_pair",
        ),
    )


class FeatureExtractionRun(Base):
    """
    Groups a set of feature extractions together for versioning and tracking.
    
    Each run represents one execution of the feature extraction pipeline,
    allowing multiple iterations without overwriting previous results.
    """
    __tablename__ = "feature_extraction_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=True)  # Optional user description
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    provider = Column(String(50), nullable=False)  # gemini, openrouter, cerebras
    model_name = Column(String(100), nullable=True)  # Model name for all providers
    runs_per_message = Column(Integer, nullable=False, default=1)
    message_count = Column(Integer, nullable=False, default=0)  # How many messages have features


class MessageFeatures(Base):
    """
    Stores extracted LLM features for a single message.
    
    Features are stored as a JSON dict mapping feature names to float values.
    Multiple rows per message are allowed when runs_per_message > 1 for
    stochastic feature extraction.
    """
    __tablename__ = "message_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Null during production when not part of a bootstrapping run
    extraction_run_id = Column(
        Integer,
        ForeignKey("feature_extraction_runs.id"),
        nullable=True,
        index=True
    )
    message_id = Column(
        BigInteger,
        ForeignKey("flagged_messages.message_id"),
        nullable=False,
        index=True
    )
    run_index = Column(Integer, nullable=False, default=0)  # For multiple stochastic runs
    features = Column(JSON, nullable=False)  # Feature dict mapping names to float values
    target_username = Column(String(50), nullable=True)  # For stat feature refresh
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint(
            "extraction_run_id",
            "message_id",
            "run_index",
            name="uq_extraction_message_run",
        ),
    )
