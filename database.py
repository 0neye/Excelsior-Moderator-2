"""SQLAlchemy ORM models for the Discord moderation bot."""

from datetime import datetime, timezone
from typing import Optional

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
    
    # Timestamps
    timestamp = Column(DateTime, nullable=False)  # Original message timestamp
    flagged_at = Column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    
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
    started_at = Column(DateTime, nullable=False, default=datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)


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
    
    # When the user joined the server
    join_timestamp = Column(DateTime, nullable=True)
    
    # Activity metrics
    message_count = Column(Integer, nullable=False, default=0)
    character_count = Column(Integer, nullable=False, default=0)

