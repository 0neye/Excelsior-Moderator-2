"""Bootstrapping script for training moderation models from rated message data.

Provides a REPL menu for running full or partial training pipelines including:
- Loading rated messages from rating_system_log.json
- Fetching Discord context around flagged messages
- Extracting LLM features from message history
- Training LightGBM, MLP, or logistic regression classifiers
- Evaluating model performance with metrics
"""

import asyncio
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import discord
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from config import CHANNEL_ALLOW_LIST, DISCORD_BOT_TOKEN, HISTORY_PER_CHECK
from database import FlaggedMessage, FlaggedMessageRating, RatingCategory
from db_config import get_session, init_db
from llms import extract_features_from_formatted_history
from ml import FEATURE_NAMES, ModerationClassifier, create_classifier
from user_stats import (
    bootstrap_user_stats,
    build_author_id_map,
    build_username_to_id_map,
    ensure_user_stats_schema,
    get_familiarity_score_stat,
    get_seniority_scores,
)

# Configure logging with timestamps and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Path to the rating system log file
DATA_DIR = Path(__file__).parent / "data"
RATING_LOG_PATH = DATA_DIR / "rating_system_log.json"
MODEL_SAVE_DIR = Path(__file__).parent / "models"


@dataclass
class RatedMessage:
    """Container for a flagged message with its rating."""
    message_id: int
    channel_id: int
    guild_id: int
    author_id: int
    author_name: str
    content: str
    timestamp: str
    flagged_at: str
    jump_url: str
    category: str  # Rating category (no-flag, unsolicited, unconstructive, ambiguous, NA)
    rater_user_id: int
    rating_id: str
    channel_name: str | None = None
    thread_name: str | None = None
    context_message_ids: list[int] = field(default_factory=list)
    context_messages: list[dict[str, Any]] = field(default_factory=list)
    # Can hold multiple feature vectors when we repeat extraction for a single rating
    features: list[dict[str, float]] | None = None


@dataclass
class BootstrapState:
    """Holds the current state of the bootstrapping pipeline."""
    rated_messages: list[RatedMessage] = field(default_factory=list)
    messages_with_context: list[RatedMessage] = field(default_factory=list)
    messages_with_features: list[RatedMessage] = field(default_factory=list)
    X_train: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None
    model: Any = None
    model_type: str = "lightgbm"
    collapse_categories: bool = False
    collapse_ambiguous_to_no_flag: bool = False
    active_feature_names: list[str] = field(default_factory=lambda: FEATURE_NAMES.copy())
    ignored_features: set[str] = field(default_factory=set)


# Global state for the REPL
state = BootstrapState()


def ensure_user_stats_ready() -> None:
    """Ensure user stats tables exist before stat-based feature lookups."""
    init_db()
    ensure_user_stats_schema()


def extract_channel_thread_from_context(
    context_messages: list[dict[str, Any]]
) -> tuple[str | None, str | None]:
    """
    Pull channel and thread names from stored context metadata if present.
    
    Args:
        context_messages: Serialized context message payloads
        
    Returns:
        Tuple of (channel_name, thread_name) if available, otherwise (None, None)
    """
    if not context_messages:
        return None, None
    
    first_context = context_messages[0]
    channel_name = first_context.get("channel_name") or first_context.get("parent_channel_name")
    thread_name = first_context.get("thread_name")
    return channel_name, thread_name


def resolve_channel_context(rated_msg: RatedMessage) -> tuple[str, str | None]:
    """
    Resolve channel and thread names for feature extraction with fallbacks.
    
    Prefers values stored directly on the rated message, then falls back to
    the serialized context payload. Ensures a non-empty channel name is
    always returned for the LLM prompt.
    
    Args:
        rated_msg: Rated message containing serialized context
        
    Returns:
        Tuple of (channel_name, thread_name)
    """
    channel_name = rated_msg.channel_name
    thread_name = rated_msg.thread_name
    
    stored_channel, stored_thread = extract_channel_thread_from_context(rated_msg.context_messages)
    if channel_name is None:
        channel_name = stored_channel
    if thread_name is None:
        thread_name = stored_thread
    
    # Ensure channel_name is populated for LLM calls
    if channel_name is None and stored_channel is not None:
        channel_name = stored_channel
    if channel_name is None:
        channel_name = "Unknown Channel"
    
    return channel_name, thread_name


# =============================================================================
# STEP 1: Load Data from rating_system_log.json
# =============================================================================

def load_rating_data() -> list[RatedMessage]:
    """
    Load and parse rated messages from rating_system_log.json.
    
    Joins flagged_messages with their ratings, filtering to only completed ratings.
    
    Returns:
        List of RatedMessage objects with category labels
    """
    logger.info(f"Loading rating data from {RATING_LOG_PATH}")
    
    if not RATING_LOG_PATH.exists():
        logger.error(f"Rating log file not found: {RATING_LOG_PATH}")
        return []
    
    with open(RATING_LOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    flagged_messages = data.get("flagged_messages", {})
    ratings = data.get("ratings", {})
    
    logger.info(f"Found {len(flagged_messages)} flagged messages and {len(ratings)} ratings")
    
    # Build a mapping from flagged_message_id to rating(s)
    message_to_ratings: dict[int, list[dict]] = defaultdict(list)
    for rating_id, rating_data in ratings.items():
        msg_id = rating_data.get("flagged_message_id")
        if msg_id and rating_data.get("category"):  # Only include completed ratings
            message_to_ratings[msg_id].append(rating_data)
    
    logger.info(f"Messages with completed ratings: {len(message_to_ratings)}")
    
    # Join messages with their ratings
    rated_messages: list[RatedMessage] = []
    category_counts: dict[str, int] = defaultdict(int)
    
    for msg_id_str, msg_data in flagged_messages.items():
        msg_id = int(msg_id_str)
        
        # Get ratings for this message
        msg_ratings = message_to_ratings.get(msg_id, [])
        if not msg_ratings:
            continue  # Skip messages without completed ratings
        
        # Use the first rating (could also implement majority voting for multiple ratings)
        rating = msg_ratings[0]
        category = rating["category"]
        category_counts[category] += 1
        
        rated_msg = RatedMessage(
            message_id=msg_id,
            channel_id=msg_data["channel_id"],
            guild_id=msg_data["guild_id"],
            author_id=msg_data["author_id"],
            author_name=msg_data["author_name"],
            content=msg_data["content"],
            timestamp=msg_data["timestamp"],
            flagged_at=msg_data["flagged_at"],
            jump_url=msg_data["jump_url"],
            channel_name=msg_data.get("channel_name"),
            thread_name=msg_data.get("thread_name"),
            category=category,
            rater_user_id=rating["rater_user_id"],
            rating_id=rating["rating_id"],
        )
        rated_messages.append(rated_msg)
    
    # Log category distribution
    logger.info("Category distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"  {category}: {count} ({count/len(rated_messages)*100:.1f}%)")
    
    logger.info(f"Total rated messages loaded: {len(rated_messages)}")
    return rated_messages


# =============================================================================
# STEP 2: Fetch Discord Context
# =============================================================================

async def fetch_discord_context(rated_messages: list[RatedMessage]) -> list[RatedMessage]:
    """
    Fetch message context from Discord API for each rated message.
    
    First checks if context already exists in the database, using cached data where available.
    For messages without cached context, connects to Discord and fetches HISTORY_PER_CHECK
    messages around each flagged message.
    
    Args:
        rated_messages: List of rated messages to fetch context for
        
    Returns:
        List of messages that successfully had context fetched
    """
    logger.info(f"Fetching Discord context for {len(rated_messages)} messages...")
    logger.info(f"Context window size: {HISTORY_PER_CHECK} messages")
    
    # Check database for existing context
    init_db()
    session = get_session()
    
    messages_needing_fetch: list[RatedMessage] = []
    messages_with_context: list[RatedMessage] = []
    
    try:
        for rated_msg in rated_messages:
            existing = session.query(FlaggedMessage).filter_by(
                message_id=rated_msg.message_id
            ).first()
            
            # Use cached context if it exists and has data
            if existing and existing.context_messages:  # type: ignore[truthy-bool]
                rated_msg.context_message_ids = existing.context_message_ids or []  # type: ignore[assignment]
                rated_msg.context_messages = existing.context_messages  # type: ignore[assignment]
                cached_channel_name, cached_thread_name = extract_channel_thread_from_context(
                    rated_msg.context_messages
                )
                rated_msg.channel_name = rated_msg.channel_name or cached_channel_name
                rated_msg.thread_name = rated_msg.thread_name or cached_thread_name
                messages_with_context.append(rated_msg)
                logger.debug(f"Using cached context for message {rated_msg.message_id}")
            else:
                messages_needing_fetch.append(rated_msg)
    finally:
        session.close()
    
    logger.info(f"Found {len(messages_with_context)} messages with cached context")
    logger.info(f"Need to fetch context for {len(messages_needing_fetch)} messages")
    
    # If all messages have cached context, return early
    if not messages_needing_fetch:
        logger.info("All messages have cached context, skipping Discord fetch")
        return messages_with_context
    
    # Set up Discord client with minimal intents
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    
    fetch_errors: dict[str, int] = defaultdict(int)
    
    # List to collect newly fetched messages (accessible in on_ready closure)
    newly_fetched: list[RatedMessage] = []
    
    @client.event
    async def on_ready():
        """Triggered when Discord client is ready."""
        logger.info(f"Connected to Discord as {client.user}")
        logger.info(f"Connected to {len(client.guilds)} guild(s)")
        
        # Group messages by channel to minimize API calls
        messages_by_channel: dict[int, list[RatedMessage]] = defaultdict(list)
        for msg in messages_needing_fetch:
            messages_by_channel[msg.channel_id].append(msg)
        
        logger.info(f"Messages spread across {len(messages_by_channel)} channels")
        
        processed = 0
        for channel_id, channel_messages in messages_by_channel.items():
            logger.info(f"Processing channel {channel_id} ({len(channel_messages)} messages)...")
            
            # Try to get the channel
            channel = client.get_channel(channel_id)
            if channel is None:
                try:
                    channel = await client.fetch_channel(channel_id)
                except discord.NotFound:
                    logger.warning(f"Channel {channel_id} not found, skipping {len(channel_messages)} messages")
                    fetch_errors["channel_not_found"] += len(channel_messages)
                    continue
                except discord.Forbidden:
                    logger.warning(f"No access to channel {channel_id}, skipping {len(channel_messages)} messages")
                    fetch_errors["channel_forbidden"] += len(channel_messages)
                    continue
            
            # Cast to messageable channel type for type checker
            messageable_channel = channel
            if not isinstance(channel, (discord.TextChannel, discord.Thread)):
                logger.warning(f"Channel {channel_id} is not a text channel, skipping")
                continue
            
            # Capture channel/thread names once per channel to reuse for each message
            parent_channel_name: str | None = None
            thread_name: str | None = None
            if isinstance(channel, discord.Thread):
                thread_name = channel.name
                parent_channel_name = channel.parent.name if channel.parent else None
                channel_name = parent_channel_name or channel.name
            else:
                channel_name = channel.name
            
            for rated_msg in channel_messages:
                try:
                    # Fetch messages around the flagged message
                    context_messages: list[discord.Message] = []
                    
                    # Fetch messages before the flagged message
                    before_count = HISTORY_PER_CHECK // 2
                    async for msg in messageable_channel.history(
                        limit=before_count,
                        before=discord.Object(id=rated_msg.message_id),
                        oldest_first=False
                    ):
                        context_messages.append(msg)
                    
                    # Reverse to get chronological order
                    context_messages.reverse()
                    
                    # Fetch the flagged message itself
                    try:
                        flagged_msg = await messageable_channel.fetch_message(rated_msg.message_id)
                        context_messages.append(flagged_msg)
                    except discord.NotFound:
                        logger.warning(f"Flagged message {rated_msg.message_id} not found")
                        fetch_errors["message_not_found"] += 1
                        continue
                    
                    # Fetch messages after the flagged message
                    after_count = HISTORY_PER_CHECK - before_count - 1
                    after_messages: list[discord.Message] = []
                    async for msg in messageable_channel.history(
                        limit=after_count,
                        after=discord.Object(id=rated_msg.message_id),
                        oldest_first=True
                    ):
                        after_messages.append(msg)
                    context_messages.extend(after_messages)
                    
                    # Store context message IDs and serialized data
                    rated_msg.context_message_ids = [m.id for m in context_messages]
                    rated_msg.channel_name = channel_name
                    rated_msg.thread_name = thread_name
                    rated_msg.context_messages = [
                        {
                            "id": m.id,
                            "content": m.content,
                            "author_id": m.author.id,
                            "author_name": m.author.display_name,
                            "author_username": m.author.name,
                            "timestamp": m.created_at.isoformat(),
                            "edited_at": m.edited_at.isoformat() if m.edited_at else None,
                            "reference_id": m.reference.message_id if m.reference else None,
                            "attachments": len(m.attachments) > 0,
                            "reactions": [(str(r.emoji), r.count) for r in m.reactions],
                            "channel_name": channel_name,
                            "parent_channel_name": parent_channel_name,
                            "thread_name": thread_name,
                        }
                        for m in context_messages
                    ]
                    
                    newly_fetched.append(rated_msg)
                    processed += 1
                    
                    if processed % 10 == 0:
                        logger.info(f"Processed {processed}/{len(messages_needing_fetch)} messages")
                    
                    # Rate limiting - small delay between fetches
                    await asyncio.sleep(0.5)
                    
                except discord.errors.RateLimited as e:
                    logger.warning(f"Rate limited, waiting {e.retry_after}s...")
                    await asyncio.sleep(e.retry_after)
                    fetch_errors["rate_limited"] += 1
                except Exception as e:
                    logger.error(f"Error fetching context for message {rated_msg.message_id}: {e}")
                    fetch_errors["other_error"] += 1
        
        logger.info(f"Discord fetch complete: {len(newly_fetched)}/{len(messages_needing_fetch)} successful")
        
        # Log error summary
        if fetch_errors:
            logger.info("Fetch errors:")
            for error_type, count in fetch_errors.items():
                logger.info(f"  {error_type}: {count}")
        
        # Close the client
        await client.close()
    
    # Run the client
    try:
        if DISCORD_BOT_TOKEN is None:
            raise ValueError("DISCORD_BOT_TOKEN is not set in environment")
        await client.start(DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.error(f"Discord client error: {e}")
    
    # Combine cached and newly fetched messages
    messages_with_context.extend(newly_fetched)
    logger.info(f"Total messages with context: {len(messages_with_context)} ({len(messages_with_context) - len(newly_fetched)} cached, {len(newly_fetched)} fetched)")
    
    return messages_with_context


def save_to_database(rated_messages: list[RatedMessage]) -> None:
    """
    Save rated messages and their context to the database.
    
    Args:
        rated_messages: List of rated messages with context to save
    """
    logger.info(f"Saving {len(rated_messages)} messages to database...")
    
    init_db()
    session = get_session()
    
    saved_count = 0
    updated_count = 0
    
    try:
        for rated_msg in rated_messages:
            # Check if message already exists
            existing = session.query(FlaggedMessage).filter_by(
                message_id=rated_msg.message_id
            ).first()
            
            if existing:
                # Update existing record with context data
                existing.context_message_ids = rated_msg.context_message_ids
                existing.context_messages = rated_msg.context_messages
                updated_count += 1
            else:
                # Create new record
                flagged_msg = FlaggedMessage(
                    message_id=rated_msg.message_id,
                    author_id=rated_msg.author_id,
                    author_username=rated_msg.author_name,
                    content=rated_msg.content,
                    channel_id=rated_msg.channel_id,
                    guild_id=rated_msg.guild_id,
                    context_message_ids=rated_msg.context_message_ids,
                    context_messages=rated_msg.context_messages,
                    timestamp=datetime.fromisoformat(rated_msg.timestamp.replace("Z", "+00:00")),
                    flagged_at=datetime.fromisoformat(rated_msg.flagged_at.replace("Z", "+00:00")),
                )
                session.add(flagged_msg)
                saved_count += 1
            
            # Check if rating already exists
            existing_rating = session.query(FlaggedMessageRating).filter_by(
                rating_id=rated_msg.rating_id
            ).first()
            
            if not existing_rating:
                # Map category string to enum
                category_map = {
                    "no-flag": RatingCategory.NO_FLAG,
                    "unsolicited": RatingCategory.UNSOLICITED,
                    "unconstructive": RatingCategory.UNCONSTRUCTIVE,
                    "ambiguous": RatingCategory.AMBIGUOUS,
                    "NA": RatingCategory.NA,
                }
                
                rating = FlaggedMessageRating(
                    rating_id=rated_msg.rating_id,
                    flagged_message_id=rated_msg.message_id,
                    rater_user_id=rated_msg.rater_user_id,
                    category=category_map.get(rated_msg.category),
                )
                session.add(rating)
        
        session.commit()
        logger.info(f"Database save complete: {saved_count} new, {updated_count} updated")
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


# =============================================================================
# STEP 3: Extract LLM Features
# =============================================================================

async def extract_features(
    rated_messages: list[RatedMessage],
    max_concurrent: int = 5,
    runs_per_message: int = 1,
    provider: str = "gemini",
    openrouter_model: str = "openai/gpt-oss-120b"
) -> list[RatedMessage]:
    """
    Extract LLM features from message context using LLM API.
    
    Formats the context messages and calls extract_features_from_formatted_history
    to get feature vectors for each flagged message. Runs API calls in parallel with
    configurable concurrency. When runs_per_message > 1, repeats extraction to collect
    multiple stochastic feature vectors per rated message.
    
    Args:
        rated_messages: List of rated messages with context
        max_concurrent: Maximum number of concurrent API calls
        runs_per_message: How many times to run feature extraction per message
        provider: LLM provider to use ("gemini", "openrouter", or "cerebras")
        openrouter_model: Model to use when provider is "openrouter"
            (Gemini uses the default model configured in llms.py)
        
    Returns:
        List of messages with features extracted
    """
    logger.info(f"Extracting LLM features for {len(rated_messages)} messages...")
    logger.info(f"Provider: {provider}, Max concurrent: {max_concurrent}")
    logger.info(f"Runs per message: {runs_per_message}")
    if provider == "openrouter":
        logger.info(f"OpenRouter model: {openrouter_model}")
    
    # Ensure stats tables are present so stat features can be populated
    ensure_user_stats_ready()
    
    # Filter messages that have context
    messages_to_process = [m for m in rated_messages if m.context_messages]
    skipped = len(rated_messages) - len(messages_to_process)
    if skipped > 0:
        logger.warning(f"Skipping {skipped} messages without context")
    
    # Semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Track results and errors
    results: list[tuple[int, RatedMessage | None]] = []
    error_count = 0
    processed_count = 0
    lock = asyncio.Lock()
    
    async def process_message(idx: int, rated_msg: RatedMessage) -> tuple[int, RatedMessage | None]:
        """Process a single message with semaphore-limited concurrency."""
        nonlocal error_count, processed_count
        
        async with semaphore:
            try:
                # Format context messages for LLM
                formatted_lines = []
                message_id_to_rel_id = {m["id"]: i + 1 for i, m in enumerate(rated_msg.context_messages)}
                
                for i, ctx_msg in enumerate(rated_msg.context_messages):
                    rel_id = i + 1
                    
                    # Build timestamp
                    timestamp = ""
                    if ctx_msg.get("timestamp"):
                        dt = datetime.fromisoformat(ctx_msg["timestamp"].replace("Z", "+00:00"))
                        timestamp = f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                    
                    # Build reply reference
                    reply = ""
                    if ctx_msg.get("reference_id"):
                        ref_rel_id = message_id_to_rel_id.get(ctx_msg["reference_id"])
                        if ref_rel_id:
                            reply = f"[reply to {ref_rel_id}] "
                    
                    # Build content with attachment indicator
                    content = ctx_msg["content"]
                    if ctx_msg.get("attachments"):
                        content += " [uploaded attachment/image]"
                    
                    # Build edited indicator
                    edited = " (edited)" if ctx_msg.get("edited_at") else ""
                    
                    # Build reactions
                    reactions = ""
                    if ctx_msg.get("reactions"):
                        reaction_strs = [f"{emoji} {count}" for emoji, count in ctx_msg["reactions"]]
                        reactions = f"\n[reactions: {', '.join(reaction_strs)}]"
                    
                    author_name = ctx_msg.get("author_username", ctx_msg.get("author_name", "Unknown"))
                    line = f"{timestamp}({rel_id}) {reply}{author_name}: ❝{content}❞{edited}{reactions}"
                    formatted_lines.append(line.strip())
                
                formatted_history = "\n".join(formatted_lines)
                
                # Determine the flagged message's relative ID so we can require it
                flagged_rel_id = message_id_to_rel_id.get(rated_msg.message_id)
                required_indexes = [flagged_rel_id] if flagged_rel_id else None
                message_id_to_author = {
                    str(ctx_msg["id"]): ctx_msg["author_id"]
                    for ctx_msg in rated_msg.context_messages
                    if isinstance(ctx_msg.get("author_id"), int)
                }
                rel_id_to_author = {
                    rel_id: ctx_msg["author_id"]
                    for rel_id, ctx_msg in enumerate(rated_msg.context_messages, start=1)
                    if isinstance(ctx_msg.get("author_id"), int)
                }
                author_id_map = build_author_id_map(message_id_to_author, rel_id_to_author)
                username_id_map = build_username_to_id_map(rated_msg.context_messages)
                
                # Call LLM to extract features with channel/thread context
                channel_name, thread_name = resolve_channel_context(rated_msg)
                
                feature_runs: list[dict[str, float]] = []
                for run_idx in range(runs_per_message):
                    candidates = await extract_features_from_formatted_history(
                        formatted_history,
                        channel_name,
                        thread_name,
                        provider=provider,  # type: ignore[arg-type]
                        openrouter_model=openrouter_model,
                        required_message_indexes=required_indexes,
                        author_id_map=author_id_map,
                        username_id_map=username_id_map,
                        stats_session_factory=get_session,
                    )
                    
                    # Find features for the flagged message in this run
                    flagged_msg_id_str = str(rated_msg.message_id)
                    
                    features_found = False
                    selected_features: dict[str, float] = {}
                    target_username: str | None = None
                    for candidate in candidates:
                        # Match by message_id or relative_id
                        if (candidate.get("message_id") == flagged_msg_id_str or 
                            candidate.get("message_id") == str(flagged_rel_id)):
                            selected_features = candidate.get("features", {})
                            target_username = candidate.get("target_username")
                            features_found = True
                            break
                    
                    if not features_found:
                        # If LLM didn't identify the flagged message as a candidate, use zero features
                        logger.warning(
                            f"LLM did not identify message {rated_msg.message_id} as candidate on run {run_idx + 1}, using zero features"
                        )
                        selected_features = {name: 0.0 for name in FEATURE_NAMES}
                    
                    # Store target_username in features for later stat refreshes
                    if target_username:
                        selected_features["target_username"] = target_username  # type: ignore[assignment]
                    
                    feature_runs.append(selected_features)
                
                rated_msg.features = feature_runs
                
                # Update progress
                async with lock:
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"Processed {processed_count}/{len(messages_to_process)} messages")
                
                return (idx, rated_msg)
                
            except Exception as e:
                logger.error(f"Error extracting features for message {rated_msg.message_id}: {e}")
                async with lock:
                    error_count += 1
                return (idx, None)
    
    # Create tasks for all messages
    tasks = [process_message(i, msg) for i, msg in enumerate(messages_to_process)]
    
    # Run all tasks concurrently (semaphore limits actual concurrency)
    results = await asyncio.gather(*tasks)
    
    # Collect successful results, preserving original order
    messages_with_features: list[RatedMessage] = []
    for idx, result in sorted(results, key=lambda x: x[0]):
        if result is not None:
            messages_with_features.append(result)
    
    logger.info(f"Feature extraction complete: {len(messages_with_features)} successful, {error_count} errors")
    return messages_with_features


# =============================================================================
# STEP 4: Train Model
# =============================================================================

def collapse_category_label(
    category: str,
    collapse_ambiguous_to_no_flag: bool = False
) -> str:
    """
    Normalize category labels when collapsing classes.
    
    Args:
        category: Original rating category
        collapse_ambiguous_to_no_flag: Whether to map ambiguous to no-flag
        
    Returns:
        Collapsed category label following the mapping rules
    """
    if category in {"NA", "no-flag"}:
        return "no-flag"
    if category in {"unsolicited", "unconstructive"}:
        return "flag"
    if collapse_ambiguous_to_no_flag and category == "ambiguous":
        return "no-flag"
    return category


def refresh_stat_features(messages_with_features: list[RatedMessage]) -> None:
    """
    Refresh seniority and familiarity stats from the database for all messages.
    
    Updates the stat-based features (seniority_score_messages, seniority_score_characters,
    familiarity_score_stat) in-place using the current user_stats and user_co_occurrences tables.
    This allows training to use freshly collected stats without re-running LLM feature extraction.
    """
    ensure_user_stats_ready()
    session = get_session()
    updated_count = 0
    missing_target_count = 0
    missing_author_count = 0
    total_vectors = 0
    
    try:
        for rated_msg in messages_with_features:
            if rated_msg.features is None or not rated_msg.context_messages:
                continue
            
            # Build username-to-ID mapping from context messages
            username_id_map = build_username_to_id_map(rated_msg.context_messages)
            
            # Build author ID map from context messages
            rel_id_to_author = {
                rel_id: ctx_msg["author_id"]
                for rel_id, ctx_msg in enumerate(rated_msg.context_messages, start=1)
                if isinstance(ctx_msg.get("author_id"), int)
            }
            message_id_to_author = {
                str(ctx_msg["id"]): ctx_msg["author_id"]
                for ctx_msg in rated_msg.context_messages
                if isinstance(ctx_msg.get("author_id"), int)
            }
            author_id_map = build_author_id_map(message_id_to_author, rel_id_to_author)
            
            # Find the flagged message's relative ID
            flagged_rel_id = None
            for rel_id, ctx_msg in enumerate(rated_msg.context_messages, start=1):
                if ctx_msg.get("id") == rated_msg.message_id:
                    flagged_rel_id = rel_id
                    break
            
            # Get author ID from the flagged message
            author_id = author_id_map.get(str(rated_msg.message_id))
            if author_id is None and flagged_rel_id is not None:
                author_id = author_id_map.get(str(flagged_rel_id))
            
            # Update each feature payload
            for feature_payload in rated_msg.features:
                total_vectors += 1
                
                # Try to find target user from the stored target_username if available
                target_username = feature_payload.get("target_username")
                target_id = (
                    username_id_map.get(target_username)  # type: ignore[arg-type]
                    if isinstance(target_username, str)
                    else None
                )
                
                # Compute fresh stats if we have both author and target
                if author_id is not None and target_id is not None:
                    msg_score, char_score = get_seniority_scores(author_id, target_id, session)
                    fam_score = get_familiarity_score_stat(author_id, target_id, session)
                    feature_payload["seniority_score_messages"] = msg_score
                    feature_payload["seniority_score_characters"] = char_score
                    feature_payload["familiarity_score_stat"] = fam_score
                    updated_count += 1
                else:
                    # Track why we couldn't update
                    if author_id is None:
                        missing_author_count += 1
                    elif target_id is None:
                        missing_target_count += 1
                    
                    # Ensure keys exist with default values
                    feature_payload.setdefault("seniority_score_messages", 0.0)
                    feature_payload.setdefault("seniority_score_characters", 0.0)
                    feature_payload.setdefault("familiarity_score_stat", 0.0)
    finally:
        session.close()
    
    logger.info(f"Refreshed stat features for {updated_count}/{total_vectors} feature vectors")
    if missing_target_count > 0:
        logger.warning(
            f"  {missing_target_count} vectors missing target_username (re-extract features to fix)"
        )
    if missing_author_count > 0:
        logger.warning(f"  {missing_author_count} vectors missing author_id")


def build_feature_subset(
    ignored_features: set[str] | None = None,
    available_features: list[str] | None = None
) -> tuple[list[str], list[str], list[str]]:
    """
    Compute active features after applying an ignore list.
    
    Args:
        ignored_features: Feature names to exclude from training
        available_features: Baseline ordered feature list (defaults to FEATURE_NAMES)
        
    Returns:
        Tuple of (active_features, applied_ignored, invalid_requested)
    """
    base_features = available_features or FEATURE_NAMES
    ignored_set = {feat.strip() for feat in ignored_features} if ignored_features else set()
    invalid_requested = sorted(ignored_set - set(base_features))
    applied_ignored = sorted(ignored_set & set(base_features))
    active_features = [feat for feat in base_features if feat not in applied_ignored]
    
    if not active_features:
        raise ValueError("At least one feature must remain active for training")
    
    return active_features, applied_ignored, invalid_requested


def prepare_training_data(
    messages_with_features: list[RatedMessage],
    test_size: float = 0.2,
    random_state: int = 42,
    collapse_categories: bool = False,
    collapse_ambiguous_to_no_flag: bool = False,
    refresh_stats: bool = True,
    active_feature_names: list[str] | None = None,
    ignored_features: set[str] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare feature matrices and labels for training.
    
    Args:
        messages_with_features: List of messages with extracted features
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        collapse_categories: Whether to merge rating categories into broader buckets
        collapse_ambiguous_to_no_flag: Whether to map ambiguous to no-flag when collapsing
        refresh_stats: Whether to refresh seniority/familiarity stats from the database
        active_feature_names: Ordered feature list to include (defaults to FEATURE_NAMES)
        ignored_features: Feature names to drop (ignored if active_feature_names supplied)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Preparing training data from {len(messages_with_features)} messages...")
    
    # Refresh stat features from database if requested
    if refresh_stats:
        refresh_stat_features(messages_with_features)
    
    applied_ignored: list[str] = []
    invalid_requested: list[str] = []
    
    if active_feature_names is not None:
        # Validate provided active feature list
        invalid_requested = [feat for feat in active_feature_names if feat not in FEATURE_NAMES]
        if invalid_requested:
            logger.warning("Dropping unknown feature(s) from active list: %s", ", ".join(invalid_requested))
        active_feature_names = [feat for feat in active_feature_names if feat in FEATURE_NAMES]
        applied_ignored = sorted(ignored_features) if ignored_features else []
        if not active_feature_names:
            raise ValueError("Active feature list is empty after validation")
    else:
        try:
            # Determine which features will be used for vector construction
            active_feature_names, applied_ignored, invalid_requested = build_feature_subset(
                ignored_features=ignored_features,
                available_features=FEATURE_NAMES,
            )
        except ValueError as exc:
            logger.error(str(exc))
            raise
    
    if applied_ignored:
        logger.info("Ignoring %d feature(s): %s", len(applied_ignored), ", ".join(applied_ignored))
    if invalid_requested:
        logger.warning("Requested invalid feature(s): %s", ", ".join(invalid_requested))
    
    def filter_discusses_ellie_messages(
        messages: list[RatedMessage],
        threshold: float = 0.2
    ) -> list[RatedMessage]:
        """
        Drop messages that mention the Ellie topic above a threshold.
        
        Args:
            messages: Messages with extracted features
            threshold: Score cutoff for excluding a message
            
        Returns:
            Filtered messages safe for model training
        """
        filtered: list[RatedMessage] = []
        removed_count = 0
        
        for msg in messages:
            if msg.features is None:
                filtered.append(msg)
                continue
            
            feature_payloads = msg.features if isinstance(msg.features, list) else [msg.features]
            exceeds_threshold = any(
                payload.get("discusses_ellie", 0.0) > threshold for payload in feature_payloads
            )
            
            if exceeds_threshold:
                removed_count += 1
                continue
            
            filtered.append(msg)
        
        logger.info(
            "Filtered %d messages with discusses_ellie > %.2f (remaining: %d)",
            removed_count,
            threshold,
            len(filtered),
        )
        return filtered

    # Remove high discusses_ellie messages to keep training data clean
    messages_with_features = filter_discusses_ellie_messages(messages_with_features)

    # Build feature matrix
    X = []
    y = []
    total_feature_vectors = 0
    
    for msg in messages_with_features:
        if msg.features is None:
            continue
        
        feature_payloads = msg.features if isinstance(msg.features, list) else [msg.features]
        
        for feature_payload in feature_payloads:
            # Extract features in consistent order
            feature_vector = [feature_payload.get(name, 0.0) for name in active_feature_names]
            X.append(feature_vector)
            
            label = msg.category
            if collapse_categories:
                label = collapse_category_label(
                    label,
                    collapse_ambiguous_to_no_flag=collapse_ambiguous_to_no_flag
                )
            y.append(label)
            total_feature_vectors += 1
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(
        "Feature matrix shape: %s built from %d feature vectors using %d features",
        X.shape,
        total_feature_vectors,
        len(active_feature_names),
    )
    label_distribution = dict(zip(*np.unique(y, return_counts=True)))
    distribution_label = "collapsed label distribution" if collapse_categories else "label distribution"
    logger.info(f"{distribution_label}: {label_distribution}")
    
    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set size: {len(y_train)}")
    logger.info(f"Test set size: {len(y_test)}")
    
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "lightgbm",
    feature_names: list[str] | None = None
) -> ModerationClassifier:
    """
    Train a classifier on the training data.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        model_type: Type of model to train ('lightgbm', 'mlp', or 'logistic')
        feature_names: Ordered feature names used for training
        
    Returns:
        Trained classifier
    """
    logger.info(
        "Training %s model with %d features...",
        model_type,
        len(feature_names or FEATURE_NAMES),
    )
    
    model = create_classifier(model_type)
    # Attach active feature names for downstream importance reporting
    if feature_names:
        model.feature_names = feature_names  # type: ignore[attr-defined]
    model.fit(X_train, y_train)
    
    # Save model
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    model_path = MODEL_SAVE_DIR / f"{model_type}_model.joblib"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model


# =============================================================================
# STEP 5: Evaluate Model
# =============================================================================

def evaluate_model(
    model: ModerationClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict[str, Any]:
    """
    Evaluate model performance on test set.
    
    Computes confusion matrix, balanced accuracy, per-class metrics,
    and feature importance.
    
    Args:
        model: Trained classifier
        X_test: Test feature matrix
        y_test: Test labels
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    logger.info("Evaluating model performance...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    _ = model.predict_proba(X_test)  # Computed but not displayed in this version
    
    # Compute metrics
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    feature_importance = model.get_feature_importance()
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    
    logger.info(f"\nBalanced Accuracy: {bal_acc:.4f}")
    
    logger.info("\nConfusion Matrix:")
    classes = sorted(set(y_test))
    # Header
    header = "Actual\\Pred  " + "  ".join(f"{c[:8]:>8}" for c in classes)
    logger.info(header)
    # Rows
    for i, actual_class in enumerate(classes):
        row = f"{actual_class[:12]:<12} " + "  ".join(f"{conf_matrix[i][j]:>8}" for j in range(len(classes)))
        logger.info(row)
    
    logger.info("\nPer-Class Metrics:")
    logger.info(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    logger.info("-" * 55)
    for cls in classes:
        if cls in class_report:
            metrics = class_report[cls]
            # Access dict values safely
            precision = metrics["precision"] if isinstance(metrics, dict) else 0.0
            recall = metrics["recall"] if isinstance(metrics, dict) else 0.0
            f1 = metrics["f1-score"] if isinstance(metrics, dict) else 0.0
            support = int(metrics["support"]) if isinstance(metrics, dict) else 0
            logger.info(
                f"{cls:<15} {precision:>10.4f} {recall:>10.4f} "
                f"{f1:>10.4f} {support:>10}"
            )
    
    logger.info("\nFeature Importance (sorted):")
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for name, importance in sorted_importance:
        bar = "█" * int(importance * 50)
        logger.info(f"  {name:<35} {importance:.4f} {bar}")
    
    logger.info(f"\n{'='*60}")
    
    return {
        "balanced_accuracy": bal_acc,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "feature_importance": feature_importance,
        "classes": classes,
    }


# =============================================================================
# REPL Menu
# =============================================================================

def print_menu():
    """Print the REPL menu options."""
    print("\n" + "=" * 50)
    print("       BOOTSTRAPPING PIPELINE MENU")
    print("=" * 50)
    print("1. Run full pipeline (load -> fetch -> extract -> train -> eval)")
    print("2. Load data only (from rating_system_log.json)")
    print("3. Fetch Discord context (requires bot connection)")
    print("4. Extract LLM features (from stored context)")
    print("5. Train model (LightGBM, MLP, or logistic regression)")
    print("6. Evaluate model (on held-out test set)")
    print("7. Show current state")
    print("8. Save/Load state to file")
    print("9. Exit")
    print("10. Collect user statistics from Discord (channels & threads)")
    print("=" * 50)


def print_state():
    """Print the current pipeline state."""
    print("\n--- Current Pipeline State ---")
    print(f"Rated messages loaded: {len(state.rated_messages)}")
    print(f"Messages with context: {len(state.messages_with_context)}")
    print(f"Messages with features: {len(state.messages_with_features)}")
    total_feature_vectors = sum(len(msg.features or []) for msg in state.messages_with_features)
    if total_feature_vectors:
        print(f"Feature vectors available: {total_feature_vectors}")
    print(f"Training data prepared: {state.X_train is not None}")
    if state.X_train is not None and state.y_train is not None and state.y_test is not None:
        print(f"  Train samples: {len(state.y_train)}")
        print(f"  Test samples: {len(state.y_test)}")
    print(f"Model trained: {state.model is not None}")
    print(f"Model type: {state.model_type}")
    print(f"Collapse categories: {state.collapse_categories}")
    if state.collapse_categories:
        print(f"  Collapse ambiguous to no-flag: {state.collapse_ambiguous_to_no_flag}")
    print(
        f"Active features: {len(state.active_feature_names)} "
        f"(ignored: {len(state.ignored_features)})"
    )
    if state.ignored_features:
        print(f"  Ignored features: {', '.join(sorted(state.ignored_features))}")
    print("---")


async def run_full_pipeline(
    max_concurrent: int = 5,
    runs_per_message: int = 1,
    provider: str = "gemini",
    openrouter_model: str = "openai/gpt-oss-120b",
    collapse_categories: bool = False,
    collapse_ambiguous_to_no_flag: bool = False
):
    """
    Run the complete bootstrapping pipeline.
    
    Args:
        max_concurrent: Maximum concurrent API calls for feature extraction
        runs_per_message: How many times to extract features per rated message
        provider: LLM provider ("gemini" default, or "openrouter"/"cerebras")
        openrouter_model: Model to use with OpenRouter
        collapse_categories: Whether to collapse rating categories before training
        collapse_ambiguous_to_no_flag: Whether to map ambiguous to no-flag when collapsing
    """
    logger.info("Starting full bootstrapping pipeline...")
    # Full pipeline always starts with all features unless overridden later
    state.ignored_features = set()
    state.active_feature_names = FEATURE_NAMES.copy()
    
    # Step 1: Load data
    state.rated_messages = load_rating_data()
    if not state.rated_messages:
        logger.error("No rated messages found, aborting pipeline")
        return
    
    # Step 2: Fetch Discord context (uses DB cache when available)
    state.messages_with_context = await fetch_discord_context(state.rated_messages)
    if not state.messages_with_context:
        logger.error("No context fetched, aborting pipeline")
        return
    
    # Save to database
    save_to_database(state.messages_with_context)
    
    # Step 3: Extract features (Gemini by default, concurrent requests)
    state.messages_with_features = await extract_features(
        state.messages_with_context,
        max_concurrent=max_concurrent,
        runs_per_message=runs_per_message,
        provider=provider,
        openrouter_model=openrouter_model
    )
    if not state.messages_with_features:
        logger.error("No features extracted, aborting pipeline")
        return
    
    # Step 4: Prepare data and train
    logger.info(f"Collapsing categories for training: {collapse_categories}")
    X_train, X_test, y_train, y_test = prepare_training_data(
        state.messages_with_features,
        collapse_categories=collapse_categories,
        collapse_ambiguous_to_no_flag=collapse_ambiguous_to_no_flag,
        active_feature_names=state.active_feature_names,
        ignored_features=state.ignored_features
    )
    state.X_train = X_train
    state.X_test = X_test
    state.y_train = y_train
    state.y_test = y_test
    state.collapse_categories = collapse_categories
    state.collapse_ambiguous_to_no_flag = collapse_ambiguous_to_no_flag
    if collapse_categories:
        logger.info(
            "Collapse ambiguous into no-flag enabled: %s",
            collapse_ambiguous_to_no_flag
        )
    
    state.model = train_model(
        X_train,
        y_train,
        state.model_type,
        feature_names=state.active_feature_names,
    )
    
    # Step 5: Evaluate
    evaluate_model(state.model, X_test, y_test)
    
    # Step 6: Persist pipeline state for resumption/debugging
    save_state_to_file()
    
    logger.info("Full pipeline complete!")


def save_state_to_file(filepath: str = "bootstrap_state.json"):
    """Save current state to a JSON file for later resumption."""
    logger.info(f"Saving state to {filepath}...")
    
    # Serialize rated messages with context
    serializable_messages = []
    for msg in state.messages_with_features or state.messages_with_context or state.rated_messages:
        serializable_messages.append({
            "message_id": msg.message_id,
            "channel_id": msg.channel_id,
            "guild_id": msg.guild_id,
            "author_id": msg.author_id,
            "author_name": msg.author_name,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "flagged_at": msg.flagged_at,
            "jump_url": msg.jump_url,
            "channel_name": msg.channel_name,
            "thread_name": msg.thread_name,
            "category": msg.category,
            "rater_user_id": msg.rater_user_id,
            "rating_id": msg.rating_id,
            "context_message_ids": msg.context_message_ids,
            "context_messages": msg.context_messages,
            "features": msg.features,
        })
    
    state_data = {
        "messages": serializable_messages,
        "model_type": state.model_type,
        "collapse_categories": state.collapse_categories,
        "collapse_ambiguous_to_no_flag": state.collapse_ambiguous_to_no_flag,
        "active_feature_names": state.active_feature_names,
        "ignored_features": sorted(state.ignored_features),
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)
    
    logger.info(f"State saved: {len(serializable_messages)} messages")


def load_state_from_file(filepath: str = "bootstrap_state.json"):
    """Load state from a JSON file."""
    logger.info(f"Loading state from {filepath}...")
    
    if not Path(filepath).exists():
        logger.error(f"State file not found: {filepath}")
        return
    
    with open(filepath, "r", encoding="utf-8") as f:
        state_data = json.load(f)
    
    # Deserialize messages
    messages = []
    for msg_data in state_data["messages"]:
        raw_features = msg_data.get("features")
        normalized_features: list[dict[str, float]] | None = None
        if isinstance(raw_features, list):
            normalized_features = raw_features
        elif isinstance(raw_features, dict):
            # Backward compatibility for states saved before multi-run support
            normalized_features = [raw_features]
        
        msg = RatedMessage(
            message_id=msg_data["message_id"],
            channel_id=msg_data["channel_id"],
            guild_id=msg_data["guild_id"],
            author_id=msg_data["author_id"],
            author_name=msg_data["author_name"],
            content=msg_data["content"],
            timestamp=msg_data["timestamp"],
            flagged_at=msg_data["flagged_at"],
            jump_url=msg_data["jump_url"],
            channel_name=msg_data.get("channel_name"),
            thread_name=msg_data.get("thread_name"),
            category=msg_data["category"],
            rater_user_id=msg_data["rater_user_id"],
            rating_id=msg_data["rating_id"],
            context_message_ids=msg_data.get("context_message_ids", []),
            context_messages=msg_data.get("context_messages", []),
            features=normalized_features,
        )
        messages.append(msg)
    
    # Categorize by what data is available
    state.rated_messages = messages
    state.messages_with_context = [m for m in messages if m.context_messages]
    state.messages_with_features = [m for m in messages if m.features]
    state.model_type = state_data.get("model_type", "lightgbm")
    state.collapse_categories = state_data.get("collapse_categories", False)
    state.collapse_ambiguous_to_no_flag = state_data.get("collapse_ambiguous_to_no_flag", False)
    state.active_feature_names = state_data.get("active_feature_names", FEATURE_NAMES)
    state.ignored_features = set(state_data.get("ignored_features", []))
    
    logger.info(f"State loaded: {len(messages)} messages")
    logger.info(f"  With context: {len(state.messages_with_context)}")
    logger.info(f"  With features: {len(state.messages_with_features)}")


async def repl():
    """Run the interactive REPL menu."""
    print("\nWelcome to the Bootstrapping Pipeline!")
    print("This tool helps train moderation models from rated message data.")
    
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter choice (1-9): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        
        if choice == "1":
            # Full pipeline
            print("\nFeature extraction settings for full pipeline:")
            print("1. gemini (default)")
            print("2. openrouter")
            print("3. cerebras")
            provider_choice = input("Select provider (1-3, default 1): ").strip()
            if provider_choice == "2":
                provider = "openrouter"
            elif provider_choice == "3":
                provider = "cerebras"
            else:
                provider = "gemini"
            
            openrouter_model = "openai/gpt-oss-120b"
            if provider == "openrouter":
                model_input = input(f"OpenRouter model (default: {openrouter_model}): ").strip()
                if model_input:
                    openrouter_model = model_input
            
            max_concurrent_input = input("Max concurrent API calls (default: 5): ").strip()
            max_concurrent = int(max_concurrent_input) if max_concurrent_input.isdigit() else 5
            runs_input = input("Feature extraction runs per message (default: 1): ").strip()
            runs_per_message = int(runs_input) if runs_input.isdigit() else 1
            collapse_input = input("Collapse categories for training? (y/N): ").strip().lower()
            collapse_categories = collapse_input in {"y", "yes"}
            collapse_ambiguous_to_no_flag = False
            if collapse_categories:
                collapse_ambiguous_input = input(
                    "Collapse ambiguous into no-flag? (y/N): "
                ).strip().lower()
                collapse_ambiguous_to_no_flag = collapse_ambiguous_input in {"y", "yes"}
            
            print("\nSelect model type for training:")
            print("1. LightGBM (faster, interpretable)")
            print("2. MLP (neural network)")
            print("3. Logistic regression (linear baseline)")
            model_choice_full = input("Enter choice (1-3): ").strip()
            if model_choice_full == "2":
                state.model_type = "mlp"
            elif model_choice_full == "3":
                state.model_type = "logistic"
            else:
                state.model_type = "lightgbm"

            await run_full_pipeline(
                max_concurrent=max_concurrent,
                runs_per_message=runs_per_message,
                provider=provider,
                openrouter_model=openrouter_model,
                collapse_categories=collapse_categories,
                collapse_ambiguous_to_no_flag=collapse_ambiguous_to_no_flag
            )
            
        elif choice == "2":
            # Load data only
            state.rated_messages = load_rating_data()
            print(f"Loaded {len(state.rated_messages)} rated messages")
            
        elif choice == "3":
            # Fetch Discord context
            if not state.rated_messages:
                print("No messages loaded. Run option 2 first.")
                continue
            state.messages_with_context = await fetch_discord_context(state.rated_messages)
            if state.messages_with_context:
                save_to_database(state.messages_with_context)
            print(f"Fetched context for {len(state.messages_with_context)} messages")
            
        elif choice == "4":
            # Extract features
            messages = state.messages_with_context or state.rated_messages
            if not messages:
                print("No messages available. Run options 2-3 first.")
                continue
            
            # Prompt for extraction settings
            print("\nFeature extraction settings:")
            print("1. gemini (default)")
            print("2. openrouter")
            print("3. cerebras")
            provider_choice = input("Select provider (1-3, default 1): ").strip()
            if provider_choice == "2":
                provider = "openrouter"
            elif provider_choice == "3":
                provider = "cerebras"
            else:
                provider = "gemini"
            
            openrouter_model = "openai/gpt-oss-120b"
            if provider == "openrouter":
                model_input = input(f"OpenRouter model (default: {openrouter_model}): ").strip()
                if model_input:
                    openrouter_model = model_input
            
            max_concurrent_input = input("Max concurrent API calls (default: 5): ").strip()
            max_concurrent = int(max_concurrent_input) if max_concurrent_input.isdigit() else 5
            runs_input = input("Feature extraction runs per message (default: 1): ").strip()
            runs_per_message = int(runs_input) if runs_input.isdigit() else 1
            
            state.messages_with_features = await extract_features(
                messages,
                max_concurrent=max_concurrent,
                runs_per_message=runs_per_message,
                provider=provider,
                openrouter_model=openrouter_model
            )
            print(f"Extracted features for {len(state.messages_with_features)} messages")
            
        elif choice == "5":
            # Train model
            if not state.messages_with_features:
                print("No features available. Run options 2-4 first.")
                continue
            
            # Ask for model type
            print("\nSelect model type:")
            print("1. LightGBM (faster, interpretable)")
            print("2. MLP (neural network)")
            print("3. Logistic regression (linear baseline)")
            model_choice = input("Enter choice (1-3): ").strip()
            if model_choice == "2":
                state.model_type = "mlp"
            elif model_choice == "3":
                state.model_type = "logistic"
            else:
                state.model_type = "lightgbm"
            
            collapse_input = input("Collapse categories for training? (y/N): ").strip().lower()
            collapse_categories = collapse_input in {"y", "yes"}
            state.collapse_categories = collapse_categories
            collapse_ambiguous_to_no_flag = False
            if collapse_categories:
                collapse_ambiguous_input = input(
                    "Collapse ambiguous into no-flag? (y/N): "
                ).strip().lower()
                collapse_ambiguous_to_no_flag = collapse_ambiguous_input in {"y", "yes"}
            state.collapse_ambiguous_to_no_flag = collapse_ambiguous_to_no_flag
            
            # Prompt for feature subset
            print("\nAvailable features:")
            print(", ".join(FEATURE_NAMES))
            ignore_input = input(
                "Features to ignore for training (comma-separated, blank for none): "
            ).strip()
            requested_ignored = {
                feat.strip() for feat in ignore_input.split(",") if feat.strip()
            } if ignore_input else set()
            
            try:
                active_feature_names, applied_ignored, invalid_requested = build_feature_subset(
                    ignored_features=requested_ignored,
                    available_features=FEATURE_NAMES,
                )
            except ValueError:
                print("Cannot ignore all features. Using all features instead.")
                active_feature_names, applied_ignored, invalid_requested = build_feature_subset(
                    ignored_features=set(),
                    available_features=FEATURE_NAMES,
                )
            
            state.ignored_features = set(applied_ignored)
            state.active_feature_names = active_feature_names
            
            if invalid_requested:
                print(f"Unknown feature names skipped: {', '.join(invalid_requested)}")
            if state.ignored_features:
                print(f"Ignoring features: {', '.join(sorted(state.ignored_features))}")
            else:
                print("Using all features for training.")
            
            # Always prepare data with the selected category handling
            state.X_train, state.X_test, state.y_train, state.y_test = prepare_training_data(
                state.messages_with_features,
                collapse_categories=collapse_categories,
                collapse_ambiguous_to_no_flag=collapse_ambiguous_to_no_flag,
                active_feature_names=state.active_feature_names,
                ignored_features=state.ignored_features
            )
            
            if state.X_train is not None and state.y_train is not None:
                state.model = train_model(
                    state.X_train,
                    state.y_train,
                    state.model_type,
                    feature_names=state.active_feature_names,
                )
                print(f"Model trained: {state.model_type}")
            else:
                print("Training data not prepared. Run option 4 first.")
            
        elif choice == "6":
            # Evaluate model
            if state.model is None or state.X_test is None or state.y_test is None:
                print("No model or test data available. Run option 5 first.")
                continue
            evaluate_model(state.model, state.X_test, state.y_test)
            
        elif choice == "7":
            # Show state
            print_state()
            
        elif choice == "8":
            # Save/Load state
            print("\n1. Save state")
            print("2. Load state")
            sl_choice = input("Enter choice (1-2): ").strip()
            
            if sl_choice == "1":
                filepath = input("Enter filename (default: bootstrap_state.json): ").strip()
                save_state_to_file(filepath or "bootstrap_state.json")
            elif sl_choice == "2":
                filepath = input("Enter filename (default: bootstrap_state.json): ").strip()
                load_state_from_file(filepath or "bootstrap_state.json")
            
        elif choice == "9":
            print("Exiting...")
            break
        elif choice == "10":
            print("\nCollecting user statistics from tracked channels.")
            print("This will fetch up to ~10000 messages per channel and update the stats tables.")
            limit_input = input("Max messages per channel (default 10000): ").strip()
            window_input = input("Rolling window size for co-occurrence (default 30): ").strip()
            limit = int(limit_input) if limit_input.isdigit() else 10_000
            window_size = int(window_input) if window_input.isdigit() else 30

            def _log_progress(msg: str) -> None:
                print(msg)

            summary = await bootstrap_user_stats(
                limit_per_channel=limit,
                window_size=window_size,
                # Only collect stats from explicitly allowed channels
                channel_ids=CHANNEL_ALLOW_LIST,
                progress_callback=_log_progress,
            )
            print(
                f"User stats collection complete. "
                f"Channels: {summary['channels_processed']}, "
                f"Messages: {summary['messages_processed']}, "
                f"Windows updated: {summary['windows_processed']}"
            )
            
        else:
            print("Invalid choice. Please enter 1-9.")


def main():
    """Main entry point for the bootstrapping script."""
    logger.info("Starting bootstrapping script...")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            # Run full pipeline non-interactively
            asyncio.run(run_full_pipeline())
        elif sys.argv[1] == "--help":
            print("Usage: python bootstrapping.py [--full | --help]")
            print("  --full   Run full pipeline non-interactively")
            print("  --help   Show this help message")
            print("  (none)   Start interactive REPL menu")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Start interactive REPL
        asyncio.run(repl())


if __name__ == "__main__":
    main()
