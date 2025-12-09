"""Utilities for collecting and querying user activity statistics."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Callable, Iterable, Mapping

import discord
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import CHANNEL_ALLOW_LIST, DISCORD_BOT_TOKEN
from database import UserCoOccurrence, UserStats
from db_config import Base, engine, get_session, init_db

# Defaults chosen to balance coverage and API limits
DEFAULT_HISTORY_LIMIT = 10_000
DEFAULT_WINDOW_SIZE = 30


def _normalize_ratio(ratio: float) -> float:
    """Clamp any positive ratio into a 0-1 score."""
    if ratio <= 0:
        return 0.0
    return ratio / (ratio + 1.0)


def _normalize_count(count: int, scale: float = 10.0) -> float:
    """Convert a count into a 0-1 score with diminishing returns."""
    if count <= 0:
        return 0.0
    return min(count / (count + scale), 1.0)


def ensure_user_stats_schema() -> None:
    """
    Ensure the user stats schema has the latest columns/tables.
    Safe to call repeatedly; no-ops if already migrated.
    """
    inspector = inspect(engine)
    existing_columns = {col["name"] for col in inspector.get_columns("user_stats")}
    if "display_name" not in existing_columns:
        # Add the missing column for display names
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE user_stats ADD COLUMN display_name VARCHAR(100)"))

    # Create co-occurrence table if it doesn't exist
    existing_tables = set(inspector.get_table_names())
    if "user_co_occurrences" not in existing_tables:
        # Import models already done; create_all will no-op on existing tables
        Base.metadata.create_all(bind=engine)


def _get_or_create_user_stats(
    session: Session,
    user_id: int,
    username: str,
    display_name: str | None = None,
) -> UserStats:
    """
    Fetch or create a UserStats row, updating stored names if they changed.
    Uses savepoints to avoid rolling back entire transaction on IntegrityError.
    """
    # Check pending objects first to avoid duplicate inserts within the same session
    for pending in session.new:
        if isinstance(pending, UserStats) and pending.user_id == user_id:
            # Update names on the pending object if needed
            pending.username = username
            if display_name:
                pending.display_name = display_name
            return pending

    user_stats = session.query(UserStats).filter_by(user_id=user_id).first()
    if user_stats is None:
        user_stats = UserStats(
            user_id=user_id,
            username=username,
            display_name=display_name,
            message_count=0,
            character_count=0,
        )
        # Use savepoint so IntegrityError only rolls back this insert, not entire session
        savepoint = session.begin_nested()
        try:
            session.add(user_stats)
            session.flush()
        except IntegrityError:
            # User was inserted by another session/run; rollback savepoint and re-query
            savepoint.rollback()
            user_stats = session.query(UserStats).filter_by(user_id=user_id).first()
            if user_stats is None:
                raise
    else:
        # Keep username/display_name fresh for downstream lookups
        user_stats.username = username
        if display_name:
            user_stats.display_name = display_name
    return user_stats


def _get_or_create_co_occurrence(session: Session, user_a_id: int, user_b_id: int) -> UserCoOccurrence:
    """
    Fetch or create a co-occurrence row for an unordered pair (lowest id first).
    Uses savepoints to avoid rolling back entire transaction on IntegrityError.
    """
    if user_a_id == user_b_id:
        raise ValueError("Co-occurrence pair must involve two distinct users")
    low_id, high_id = sorted((user_a_id, user_b_id))

    # Check pending objects to avoid duplicate inserts when autoflush=False
    for pending in session.new:
        if (
            isinstance(pending, UserCoOccurrence)
            and pending.user_a_id == low_id
            and pending.user_b_id == high_id
        ):
            return pending

    pair = (
        session.query(UserCoOccurrence)
        .filter_by(user_a_id=low_id, user_b_id=high_id)
        .first()
    )
    if pair is None:
        pair = UserCoOccurrence(user_a_id=low_id, user_b_id=high_id, co_occurrence_count=0)
        # Use savepoint so IntegrityError only rolls back this insert, not entire session
        savepoint = session.begin_nested()
        try:
            session.add(pair)
            session.flush()
        except IntegrityError:
            # Pair was inserted by another session/run; rollback savepoint and re-query
            savepoint.rollback()
            pair = (
                session.query(UserCoOccurrence)
                .filter_by(user_a_id=low_id, user_b_id=high_id)
                .first()
            )
            if pair is None:
                raise
    return pair


def update_user_stats_from_message(message: discord.Message, session: Session) -> None:
    """
    Increment message and character counts for the author of a message.
    """
    author = message.author
    user_stats = _get_or_create_user_stats(
        session,
        user_id=author.id,
        username=author.name,
        display_name=getattr(author, "display_name", None),
    )
    user_stats.message_count += 1
    user_stats.character_count += len(message.content or "")


def update_co_occurrences_from_window(
    messages: Iterable[discord.Message],
    session: Session,
) -> None:
    """
    Increment co-occurrence counts for every unique user pair in a window of messages.
    """
    user_ids = {msg.author.id for msg in messages}
    if len(user_ids) < 2:
        return

    id_list = sorted(user_ids)
    for idx, user_a in enumerate(id_list):
        for user_b in id_list[idx + 1 :]:
            pair = _get_or_create_co_occurrence(session, user_a, user_b)
            pair.co_occurrence_count += 1


def process_messages_for_stats(
    messages: list[discord.Message],
    session: Session,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> None:
    """
    Process a chronological list of messages to update user stats and co-occurrences.
    """
    window: deque[discord.Message] = deque(maxlen=window_size)
    for message in messages:
        update_user_stats_from_message(message, session)
        window.append(message)
        update_co_occurrences_from_window(window, session)


def get_seniority_scores(
    author_id: int,
    target_id: int,
    session: Session,
) -> tuple[float, float]:
    """
    Compute seniority scores between author and target based on activity ratios.
    
    Returns:
        Tuple of (seniority_score_messages, seniority_score_characters)
    """
    author_stats = session.query(UserStats).filter_by(user_id=author_id).first()
    target_stats = session.query(UserStats).filter_by(user_id=target_id).first()

    if author_stats is None or target_stats is None:
        return 0.0, 0.0

    msg_ratio = author_stats.message_count / max(target_stats.message_count, 1)
    char_ratio = author_stats.character_count / max(target_stats.character_count, 1)
    return _normalize_ratio(msg_ratio), _normalize_ratio(char_ratio)


def get_familiarity_score_stat(
    author_id: int,
    target_id: int,
    session: Session,
) -> float:
    """
    Compute familiarity based on historical co-occurrence counts.
    """
    low_id, high_id = sorted((author_id, target_id))
    pair = (
        session.query(UserCoOccurrence)
        .filter_by(user_a_id=low_id, user_b_id=high_id)
        .first()
    )
    count = pair.co_occurrence_count if pair else 0
    return _normalize_count(count)


async def _collect_channel_messages(
    channel: discord.abc.Messageable,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> list[discord.Message]:
    """
    Fetch up to `limit` messages from newest to oldest, returning oldest-first order.
    """
    fetched: list[discord.Message] = []
    backoff = 1.0

    while len(fetched) < limit:
        remaining = limit - len(fetched)
        try:
            async for message in channel.history(limit=remaining, oldest_first=False):
                fetched.append(message)
            break
        except discord.errors.HTTPException as exc:
            # Handle rate limits or transient HTTP errors
            retry_after = getattr(exc, "retry_after", None)
            wait_for = float(retry_after) if retry_after is not None else backoff
            await asyncio.sleep(wait_for)
            backoff = min(backoff * 2, 10.0)

    fetched.reverse()  # Process oldest first
    return fetched


async def _process_single_channel(
    channel: discord.TextChannel | discord.Thread,
    session_factory: Callable[[], Session],
    limit: int,
    window_size: int,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    """
    Collect and process messages for a single channel or thread.
    Returns (message_count, co_occurrence_updates_estimate).
    """
    messages = await _collect_channel_messages(channel, limit=limit)
    if progress_callback:
        progress_callback(f"Fetched {len(messages)} messages from #{channel.name}")

    session = session_factory()
    try:
        process_messages_for_stats(messages, session, window_size=window_size)
        session.commit()
    finally:
        session.close()

    # Estimate updates as number of windows processed
    return len(messages), max(0, len(messages) - 1)


async def _process_channel_and_threads(
    channel: discord.TextChannel | discord.ForumChannel,
    session_factory: Callable[[], Session],
    limit: int,
    window_size: int,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[int, int]:
    """
    Process a text or forum channel and all accessible threads.
    """
    total_messages = 0
    total_windows = 0

    # Only process the channel directly if it's a TextChannel (ForumChannels have no history)
    if isinstance(channel, discord.TextChannel):
        msg_count, windows = await _process_single_channel(
            channel, session_factory, limit, window_size, progress_callback
        )
        total_messages += msg_count
        total_windows += windows

    # Active threads
    for thread in getattr(channel, "threads", []):
        msg_count, windows = await _process_single_channel(
            thread, session_factory, limit, window_size, progress_callback
        )
        total_messages += msg_count
        total_windows += windows

    # Archived threads
    try:
        async for thread in channel.archived_threads(limit=None):
            msg_count, windows = await _process_single_channel(
                thread, session_factory, limit, window_size, progress_callback
            )
            total_messages += msg_count
            total_windows += windows
    except (discord.Forbidden, AttributeError):
        pass

    try:
        async for thread in channel.archived_threads(limit=None, private=True):
            msg_count, windows = await _process_single_channel(
                thread, session_factory, limit, window_size, progress_callback
            )
            total_messages += msg_count
            total_windows += windows
    except (discord.Forbidden, AttributeError, TypeError):
        pass

    return total_messages, total_windows


async def bootstrap_user_stats(
    limit_per_channel: int = DEFAULT_HISTORY_LIMIT,
    window_size: int = DEFAULT_WINDOW_SIZE,
    channel_ids: list[int] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """
    Collect and persist user stats for tracked channels using the Discord API.
    
    Returns summary counts for reporting.
    """
    init_db()
    ensure_user_stats_schema()
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    client = discord.Client(intents=intents)

    processed_channels = 0
    processed_messages = 0
    processed_windows = 0
    channel_scope = channel_ids or CHANNEL_ALLOW_LIST

    @client.event
    async def on_ready():
        nonlocal processed_channels, processed_messages, processed_windows
        if progress_callback and client.user:
            progress_callback(f"Connected as {client.user}")

        for guild in client.guilds:
            for channel_id in channel_scope:
                channel = guild.get_channel(channel_id)
                if channel is None:
                    continue
                if isinstance(channel, discord.ForumChannel):
                    msg_count, windows = await _process_channel_and_threads(
                        channel,
                        get_session,
                        limit_per_channel,
                        window_size,
                        progress_callback,
                    )
                elif isinstance(channel, discord.TextChannel):
                    msg_count, windows = await _process_channel_and_threads(
                        channel,
                        get_session,
                        limit_per_channel,
                        window_size,
                        progress_callback,
                    )
                else:
                    continue

                processed_channels += 1
                processed_messages += msg_count
                processed_windows += windows

        await client.close()

    if DISCORD_BOT_TOKEN is None:
        raise ValueError("DISCORD_BOT_TOKEN is not set")

    await client.start(DISCORD_BOT_TOKEN)

    return {
        "channels_processed": processed_channels,
        "messages_processed": processed_messages,
        "windows_processed": processed_windows,
    }


def build_author_id_map(
    message_id_to_author: Mapping[str, int],
    rel_id_to_author: Mapping[int, int],
) -> dict[str, int]:
    """
    Build a composite map usable by the llm feature pipeline.
    """
    composite: dict[str, int] = {}
    composite.update(message_id_to_author)
    composite.update({str(rel_id): author_id for rel_id, author_id in rel_id_to_author.items()})
    return composite


def build_username_to_id_map(messages: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """
    Build a username/display-name map from serialized context messages.
    """
    username_map: dict[str, int] = {}
    for ctx in messages:
        author_id = ctx.get("author_id")
        if not isinstance(author_id, int):
            continue
        for key in ("author_username", "author_name"):
            name = ctx.get(key)
            if isinstance(name, str):
                username_map.setdefault(name, author_id)
    return username_map
