"""Main entry point for the Excelsior Moderator Discord bot."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import discord
import numpy as np
from sqlalchemy.orm import Session

from pathlib import Path

from config import (
    DISCORD_BOT_TOKEN,
    CHANNEL_ALLOW_LIST,
    MESSAGES_PER_CHECK,
    WAIVER_ROLE_NAME,
    REACTION_EMOJI,
    LOG_CHANNEL_ID,
    SECS_BETWEEN_AUTO_CHECKS,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    get_logger,
)
from database import FlaggedMessage, LogChannelRatingPost, MessageFeatures
from db_config import init_db, get_session
from history import MessageStore
from llms import get_candidate_features
from ml import FEATURE_NAMES, load_classifier
from utils import serialize_context_messages, is_tracked_channel

# Set up intents for the bot
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.members = True
intents.reactions = True
intents.message_content = True

# Module-level logger configured via config.get_logger
logger = get_logger(__name__)

@dataclass
class ChannelModerationState:
    """Keeps moderation scheduling state for a channel or thread."""

    channel_id: int
    messages_since_check: int = 0
    has_new_message_since_check: bool = False
    idle_timer_started_at: datetime | None = None
    last_checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_checked_message_id: int | None = None
    most_recent_message_id: int | None = None
    trigger_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


@dataclass
class ModerationResult:
    success: bool
    reason: str
    flagged_new_count: int = 0
    flagged_existing_count: int = 0
    candidates_considered: int = 0
    candidates_after_filters: int = 0

    @property
    def total_flagged(self) -> int:
        """Return the total number of flagged messages including existing ones."""
        return self.flagged_new_count + self.flagged_existing_count


class ExcelsiorBot(discord.Bot):
    """Custom bot class with message store and DB session access."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Global in-memory message store
        self.message_store: MessageStore = MessageStore()
        # Function to get DB sessions
        self.get_db_session: Callable[[], Session] = get_session
        # Per-channel moderation scheduler state
        self.channel_states: dict[int, ChannelModerationState] = {}

    def _get_or_create_channel_state(self, channel_id: int) -> ChannelModerationState:
        """
        Return existing channel state or create a new one with defaults.
        
        Args:
            channel_id: The Discord channel/thread ID.
            
        Returns:
            ChannelModerationState: Mutable state for scheduling moderation.
        """
        if channel_id not in self.channel_states:
            self.channel_states[channel_id] = ChannelModerationState(channel_id=channel_id)
        return self.channel_states[channel_id]

    def _compute_idle_timeout(self, state: ChannelModerationState) -> float | None:
        """
        Calculate seconds remaining before the idle timer should trigger moderation.
        
        Args:
            state: Channel moderation state to inspect.
            
        Returns:
            Seconds remaining or None when no idle timer is running.
        """
        if not (state.has_new_message_since_check and state.idle_timer_started_at):
            return None
        elapsed = (datetime.now(timezone.utc) - state.idle_timer_started_at).total_seconds()
        remaining = SECS_BETWEEN_AUTO_CHECKS - elapsed
        return remaining

    def _should_moderate(self, state: ChannelModerationState) -> bool:
        """
        Decide if the moderation workflow should run based on counters and timers.
        
        Args:
            state: Channel moderation state to evaluate.
            
        Returns:
            True when thresholds are met, False otherwise.
        """
        if state.messages_since_check >= MESSAGES_PER_CHECK:
            return True
        remaining = self._compute_idle_timeout(state)
        if remaining is not None and remaining <= 0:
            return True
        return False

    def _determine_moderation_reason(self, state: ChannelModerationState) -> str:
        """
        Identify why moderation is being triggered (message count vs idle timer).
        
        Args:
            state: Channel moderation state to inspect.
            
        Returns:
            String reason indicating trigger source.
        """
        if state.messages_since_check >= MESSAGES_PER_CHECK:
            return "message_count"
        remaining = self._compute_idle_timeout(state)
        if remaining is not None and remaining <= 0:
            return "idle_timer"
        return "idle_timer"

    def _ensure_scheduler_task(self, channel: discord.TextChannel | discord.Thread) -> None:
        """
        Ensure a scheduler task exists for a tracked channel/thread.
        
        Args:
            channel: The channel or thread requiring a scheduler.
        """
        if not is_tracked_channel(channel):
            return
        # Skip forum parents; they only contain threads
        if isinstance(channel, discord.ForumChannel):
            return
        state = self._get_or_create_channel_state(channel.id)
        # If an existing task is alive, nothing to do
        if state.task and not state.task.done():
            return

        async def _runner(channel_id: int):
            try:
                await self._moderation_scheduler(channel_id)
            except asyncio.CancelledError:
                # Allow graceful shutdown without noisy tracebacks
                return
            except Exception:
                logger.exception("Scheduler for channel %s crashed", channel_id)

        state.task = asyncio.create_task(_runner(channel.id))

    async def trigger_manual_moderation(
        self, channel: discord.TextChannel | discord.Thread
    ) -> bool:
        """
        Run a manual moderation pass immediately for a channel or thread.

        Args:
            channel: Channel or thread where moderation should be executed.

        Returns:
            True when the request succeeds, False when channel is unsupported or not tracked.
        """
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return False
        if not is_tracked_channel(channel):
            return False
        
        # Ensure scheduler exists for future automatic checks
        self._ensure_scheduler_task(channel)
        result = await self.run_moderation_now(channel)
        return result.success

    async def run_moderation_now(
        self, channel: discord.TextChannel | discord.Thread
    ) -> ModerationResult:
        """
        Run moderation immediately for a channel or thread and return detailed results.
        
        Args:
            channel: The channel or thread where moderation should run.
        
        Returns:
            ModerationResult describing the outcome of the run.
        """
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return ModerationResult(
                success=False,
                reason="Unsupported channel type for moderation.",
            )
        if not is_tracked_channel(channel):
            return ModerationResult(
                success=False,
                reason="This channel is not tracked by the moderation system.",
            )

        state = self._get_or_create_channel_state(channel.id)
        return await self._run_moderation(channel, state, "manual")

    async def _run_moderation(
        self,
        channel: discord.TextChannel | discord.Thread,
        state: ChannelModerationState,
        trigger_reason: str,
    ) -> ModerationResult:
        """
        Run moderate_channel and update state on completion.
        
        Args:
            channel: Target channel or thread.
            state: Channel state to reset after completion.
            trigger_reason: Reason moderation was triggered.
        """
        logger.info(
            "Starting moderation for channel %s (reason=%s, messages_since_check=%s, last_checked_message_id=%s)",
            channel.id,
            trigger_reason,
            state.messages_since_check,
            state.last_checked_message_id,
        )
        result = await self.moderate_channel(channel)

        if result.success:
            # Reset counters after a successful moderation pass
            state.messages_since_check = 0
            state.has_new_message_since_check = False
            state.idle_timer_started_at = None
            state.last_checked_at = datetime.now(timezone.utc)
            recent_message = self.message_store.get_most_recent_message(channel.id)
            state.last_checked_message_id = recent_message.id if recent_message else None
            logger.info(
                "Moderation completed for channel %s (reason=%s); counters reset (flagged_new=%s, total_flagged=%s)",
                channel.id,
                trigger_reason,
                result.flagged_new_count,
                result.total_flagged,
            )
        else:
            logger.warning(
                "Moderation run returned False for channel %s (reason=%s, details=%s)",
                channel.id,
                trigger_reason,
                result.reason,
            )
        return result

    async def _moderation_scheduler(self, channel_id: int) -> None:
        """
        Scheduler loop that triggers moderation based on message count or idle time.
        
        Args:
            channel_id: The channel or thread ID this scheduler manages.
        """
        state = self._get_or_create_channel_state(channel_id)
        while True:
            # Quick check: if thresholds are already met, skip waiting
            if not self._should_moderate(state):
                timeout = self._compute_idle_timeout(state)
                state.trigger_event.clear()
                try:
                    if timeout is None:
                        await state.trigger_event.wait()
                    elif timeout <= 0:
                        # Timer already expired; fall through to evaluation
                        pass
                    else:
                        await asyncio.wait_for(state.trigger_event.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    # Idle timer fired
                    pass
                except asyncio.CancelledError:
                    raise
                finally:
                    # Ensure event is cleared so new messages can wake the loop
                    state.trigger_event.clear()

            if not self._should_moderate(state):
                # Nothing to do yet; loop back to wait for next signal
                continue

            channel = self.get_channel(channel_id)
            if not isinstance(channel, (discord.TextChannel, discord.Thread)):
                # Channel unavailable; pause until next activity
                state.has_new_message_since_check = False
                state.messages_since_check = 0
                state.idle_timer_started_at = None
                logger.warning(
                    "Channel %s unavailable for moderation; state reset until next activity",
                    channel_id,
                )
                continue

            trigger_reason = self._determine_moderation_reason(state)
            try:
                await self._run_moderation(channel, state, trigger_reason)
            except Exception:
                # Log and continue loop so the scheduler keeps running
                logger.exception(
                    "Moderation run failed for channel %s (reason=%s)",
                    channel_id,
                    trigger_reason,
                )

    async def notify_moderation_on_message(self, message: discord.Message) -> None:
        """
        Record new message activity and wake the scheduler for the message's channel.
        
        Args:
            message: The new Discord message.
        """
        channel = message.channel
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return
        if not is_tracked_channel(channel):
            return

        self._ensure_scheduler_task(channel)
        state = self._get_or_create_channel_state(channel.id)
        state.messages_since_check += 1
        state.most_recent_message_id = message.id
        # Only start the idle timer and flag once per post-check batch
        if not state.has_new_message_since_check:
            state.has_new_message_since_check = True
            state.idle_timer_started_at = datetime.now(timezone.utc)
            logger.info(
                "Idle timer started for channel %s due to message %s by user %s",
                channel.id,
                message.id,
                getattr(message.author, "id", "unknown"),
            )

        # Wake scheduler to consider thresholds immediately
        state.trigger_event.set()

    async def initialize_moderation_tasks(self) -> None:
        """
        Ensure schedulers are running for all tracked channels and threads seen at startup.
        """
        # Start tasks for allowed parent channels present in guilds
        for guild in self.guilds:
            for channel_id in CHANNEL_ALLOW_LIST:
                channel = guild.get_channel(channel_id)
                if isinstance(channel, discord.ForumChannel):
                    # Forum parents are skipped; their threads are handled separately
                    continue
                if isinstance(channel, (discord.TextChannel, discord.Thread)):
                    self._ensure_scheduler_task(channel)

        # Start tasks for any threads or channels discovered during backfill
        for channel_id in self.message_store.get_all_channel_info().keys():
            channel = self.get_channel(channel_id)
            if isinstance(channel, (discord.TextChannel, discord.Thread)) and is_tracked_channel(channel):
                self._ensure_scheduler_task(channel)
    

    async def flag_message(self, message: discord.Message, db_session=None) -> None:
        """
        Flags a message by adding a reaction and posting to the log channel for mod rating.

        Args:
            message: The Discord message to flag.
            db_session: Optional DB session. If not provided, creates one internally.
        """
        # Add reaction to the original message
        await message.add_reaction(REACTION_EMOJI)

        # Post to log channel for mod rating
        log_channel = self.get_channel(LOG_CHANNEL_ID)
        if not log_channel:
            logger.warning(
                "Log channel %s not found; cannot post flagged message %s",
                LOG_CHANNEL_ID,
                message.id,
            )
            return

        # Build the log message content
        author_name = message.author.display_name or message.author.name
        jump_url = message.jump_url
        content_preview = (message.content or "")[:500]
        if len(message.content or "") > 500:
            content_preview += "..."

        log_content = (
            f"**Flagged Message**\n"
            f"**Author:** {author_name}\n"
            f"**Link:** {jump_url}\n"
            f"**Content:**\n```\n{content_preview}\n```\n"
            f"React with: 1️⃣ No Flag | 2️⃣ Ambiguous | 3️⃣ Unconstructive | 4️⃣ Unsolicited | 5️⃣ N/A"
        )

        # Send to log channel (type-check that it's a text channel)
        if not isinstance(log_channel, discord.TextChannel):
            logger.warning(
                "Log channel %s is not a text channel; cannot post flagged message %s",
                LOG_CHANNEL_ID,
                message.id,
            )
            return
        log_message = await log_channel.send(log_content)
        logger.info(
            "Posted flagged message %s to log channel %s as message %s",
            message.id,
            log_channel.id,
            log_message.id,
        )

        # Add rating reaction emojis
        rating_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
        for emoji in rating_emojis:
            await log_message.add_reaction(emoji)

        # Store the mapping in DB
        close_session = False
        if db_session is None:
            db_session = self.get_db_session()
            close_session = True

        try:
            post = LogChannelRatingPost(
                bot_message_id=log_message.id,
                flagged_message_id=message.id,
            )
            db_session.add(post)
            db_session.commit()
        finally:
            if close_session:
                db_session.close()


    async def moderate_channel(self, channel: discord.TextChannel | discord.Thread) -> ModerationResult:
        """
        Moderates a channel by running the moderation workflow on the in-memory message store for that channel.
        
        Args:
            channel: The channel to moderate.
        
        Returns:
            ModerationResult describing the run outcome and any flagged messages.
        """
        result = ModerationResult(
            success=False,
            reason="Moderation did not start.",
        )

        # Copy at time of call to avoid race conditions
        store_history_copy = self.message_store.get_whole_history(channel.id)

        candidates = await get_candidate_features(
            self.message_store,
            channel.id,
            provider=DEFAULT_LLM_PROVIDER,
            model=DEFAULT_LLM_MODEL,
            ignore_first_message_count=MESSAGES_PER_CHECK,
        )
        result.candidates_considered = len(candidates)

        logger.info(
            "Retrieved %d moderation candidate(s) for channel %s",
            len(candidates),
            channel.id,
        )

        # Filter anything with discusses_ellie > 0.2 since that's likely bot-related noise
        candidates = [candidate for candidate in candidates if candidate["features"]["discusses_ellie"] <= 0.2]

        # Filter out candidates that don't have a resolved discord_message_id
        candidates = [c for c in candidates if c.get("discord_message_id") is not None]

        if not candidates:
            logger.info(
                "No candidates with discord_message_id for channel %s; skipping moderation",
                channel.id,
            )
            result.candidates_after_filters = 0
            result.success = True
            result.reason = "No candidates with Discord message IDs were found."
            return result

        # Filter out anything directed at a user with the WAIVER_ROLE_NAME role
        candidate_messages = [next((msg for msg in store_history_copy if msg.id == candidate["discord_message_id"]), None) for candidate in candidates]

        candidate_message_authors = [message.author for message in candidate_messages if message is not None]

        # Build a filtered list to avoid mutating while iterating
        filtered_candidates: list[dict] = []
        filtered_candidate_messages: list[discord.Message | None] = []
        for candidate, message_author in zip(candidates, candidate_message_authors):
            if isinstance(message_author, discord.Member) and message_author.roles:
                if any(role.name == WAIVER_ROLE_NAME for role in message_author.roles):
                    continue
            filtered_candidates.append(candidate)
            filtered_candidate_messages.append(
                next((msg for msg in store_history_copy if msg.id == candidate["discord_message_id"]), None)
            )

        candidates = filtered_candidates
        candidate_messages = filtered_candidate_messages

        if not candidates:
            logger.info(
                "No candidates remain after waiver-role filter for channel %s",
                channel.id,
            )
            result.candidates_after_filters = 0
            result.success = True
            result.reason = "No candidates remain after waiver-role filter."
            return result

        # Filter out candidates that are too close to the beginning of the channel history
        # since the model is unlikely to have enough context to extract good features
        # This is redundant, but can't hurt
        def _safe_message_index(candidate: dict) -> int:
            # Coerce message_id to int to tolerate string or missing values
            try:
                return int(candidate.get("message_id", 0))
            except (TypeError, ValueError):
                return 0

        candidates = [
            candidate for candidate in candidates
            if _safe_message_index(candidate) > MESSAGES_PER_CHECK # message_id is 1-based index
        ]

        if not candidates:
            logger.info(
                "No candidates remain after early-history filter for channel %s",
                channel.id,
            )
            result.candidates_after_filters = 0
            result.success = True
            result.reason = "No candidates remain after early-history filter."
            return result
        
        result.candidates_after_filters = len(candidates)
            
        # Get the actual message objects for the candidates
        candidate_messages = [next((msg for msg in store_history_copy if msg.id == candidate["discord_message_id"]), None) for candidate in candidates]

        logger.info(
            "Processing %d filtered candidate(s) for channel %s",
            len(candidates),
            channel.id,
        )

        # Load the ML model, checking that it exists first
        model_path = Path("models/lightgbm_model.joblib")
        if not model_path.exists():
            logger.warning(
                "Model file not found at %s; skipping moderation for channel %s",
                model_path,
                channel.id,
            )
            result.reason = f"Model file not found at {model_path}"
            return result
        classifier = load_classifier(model_path)
        filter_out_feature_names = [] #["discusses_ellie", "includes_positive_takeaways"]
        # Align feature order with the model's training order; fall back to defaults
        model_feature_names = getattr(classifier, "feature_names", FEATURE_NAMES)
        active_feature_names = [
            name for name in model_feature_names if name not in filter_out_feature_names
        ]
        if not active_feature_names:
            logger.warning(
                "No active features available for model; skipping moderation for channel %s",
                channel.id,
            )
            result.reason = "No active features available for model."
            return result

        def _safe_feature_value(raw: Any) -> float:
            """Convert raw feature to float, defaulting to 0 on bad inputs."""
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0

        feature_matrix = np.array(
            [
                [
                    _safe_feature_value(candidate["features"].get(name, 0.0))
                    for name in active_feature_names
                ]
                for candidate in candidates
            ],
            dtype=float,
        )

        # Predict using a numpy feature matrix aligned to training order
        predictions = classifier.predict(feature_matrix)
        
        db_session = self.get_db_session()
        try:
            # Collect all candidate IDs that are predicted to be flagged
            candidate_message_ids_to_flag = [
                candidate_message.id
                for candidate, candidate_message, prediction in zip(candidates, candidate_messages, predictions)
                if prediction == "flag" and candidate_message is not None
            ]
            
            # Query for existing flagged message IDs to avoid duplicate inserts
            existing_flagged_ids: set[int] = set()
            runtime_feature_ids: set[int] = set()
            if candidate_message_ids_to_flag:
                existing_rows = db_session.query(FlaggedMessage.message_id).filter(
                    FlaggedMessage.message_id.in_(candidate_message_ids_to_flag)
                ).all()
                existing_flagged_ids = {row[0] for row in existing_rows}
                result.flagged_existing_count = len(existing_flagged_ids)
                if existing_flagged_ids:
                    logger.debug(
                        "Skipping %d already-flagged message(s) for channel %s",
                        len(existing_flagged_ids),
                        channel.id,
                    )
                # Check which runtime feature rows already exist so we do not violate unique constraints
                existing_runtime_feature_rows = (
                    db_session.query(MessageFeatures.message_id)
                    .filter(
                        MessageFeatures.extraction_run_id.is_(None),
                        MessageFeatures.message_id.in_(candidate_message_ids_to_flag),
                    )
                    .all()
                )
                runtime_feature_ids = {row[0] for row in existing_runtime_feature_rows}
            
            # Collect all flagged messages first, then commit once at the end
            flagged_messages_to_add: list[FlaggedMessage] = []
            # Capture runtime feature rows so we retain the model inputs alongside flags
            feature_records_to_add: list[MessageFeatures] = []
            
            for candidate, candidate_message, prediction in zip(candidates, candidate_messages, predictions):
                if prediction == "flag" and candidate_message is not None:
                    # Skip flag insert when it already exists, but still consider persisting features
                    if candidate_message.id not in existing_flagged_ids:
                        logger.info(
                            "Flagging message %s in channel %s (author_id=%s)",
                            candidate_message.id,
                            channel.id,
                            getattr(candidate_message.author, "id", "unknown"),
                        )
                        await self.flag_message(candidate_message, db_session)
                        surrounding_context = [msg for msg in store_history_copy if msg.id != candidate_message.id]
                        # Serialize surrounding context for persistent storage
                        context_ids, serialized_context = serialize_context_messages(surrounding_context)
                        
                        flagged_message = FlaggedMessage(
                            message_id=candidate_message.id,
                            channel_id=channel.id,
                            guild_id=channel.guild.id,
                            author_id=candidate_message.author.id,
                            # Store both display_name and username for user-friendly rendering
                            author_display_name=getattr(
                                candidate_message.author, "display_name", None
                            ),
                            author_username=candidate_message.author.name,
                            content=candidate_message.content,
                            context_message_ids=context_ids,
                            context_messages=serialized_context,
                            # Store native datetimes to match the ORM schema
                            timestamp=candidate_message.created_at if candidate_message.created_at else datetime.now(timezone.utc),
                            flagged_at=datetime.now(timezone.utc),
                            target_user_id=candidate.get("target_user_id"),
                            target_username=candidate.get("target_username"),
                        )
                        flagged_messages_to_add.append(flagged_message)
                    
                    # Persist runtime feature vector for this flagged message when not already saved
                    if candidate_message.id not in runtime_feature_ids:
                        feature_payload = candidate.get("features") or {}
                        if feature_payload:
                            target_username = feature_payload.get("target_username")
                            feature_record = MessageFeatures(
                                extraction_run_id=None,  # Null signals runtime (non-batch) extraction
                                message_id=candidate_message.id,
                                run_index=0,
                                features=feature_payload,
                                target_username=target_username if isinstance(target_username, str) else None,
                            )
                            feature_records_to_add.append(feature_record)
            
            # Add all flagged messages and runtime feature rows then commit once
            if flagged_messages_to_add or feature_records_to_add:
                logger.info(
                    "Persisting %d flagged message(s) and %d runtime feature row(s) for channel %s",
                    len(flagged_messages_to_add),
                    len(feature_records_to_add),
                    channel.id,
                )
                for flagged_message in flagged_messages_to_add:
                    db_session.add(flagged_message)
                for feature_record in feature_records_to_add:
                    db_session.add(feature_record)
                db_session.commit()
                result.flagged_new_count = len(flagged_messages_to_add)
            else:
                logger.info(
                    "No messages flagged for channel %s in this moderation pass",
                    channel.id,
                )
        finally:
            db_session.close()

        result.success = True
        result.reason = "Moderation completed successfully."
        return result


# Initialize the bot with a command prefix and intents
bot = ExcelsiorBot(intents=intents)


async def backfill_channel_history(
    channel: discord.TextChannel | discord.Thread,
    message_store: MessageStore,
    max_messages: int
) -> int:
    """
    Backfill message history for a single channel or thread.
    
    Args:
        channel: The channel or thread to backfill.
        message_store: The message store to add messages to.
        max_messages: Maximum number of messages to fetch.
        
    Returns:
        The number of messages backfilled.
    """
    messages = []
    async for message in channel.history(limit=max_messages):
        messages.append(message)
    
    # Add in chronological order (oldest first) so deque maintains proper order
    for message in reversed(messages):
        message_store.add_message(message)
    
    return len(messages)


async def backfill_threads_for_channel(
    channel: discord.TextChannel | discord.ForumChannel,
    message_store: MessageStore,
    max_messages: int
) -> tuple[int, int, list[str]]:
    """
    Backfill message history for all currently active threads in a channel.
    
    Args:
        channel: The parent channel containing threads.
        message_store: The message store to add messages to.
        max_messages: Maximum number of messages per thread.
        
    Returns:
        Tuple of (thread_count, total_messages_count, list_of_thread_names).
    """
    thread_count = 0
    total_messages = 0
    thread_names: list[str] = []
    
    # Only backfill active threads from the channel's cached threads list
    for thread in channel.threads:
        count = await backfill_channel_history(thread, message_store, max_messages)
        total_messages += count
        thread_count += 1
        thread_names.append(thread.name)
    
    return thread_count, total_messages, thread_names


async def backfill_message_store():
    """
    Backfill message history for all tracked channels and their threads.
    Handles both regular text channels (with threads) and forum channels (where all posts are threads).
    """
    max_messages = bot.message_store._max_size
    total_channels = 0
    total_threads = 0
    total_messages = 0
    
    for guild in bot.guilds:
        for channel_id in CHANNEL_ALLOW_LIST:
            channel = guild.get_channel(channel_id)
            if channel is None:
                continue
            
            if isinstance(channel, discord.ForumChannel):
                # Forum channels only have threads (posts), no direct messages
                thread_count, msg_count, thread_names = await backfill_threads_for_channel(
                    channel, bot.message_store, max_messages
                )
                total_threads += thread_count
                total_messages += msg_count
                total_channels += 1
                logger.info(
                    "Backfilled forum #%s: %s threads, %s messages",
                    channel.name,
                    thread_count,
                    msg_count,
                )
                # Log thread names with their parent forum
                for name in thread_names:
                    logger.debug(
                        "Thread '%s' (parent: #%s) backfilled",
                        name,
                        channel.name,
                    )
                
            elif isinstance(channel, discord.TextChannel):
                # Backfill the channel itself
                msg_count = await backfill_channel_history(channel, bot.message_store, max_messages)
                total_messages += msg_count
                total_channels += 1
                
                # Backfill threads within this channel
                thread_count, thread_msgs, thread_names = await backfill_threads_for_channel(
                    channel, bot.message_store, max_messages
                )
                total_threads += thread_count
                total_messages += thread_msgs
                
                logger.info(
                    "Backfilled #%s: %s messages, %s threads (%s thread messages)",
                    channel.name,
                    msg_count,
                    thread_count,
                    thread_msgs,
                )
                # Log thread names with their parent channel
                for name in thread_names:
                    logger.debug(
                        "Thread '%s' (parent: #%s) backfilled",
                        name,
                        channel.name,
                    )
    
    logger.info(
        "Backfill complete: %s channels, %s threads, %s total messages",
        total_channels,
        total_threads,
        total_messages,
    )


@bot.event
async def on_ready():
    """Called when the bot has successfully connected to Discord."""
    if bot.user:
        logger.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
    logger.info("Connected to %s guild(s)", len(bot.guilds))
    
    # Initialize database tables
    init_db()
    logger.info("Database initialized")
    
    # Backfill message history for tracked channels and threads
    logger.info("Backfilling message history...")
    await backfill_message_store()

    # Kick off moderation schedulers for tracked channels/threads
    logger.info("Starting moderation schedulers...")
    await bot.initialize_moderation_tasks()

    # Initialize rating system channel (instructions and leaderboard)
    from cogs.rating import Rating
    rating_cog = bot.get_cog("Rating")
    if rating_cog and isinstance(rating_cog, Rating):
        logger.info("Initializing rating channel...")
        await rating_cog.init_rating_channel()
    
    logger.info("Bot is ready!")


def load_cogs():
    """Load all cog extensions from the cogs folder."""
    cog_list = [
        "cogs.public",
        "cogs.restricted",
        "cogs.events",
        "cogs.rating",
    ]
    
    for cog in cog_list:
        try:
            bot.load_extension(cog)
            logger.info("Loaded cog: %s", cog)
        except Exception:
            logger.exception("Failed to load cog %s", cog)


if __name__ == "__main__":
    load_cogs()
    bot.run(DISCORD_BOT_TOKEN)

