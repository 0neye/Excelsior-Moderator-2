"""Main entry point for the Excelsior Moderator Discord bot."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable

import discord
import numpy as np
from sqlalchemy.orm import Session

from pathlib import Path

from config import (
    DISCORD_BOT_TOKEN,
    CHANNEL_ALLOW_LIST,
    MESSAGES_PER_CHECK,
    NEW_MESSAGES_BEFORE_TIMER_START,
    WAIVER_ROLE_NAME,
    SAVE_WAIVER_FILTERED_FLAGS,
    REACTION_EMOJI,
    LOG_CHANNEL_ID,
    SECS_BETWEEN_AUTO_CHECKS,
    MODERATION_FAILURE_BACKOFF_BASE,
    MODERATION_FAILURE_BACKOFF_MAX,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    get_logger,
)
from database import FlaggedMessage, LogChannelRatingPost, MessageFeatures
from db_config import init_db, get_session
from history import MessageStore
from llms import get_candidate_features
from ml import FEATURE_NAMES, ModelLoadError, load_classifier
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
    moderation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    in_flight_message_count: int = 0
    consecutive_failures: int = 0
    cooldown_until: datetime | None = None


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

    def _cooldown_remaining(self, state: ChannelModerationState) -> float | None:
        """
        Return remaining cooldown seconds after a failed moderation run.
        """
        if state.cooldown_until is None:
            return None
        remaining = (state.cooldown_until - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            state.cooldown_until = None
            return None
        return remaining

    def _compute_idle_timeout(self, state: ChannelModerationState) -> float | None:
        """
        Calculate seconds remaining before the idle timer should trigger moderation.
        
        Args:
            state: Channel moderation state to inspect.
            
        Returns:
            Seconds remaining or None when no idle timer is running.
        """
        idle_remaining: float | None = None
        if state.has_new_message_since_check and state.idle_timer_started_at:
            elapsed = (datetime.now(timezone.utc) - state.idle_timer_started_at).total_seconds()
            idle_remaining = SECS_BETWEEN_AUTO_CHECKS - elapsed
            if idle_remaining < 0:
                idle_remaining = 0

        cooldown_remaining = self._cooldown_remaining(state)
        if cooldown_remaining is None:
            return idle_remaining
        if state.messages_since_check >= MESSAGES_PER_CHECK:
            return cooldown_remaining
        if idle_remaining is None:
            return cooldown_remaining
        if idle_remaining == 0:
            return cooldown_remaining
        return min(idle_remaining, cooldown_remaining)

    def _get_idle_timer_start_threshold(self) -> int:
        """
        Return a safe minimum message threshold for starting the idle timer.

        Returns:
            Threshold value clamped to at least 1.
        """
        return max(1, NEW_MESSAGES_BEFORE_TIMER_START)

    def _register_moderation_failure(self, state: ChannelModerationState, reason: str) -> None:
        """
        Record a failed moderation run and apply a backoff cooldown.
        """
        state.consecutive_failures += 1
        backoff = min(
            MODERATION_FAILURE_BACKOFF_BASE * (2 ** (state.consecutive_failures - 1)),
            MODERATION_FAILURE_BACKOFF_MAX,
        )
        state.cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=backoff)
        logger.warning(
            "Moderation backoff for channel %s after failure (%s); cooldown %ss",
            state.channel_id,
            reason,
            backoff,
        )

    def _reset_moderation_failures(self, state: ChannelModerationState) -> None:
        """
        Clear failure counters after a successful moderation run.
        """
        state.consecutive_failures = 0
        state.cooldown_until = None

    def _should_moderate(self, state: ChannelModerationState) -> bool:
        """
        Decide if the moderation workflow should run based on counters and timers.
        
        Args:
            state: Channel moderation state to evaluate.
            
        Returns:
            True when thresholds are met, False otherwise.
        """
        if self._cooldown_remaining(state) is not None:
            return False
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
            state: Channel scheduling state to update after completion.
            trigger_reason: Reason moderation was triggered.
        """
        async with state.moderation_lock:
            # A manual pass may have consumed the work while the scheduler waited.
            if trigger_reason != "manual" and not self._should_moderate(state):
                return ModerationResult(success=True, reason="Scheduled work already consumed.")

            consumed_count = state.messages_since_check
            recent_message = self.message_store.get_most_recent_message(channel.id)
            consumed_message_id = recent_message.id if recent_message else None
            previous_idle_timer = state.idle_timer_started_at
            state.in_flight_message_count = consumed_count
            succeeded = False
            logger.info(
                "Starting moderation for channel %s (reason=%s, messages_since_check=%s, last_checked_message_id=%s)",
                channel.id,
                trigger_reason,
                consumed_count,
                state.last_checked_message_id,
            )
            try:
                result = await self.moderate_channel(channel)
                succeeded = result.success
                if result.success:
                    self._reset_moderation_failures(state)
                    state.messages_since_check -= consumed_count
                    state.has_new_message_since_check = (
                        state.messages_since_check >= self._get_idle_timer_start_threshold()
                    )
                    if not state.has_new_message_since_check:
                        state.idle_timer_started_at = None
                    state.last_checked_at = datetime.now(timezone.utc)
                    state.last_checked_message_id = consumed_message_id
                    logger.info(
                        "Moderation completed for channel %s (reason=%s, pending=%s, flagged_new=%s, total_flagged=%s)",
                        channel.id,
                        trigger_reason,
                        state.messages_since_check,
                        result.flagged_new_count,
                        result.total_flagged,
                    )
                else:
                    self._register_moderation_failure(state, result.reason)
                    logger.warning(
                        "Moderation run returned False for channel %s (reason=%s, details=%s)",
                        channel.id,
                        trigger_reason,
                        result.reason,
                    )
                return result
            except Exception:
                self._register_moderation_failure(state, "exception")
                raise
            finally:
                state.in_flight_message_count = 0
                if not succeeded and previous_idle_timer is not None:
                    state.idle_timer_started_at = previous_idle_timer

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
        # Arrivals during extraction belong to the next batch and need their own timer.
        pending_count = state.messages_since_check - state.in_flight_message_count
        threshold = self._get_idle_timer_start_threshold()
        if (
            not state.has_new_message_since_check and state.messages_since_check >= threshold
        ) or pending_count == threshold:
            state.has_new_message_since_check = True
            state.idle_timer_started_at = datetime.now(timezone.utc)
            logger.info(
                "Idle timer started for channel %s due to message %s by user %s",
                channel.id,
                message.id,
                getattr(message.author, "id", "unknown"),
            )
        elif not state.has_new_message_since_check:
            # Defer timer startup until enough new messages accumulate in this channel
            logger.debug(
                "Idle timer not started for channel %s yet (%d/%d new messages since check)",
                channel.id,
                state.messages_since_check,
                self._get_idle_timer_start_threshold(),
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
    

    async def flag_message(self, flagged_message: FlaggedMessage, db_session: Session) -> int | None:
        """Deliver a stored flag and persist its log mapping before adding rating reactions."""
        existing_post = db_session.query(LogChannelRatingPost).filter_by(
            flagged_message_id=flagged_message.message_id,
        ).first()
        if existing_post is not None:
            return existing_post.bot_message_id

        # A reaction on the original message is separate from moderator-log delivery.
        if flagged_message.was_acted_upon:
            flagged_message.was_acted_upon = False
            db_session.commit()
        message = self.message_store.get_message_by_id(
            flagged_message.message_id, flagged_message.channel_id,
        )
        if message is not None:
            try:
                await message.add_reaction(REACTION_EMOJI)
            except discord.HTTPException as error:
                logger.warning("Unable to react to flagged message %s: %s", flagged_message.message_id, error)

        log_channel = self.get_channel(LOG_CHANNEL_ID)
        if not isinstance(log_channel, discord.TextChannel):
            logger.warning("Moderator log channel unavailable for flagged message %s", flagged_message.message_id)
            return None

        author_name = flagged_message.author_display_name or flagged_message.author_username
        target_display = (
            f"<@{flagged_message.target_user_id}>"
            if flagged_message.target_user_id is not None
            else (flagged_message.target_username or "Unknown / None")
        )
        jump_url = (
            f"https://discord.com/channels/{flagged_message.guild_id}/"
            f"{flagged_message.channel_id}/{flagged_message.message_id}"
        )
        content_preview = flagged_message.content[:500]
        if len(flagged_message.content) > 500:
            content_preview += "..."
        log_content = (
            f"**Flagged Message**\n"
            f"**Author:** {author_name}\n"
            f"**Target:** {target_display}\n"
            f"**Link:** {jump_url}\n"
            f"**Content:**\n```\n{content_preview}\n```\n"
            f"React with: 1️⃣ No Flag | 2️⃣ Ambiguous | 3️⃣ Unconstructive | 4️⃣ Unsolicited | 5️⃣ N/A"
        )
        try:
            log_message = await log_channel.send(log_content)
        except discord.HTTPException as error:
            logger.warning("Moderator log delivery failed for message %s: %s", flagged_message.message_id, error)
            return None

        db_session.add(LogChannelRatingPost(
            bot_message_id=log_message.id, flagged_message_id=flagged_message.message_id,
        ))
        flagged_message.was_acted_upon = True
        db_session.commit()
        logger.info("Delivered flagged message %s as log post %s", flagged_message.message_id, log_message.id)

        for emoji in ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]:
            try:
                await log_message.add_reaction(emoji)
            except discord.HTTPException as error:
                logger.warning("Unable to add rating reaction to log post %s: %s", log_message.id, error)
        return log_message.id

    async def _retry_pending_log_deliveries(self, channel_id: int) -> bool:
        """Retry stored flags independently of candidate extraction or the history window."""
        db_session = self.get_db_session()
        try:
            pending_flags = (
                db_session.query(FlaggedMessage)
                .outerjoin(LogChannelRatingPost, LogChannelRatingPost.flagged_message_id == FlaggedMessage.message_id)
                .filter(
                    FlaggedMessage.channel_id == channel_id,
                    FlaggedMessage.waiver_filtered.is_(False),
                    LogChannelRatingPost.id.is_(None),
                )
                .order_by(FlaggedMessage.id)
                .all()
            )
            for flagged_message in pending_flags:
                if await self.flag_message(flagged_message, db_session) is None:
                    return False
            return True
        finally:
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

        if not await self._retry_pending_log_deliveries(channel.id):
            result.reason = "Moderator log delivery is pending."
            return result

        # Copy at time of call to avoid race conditions
        store_history_copy = self.message_store.get_whole_history(channel.id)
        # Keep only the newest MESSAGES_PER_CHECK messages eligible for flagging.
        # Older messages in the history window are context-only.
        context_only_message_count = max(0, len(store_history_copy) - MESSAGES_PER_CHECK)

        candidates = await get_candidate_features(
            self.message_store,
            channel.id,
            provider=DEFAULT_LLM_PROVIDER,
            model=DEFAULT_LLM_MODEL,
            ignore_first_message_count=context_only_message_count,
        )
        result.candidates_considered = len(candidates)

        logger.info(
            "Retrieved %d moderation candidate(s) for channel %s",
            len(candidates),
            channel.id,
        )

        # Track how many each filter removes (for logging when no candidates remain)
        filter_counts: dict[str, int] = {}

        # Filter anything with discusses_ellie > 0.2 since that's likely bot-related noise
        before_ellie = len(candidates)
        candidates = [candidate for candidate in candidates if candidate["features"]["discusses_ellie"] <= 0.2]
        removed = before_ellie - len(candidates)
        if removed > 0:
            filter_counts["discusses_ellie"] = removed

        # Filter out candidates that don't have a resolved discord_message_id
        before_discord_id = len(candidates)
        candidates = [c for c in candidates if c.get("discord_message_id") is not None]
        removed = before_discord_id - len(candidates)
        if removed > 0:
            filter_counts["discord_message_id"] = removed

        if not candidates:
            logger.info(
                "No candidates with discord_message_id for channel %s; skipping moderation",
                channel.id,
            )
            result.candidates_after_filters = 0
            result.success = True
            result.reason = "No candidates with Discord message IDs were found."
            return result

        # Filter out candidates that fall in the context-only prefix of the current
        # history window. This mirrors ignore_first_message_count above.
        # Build a map once so we can recover relative indexes from Discord IDs.
        discord_message_id_to_rel_index = {
            message.id: index + 1 for index, message in enumerate(store_history_copy)
        }

        def _safe_message_index(candidate: dict) -> int:
            """
            Resolve a candidate's relative message index in the current history window.

            Returns:
                1-based index when resolvable, otherwise 0.
            """
            # Prefer the canonical index produced during candidate resolution.
            candidate_relative_index = candidate.get("relative_message_index")
            if isinstance(candidate_relative_index, int):
                return candidate_relative_index
            if isinstance(candidate_relative_index, str) and candidate_relative_index.isdigit():
                return int(candidate_relative_index)

            # Fallback for older candidate payloads that only include message_id.
            raw_message_id = candidate.get("message_id")
            if isinstance(raw_message_id, int):
                numeric_message_id = raw_message_id
            elif isinstance(raw_message_id, str):
                try:
                    numeric_message_id = int(raw_message_id)
                except ValueError:
                    numeric_message_id = 0
            else:
                numeric_message_id = 0

            if 1 <= numeric_message_id <= len(store_history_copy):
                return numeric_message_id

            # When message_id is a Discord snowflake, map it back to a relative index.
            discord_message_id = candidate.get("discord_message_id")
            if isinstance(discord_message_id, int):
                return discord_message_id_to_rel_index.get(discord_message_id, 0)

            return 0

        def _is_self_referential_candidate(candidate: dict, candidate_message: discord.Message) -> bool:
            """
            Detect candidates where criticism is directed at the candidate's own author.

            Args:
                candidate: Candidate payload produced by the LLM feature extractor
                candidate_message: Discord message resolved for this candidate

            Returns:
                True when the candidate appears self-referential and should be filtered
            """
            target_user_id = candidate.get("target_user_id")
            if not isinstance(target_user_id, int):
                return False
            return target_user_id == candidate_message.author.id

        # Build a lookup of users who currently hold the waiver role
        # Behavior is configurable:
        # - SAVE_WAIVER_FILTERED_FLAGS=True: keep waived flags in DB, suppress Discord action
        # - SAVE_WAIVER_FILTERED_FLAGS=False: drop waived targets early (legacy behavior)
        waiver_target_user_ids: set[int] = set()
        waiver_role = next(
            (role for role in channel.guild.roles if role.name == WAIVER_ROLE_NAME),
            None,
        )
        if waiver_role is not None:
            waiver_target_user_ids = {member.id for member in waiver_role.members}

        # Keep candidate/message pairs aligned through all downstream filtering.
        filtered_pairs: list[tuple[dict, discord.Message]] = []
        count_message_not_in_history = 0
        count_self_referential = 0
        count_waiver = 0
        count_waiver_targets_seen = 0
        count_too_close = 0
        for candidate in candidates:
            candidate_message = next(
                (msg for msg in store_history_copy if msg.id == candidate["discord_message_id"]),
                None,
            )
            if candidate_message is None:
                count_message_not_in_history += 1
                continue

            if _is_self_referential_candidate(candidate, candidate_message):
                count_self_referential += 1
                continue

            target_user_id = candidate.get("target_user_id")
            is_waiver_target = (
                isinstance(target_user_id, int)
                and target_user_id in waiver_target_user_ids
            )
            if is_waiver_target:
                count_waiver_targets_seen += 1
            if (
                not SAVE_WAIVER_FILTERED_FLAGS
                and is_waiver_target
            ):
                count_waiver += 1
                continue

            if _safe_message_index(candidate) <= context_only_message_count:  # message_id is 1-based index
                count_too_close += 1
                continue

            filtered_pairs.append((candidate, candidate_message))

        if count_message_not_in_history > 0:
            filter_counts["message_not_in_history"] = count_message_not_in_history
        if count_self_referential > 0:
            filter_counts["self_referential"] = count_self_referential
        if count_waiver > 0:
            filter_counts["waiver"] = count_waiver
        if count_too_close > 0:
            filter_counts["too_close_to_beginning"] = count_too_close
        if count_waiver_targets_seen > 0:
            logger.info(
                "Waiver-role impact for channel %s: %d candidate(s) target waived users, %d filtered pre-model (SAVE_WAIVER_FILTERED_FLAGS=%s)",
                channel.id,
                count_waiver_targets_seen,
                count_waiver,
                SAVE_WAIVER_FILTERED_FLAGS,
            )

        if not filtered_pairs:
            filter_summary = ", ".join(f"{name}: {n}" for name, n in filter_counts.items())
            logger.info(
                "No candidates remain after moderation filters for channel %s (filtered out: %s)",
                channel.id,
                filter_summary if filter_summary else "none",
            )
            result.candidates_after_filters = 0
            result.success = True
            result.reason = "No candidates remain after moderation filters."
            return result

        candidates = [candidate for candidate, _ in filtered_pairs]
        candidate_messages = [candidate_message for _, candidate_message in filtered_pairs]
        result.candidates_after_filters = len(candidates)

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
        try:
            classifier = load_classifier(model_path)
        except ModelLoadError as exc:
            # Keep scheduler healthy and report an actionable infrastructure error
            logger.error(
                "Model load failed for channel %s: %s",
                channel.id,
                exc,
            )
            result.reason = str(exc)
            return result
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
        model_n_features_in = getattr(getattr(classifier, "model", None), "n_features_in_", None)
        if isinstance(model_n_features_in, int) and feature_matrix.shape[1] != model_n_features_in:
            logger.error(
                "Feature shape mismatch for channel %s: runtime has %d feature(s), model expects %d; skipping moderation pass",
                channel.id,
                feature_matrix.shape[1],
                model_n_features_in,
            )
            result.reason = (
                f"Feature shape mismatch: runtime={feature_matrix.shape[1]}, "
                f"model={model_n_features_in}"
            )
            return result

        label_classes = getattr(getattr(classifier, "label_encoder", None), "classes_", None)
        if label_classes is not None:
            class_names = [str(class_name) for class_name in label_classes]
            logger.info("Loaded moderation classes for channel %s: %s", channel.id, class_names)
            if "flag" not in class_names:
                logger.warning(
                    "Model classes for channel %s do not include 'flag'; runtime will only action explicit 'flag' predictions",
                    channel.id,
                )

        predictions = classifier.predict(feature_matrix)
        
        db_session = self.get_db_session()
        try:
            candidate_message_ids_to_flag = {
                message.id
                for message, prediction in zip(candidate_messages, predictions)
                if prediction == "flag"
            }
            existing_flags = {
                row.message_id: row
                for row in db_session.query(FlaggedMessage).filter(
                    FlaggedMessage.message_id.in_(candidate_message_ids_to_flag),
                ).all()
            } if candidate_message_ids_to_flag else {}
            result.flagged_existing_count = len(existing_flags)
            runtime_feature_ids = {
                row[0] for row in db_session.query(MessageFeatures.message_id).filter(
                    MessageFeatures.extraction_run_id.is_(None),
                    MessageFeatures.message_id.in_(candidate_message_ids_to_flag),
                ).all()
            } if candidate_message_ids_to_flag else set()

            for candidate, candidate_message, prediction in zip(candidates, candidate_messages, predictions):
                if prediction != "flag":
                    continue
                target_user_id = candidate.get("target_user_id")
                is_waiver_filtered = target_user_id in waiver_target_user_ids
                if is_waiver_filtered and not SAVE_WAIVER_FILTERED_FLAGS:
                    continue

                flagged_message = existing_flags.get(candidate_message.id)
                is_new = flagged_message is None
                if is_new:
                    context_ids, serialized_context = serialize_context_messages(
                        [message for message in store_history_copy if message.id != candidate_message.id],
                    )
                    flagged_message = FlaggedMessage(
                        message_id=candidate_message.id,
                        channel_id=channel.id,
                        guild_id=channel.guild.id,
                        author_id=candidate_message.author.id,
                        author_display_name=getattr(candidate_message.author, "display_name", None),
                        author_username=candidate_message.author.name,
                        content=candidate_message.content,
                        context_message_ids=context_ids,
                        context_messages=serialized_context,
                        timestamp=candidate_message.created_at or datetime.now(timezone.utc),
                        flagged_at=datetime.now(timezone.utc),
                        target_user_id=target_user_id,
                        target_username=candidate.get("target_username"),
                        was_acted_upon=False,
                        waiver_filtered=is_waiver_filtered,
                    )
                    db_session.add(flagged_message)
                    existing_flags[candidate_message.id] = flagged_message

                if candidate_message.id not in runtime_feature_ids and candidate.get("features"):
                    target_username = candidate.get("target_username")
                    db_session.add(MessageFeatures(
                        extraction_run_id=None,
                        message_id=candidate_message.id,
                        run_index=0,
                        features=candidate["features"],
                        target_username=target_username if isinstance(target_username, str) else None,
                    ))
                    runtime_feature_ids.add(candidate_message.id)

                result.flagged_new_count += int(is_new)

            # Retain every classified flag, including candidates after a failed send.
            db_session.commit()
            for flagged_message in existing_flags.values():
                if not flagged_message.waiver_filtered:
                    if await self.flag_message(flagged_message, db_session) is None:
                        result.reason = "Moderator log delivery is pending."
                        return result
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

