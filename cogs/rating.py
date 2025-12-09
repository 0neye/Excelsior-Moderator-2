"""Rating system cog for moderator and public message rating."""

import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import LOG_CHANNEL_ID, RATING_CHANNEL_ID, MODERATOR_ROLES
from database import FlaggedMessage, FlaggedMessageRating, LogChannelRatingPost, RatingCategory
from utils import format_markdown_table


# Emoji mappings for rating categories (used in log channel reactions)
RATING_EMOJIS = {
    "1️⃣": RatingCategory.NO_FLAG,
    "2️⃣": RatingCategory.AMBIGUOUS,
    "3️⃣": RatingCategory.UNCONSTRUCTIVE,
    "4️⃣": RatingCategory.UNSOLICITED,
    "5️⃣": RatingCategory.NA,
}

# Ordered list of emojis for adding reactions
RATING_EMOJI_ORDER = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]

# Path to leaderboard metadata JSON file
RATING_METADATA_FILE = Path(__file__).parent.parent / "data" / "rating_metadata.json"


def _load_rating_metadata() -> dict:
    """Load leaderboard metadata from JSON file."""
    if RATING_METADATA_FILE.exists():
        with open(RATING_METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_rating_metadata(metadata: dict) -> None:
    """Save leaderboard metadata to JSON file."""
    RATING_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RATING_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


class RatingView(discord.ui.View):
    """View with buttons for rating a flagged message in the public rating channel."""

    def __init__(
        self,
        flagged_message_id: int,
        user_id: int,
        message_details: str,
        get_db_session,
        on_rating_complete,
    ):
        """
        Initialize the rating view.

        Args:
            flagged_message_id: The message_id of the flagged message being rated.
            user_id: Discord ID of the rater.
            message_details: Pre-formatted message details to display.
            get_db_session: Function to get DB sessions.
            on_rating_complete: Callback when rating is submitted.
        """
        super().__init__(timeout=300)  # 5 minute timeout
        # Store only the ID to avoid detached session issues
        self.flagged_message_id = flagged_message_id
        self.user_id = user_id
        self.message_details = message_details
        self.get_db_session = get_db_session
        self.on_rating_complete = on_rating_complete
        self._rating_submitted = False
        self._interaction: Optional[discord.Interaction] = None

    async def on_timeout(self) -> None:
        """Handle view timeout by notifying user and cleaning up."""
        if self._interaction is not None and not self._rating_submitted:
            try:
                await self._interaction.edit_original_response(
                    content="⏰ Rating session timed out. Use `/rate` to start a new one.",
                    view=None,
                )
            except discord.NotFound:
                pass
            except discord.HTTPException as e:
                print(f"[RatingView] Failed to edit timeout message: {e}")

    async def _handle_rating(
        self, interaction: discord.Interaction, category: RatingCategory
    ):
        """
        Handle a rating selection from the user.

        Args:
            interaction: The Discord interaction from the button press.
            category: The RatingCategory the user selected.
        """
        # Ensure only the user who invoked /rate can use these buttons
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "This rating is not for you!", ephemeral=True
            )
            return

        # Prevent duplicate submissions
        if self._rating_submitted:
            await interaction.response.send_message(
                "You've already submitted a rating for this message!", ephemeral=True
            )
            return
        self._rating_submitted = True
        self._interaction = interaction

        # Generate unique rating ID
        rating_id = f"{self.user_id}_{self.flagged_message_id}_{uuid.uuid4().hex[:8]}"

        # Save rating to DB
        session = self.get_db_session()
        try:
            now = datetime.now(timezone.utc)
            rating = FlaggedMessageRating(
                rating_id=rating_id,
                flagged_message_id=self.flagged_message_id,
                rater_user_id=self.user_id,
                category=category,
                started_at=now,
                completed_at=now,
            )
            session.add(rating)
            session.commit()
        finally:
            session.close()

        # Update the message
        await interaction.response.edit_message(
            content=f"✅ Rating submitted: **{category.value}**\n\n{self.message_details}",
            view=None,
        )

        # Trigger leaderboard update callback
        if self.on_rating_complete:
            await self.on_rating_complete()

    @discord.ui.button(label="No Flag", style=discord.ButtonStyle.green, emoji="✅")
    async def no_flag_button(
        self, button: discord.ui.Button, interaction: discord.Interaction
    ):
        """Handle No Flag category selection."""
        await self._handle_rating(interaction, RatingCategory.NO_FLAG)

    @discord.ui.button(label="Ambiguous", style=discord.ButtonStyle.gray, emoji="❓")
    async def ambiguous_button(
        self, button: discord.ui.Button, interaction: discord.Interaction
    ):
        """Handle Ambiguous category selection."""
        await self._handle_rating(interaction, RatingCategory.AMBIGUOUS)

    @discord.ui.button(label="Unconstructive", style=discord.ButtonStyle.red, emoji="⚠️")
    async def unconstructive_button(
        self, button: discord.ui.Button, interaction: discord.Interaction
    ):
        """Handle Unconstructive category selection."""
        await self._handle_rating(interaction, RatingCategory.UNCONSTRUCTIVE)

    @discord.ui.button(label="Unsolicited", style=discord.ButtonStyle.red, emoji="📢")
    async def unsolicited_button(
        self, button: discord.ui.Button, interaction: discord.Interaction
    ):
        """Handle Unsolicited category selection."""
        await self._handle_rating(interaction, RatingCategory.UNSOLICITED)

    @discord.ui.button(label="Not Applicable", style=discord.ButtonStyle.blurple, emoji="🚫")
    async def not_applicable_button(
        self, button: discord.ui.Button, interaction: discord.Interaction
    ):
        """Handle Not Applicable category selection."""
        await self._handle_rating(interaction, RatingCategory.NA)


class Rating(commands.Cog):
    """Cog for rating flagged messages via reactions (log channel) and buttons (public channel)."""

    def __init__(self, bot):
        self.bot = bot
        self.get_db_session = bot.get_db_session

    # -------------------------------------------------------------------------
    # DB Query Helpers
    # -------------------------------------------------------------------------

    def _has_user_rated_message(
        self, session: Session, user_id: int, message_id: int
    ) -> bool:
        """Check if a user has already rated a specific message."""
        existing = (
            session.query(FlaggedMessageRating)
            .filter(
                FlaggedMessageRating.flagged_message_id == message_id,
                FlaggedMessageRating.rater_user_id == user_id,
            )
            .first()
        )
        return existing is not None

    def _get_random_least_rated_message(
        self, session: Session, exclude_user_id: Optional[int] = None
    ) -> Optional[FlaggedMessage]:
        """
        Get a random flagged message from the pool with the fewest ratings.

        Args:
            session: DB session.
            exclude_user_id: User ID to exclude (don't show messages they've already rated).

        Returns:
            A random FlaggedMessage from the least-rated pool, or None if no messages.
        """
        # Subquery to count ratings per flagged message
        rating_counts = (
            session.query(
                FlaggedMessageRating.flagged_message_id,
                func.count(FlaggedMessageRating.id).label("rating_count"),
            )
            .group_by(FlaggedMessageRating.flagged_message_id)
            .subquery()
        )

        # Get all flagged messages with their rating counts (0 if none)
        query = session.query(
            FlaggedMessage,
            func.coalesce(rating_counts.c.rating_count, 0).label("count"),
        ).outerjoin(
            rating_counts,
            FlaggedMessage.message_id == rating_counts.c.flagged_message_id,
        )

        # Exclude messages the user has already rated
        if exclude_user_id is not None:
            user_rated_subquery = (
                session.query(FlaggedMessageRating.flagged_message_id)
                .filter(FlaggedMessageRating.rater_user_id == exclude_user_id)
                .subquery()
            )
            query = query.filter(
                ~FlaggedMessage.message_id.in_(
                    session.query(user_rated_subquery.c.flagged_message_id)
                )
            )

        results = query.all()
        if not results:
            return None

        # Find minimum rating count
        min_count = min(r[1] for r in results)

        # Filter to only messages with minimum count
        least_rated = [r[0] for r in results if r[1] == min_count]

        return random.choice(least_rated) if least_rated else None

    def _get_random_most_contested_message(
        self, session: Session, exclude_user_id: Optional[int] = None
    ) -> Optional[FlaggedMessage]:
        """
        Get a random flagged message that has contested ratings (high disagreement).

        Args:
            session: DB session.
            exclude_user_id: User ID to exclude.

        Returns:
            A random FlaggedMessage from contested pool, or None.
        """
        # Get messages that have at least 2 ratings
        rating_counts = (
            session.query(
                FlaggedMessageRating.flagged_message_id,
                func.count(FlaggedMessageRating.id).label("rating_count"),
            )
            .group_by(FlaggedMessageRating.flagged_message_id)
            .having(func.count(FlaggedMessageRating.id) >= 2)
            .subquery()
        )

        query = session.query(FlaggedMessage).join(
            rating_counts,
            FlaggedMessage.message_id == rating_counts.c.flagged_message_id,
        )

        # Exclude messages the user has already rated
        if exclude_user_id is not None:
            user_rated_subquery = (
                session.query(FlaggedMessageRating.flagged_message_id)
                .filter(FlaggedMessageRating.rater_user_id == exclude_user_id)
                .subquery()
            )
            query = query.filter(
                ~FlaggedMessage.message_id.in_(
                    session.query(user_rated_subquery.c.flagged_message_id)
                )
            )

        candidates = query.all()
        if not candidates:
            return None

        # Calculate variance for each candidate and find most contested
        best_candidates = []
        best_variance = None

        for message in candidates:
            ratings = (
                session.query(FlaggedMessageRating.category)
                .filter(FlaggedMessageRating.flagged_message_id == message.message_id)
                .all()
            )

            # Group into meta-classes: no_flag, ambiguous, flag (unconstructive/unsolicited)
            counts = {"no_flag": 0, "ambiguous": 0, "flag": 0}
            for (cat,) in ratings:
                if cat in (RatingCategory.NO_FLAG, RatingCategory.NA):
                    counts["no_flag"] += 1
                elif cat == RatingCategory.AMBIGUOUS:
                    counts["ambiguous"] += 1
                elif cat in (RatingCategory.UNCONSTRUCTIVE, RatingCategory.UNSOLICITED):
                    counts["flag"] += 1

            total = sum(counts.values())
            if total < 2:
                continue

            # Lower variance = more contested
            normalized = [c / total for c in counts.values()]
            active = [n for n in normalized if n > 0]
            if len(active) < 2:
                continue

            # Calculate population variance
            mean = sum(normalized) / len(normalized)
            variance = sum((x - mean) ** 2 for x in normalized) / len(normalized)

            if best_variance is None or variance < best_variance:
                best_variance = variance
                best_candidates = [message]
            elif abs(variance - best_variance) < 1e-9:
                best_candidates.append(message)

        return random.choice(best_candidates) if best_candidates else None

    def _get_user_stats(self, session: Session, user_id: int) -> dict:
        """
        Get detailed rating statistics for a specific user.

        Args:
            session: DB session.
            user_id: Discord user ID.

        Returns:
            Dict with total_ratings, category_breakdown, rank, etc.
        """
        # Count user's ratings by category
        user_ratings = (
            session.query(FlaggedMessageRating)
            .filter(FlaggedMessageRating.rater_user_id == user_id)
            .all()
        )

        total_ratings = len(user_ratings)
        category_counts = {cat: 0 for cat in RatingCategory}
        for rating in user_ratings:
            if rating.category:
                category_counts[rating.category] += 1

        # Get total flagged messages
        total_flagged = session.query(FlaggedMessage).count()
        coverage_pct = (total_ratings / total_flagged * 100) if total_flagged > 0 else 0.0

        # Get user's rank
        all_counts = (
            session.query(
                FlaggedMessageRating.rater_user_id,
                func.count(FlaggedMessageRating.id).label("count"),
            )
            .group_by(FlaggedMessageRating.rater_user_id)
            .order_by(func.count(FlaggedMessageRating.id).desc())
            .all()
        )

        user_rank = None
        for rank, (uid, _) in enumerate(all_counts, start=1):
            if uid == user_id:
                user_rank = rank
                break

        return {
            "total_ratings": total_ratings,
            "category_breakdown": {cat.value: count for cat, count in category_counts.items()},
            "total_flagged_messages": total_flagged,
            "coverage_percentage": coverage_pct,
            "rank": user_rank,
            "total_raters": len(all_counts),
        }

    def _get_top_raters(self, session: Session, limit: int = 10) -> list[tuple[int, int]]:
        """
        Get the top raters ordered by rating count.

        Args:
            session: DB session.
            limit: Maximum number of raters to return.

        Returns:
            List of (user_id, rating_count) tuples.
        """
        results = (
            session.query(
                FlaggedMessageRating.rater_user_id,
                func.count(FlaggedMessageRating.id).label("count"),
            )
            .group_by(FlaggedMessageRating.rater_user_id)
            .order_by(func.count(FlaggedMessageRating.id).desc())
            .limit(limit)
            .all()
        )
        return [(uid, count) for uid, count in results]

    # -------------------------------------------------------------------------
    # Log Channel Reaction Handling
    # -------------------------------------------------------------------------

    async def handle_log_channel_reaction(
        self, payload: discord.RawReactionActionEvent
    ) -> bool:
        """
        Process a rating reaction on a log channel post.

        Args:
            payload: The raw reaction event data.

        Returns:
            True if the reaction was processed, False otherwise.
        """
        # Check if this is a rating emoji
        emoji_str = str(payload.emoji)
        if emoji_str not in RATING_EMOJIS:
            return False

        # Ignore bot reactions
        if payload.user_id == self.bot.user.id:
            return False

        session = self.get_db_session()
        try:
            # Look up the log channel post
            post = (
                session.query(LogChannelRatingPost)
                .filter(LogChannelRatingPost.bot_message_id == payload.message_id)
                .first()
            )

            if not post:
                return False

            # Look up existing rating to allow overrides instead of ignoring
            existing_rating = (
                session.query(FlaggedMessageRating)
                .filter(
                    FlaggedMessageRating.flagged_message_id == post.flagged_message_id,
                    FlaggedMessageRating.rater_user_id == payload.user_id,
                )
                .first()
            )

            category = RATING_EMOJIS[emoji_str]
            now = datetime.now(timezone.utc)

            if existing_rating:
                # Update the user's previous rating with the new category
                existing_rating.category = category
                existing_rating.completed_at = now
                session.commit()
                print(
                    f"[Rating] User {payload.user_id} updated rating for message {post.flagged_message_id} to {category.value}"
                )
            else:
                # Create a new rating entry for this user/message pair
                rating_id = f"{payload.user_id}_{post.flagged_message_id}_{uuid.uuid4().hex[:8]}"
                rating = FlaggedMessageRating(
                    rating_id=rating_id,
                    flagged_message_id=post.flagged_message_id,
                    rater_user_id=payload.user_id,
                    category=category,
                    started_at=now,
                    completed_at=now,
                )
                session.add(rating)
                session.commit()

                print(
                    f"[Rating] User {payload.user_id} rated message {post.flagged_message_id} as {category.value}"
                )

            # Update leaderboard
            await self.update_leaderboard()
            return True
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # Leaderboard Management
    # -------------------------------------------------------------------------

    async def _generate_scoreboard(self, guild: Optional[discord.Guild]) -> str:
        """
        Generate a markdown table showing the top 10 raters with display names.

        Args:
            guild: Discord guild for resolving user display names.

        Returns:
            Formatted markdown table string.
        """
        session = self.get_db_session()
        try:
            top_raters = self._get_top_raters(session, limit=10)
            total_flagged = session.query(FlaggedMessage).count()
        finally:
            session.close()

        data_rows = []
        for rank, (user_id, rating_count) in enumerate(top_raters, start=1):
            # Resolve display name
            display_name = "Unknown User"
            if guild:
                member = guild.get_member(user_id)
                if member:
                    display_name = member.display_name
                else:
                    try:
                        member = await guild.fetch_member(user_id)
                        display_name = member.display_name
                    except discord.NotFound:
                        display_name = f"User {user_id}"
                    except Exception:
                        display_name = f"User {user_id}"

            percentage = (rating_count / total_flagged * 100) if total_flagged > 0 else 0.0
            data_rows.append((str(rank), display_name, str(rating_count), f"{percentage:.1f}%"))

        if not top_raters:
            data_rows.append(("-", "-", "-", "-"))

        headers = ["Rank", "User", "Ratings", "% of Available"]
        return format_markdown_table(headers, data_rows)

    def _build_instructions_content(self) -> str:
        """Build the instructions message for the rating channel."""
        return """# 📊 Message Rating System

Help improve Ellie by rating flagged messages!

## How to Rate Messages

1. Use `/rate` to get a random flagged message to evaluate
2. Read the message (*and context*) and choose one of five categories:
   • **No Flag** ✅ - Message is fine, should not have been flagged
   • **Ambiguous** ❓ - Unclear whether the message violates guidelines
   • **Unconstructive** ⚠️ - Message contains unconstructive criticism
   • **Unsolicited** 📢 - Feedback was not requested/wanted
   • **Not Applicable** 🚫 - Message falls outside the scope of the feedback guidelines

### More context on the categories

**No Flag** should be for when you can see how it could be interpreted as criticism towards someone in the conversation but you think it doesn't break the feedback guidelines.
**Ambiguous** should be for when you're not sure if the message breaks the feedback guidelines, but you can see it going either way.
**Not Applicable** should be for when the message is clearly about something like game mechanics, off-topic convo, or criticism of nebulous targets.

## Important notes:

**For Unsolicited:** Even if the person didn't mind the criticism, if it was not directly asked for, it should be marked as unsolicited!

**Considering your biases:** Try to put aside what you know of the people in these conversations, your "tacit knowledge" of them. If it helps, imagine everyone is a new player who just joined the server.

**Quality Control:** Your ratings will be analyzed to ensure quality responses! Spamming low-effort ratings will be penalized.

Use `/view_score` to see your personal statistics."""

    def _build_leaderboard_content(self, scoreboard: str) -> str:
        """Build the leaderboard message content."""
        return f"""## 🏆 Leaderboard (Top 10)

```
{scoreboard}
```"""

    async def init_rating_channel(self) -> None:
        """Initialize the rating channel with instructions and leaderboard messages."""
        channel = self.bot.get_channel(RATING_CHANNEL_ID)
        if not channel:
            print(f"Warning: Rating channel not found: {RATING_CHANNEL_ID}")
            return

        guild = channel.guild if hasattr(channel, "guild") else None
        metadata = _load_rating_metadata()

        # Generate content
        scoreboard = await self._generate_scoreboard(guild)
        instructions_content = self._build_instructions_content()
        leaderboard_content = self._build_leaderboard_content(scoreboard)

        # Handle instructions message
        instructions_message_id = metadata.get("instructions_message_id")
        instructions_message = None

        if instructions_message_id:
            try:
                instructions_message = await channel.fetch_message(instructions_message_id)
                await instructions_message.edit(content=instructions_content)
            except discord.NotFound:
                print(f"Instructions message {instructions_message_id} not found, creating new")
                metadata.pop("instructions_message_id", None)
                instructions_message = None
            except Exception as e:
                print(f"Error refreshing instructions message: {e}")

        if instructions_message is None:
            new_msg = await channel.send(instructions_content)
            metadata["instructions_message_id"] = new_msg.id

        # Handle leaderboard message
        leaderboard_message_id = metadata.get("leaderboard_message_id")
        leaderboard_message = None

        if leaderboard_message_id:
            try:
                leaderboard_message = await channel.fetch_message(leaderboard_message_id)
                await leaderboard_message.edit(content=leaderboard_content)
            except discord.NotFound:
                print(f"Leaderboard message {leaderboard_message_id} not found, creating new")
                metadata.pop("leaderboard_message_id", None)
                leaderboard_message = None
            except Exception as e:
                print(f"Error refreshing leaderboard message: {e}")

        if leaderboard_message is None:
            new_msg = await channel.send(leaderboard_content)
            metadata["leaderboard_message_id"] = new_msg.id

        # Save metadata
        metadata["last_scoreboard"] = scoreboard
        _save_rating_metadata(metadata)

        print("Rating channel initialized. Instructions and leaderboard messages ready.")

    async def update_leaderboard(self) -> None:
        """Update the leaderboard message if the scoreboard has changed."""
        channel = self.bot.get_channel(RATING_CHANNEL_ID)
        if not channel:
            print(f"Rating channel not found: {RATING_CHANNEL_ID}")
            return

        guild = channel.guild if hasattr(channel, "guild") else None
        metadata = _load_rating_metadata()

        # Generate new scoreboard
        new_scoreboard = await self._generate_scoreboard(guild)

        # Check if changed
        if new_scoreboard == metadata.get("last_scoreboard", ""):
            return

        leaderboard_message_id = metadata.get("leaderboard_message_id")
        if not leaderboard_message_id:
            print("No leaderboard message ID found, reinitializing")
            await self.init_rating_channel()
            return

        try:
            leaderboard_message = await channel.fetch_message(leaderboard_message_id)
            content = self._build_leaderboard_content(new_scoreboard)
            await leaderboard_message.edit(content=content)

            metadata["last_scoreboard"] = new_scoreboard
            _save_rating_metadata(metadata)
            print("Leaderboard updated successfully")
        except discord.NotFound:
            print(f"Leaderboard message {leaderboard_message_id} not found")
            metadata.pop("leaderboard_message_id", None)
            _save_rating_metadata(metadata)
            await self.init_rating_channel()
        except Exception as e:
            print(f"Error updating leaderboard: {e}")

    # -------------------------------------------------------------------------
    # Slash Commands
    # -------------------------------------------------------------------------

    @discord.slash_command(name="rate", description="Rate a flagged message")
    async def rate(self, ctx: discord.ApplicationContext):
        """Present a random flagged message for rating (public channel only)."""
        # Check channel
        if ctx.channel.id != RATING_CHANNEL_ID:
            await ctx.respond(
                "This command can only be used in the rating channel!",
                ephemeral=True,
            )
            return

        await ctx.defer(ephemeral=True)

        session = self.get_db_session()
        try:
            # Choose selection method randomly
            selection_method = random.choice(["least_rated", "most_contested"])

            if selection_method == "least_rated":
                message_record = self._get_random_least_rated_message(
                    session, exclude_user_id=ctx.author.id
                )
            else:
                message_record = self._get_random_most_contested_message(
                    session, exclude_user_id=ctx.author.id
                )
                # Fallback to least rated if most contested returns None
                if not message_record:
                    message_record = self._get_random_least_rated_message(
                        session, exclude_user_id=ctx.author.id
                    )
                    selection_method = "least_rated (fallback)"

            if not message_record:
                total_flagged = session.query(FlaggedMessage).count()
                await ctx.followup.send(
                    f"No flagged messages available to rate! (Total in system: {total_flagged})",
                    ephemeral=True,
                )
                return

            # Build jump URL
            jump_url = f"https://discord.com/channels/{message_record.guild_id}/{message_record.channel_id}/{message_record.message_id}"

            # Build message details
            author_name = message_record.author_display_name or message_record.author_username or "Unknown"
            content_preview = (message_record.content or "")[:500]
            if len(message_record.content or "") > 500:
                content_preview += "..."

            message_details = (
                f"**Author:** {author_name}\n"
                f"**Message:** {jump_url}\n"
                f"**Content:**\n```\n{content_preview}\n```\n"
                f"**Selection Method:** {selection_method.replace('_', ' ').title()}\n"
            )

            content = (
                "**Rate this message:**\n\n"
                f"{message_details}\n"
                "Choose a category to classify this message:"
            )

            view = RatingView(
                flagged_message_id=message_record.message_id,
                user_id=ctx.author.id,
                message_details=message_details,
                get_db_session=self.get_db_session,
                on_rating_complete=self.update_leaderboard,
            )

            await ctx.followup.send(content, view=view, ephemeral=True)
        finally:
            session.close()

    @discord.slash_command(name="view_score", description="View your rating statistics")
    async def view_score(self, ctx: discord.ApplicationContext):
        """Display personal rating statistics."""
        # Check channel
        if ctx.channel.id != RATING_CHANNEL_ID:
            await ctx.respond(
                "This command can only be used in the rating channel!",
                ephemeral=True,
            )
            return

        session = self.get_db_session()
        try:
            stats = self._get_user_stats(session, ctx.author.id)
        finally:
            session.close()

        response = "**📊 Your Rating Statistics**\n\n"
        response += f"**Total Ratings:** {stats['total_ratings']}\n"
        response += f"**Coverage:** {stats['coverage_percentage']:.1f}% ({stats['total_ratings']}/{stats['total_flagged_messages']} messages)\n"

        if stats["rank"]:
            response += f"**Rank:** #{stats['rank']} of {stats['total_raters']} raters\n"
        else:
            response += "**Rank:** Unranked (no ratings yet)\n"

        response += "\n**Category Breakdown:**\n"
        breakdown = stats["category_breakdown"]
        response += f"• No Flag: {breakdown.get('no-flag', 0)}\n"
        response += f"• Ambiguous: {breakdown.get('ambiguous', 0)}\n"
        response += f"• Unconstructive: {breakdown.get('unconstructive', 0)}\n"
        response += f"• Unsolicited: {breakdown.get('unsolicited', 0)}\n"
        response += f"• Not Applicable: {breakdown.get('NA', 0)}\n"

        await ctx.respond(response, ephemeral=True)


def setup(bot):
    """Called by Pycord to load this cog."""
    bot.add_cog(Rating(bot))

