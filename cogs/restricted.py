"""Restricted commands available only to moderators."""

from typing import TYPE_CHECKING

import asyncio
from datetime import datetime, timedelta, timezone
import discord
from discord.ext import commands
from sqlalchemy import func

from config import ADMIN_ROLE, MODERATOR_ROLES
from database import FlaggedMessage, FlaggedMessageRating, RatingCategory
from utils import is_tracked_channel

if TYPE_CHECKING:
    from bot import ModerationResult


def is_moderator():
    """Check if the user has one of the moderator roles."""
    async def predicate(ctx: discord.ApplicationContext) -> bool:
        if not ctx.guild or not isinstance(ctx.author, discord.Member):
            return False
        
        # Get the member's role names
        member_role_names = [role.name for role in ctx.author.roles]
        
        # Check if any moderator role is in the member's roles
        return any(role in member_role_names for role in MODERATOR_ROLES)
    
    return commands.check(predicate)  # type: ignore[arg-type]


def is_admin():
    """Check if the user has the admin role."""
    async def predicate(ctx: discord.ApplicationContext) -> bool:
        if not ctx.guild or not isinstance(ctx.author, discord.Member):
            return False
        
        member_role_names = [role.name for role in ctx.author.roles]
        return ADMIN_ROLE in member_role_names
    
    return commands.check(predicate)  # type: ignore[arg-type]


class Restricted(commands.Cog):
    """Cog containing commands restricted to moderators."""
    
    def __init__(self, bot):
        self.bot = bot
        # Access to global message store and DB connection
        self.message_store = bot.message_store
        self.get_db_session = bot.get_db_session

    def _get_flagged_message_counts(self, guild_id: int, days: int) -> tuple[int, int]:
        """
        Count recently flagged messages and waiver-filtered messages.

        Args:
            guild_id: Guild ID to scope analytics to the active server
            days: Number of trailing days to include in the analytics window

        Returns:
            Tuple of (total_flagged_count, waiver_filtered_count)
        """
        # Build an absolute UTC cutoff so time-window logic is consistent
        cutoff_timestamp = datetime.now(timezone.utc) - timedelta(days=days)

        session = self.get_db_session()
        try:
            # Apply one shared time filter so both counts use the same window
            base_query = session.query(FlaggedMessage).filter(
                FlaggedMessage.guild_id == guild_id,
                FlaggedMessage.flagged_at >= cutoff_timestamp
            )

            # Count all flagged messages in the requested timeframe
            total_flagged_count = base_query.count()

            # Count only records marked as waiver-filtered in that same timeframe
            waiver_filtered_count = base_query.filter(
                FlaggedMessage.waiver_filtered.is_(True)
            ).count()
        finally:
            session.close()

        return total_flagged_count, waiver_filtered_count

    def _get_flagged_author_leaderboard(
        self,
        guild_id: int,
        days: int,
        include_waiver_filtered: bool,
        include_all_flagged_messages: bool,
        limit: int = 10,
    ) -> list[tuple[int, int]]:
        """
        Build a leaderboard of authors with the most flagged messages.

        Args:
            guild_id: Guild ID to scope the leaderboard to the active server
            days: Number of trailing days to include in the leaderboard window
            include_waiver_filtered: Whether to include waiver-filtered flag rows
            include_all_flagged_messages: If False, only count flags with confirming ratings
            limit: Maximum number of leaderboard rows to return

        Returns:
            List of (author_id, flagged_count) sorted by count descending
        """
        session = self.get_db_session()
        try:
            # Build an absolute UTC cutoff so timeframe filtering is deterministic
            cutoff_timestamp = datetime.now(timezone.utc) - timedelta(days=days)

            # Start from all flagged messages for this guild so cross-server rows are excluded
            flagged_query = session.query(FlaggedMessage).filter(
                FlaggedMessage.guild_id == guild_id,
                FlaggedMessage.flagged_at >= cutoff_timestamp,
            )

            # Optionally remove waiver-filtered rows when building the ranking
            if not include_waiver_filtered:
                flagged_query = flagged_query.filter(FlaggedMessage.waiver_filtered.is_(False))

            # Optionally keep only rows that have at least one rating confirming the flag
            if not include_all_flagged_messages:
                confirming_rating_message_ids = (
                    session.query(FlaggedMessageRating.flagged_message_id)
                    .filter(
                        FlaggedMessageRating.category.in_(
                            [RatingCategory.UNCONSTRUCTIVE, RatingCategory.UNSOLICITED]
                        )
                    )
                    .distinct()
                    .subquery()
                )
                flagged_query = flagged_query.filter(
                    FlaggedMessage.message_id.in_(
                        session.query(confirming_rating_message_ids.c.flagged_message_id)
                    )
                )

            # Aggregate by author to rank the members most frequently flagged
            leaderboard_rows = (
                flagged_query.with_entities(
                    FlaggedMessage.author_id,
                    func.count(FlaggedMessage.id).label("flagged_count"),
                )
                .group_by(FlaggedMessage.author_id)
                .order_by(func.count(FlaggedMessage.id).desc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()

        return [(author_id, flagged_count) for author_id, flagged_count in leaderboard_rows]

    def _get_recent_flagged_messages(
        self,
        guild_id: int,
        limit: int,
        include_waiver_filtered: bool,
    ) -> list[tuple[int, int, int, datetime, bool]]:
        """
        Fetch the most recent flagged messages for a single guild.

        Args:
            guild_id: Guild ID used to scope results to the current server
            limit: Maximum number of flagged message rows to return
            include_waiver_filtered: Whether rows marked waiver-filtered should be included

        Returns:
            List of tuples containing (message_id, channel_id, author_id, flagged_at, waiver_filtered)
            sorted by newest first
        """
        session = self.get_db_session()
        try:
            # Query only the fields needed for the moderator list to keep memory usage low
            recent_query = (
                session.query(
                    FlaggedMessage.message_id,
                    FlaggedMessage.channel_id,
                    FlaggedMessage.author_id,
                    FlaggedMessage.flagged_at,
                    FlaggedMessage.waiver_filtered,
                )
                .filter(FlaggedMessage.guild_id == guild_id)
            )

            # Allow moderators to include or exclude waiver-filtered rows on demand
            if not include_waiver_filtered:
                recent_query = recent_query.filter(FlaggedMessage.waiver_filtered.is_(False))

            recent_rows = recent_query.order_by(FlaggedMessage.flagged_at.desc()).limit(limit).all()
        finally:
            session.close()

        return [
            (message_id, channel_id, author_id, flagged_at, waiver_filtered)
            for message_id, channel_id, author_id, flagged_at, waiver_filtered in recent_rows
        ]

    @discord.slash_command(name="mod_ping", description="Moderator-only ping command")
    @is_moderator()
    async def mod_ping(self, ctx: discord.ApplicationContext):
        """A moderator-only ping command for testing permissions."""
        await ctx.respond("Moderator ping received!", ephemeral=True)

    @discord.slash_command(name="mod_info", description="Get moderator information")
    @is_moderator()
    async def mod_info(self, ctx: discord.ApplicationContext):
        """Displays moderator-specific information."""
        embed = discord.Embed(
            title="Moderator Info",
            description="Moderator-only bot information.",
            color=discord.Color.red()
        )
        embed.add_field(name="Moderator Roles", value=", ".join(MODERATOR_ROLES), inline=False)
        if isinstance(ctx.author, discord.Member):
            embed.add_field(name="Your Roles", value=", ".join([r.name for r in ctx.author.roles if r.name != "@everyone"]), inline=False)
        
        await ctx.respond(embed=embed, ephemeral=True)

    @discord.slash_command(
        name="mod_flag_analytics",
        description="View flagged message counts for a timeframe",
    )
    @is_moderator()
    async def mod_flag_analytics(
        self,
        ctx: discord.ApplicationContext,
        days: int = 1,
    ):
        """
        Show flagged message analytics for the last N days.

        Args:
            ctx: Discord application context for the command
            days: Number of trailing days to include in the report
        """
        # Guard against invalid or extreme values that don't make operational sense
        if days < 1:
            await ctx.respond(
                "Please provide a timeframe of at least 1 day.",
                ephemeral=True,
            )
            return

        if ctx.guild is None:
            await ctx.respond(
                "This command can only be used in a server.",
                ephemeral=True,
            )
            return

        # Query aggregate counts for the same timeframe window
        total_flagged_count, waiver_filtered_count = self._get_flagged_message_counts(
            guild_id=ctx.guild.id,
            days=days,
        )
        waiver_percentage = (
            (waiver_filtered_count / total_flagged_count) * 100
            if total_flagged_count
            else 0.0
        )

        # Present the analytics in an embed for readability
        embed = discord.Embed(
            title="Flagged Message Analytics",
            color=discord.Color.orange(),
        )
        embed.add_field(
            name="Timeframe",
            value=f"Last {days} day{'s' if days != 1 else ''}",
            inline=False,
        )
        embed.add_field(
            name="Total Flagged Messages",
            value=str(total_flagged_count),
            inline=True,
        )
        embed.add_field(
            name="Waiver-Filtered Flags",
            value=f"{waiver_filtered_count} ({waiver_percentage:.1f}%)",
            inline=True,
        )

        await ctx.respond(embed=embed, ephemeral=True)

    @discord.slash_command(
        name="mod_flag_leaderboard",
        description="Show top members most frequently flagged by moderation",
    )
    @is_moderator()
    async def mod_flag_leaderboard(
        self,
        ctx: discord.ApplicationContext,
        days: int = 30,
        include_waiver_filtered: bool = True,
        include_all_flagged_messages: bool = True,
    ):
        """
        Show the top 10 members most likely to be flagged by the bot.

        Args:
            ctx: Discord application context for the command
            days: Number of trailing days to include in the leaderboard
            include_waiver_filtered: Whether to include waiver-filtered flagged messages
            include_all_flagged_messages: Whether to include all flagged rows instead of only confirmed ones
        """
        # Ensure timeframe input is valid before running DB queries
        if days < 1:
            await ctx.respond(
                "Please provide a timeframe of at least 1 day.",
                ephemeral=True,
            )
            return

        if ctx.guild is None:
            await ctx.respond(
                "This command can only be used in a server.",
                ephemeral=True,
            )
            return

        # Pull a wider pool first, then keep the top ten current guild members
        leaderboard_rows = self._get_flagged_author_leaderboard(
            guild_id=ctx.guild.id,
            days=days,
            include_waiver_filtered=include_waiver_filtered,
            include_all_flagged_messages=include_all_flagged_messages,
            limit=100,
        )

        if not leaderboard_rows:
            await ctx.respond(
                "No flagged message data matched the selected filters.",
                ephemeral=True,
            )
            return

        # Keep leaderboard scoped to current server members as requested
        top_member_rows = [
            (author_id, flagged_count)
            for author_id, flagged_count in leaderboard_rows
            if ctx.guild.get_member(author_id) is not None
        ][:10]

        if not top_member_rows:
            await ctx.respond(
                "No current server members matched the selected filters.",
                ephemeral=True,
            )
            return

        # Render a compact top-10 leaderboard with mentions for easy moderation follow-up
        leaderboard_lines = [
            f"{rank}. <@{author_id}> - {flagged_count} flagged"
            for rank, (author_id, flagged_count) in enumerate(top_member_rows, start=1)
        ]

        embed = discord.Embed(
            title="Flag Likelihood Leaderboard",
            description="\n".join(leaderboard_lines),
            color=discord.Color.gold(),
        )
        embed.add_field(
            name="Timeframe",
            value=f"Last {days} day{'s' if days != 1 else ''}",
            inline=True,
        )
        embed.add_field(
            name="Include Waiver-Filtered",
            value="Yes" if include_waiver_filtered else "No",
            inline=True,
        )
        embed.add_field(
            name="Include All Flagged",
            value="Yes" if include_all_flagged_messages else "No (Confirmed only)",
            inline=True,
        )
        if not include_all_flagged_messages:
            embed.add_field(
                name="Confirmed Rating Definition",
                value="Rated as `unconstructive` or `unsolicited`",
                inline=False,
            )

        await ctx.respond(embed=embed, ephemeral=True)

    @discord.slash_command(
        name="mod_flag_recent",
        description="Show the most recent flagged messages with jump links",
    )
    @is_moderator()
    async def mod_flag_recent(
        self,
        ctx: discord.ApplicationContext,
        count: int = 10,
        include_waiver_filtered: bool = True,
    ):
        """
        Show the most recent flagged messages for the current guild.

        Args:
            ctx: Discord application context for the command
            count: Number of recent flagged messages to show
            include_waiver_filtered: Whether to include waiver-filtered flagged messages
        """
        # Keep the response readable and under Discord embed limits
        if count < 1 or count > 25:
            await ctx.respond(
                "Please provide a count between 1 and 25.",
                ephemeral=True,
            )
            return

        if ctx.guild is None:
            await ctx.respond(
                "This command can only be used in a server.",
                ephemeral=True,
            )
            return

        recent_rows = self._get_recent_flagged_messages(
            guild_id=ctx.guild.id,
            limit=count,
            include_waiver_filtered=include_waiver_filtered,
        )
        if not recent_rows:
            await ctx.respond(
                "No flagged messages found for this server.",
                ephemeral=True,
            )
            return

        # Build one concise line per flagged message with direct jump URLs
        recent_lines = [
            (
                f"{index}. <@{author_id}> in <#{channel_id}> at "
                f"<t:{int(flagged_at.timestamp())}:R> - "
                f"[Jump](https://discord.com/channels/{ctx.guild.id}/{channel_id}/{message_id})"
                f"{' (waiver filtered)' if waiver_filtered else ''}"
            )
            for index, (message_id, channel_id, author_id, flagged_at, waiver_filtered)
            in enumerate(recent_rows, start=1)
        ]

        embed = discord.Embed(
            title=f"Most Recent Flagged Messages ({len(recent_rows)})",
            description="\n".join(recent_lines),
            color=discord.Color.orange(),
        )
        embed.add_field(
            name="Guild Scope",
            value=ctx.guild.name,
            inline=False,
        )
        embed.add_field(
            name="Include Waiver-Filtered",
            value="Yes" if include_waiver_filtered else "No",
            inline=False,
        )
        await ctx.respond(embed=embed, ephemeral=True)

    @discord.slash_command(name="check", description="Manually trigger moderation for this channel or thread")
    @is_moderator()
    async def check(self, ctx: discord.ApplicationContext):
        """
        Manually trigger the moderation workflow for the current channel or thread.
        
        Args:
            ctx: Discord application context for the command.
        """
        channel = ctx.channel

        # Validate channel type to ensure we can moderate
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            await ctx.respond(
                "This command can only be used in a text channel or thread.",
                ephemeral=True,
            )
            return

        # Ensure the channel is part of the tracked allow list
        if not is_tracked_channel(channel):
            await ctx.respond(
                "This channel is not tracked by the moderation system.",
                ephemeral=True,
            )
            return

        # Acknowledge quickly so the interaction does not time out
        await ctx.respond(
            f"Running moderation for {channel.mention}...", ephemeral=True
        )

        # Run moderation in the background so the command handler returns quickly
        async def _run_and_report():
            try:
                result: "ModerationResult" = await self.bot.run_moderation_now(channel)
                # Build a concise summary of the moderation pass outcome
                summary_lines = [
                    f"Moderation {'succeeded' if result.success else 'failed'} for {channel.mention}",
                    f"Reason: {result.reason}",
                    f"Candidates considered: {result.candidates_considered}",
                    f"Candidates after filters: {result.candidates_after_filters}",
                    f"Flagged messages: {result.flagged_new_count} new, {result.flagged_existing_count} already flagged",
                ]
                summary = "\n".join(summary_lines)
                try:
                    await ctx.interaction.edit_original_response(content=summary)
                except Exception:
                    await ctx.interaction.followup.send(summary, ephemeral=True)
            except Exception:
                # Attempt to notify about failure without blocking the command
                try:
                    await ctx.interaction.edit_original_response(
                        content="Moderation run failed due to an unexpected error."
                    )
                except Exception:
                    await ctx.interaction.followup.send(
                        "Moderation run failed due to an unexpected error.",
                        ephemeral=True,
                    )

        asyncio.create_task(_run_and_report())

    @discord.slash_command(name="retrain", description="Manually trigger model retraining (admin only)")
    @is_admin()
    async def retrain(self, ctx: discord.ApplicationContext):
        """
        Manually trigger retraining of the moderation classifier model.

        This bypasses the normal rating count threshold and immediately
        starts a retraining run using available features.
        """
        await ctx.respond("Starting model retraining...", ephemeral=True)

        async def _run_retrain():
            try:
                from training import retrain_model

                success = await retrain_model()
                if success:
                    result_msg = "✅ Model retraining completed successfully!"
                else:
                    result_msg = "⚠️ Model retraining returned False (check logs for details)"
            except Exception as e:
                result_msg = f"❌ Model retraining failed: {e}"

            try:
                await ctx.interaction.edit_original_response(content=result_msg)
            except Exception:
                await ctx.interaction.followup.send(result_msg, ephemeral=True)

        asyncio.create_task(_run_retrain())

    async def cog_command_error(self, ctx: discord.ApplicationContext, error: Exception):
        """Handle errors for commands in this cog."""
        if isinstance(error, commands.CheckFailure):
            await ctx.respond(
                "You don't have permission to use this command. "
                f"Required roles: {', '.join(MODERATOR_ROLES)}",
                ephemeral=True
            )
        else:
            raise error


def setup(bot):
    """Called by Pycord to load this cog."""
    bot.add_cog(Restricted(bot))

