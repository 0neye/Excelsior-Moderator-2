"""Restricted commands available only to moderators."""

from typing import TYPE_CHECKING

import asyncio
import discord
from discord.ext import commands

from config import MODERATOR_ROLES
from utils import is_tracked_channel

if TYPE_CHECKING:
    from bot import ModerationResult


ADMIN_ROLE = "Custodian (admin)"


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

