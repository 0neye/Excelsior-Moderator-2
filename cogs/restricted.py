"""Restricted commands available only to moderators."""

import discord
from discord.ext import commands

from config import MODERATOR_ROLES


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

