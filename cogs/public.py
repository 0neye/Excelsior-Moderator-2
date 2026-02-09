"""Public commands available to all users."""

import discord
from discord.ext import commands


class Public(commands.Cog):
    """Cog containing public commands available to all users."""
    
    def __init__(self, bot):
        self.bot = bot
        # Access to global message store and DB connection
        self.message_store = bot.message_store
        self.get_db_session = bot.get_db_session

    @discord.slash_command(name="ping", description="Check the bot's latency")
    async def ping(self, ctx: discord.ApplicationContext):
        """Responds with the bot's current latency."""
        latency_ms = round(self.bot.latency * 1000)
        await ctx.respond(f"Pong! Latency: {latency_ms}ms")

    @discord.slash_command(name="info", description="Get information about the bot")
    async def info(self, ctx: discord.ApplicationContext):
        """Displays basic information about the bot."""
        embed = discord.Embed(
            title="Ellie",
            description="A bot for the Excelsior server! See the code and more info [here](https://github.com/0neye/Excelsior-Moderator-2).\nKeeping things wholesome.",
            color=discord.Color.blue()
        )        
        await ctx.respond(embed=embed)

def setup(bot):
    """Called by Pycord to load this cog."""
    bot.add_cog(Public(bot))

