"""Event listeners for message and reaction events."""

import discord
from discord.ext import commands
from bot import ExcelsiorBot
from config import LOG_CHANNEL_ID
from history import MessageStore
from utils import is_tracked_channel
from user_stats import (
    DEFAULT_WINDOW_SIZE,
    ensure_user_stats_schema,
    update_co_occurrences_from_window,
    update_user_stats_from_message,
)


class Events(commands.Cog):
    """Cog containing event listeners for messages and reactions."""
    
    def __init__(self, bot: ExcelsiorBot):
        self.bot = bot
        # Access to global message store and DB connection
        self.message_store: MessageStore = bot.message_store
        self.get_db_session = bot.get_db_session
        # Ensure schema is migrated before realtime updates
        ensure_user_stats_schema()

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """
        Called when a message is sent in any channel the bot can see.
        
        Args:
            message: The message that was sent.
        """
        # Ignore messages from this bot
        if message.author and message.author.id == self.bot.user.id if self.bot.user else None:
            return
        
        # Only process messages from allowed channels (including threads in those channels)
        if not is_tracked_channel(message.channel):
            return
        
        self.message_store.add_message(message)

        session = self.get_db_session()
        try:
            update_user_stats_from_message(message, session)
            recent_history = self.message_store.get_whole_history(message.channel.id)
            window = recent_history[-DEFAULT_WINDOW_SIZE :]
            update_co_occurrences_from_window(window, session)
            session.commit()
        finally:
            session.close()

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """
        Called when a reaction is added to any message.
        Uses raw event to catch reactions on uncached messages.
        
        Args:
            payload: The raw reaction event data.
        """
        # Ignore reactions from this bot
        if payload.member and payload.member.id == self.bot.user.id if self.bot.user else None:
            return

        # Check for log channel rating reactions first
        if payload.channel_id == LOG_CHANNEL_ID:
            from cogs.rating import Rating
            rating_cog = self.bot.get_cog("Rating")
            if rating_cog and isinstance(rating_cog, Rating):
                handled = await rating_cog.handle_log_channel_reaction(payload)
                if handled:
                    return
        
        # Only process reactions in allowed channels (including threads)
        channel = self.bot.get_channel(payload.channel_id)
        if channel is None or not is_tracked_channel(channel):
            return
        
        # Ensure channel supports fetching messages
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return
        
        message = await channel.fetch_message(payload.message_id)
        self.message_store.update_message(message)

    @commands.Cog.listener()
    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent):
        """
        Called when a reaction is removed from any message.
        Uses raw event to catch reactions on uncached messages.
        
        Args:
            payload: The raw reaction event data.
        """
        # Ignore reactions from this bot
        if payload.member and payload.member.id == self.bot.user.id if self.bot.user else None:
            return
        
        # Only process reactions in allowed channels (including threads)
        channel = self.bot.get_channel(payload.channel_id)
        if channel is None or not is_tracked_channel(channel):
            return
        
        # Ensure channel supports fetching messages
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return
        
        message = await channel.fetch_message(payload.message_id)
        self.message_store.update_message(message)

    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """
        Called when a message is edited.
        
        Args:
            before: The message before editing.
            after: The message after editing.
        """
        
        # Only process edits in allowed channels (including threads)
        if not is_tracked_channel(after.channel):
            return
        
        # Ignore if content didn't change (e.g., embed updates)
        # if before.content == after.content:
        #     return
        # We use embed updates...
        
        self.message_store.update_message(after)

    @commands.Cog.listener()
    async def on_message_delete(self, message: discord.Message):
        """
        Called when a message is deleted.
        Note: Only works for cached messages.
        
        Args:
            message: The deleted message (if it was cached).
        """

        # Only process deletions in allowed channels (including threads)
        if not is_tracked_channel(message.channel):
            return
        
        self.message_store.delete_message(message)

    @commands.Cog.listener()
    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent):
        """
        Called when a message is deleted, including uncached messages.
        
        Args:
            payload: The raw message delete event data.
        """
        # Only process deletions in allowed channels (including threads)
        channel = self.bot.get_channel(payload.channel_id)
        if channel is None or not is_tracked_channel(channel):
            return
        
        # Try to delete but don't error if message wasn't in the store
        try:
            self.message_store.delete_message_by_id(payload.message_id, payload.channel_id)
        except ValueError:
            pass  # Message wasn't in the store


def setup(bot):
    """Called by Pycord to load this cog."""
    bot.add_cog(Events(bot))

