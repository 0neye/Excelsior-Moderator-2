"""In-memory message store using deque for efficient message history management."""

from collections import deque
from dataclasses import dataclass
from typing import Optional
import discord
from config import HISTORY_PER_CHECK, MESSAGES_PER_CHECK


@dataclass
class ChannelInfo:
    """Stores metadata about a channel or thread in the message store."""
    channel_id: int
    channel_name: str
    is_thread: bool
    parent_channel_id: Optional[int] = None
    parent_channel_name: Optional[str] = None


class MessageStore:
    """
    In-memory message store using deques with fixed maximum size per channel.
    Automatically removes oldest messages when capacity is reached for each channel.
    Maintains separate message histories for each channel ID.
    """
    
    def __init__(self):
        """Initialize the message store with a dictionary mapping channel IDs to deques."""
        max_size = HISTORY_PER_CHECK + MESSAGES_PER_CHECK
        # Dictionary mapping channel_id to deque of messages for that channel
        self._channel_messages: dict[int, deque[discord.Message]] = {}
        # Dictionary mapping channel_id to metadata about that channel/thread
        self._channel_info: dict[int, ChannelInfo] = {}
        self._max_size = max_size
    
    def _get_channel_deque(self, channel_id: int) -> deque[discord.Message]:
        """
        Get or create the deque for a specific channel.
        
        Args:
            channel_id: The Discord channel ID.
            
        Returns:
            The deque for the specified channel.
        """
        if channel_id not in self._channel_messages:
            self._channel_messages[channel_id] = deque(maxlen=self._max_size)
        return self._channel_messages[channel_id]
    
    def _update_channel_info(self, channel: discord.TextChannel | discord.Thread) -> None:
        """
        Update or create the channel info for a channel/thread.
        Extracts channel name and parent info for threads.
        
        Args:
            channel: The Discord channel or thread.
        """
        channel_id = channel.id
        
        if isinstance(channel, discord.Thread):
            # For threads, store thread name and parent channel info
            parent = channel.parent
            self._channel_info[channel_id] = ChannelInfo(
                channel_id=channel_id,
                channel_name=channel.name,
                is_thread=True,
                parent_channel_id=parent.id if parent else None,
                parent_channel_name=parent.name if parent else None
            )
        else:
            # For regular channels, just store the channel name
            self._channel_info[channel_id] = ChannelInfo(
                channel_id=channel_id,
                channel_name=channel.name,
                is_thread=False
            )
    
    def get_channel_info(self, channel_id: int) -> Optional[ChannelInfo]:
        """
        Get the stored metadata for a channel/thread.
        
        Args:
            channel_id: The Discord channel ID.
            
        Returns:
            The ChannelInfo for the channel, or None if not found.
        """
        return self._channel_info.get(channel_id)
    
    def get_all_channel_info(self) -> dict[int, ChannelInfo]:
        """
        Get all stored channel metadata.
        
        Returns:
            Dictionary mapping channel IDs to their ChannelInfo.
        """
        return self._channel_info.copy()
    
    def add_message(self, message: discord.Message) -> None:
        """
        Add a message to the store for its channel.
        If the store is at capacity for that channel, the oldest message is automatically removed.
        Also updates channel metadata (name, parent info for threads).
        
        Args:
            message: The discord message to add.
        """
        channel = message.channel
        channel_id = channel.id
        channel_deque = self._get_channel_deque(channel_id)
        channel_deque.append(message)
        
        # Update channel info (handles both TextChannel and Thread)
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            self._update_channel_info(channel)
    
    def update_message(self, message: discord.Message) -> None:
        """
        Update a message in the store for its channel.
        If the message is not found, it is added to the store.
        
        Args:
            message: The discord message to update.
        """
        channel_id = message.channel.id
        channel_deque = self._get_channel_deque(channel_id)
        index = next((i for i, m in enumerate(channel_deque) if m.id == message.id), None)
        if index is not None:
            channel_deque[index] = message
        else:
            self.add_message(message)

    
    def delete_message(self, message: discord.Message) -> None:
        """
        Delete a message from the store for its channel.

        Args:
            message: The discord message to remove from the in-memory store.

        Note:
            We search by ID instead of relying on object identity because delete
            events can supply a fresh Message instance that differs from the one
            cached in the deque.
        """
        channel_id = message.channel.id
        channel_deque = self._get_channel_deque(channel_id)
        message_index = next((i for i, stored in enumerate(channel_deque) if stored.id == message.id), None)
        if message_index is None:
            return  # Message already evicted or never tracked; safe no-op
        del channel_deque[message_index]
    
    def delete_message_by_id(self, message_id: int, channel_id: int) -> None:
        """
        Delete a message from the store for its channel by its ID.
        """
        channel_deque = self._get_channel_deque(channel_id)
        index = next((i for i, m in enumerate(channel_deque) if m.id == message_id), None)
        if index is not None:
            del channel_deque[index]
        else:
            raise ValueError(f"Message with ID {message_id} not found in channel {channel_id}")
    
    def get_most_recent_message(self, channel_id: int) -> Optional[discord.Message]:
        """
        Get the most recently added message in a specific channel.
        
        Args:
            channel_id: The Discord channel ID to search in.
        
        Returns:
            The most recent message in the channel, or None if the channel store is empty.
        """
        channel_deque = self._channel_messages.get(channel_id)
        return channel_deque[-1] if channel_deque and channel_deque else None
    
    def get_oldest_message(self, channel_id: int) -> Optional[discord.Message]:
        """
        Get the oldest message in a specific channel's store.
        
        Args:
            channel_id: The Discord channel ID to search in.
        
        Returns:
            The oldest message in the channel, or None if the channel store is empty.
        """
        channel_deque = self._channel_messages.get(channel_id)
        return channel_deque[0] if channel_deque and channel_deque else None
    
    def get_message_by_id(self, message_id: int, channel_id: int) -> Optional[discord.Message]:
        """
        Find a message by its Discord ID within a specific channel.
        
        Args:
            message_id: The Discord message ID to search for.
            channel_id: The Discord channel ID to search in.
            
        Returns:
            The message with the matching ID in the channel, or None if not found.
        """
        channel_deque = self._channel_messages.get(channel_id)
        if not channel_deque:
            return None
        # Search from most recent to oldest (most likely to find recent messages first)
        for message in reversed(channel_deque):
            if message.id == message_id:
                return message
        return None
    
    def get_most_recent_message_by_user(self, user_id: int, channel_id: int) -> Optional[discord.Message]:
        """
        Get the most recent message from a specific user in a specific channel.
        
        Args:
            user_id: The Discord user ID to search for.
            channel_id: The Discord channel ID to search in.
            
        Returns:
            The most recent message from the user in the channel, or None if not found.
        """
        channel_deque = self._channel_messages.get(channel_id)
        if not channel_deque:
            return None
        # Search from most recent to oldest
        for message in reversed(channel_deque):
            if message.author.id == user_id:
                return message
        return None
    
    def get_whole_history(self, channel_id: int) -> list[discord.Message]:
        """
        Get all messages in a specific channel's store as a list.
        
        Args:
            channel_id: The Discord channel ID to get history for.
        
        Returns:
            A list of all messages in the channel, ordered from oldest to newest.
        """
        channel_deque = self._channel_messages.get(channel_id)
        return list(channel_deque) if channel_deque else []

