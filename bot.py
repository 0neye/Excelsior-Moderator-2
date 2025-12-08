"""Main entry point for the Excelsior Moderator Discord bot."""

from datetime import datetime, timezone
from typing import Callable

import discord
import numpy as np
from sqlalchemy.orm import Session

from config import DISCORD_BOT_TOKEN, CHANNEL_ALLOW_LIST, MESSAGES_PER_CHECK, WAIVER_ROLE_NAME, REACTION_EMOJI
from database import FlaggedMessage
from db_config import init_db, get_session
from history import MessageStore
from llms import get_candidate_features
from ml import FEATURE_NAMES, load_classifier
from utils import serialize_context_messages

# Set up intents for the bot
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.members = True
intents.reactions = True
intents.message_content = True


class ExcelsiorBot(discord.Bot):
    """Custom bot class with message store and DB session access."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Global in-memory message store
        self.message_store: MessageStore = MessageStore()
        # Function to get DB sessions
        self.get_db_session: Callable[[], Session] = get_session
    

    async def flag_message(self, message: discord.Message) -> None:
        """
        Flags a message by adding a reaction to it.
        TODO: also add a log message to the mod channel
        """
        await message.add_reaction(REACTION_EMOJI)


    async def moderate_channel(self, channel: discord.TextChannel | discord.Thread) -> None:
        """
        Moderates a channel by running the moderation workflow on the in-memory message store for that channel.
        
        Args:
            channel: The channel to moderate.
        """

        # Copy at time of call to avoid race conditions
        store_history_copy = self.message_store.get_whole_history(channel.id)

        candidates = await get_candidate_features(self.message_store, channel.id, ignore_first_message_count=MESSAGES_PER_CHECK-1)

        # Filter anything with discusses_ellie > 0.2 since that's likely bot-related noise
        candidates = [candidate for candidate in candidates if candidate["features"]["discusses_ellie"] <= 0.2]

        if not candidates:
            return

        # Filter out anything directed at a user with the WAIVER_ROLE_NAME role
        candidate_messages = [next((msg for msg in store_history_copy if msg.id == candidate["discord_message_id"]), None) for candidate in candidates]

        candidate_message_authors = [message.author for message in candidate_messages if message is not None]

        for candidate, message_author in zip(candidates, candidate_message_authors):
            if message_author.roles and any(role.name == WAIVER_ROLE_NAME for role in message_author.roles):
                candidates.remove(candidate)


        if not candidates:
            return

        # Filter out candidates that are too close to the beginning of the channel history
        # since the model is unlikely to have enough context to extract good features
        # This is redundant, but can't hurt
        candidates = [
            candidate for candidate in candidates
            if candidate.get("message_id", 0) > MESSAGES_PER_CHECK # message_id is 1-based index
        ]

        if not candidates:
            return
            
        # Get the actual message objects for the candidates
        candidate_messages = [next((msg for msg in store_history_copy if msg.id == candidate["discord_message_id"]), None) for candidate in candidates]

        # Then send to the ml pipeline
        classifier = load_classifier("models/lightgbm_model.joblib")
        feature_matrix = np.array(
            [
                [float(candidate["features"].get(name, 0.0)) for name in FEATURE_NAMES]
                for candidate in candidates
            ],
            dtype=float,
        )

        # Predict using a numpy feature matrix aligned to training order
        predictions = classifier.predict(feature_matrix)
        
        db_session = self.get_db_session()

        for candidate_message, prediction in zip(candidate_messages, predictions):
            if prediction == "flag" and candidate_message is not None:
                await self.flag_message(candidate_message)
                surrounding_context = [msg for msg in store_history_copy if msg.id != candidate_message.id]
                # Serialize surrounding context for persistent storage
                context_ids, serialized_context = serialize_context_messages(surrounding_context)

                flagged_message = FlaggedMessage(
                    message_id=candidate_message.id,
                    channel_id=channel.id,
                    guild_id=channel.guild.id,
                    author_id=candidate_message.author.id,
                    author_username=candidate_message.author.name,
                    content=candidate_message.content,
                    context_message_ids=context_ids,
                    context_messages=serialized_context,
                    timestamp=candidate_message.created_at.isoformat() if candidate_message.created_at else None,
                    flagged_at=datetime.now(timezone.utc).isoformat(),
                    target_user_id=candidate.get("target_user_id"),
                    target_username=candidate.get("target_username"),
                )
                db_session.add(flagged_message)
                db_session.commit()
            elif prediction == "no_flag":
                pass



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
    Backfill message history for all threads (active and archived) in a channel.
    
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
    
    # Get active threads from the channel's cached threads list
    for thread in channel.threads:
        count = await backfill_channel_history(thread, message_store, max_messages)
        total_messages += count
        thread_count += 1
        thread_names.append(thread.name)
    
    # Get archived threads (both public and private if accessible)
    try:
        async for thread in channel.archived_threads(limit=None):
            count = await backfill_channel_history(thread, message_store, max_messages)
            total_messages += count
            thread_count += 1
            thread_names.append(thread.name)
    except discord.Forbidden:
        pass  # Bot doesn't have permission to view archived threads
    
    # Try to get private archived threads if we have permission
    try:
        async for thread in channel.archived_threads(limit=None, private=True):
            # Skip if we already processed this thread (it might be in both lists)
            if message_store.get_most_recent_message(thread.id) is not None:
                continue
            count = await backfill_channel_history(thread, message_store, max_messages)
            total_messages += count
            thread_count += 1
            thread_names.append(thread.name)
    except (discord.Forbidden, TypeError):
        pass  # Bot doesn't have permission or channel doesn't support private threads
    
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
                print(f"  Backfilled forum #{channel.name}: {thread_count} threads, {msg_count} messages")
                # Log thread names with their parent forum
                for name in thread_names:
                    print(f"    - Thread '{name}' (parent: #{channel.name})")
                
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
                
                print(f"  Backfilled #{channel.name}: {msg_count} messages, {thread_count} threads ({thread_msgs} thread messages)")
                # Log thread names with their parent channel
                for name in thread_names:
                    print(f"    - Thread '{name}' (parent: #{channel.name})")
    
    print(f"Backfill complete: {total_channels} channels, {total_threads} threads, {total_messages} total messages")


@bot.event
async def on_ready():
    """Called when the bot has successfully connected to Discord."""
    if bot.user:
        print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print(f"Connected to {len(bot.guilds)} guild(s)")
    
    # Initialize database tables
    init_db()
    print("Database initialized")
    
    # Backfill message history for tracked channels and threads
    print("Backfilling message history...")
    await backfill_message_store()
    
    print("Bot is ready!")


def load_cogs():
    """Load all cog extensions from the cogs folder."""
    cog_list = [
        "cogs.public",
        "cogs.restricted",
        "cogs.events",
    ]
    
    for cog in cog_list:
        try:
            bot.load_extension(cog)
            print(f"Loaded cog: {cog}")
        except Exception as e:
            print(f"Failed to load cog {cog}: {e}")


if __name__ == "__main__":
    load_cogs()
    bot.run(DISCORD_BOT_TOKEN)

