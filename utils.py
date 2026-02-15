from typing import Any, Optional, Tuple, List, Union
import discord

from config import CHANNEL_ALLOW_LIST

# Type alias for channels that can be checked for tracking
# (includes PrivateChannel to handle get_channel return type, though we never track DMs)
TrackableChannel = Union[
    discord.TextChannel,
    discord.Thread,
    discord.ForumChannel,
    discord.abc.GuildChannel,
    discord.abc.PrivateChannel,
    discord.abc.Messageable,
]


def is_tracked_channel(channel: TrackableChannel) -> bool:
    """
    Check if a channel or thread should be tracked based on the allow list.
    For threads, checks if the parent channel is in the allow list.
    Returns False for DMs and other non-guild channels.
    
    Args:
        channel: The channel or thread to check.
        
    Returns:
        True if the channel should be tracked, False otherwise.
    """
    # Only track guild channels and threads, not DMs
    channel_id = getattr(channel, "id", None)
    if channel_id is None:
        return False
    
    # Direct match for channels in the allow list
    if channel_id in CHANNEL_ALLOW_LIST:
        return True
    
    # For threads, check if the parent is tracked
    if isinstance(channel, discord.Thread) and channel.parent_id in CHANNEL_ALLOW_LIST:
        return True
    
    return False


def serialize_context_messages(
    messages: List[discord.Message],
    channel_name: str | None = None,
    parent_channel_name: str | None = None,
    thread_name: str | None = None,
) -> Tuple[List[int], List[dict[str, Any]]]:
    """
    Convert Discord message objects into JSON-serializable payloads for DB storage.

    Args:
        messages: Messages to serialize, typically a contextual window around a flagged message
        channel_name: Optional channel name override to embed in every payload
        parent_channel_name: Optional parent channel name (used when messages are from a thread)
        thread_name: Optional thread name override to embed in every payload

    Returns:
        Tuple of (context_message_ids, serialized_context_messages) preserving message order
    """
    context_ids: List[int] = []
    serialized_messages: List[dict[str, Any]] = []

    for message in messages:
        context_ids.append(message.id)

        resolved_channel_name = channel_name
        resolved_thread_name = thread_name
        resolved_parent_channel = parent_channel_name

        channel_obj = getattr(message, "channel", None)
        if resolved_channel_name is None and channel_obj is not None:
            if isinstance(channel_obj, discord.Thread):
                resolved_thread_name = resolved_thread_name or channel_obj.name
                if channel_obj.parent:
                    resolved_parent_channel = resolved_parent_channel or channel_obj.parent.name
                    resolved_channel_name = resolved_parent_channel or channel_obj.name
                else:
                    resolved_channel_name = channel_obj.name
            elif hasattr(channel_obj, "name"):
                resolved_channel_name = getattr(channel_obj, "name", None)

        serialized_messages.append(
            {
                "id": message.id,
                "content": message.content,
                "author_id": message.author.id,
                "author_name": message.author.display_name,
                "author_username": message.author.name,
                "timestamp": message.created_at.isoformat(),
                "edited_at": message.edited_at.isoformat() if message.edited_at else None,
                "reference_id": message.reference.message_id if message.reference else None,
                "attachments": len(message.attachments) > 0,
                "reactions": [(str(reaction.emoji), reaction.count) for reaction in message.reactions],
                "channel_name": resolved_channel_name,
                "parent_channel_name": resolved_parent_channel,
                "thread_name": resolved_thread_name,
            }
        )

    return context_ids, serialized_messages

async def get_user_names(
    bot: discord.Bot, guild: discord.Guild, user_id: int
) -> Tuple[str, str]:
    """
    Get a user's display name, handling cases where the user is not in the guild.

    Args:
        guild (discord.Guild): The guild the user is in (hopefully).
        user_id (int): The ID of the user.

    Returns:
        Tuple[str, str]: The display name of the user and their global username. Can be identical.
    """
    try:
        member: discord.Member = guild.get_member(user_id) or await guild.fetch_member(
            user_id
        )
        return member.display_name, member.name
    except discord.errors.NotFound:
        user: Optional[discord.User] = await bot.get_or_fetch_user(user_id)
        if user is None:
            return "User", f"#{user_id}"

        return user.display_name, user.name


def _resolve_reference_display_name(message: discord.Message) -> str | None:
    """
    Safely extract the display name for a replied-to message, handling deleted references.

    Args:
        message: The message whose reference should be inspected.

    Returns:
        Optional display name string when the reference is present and resolvable.
    """
    if not (message.reference and message.reference.resolved):
        return None
    resolved = message.reference.resolved
    author = getattr(resolved, "author", None)
    if author and getattr(author, "display_name", None):
        return author.display_name
    # DeletedReferencedMessage lacks an author attribute; provide a placeholder.
    return "deleted message"


def _resolve_reference_username(message: discord.Message) -> str | None:
    """
    Safely extract the username for a replied-to message, handling deleted references.

    Args:
        message: The message whose reference should be inspected.

    Returns:
        Optional username string when the reference is present and resolvable.
    """
    if not (message.reference and message.reference.resolved):
        return None
    resolved = message.reference.resolved
    author = getattr(resolved, "author", None)
    if author and getattr(author, "name", None):
        return author.name
    # DeletedReferencedMessage lacks an author attribute; provide a placeholder.
    return "deleted message"


def format_discord_message(
    message: discord.Message,
    relative_id: int | None = None,
    reply_rel_id: int | None = None,
    use_username: bool = False,
    include_timestamp: bool = False,
) -> str:
    """
    Format a single discord message as a string for passing to llm.

    Args:
        message (discord.Message): The message to format
        relative_id (int, optional): Relative ID for the message. Defaults to None.
        reply_rel_id (int, optional): Relative ID of the message being replied to. Defaults to None.
        use_username (bool, optional): Whether to use usernames instead of display names. Defaults to False.
        include_timestamp (bool, optional): Whether to include a formatted timestamp. Defaults to False.

    Returns:
        str: Formatted string representation of the message
    """
    timestamp = ""
    if include_timestamp:
        # Format timestamp as time-only to keep conversational sequencing context
        formatted_time = message.created_at.strftime("%H:%M:%S")
        timestamp = f"[{formatted_time}] "
    
    rel_id = f"({relative_id}) " if relative_id is not None else ""
    reply = ""
    reference_name = (
        _resolve_reference_username(message)
        if use_username
        else _resolve_reference_display_name(message)
    )
    if reference_name:
        pinged = len(message.mentions) > 0
        reply_str = (
            f"{reply_rel_id}"
            if reply_rel_id
            else f"{'@' if pinged else ''}{reference_name}"
        )
        reply = f"[reply to {reply_str}] "

    content = message.content
    if message.attachments:
        content += " [uploaded attachment/image]"

    author_name = message.author.name if use_username else message.author.display_name
    msg = f"{author_name}: ❝{content}❞"
    edited = " (edited)" if message.edited_at else ""
    reactions = (
        "\n[reactions: "
        + ", ".join([f"{r.emoji} {r.count}" for r in message.reactions])
        + "]"
        if message.reactions
        else ""
    )

    return (timestamp + rel_id + reply + msg + edited + reactions).strip()


def format_message_history(
    messages: List[discord.Message], use_username: bool = False, include_timestamp: bool = False
) -> List[str]:
    """
    Format a list of discord messages as a list of strings for passing to llm.
    Assigns relative IDs to messages based on their position in the list.
    
    Args:
        messages: List of discord messages to format, ordered from oldest to newest.
        use_username (bool, optional): Whether to use usernames instead of display names. Defaults to False.
        include_timestamp (bool, optional): Whether to include formatted timestamps. Defaults to False.
        
    Returns:
        List of formatted string representations of the messages.
    """
    if not messages:
        return []
    
    # Create a mapping of message IDs to relative IDs for reply resolution
    message_id_to_rel_id = {msg.id: idx + 1 for idx, msg in enumerate(messages)}
    
    formatted = []
    for idx, message in enumerate(messages):
        relative_id = idx + 1
        reply_rel_id = None
        
        # If this message is a reply, find the relative ID of the replied-to message
        if message.reference and message.reference.message_id:
            replied_to_id = message.reference.message_id
            reply_rel_id = message_id_to_rel_id.get(replied_to_id)
        
        formatted.append(
            format_discord_message(message, relative_id, reply_rel_id, use_username, include_timestamp)
        )
    
    return formatted


async def get_discord_message_by_id(
    channel: discord.TextChannel | discord.Thread, discord_message_id: int, fetch: bool = False
) -> discord.Message | None:
    """
    Retrieve a discord message by its ID from the discord API.

    Args:
        channel: Text channel or thread to get the message from.
        discord_message_id: ID of the message to retrieve.
        fetch: Whether to force fetch the message. Defaults to False.

    Returns:
        The retrieved message, or None if not found.
    """
    if fetch:
        return await channel.fetch_message(discord_message_id)
    else:
        # Try cache first, then fetch
        cached = channel.get_partial_message(discord_message_id)
        try:
            return await cached.fetch()
        except discord.NotFound:
            return None


async def respond_long_message(
    interaction: discord.Interaction,
    text: str,
    chunk_size: int = 1800,
    use_codeblock: bool = False,
    **kwargs,
):
    """
    Sends a message longer than discord's character limit by chunking it.

    Args:
        interaction (discord.Interaction): Interaction to respond to
        text (str): Text to send
        chunk_size (int, optional): Size of each chunk. Defaults to 1800.
        use_codeblock (bool, optional): Whether to wrap text in codeblocks. Defaults to False.
        **kwargs: Additional arguments to pass to interaction.respond()
    """
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    for chunk in chunks:
        if use_codeblock:
            chunk = f"```md\n{chunk}\n```"

        await interaction.respond(chunk, **kwargs)


async def send_long_message(
    channel: discord.abc.Messageable,
    text: str,
    chunk_size: int = 1800,
    use_codeblock: bool = False,
    **kwargs,
):
    """
    Sends a message longer than discord's character limit by chunking it.

    Args:
        channel (discord.abc.Messageable): Channel to send the message to
        text (str): Text to send
        chunk_size (int, optional): Size of each chunk. Defaults to 1800.
        use_codeblock (bool, optional): Whether to wrap text in codeblocks. Defaults to False.
        **kwargs: Additional arguments to pass to channel.send()
    """
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    for chunk in chunks:
        if use_codeblock:
            chunk = f"```md\n{chunk}\n```"

        await channel.send(chunk, **kwargs)


def format_markdown_table(headers: List[str], rows: List[tuple]) -> str:
    """
    Generate a markdown table with consistent column widths matching the longest content.

    Args:
        headers: List of header strings for each column
        rows: List of tuples, where each tuple represents a data row with values for each column

    Returns:
        A formatted markdown table string with consistent column widths
    """
    if not headers:
        return ""

    # Convert all row values to strings
    string_rows = [[str(cell) for cell in row] for row in rows]

    # Calculate maximum width for each column
    column_widths = []
    for col_idx in range(len(headers)):
        header_width = len(headers[col_idx])
        max_data_width = max((len(row[col_idx]) for row in string_rows), default=0)
        column_widths.append(max(header_width, max_data_width))

    # Build header row
    header_row = (
        "| "
        + " | ".join(
            f"{header:<{column_widths[i]}}" for i, header in enumerate(headers)
        )
        + " |"
    )

    # Build separator row
    separator_row = "|" + "|".join("-" * (width + 2) for width in column_widths) + "|"

    # Build data rows
    table_lines = [header_row, separator_row]
    for row in string_rows:
        data_row = (
            "| "
            + " | ".join(f"{cell:<{column_widths[i]}}" for i, cell in enumerate(row))
            + " |"
        )
        table_lines.append(data_row)

    return "\n".join(table_lines)
