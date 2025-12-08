import asyncio
import json
from typing import Any, Callable, Literal

from cerebras.cloud.sdk import Cerebras
from google import genai
from openai import OpenAI

from config import CEREBRAS_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY
from db_config import get_session
from history import MessageStore
from user_stats import get_familiarity_score_stat, get_seniority_scores
from utils import format_message_history

cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)

# OpenRouter client using the OpenAI-compatible API
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Gemini client using the newer google-genai SDK
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# JSON schema for structured output of candidate features
def _strip_additional_properties(schema: Any) -> Any:
    """
    Recursively remove additionalProperties keys from a JSON schema.
    
    Gemini's response_schema rejects the additional_properties field, so we strip it
    while keeping the stricter version for providers that support it.
    """
    if isinstance(schema, dict):
        return {
            key: _strip_additional_properties(value)
            for key, value in schema.items()
            if key != "additionalProperties"
        }
    if isinstance(schema, list):
        return [_strip_additional_properties(item) for item in schema]
    return schema


CANDIDATE_FEATURES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "target_username": {"type": "string"},
                    "features": {
                        "type": "object",
                        "properties": {
                            "discusses_ellie": {"type": "number"},
                            "familiarity_score": {"type": "number"},
                            "tone_harshness_score": {"type": "number"},
                            "positive_framing_score": {"type": "number"},
                            "includes_positive_takeaways": {"type": "number"},
                            "explains_why_score": {"type": "number"},
                            "actionable_suggestion_score": {"type": "number"},
                            "context_is_feedback_appropriate": {"type": "number"},
                            "target_uncomfortableness_score": {"type": "number"},
                            "is_part_of_discussion": {"type": "number"},
                            "criticism_directed_at_image": {"type": "number"},
                            "criticism_directed_at_statement": {"type": "number"},
                            "criticism_directed_at_generality": {"type": "number"},
                            "reciprocity_score": {"type": "number"},
                    "solicited_score": {"type": "number"},
                    "seniority_score_messages": {"type": "number"},
                    "seniority_score_characters": {"type": "number"},
                    "familiarity_score_stat": {"type": "number"},
                        },
                        "required": [
                            "discusses_ellie", "familiarity_score", "tone_harshness_score",
                            "positive_framing_score", "includes_positive_takeaways", "explains_why_score",
                            "actionable_suggestion_score", "context_is_feedback_appropriate",
                            "target_uncomfortableness_score", "is_part_of_discussion",
                    "criticism_directed_at_image", "criticism_directed_at_statement",
                    "criticism_directed_at_generality", "reciprocity_score", "solicited_score"
                        ],
                        "additionalProperties": False
                    }
                },
                "required": ["message_id", "target_username", "features"],
                "additionalProperties": False
            }
        }
    },
    "required": ["candidates"],
    "additionalProperties": False
}

# Gemini requires a schema without additionalProperties fields
CANDIDATE_FEATURES_SCHEMA_GEMINI = _strip_additional_properties(CANDIDATE_FEATURES_SCHEMA)


async def extract_features_from_formatted_history(
    formatted_message_history: str,
    channel_name: str,
    thread_name: str | None = None,
    provider: Literal["cerebras", "openrouter", "gemini"] = "cerebras",
    openrouter_model: str = "openai/gpt-oss-120b",
    gemini_model: str = "gemini-2.5-flash",
    required_message_indexes: list[int] | None = None,
    author_id_map: dict[str, int] | None = None,
    username_id_map: dict[str, int] | None = None,
    stats_session_factory: Callable[[], Any] | None = None,
    ignore_first_message_count: int = 0,
) -> list[dict]:
    """
    Extracts candidate features from formatted message history using LLM.
    
    Args:
        formatted_message_history: The formatted message history string
        channel_name: Name of the Discord channel
        thread_name: Optional name of the thread if applicable
        provider: Which LLM provider to use ("cerebras", "openrouter", or "gemini")
        openrouter_model: The model to use when provider is "openrouter"
        gemini_model: The model to use when provider is "gemini"
        required_message_indexes: Optional list of relative message IDs that MUST have
            features extracted, regardless of whether they appear to be feedback candidates
        author_id_map: Optional mapping of message_id/rel_id strings to author IDs for stat features
        username_id_map: Optional mapping of usernames/display names to user IDs
        stats_session_factory: Optional callable to create a DB session for stat lookups
        ignore_first_message_count: Number of leading messages to treat purely as context when extracting candidates
    
    Returns:
        A list of candidate dicts, each containing message_id, target_username, and features.
    """

    def _augment_stat_features(candidates: list[dict]) -> list[dict]:
        """Attach seniority/familiarity stat features to candidates."""
        if not stats_session_factory or not author_id_map:
            # Ensure keys exist even without DB lookups
            for candidate in candidates:
                features = candidate.get("features", {})
                features.setdefault("seniority_score_messages", 0.0)
                features.setdefault("seniority_score_characters", 0.0)
                features.setdefault("familiarity_score_stat", 0.0)
                candidate["features"] = features
            return candidates

        session = stats_session_factory()
        try:
            for candidate in candidates:
                features = candidate.get("features", {})
                features.setdefault("seniority_score_messages", 0.0)
                features.setdefault("seniority_score_characters", 0.0)
                features.setdefault("familiarity_score_stat", 0.0)

                message_id = candidate.get("message_id")
                target_username = candidate.get("target_username")
                author_id = author_id_map.get(str(message_id)) if message_id is not None else None
                target_id = (
                    username_id_map.get(target_username)
                    if username_id_map and isinstance(target_username, str)
                    else None
                )

                if author_id is not None and target_id is not None:
                    msg_score, char_score = get_seniority_scores(author_id, target_id, session)
                    fam_score = get_familiarity_score_stat(author_id, target_id, session)
                    features["seniority_score_messages"] = msg_score
                    features["seniority_score_characters"] = char_score
                    features["familiarity_score_stat"] = fam_score

                candidate["features"] = features
        finally:
            session.close()

        return candidates

    # Build the required messages instruction if any are specified
    required_msg_instruction = ""
    if required_message_indexes:
        required_ids_str = ", ".join(str(idx) for idx in required_message_indexes)
        required_msg_instruction = f"""
    ### Required Messages:
    You MUST extract features for the following message IDs (rel_ids): {required_ids_str}
    These messages must appear in your output even if they don't seem like typical feedback candidates. Treat them like you would any other feedback candidate.
    If a required message doesn't seem to target anyone specifically, use "Nobody" for target_username and set all features to appropriate values based on the message content.
    """

    ignore_leading_messages_instruction = ""
    if ignore_first_message_count > 0:
        ignore_leading_messages_instruction = f"""
    ### Ignore Context-Only Messages:
    The first {ignore_first_message_count} messages in the provided history are context only. Do not extract candidates from them or treat them as targets; begin considering candidates starting with message {ignore_first_message_count + 1}.
    """

    system_prompt = f"""
    You will be given a portion of a Discord channel's message history.
    Your job is to follow the process below.

    ### Process:
    1. Identify all messages that could be considered providing feedback or criticism to another user present in the conversation.
    2. Determine the target of each candidate message. Who is the message criticizing? The target must be a user present in the conversation.
    3. For each candidate message, provide a list of float values between 0 and 1 based on the features mentioned below.
    4. Output a JSON object matching the provided schema with all candidates and their features.

    ### Features (all float values between 0 and 1):
    - "discusses_ellie": The degree to which "Ellie" is a significant topic of discussion in the message history provided. Ellie's @ mention is "<@1333546604694732892>," which also counts.
    - "familiarity_score": How familiar the author of the candidate message seems to be with the target user.
    - "tone_harshness_score": Overall harshness/condescension level independent of content.
    - "positive_framing_score": Does it use positive language ("I like X", "consider trying Y") vs negative ("X is wrong", "your layout is unoptimal")?
    - "includes_positive_takeaways": Does the message acknowledge something the target did well?
    - "explains_why_score": Does it explain the why behind suggestions, not just the what?
    - "actionable_suggestion_score": Does it give specific, actionable advice vs vague criticism?
    - "context_is_feedback_appropriate": Is this a role request, classroom post, module post, etc. where feedback is expected?
    - "target_uncomfortableness_score": How uncomfortable the target user seems to be with the candidate message. This could be trying to defend themselves or their position, sad emoji, etc.
    - "is_part_of_discussion": Whether the candidate message is part of a two-way discussion about a specific topic in which both the author and target user are engaged.
    - "criticism_directed_at_image": The degree to which the candidate message could be directed at a specific posted image/attachment.
    - "criticism_directed_at_statement": The degree to which the candidate message could be criticizing a specific statement or idea made by the target user.
    - "criticism_directed_at_generality": The degree to which the candidate message could be criticizing a general topic or concept not specific to the target user, such as the game itself or the server.
    - "reciprocity_score": The degree to which the candidate message could be criticizing a negative behavior of the target user themselves, such as being rude, disrespectful, etc. This includes scolding people for giving bad feedback, being too harsh, etc.
    - "solicited_score": How solicited (1) to unsolicited (0) the candidate message seems to be. Something as simple as ending their message with a question mark can be considered solicitation of feedback, but use your best judgement.
    
    ### Remember:
    - The message history is provided in chronological order, from oldest to newest.
    - The message history format is: [timestamp] (rel_id) [reply to rel_id or username] ❝message content❞ (edited) [reactions]
    - This is a gaming server discussing the game Cosmoteer: Starship Architect & Commander.
    {required_msg_instruction}{ignore_leading_messages_instruction}"""

    user_prompt = f"""
    ### Discord context:
    Channel name: {channel_name}
    {f"Thread name: {thread_name}" if thread_name else ""}

    Message history:
    \"\"\"
    {formatted_message_history}
    \"\"\"
    """
    # TODO: Add dictionary stuff in the future

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if provider == "cerebras":
        # Run synchronous Cerebras API call in a thread pool to avoid blocking
        response: Any = await asyncio.to_thread(
            lambda: cerebras_client.chat.completions.create(
                model="gpt-oss-120b",
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "candidate_features",
                        "strict": True,
                        "schema": CANDIDATE_FEATURES_SCHEMA
                    }
                },
                stream=False,
                max_completion_tokens=65536,
            )
        )
    elif provider == "openrouter":
        # Run synchronous OpenRouter API call in a thread pool to avoid blocking
        # Cast messages to Any since OpenAI SDK has strict typing
        response = await asyncio.to_thread(
            lambda: openrouter_client.chat.completions.create(
                model=openrouter_model,
                messages=messages,  # type: ignore[arg-type]
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "candidate_features",
                        "strict": True,
                        "schema": CANDIDATE_FEATURES_SCHEMA
                    }
                },
                max_tokens=65536,
                extra_body={"reasoning": {"enabled": True}}
            )
        )
    elif provider == "gemini":
        
        # Run synchronous Gemini API call in a thread pool to avoid blocking
        gemini_response = await asyncio.to_thread(
            lambda: gemini_client.models.generate_content(
                model=gemini_model,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CANDIDATE_FEATURES_SCHEMA_GEMINI,
                    system_instruction=system_prompt
                )
            )
        )
        
        # Parse the Gemini response directly
        gemini_content = gemini_response.text
        if gemini_content is None:
            raise ValueError("Gemini returned empty response")
        gemini_result: dict[str, Any] = json.loads(gemini_content)
        candidates = gemini_result["candidates"]
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if provider in {"cerebras", "openrouter"}:
        # Parse the structured JSON response
        content: str = response.choices[0].message.content
        result: dict[str, Any] = json.loads(content)
        candidates = result["candidates"]

    return _augment_stat_features(candidates)


async def get_candidate_features(
    message_store: MessageStore,
    channel_id: int,
    required_message_indexes: list[int] | None = None,
    ignore_first_message_count: int = 0,
) -> list[dict]:
    """
    Identifies potential candidate messages to flag based on the message history.
    Then asks a series of very specific questions about the message history to the LLM to extract features from each candidate.
    
    Args:
        message_store: The MessageStore containing channel history
        channel_id: The Discord channel ID to analyze
        required_message_indexes: Optional list of relative message IDs that MUST have
            features extracted, regardless of whether they appear to be feedback candidates
        ignore_first_message_count: Number of leading messages to treat purely as context when asking the LLM for features
    
    Returns:
        A list of candidate dicts, each containing message_id, target_username,
        target_user_id, and features.
    """

    channel_info = message_store.get_channel_info(channel_id)
    if channel_info is None:
        raise ValueError(f"Channel with ID {channel_id} not found in message store")
    channel_name = channel_info.channel_name if not channel_info.is_thread else channel_info.parent_channel_name
    thread_name = channel_info.channel_name if channel_info.is_thread else None

    message_history = message_store.get_whole_history(channel_id)
    formatted_message_history_list = format_message_history(message_history, use_username=True, include_timestamp=True)
    formatted_message_history = "\n".join(formatted_message_history_list)

    # Build author/username maps for stat-driven features
    message_id_to_author = {str(msg.id): msg.author.id for msg in message_history}
    rel_id_to_author = {idx + 1: msg.author.id for idx, msg in enumerate(message_history)}
    # Map between relative IDs (1-based) and actual Discord message IDs for later resolution
    rel_id_to_discord_id = {idx + 1: msg.id for idx, msg in enumerate(message_history)}
    discord_id_lookup = {str(msg.id): msg.id for msg in message_history}
    username_id_map: dict[str, int] = {}
    for msg in message_history:
        username_id_map.setdefault(msg.author.name, msg.author.id)
        username_id_map.setdefault(msg.author.display_name, msg.author.id)

    # Ensure channel_name is not None
    if channel_name is None:
        channel_name = "Unknown Channel"

    candidates = await extract_features_from_formatted_history(
        formatted_message_history,
        channel_name,
        thread_name,
        required_message_indexes=required_message_indexes,
        author_id_map={**message_id_to_author, **{str(k): v for k, v in rel_id_to_author.items()}},
        username_id_map=username_id_map,
        stats_session_factory=get_session,
        ignore_first_message_count=ignore_first_message_count,
    )

    # Attach the resolved target Discord user ID next to the username for downstream lookups
    # Also attach the specific Discord message ID for each candidate
    for candidate in candidates:
        target_username = candidate.get("target_username")
        candidate["target_user_id"] = (
            username_id_map.get(target_username)
            if isinstance(target_username, str)
            else None
        )
        
        # Resolve the candidate's message_id (which may be relative or a Discord ID) to the real Discord message ID
        raw_message_id = candidate.get("message_id")
        discord_message_id = None
        if isinstance(raw_message_id, str):
            if raw_message_id in discord_id_lookup:
                discord_message_id = discord_id_lookup[raw_message_id]
            elif raw_message_id.isdigit():
                rel_val = int(raw_message_id)
                discord_message_id = rel_id_to_discord_id.get(rel_val)
        elif isinstance(raw_message_id, int):
            discord_message_id = rel_id_to_discord_id.get(raw_message_id)

        if discord_message_id is not None:
            candidate["discord_message_id"] = discord_message_id

    return candidates
