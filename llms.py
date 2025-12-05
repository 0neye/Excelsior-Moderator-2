import asyncio
import json
from typing import Any, Literal

from cerebras.cloud.sdk import Cerebras
from google import genai
from openai import OpenAI

from config import CEREBRAS_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY
from history import MessageStore
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
                            "solicited_score": {"type": "number"}
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


async def extract_features_from_formatted_history(
    formatted_message_history: str,
    channel_name: str,
    thread_name: str | None = None,
    provider: Literal["cerebras", "openrouter", "gemini"] = "cerebras",
    openrouter_model: str = "openai/gpt-oss-120b",
    gemini_model: str = "gemini-2.5-flash",
    required_message_indexes: list[int] | None = None
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
    
    Returns:
        A list of candidate dicts, each containing message_id, target_username, and features.
    """

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

    system_prompt = f"""
    You will be given a portion of a Discord channel's message history.
    Your job is to follow the process below.

    ### Process:
    1. Identify all messages that could be considered providing feedback or criticism to another user present in the conversation.
    2. Determine the target of each candidate message. Who is the message criticizing? The target must be a user present in the conversation.
    3. For each candidate message, provide a list of float values between 0 and 1 based on the features mentioned below.
    4. Output a JSON object matching the provided schema with all candidates and their features.

    ### Features (all float values between 0 and 1):
    - "discusses_ellie": The degree to which "Ellie" is a significant topic of discussion in the message history provided. This is unrelated to other features, and is a one-off.
    - "familiarity_score": How familiar the author of the candidate message seems to be with the target user.
    - "tone_harshness_score": Overall harshness/condescension level independent of content.
    - "positive_framing_score": Does it use positive language ("I like X", "consider trying Y") vs negative ("X is wrong", "your layout is unoptimal")?
    - "includes_positive_takeaways": Does the message acknowledge something the target did well?
    - "explains_why_score": Does it explain the why behind suggestions, not just the what?
    - "actionable_suggestion_score": Does it give specific, actionable advice vs vague criticism?
    - "context_is_feedback_appropriate": Is this a role request, classroom post, module post, etc. where feedback is expected?
    - "target_uncomfortableness_score": How uncomfortable the target user seems to be with the candidate message. This could be trying to defend themselves or their position, sad emoji, etc.
    - "is_part_of_discussion": Whether the candidate message is part of a two-way discussion about a specific topic in which both the author and target user are engaged.
    - "criticism_directed_at_image": The degree to which the candidate message could be criticizing a specific posted image/attachment.
    - "criticism_directed_at_statement": The degree to which the candidate message could be criticizing a specific statement or idea made by the target user.
    - "criticism_directed_at_generality": The degree to which the candidate message could be criticizing a general topic or concept not specific to the target user, such as the game itself or the server.
    - "reciprocity_score": The degree to which the candidate message could be criticizing a negative behavior of the target user themselves, such as being rude, disrespectful, etc. This includes scolding people for giving bad feedback, being too harsh, etc.
    - "solicited_score": How solicited (1) to unsolicited (0) the candidate message seems to be. Something as simple as ending their message with a question mark can be considered solicitation of feedback, but use your best judgement.
    
    ### Remember:
    - The message history is provided in chronological order, from oldest to newest.
    - The message history format is: [timestamp] (rel_id) [reply to rel_id or username] ❝message content❞ (edited) [reactions]
    - This is a gaming server discussing the game Cosmoteer: Starship Architect & Commander.
    {required_msg_instruction}"""

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
                reasoning_effort="medium"
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
                    response_schema=CANDIDATE_FEATURES_SCHEMA,
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=8192),
                    system_instruction=system_prompt
                )
            )
        )
        
        # Parse and return the Gemini response directly
        gemini_content = gemini_response.text
        if gemini_content is None:
            raise ValueError("Gemini returned empty response")
        gemini_result: dict[str, Any] = json.loads(gemini_content)
        return gemini_result["candidates"]
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # Parse the structured JSON response
    content: str = response.choices[0].message.content
    result: dict[str, Any] = json.loads(content)
    return result["candidates"]


async def get_candidate_features(
    message_store: MessageStore,
    channel_id: int,
    required_message_indexes: list[int] | None = None
) -> list[dict]:
    """
    Identifies potential candidate messages to flag based on the message history.
    Then asks a series of very specific questions about the message history to the LLM to extract features from each candidate.
    
    Args:
        message_store: The MessageStore containing channel history
        channel_id: The Discord channel ID to analyze
        required_message_indexes: Optional list of relative message IDs that MUST have
            features extracted, regardless of whether they appear to be feedback candidates
    
    Returns:
        A list of candidate dicts, each containing message_id, target_username, and features.
    """

    channel_info = message_store.get_channel_info(channel_id)
    if channel_info is None:
        raise ValueError(f"Channel with ID {channel_id} not found in message store")
    channel_name = channel_info.channel_name if not channel_info.is_thread else channel_info.parent_channel_name
    thread_name = channel_info.channel_name if channel_info.is_thread else None

    message_history = message_store.get_whole_history(channel_id)
    formatted_message_history_list = format_message_history(message_history, use_username=True, include_timestamp=True)
    formatted_message_history = "\n".join(formatted_message_history_list)

    # Ensure channel_name is not None
    if channel_name is None:
        channel_name = "Unknown Channel"

    return await extract_features_from_formatted_history(
        formatted_message_history,
        channel_name,
        thread_name,
        required_message_indexes=required_message_indexes
    )