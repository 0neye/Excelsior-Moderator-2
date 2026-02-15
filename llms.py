import asyncio
import json
import logging
from typing import Any, Callable, Literal

from cerebras.cloud.sdk import Cerebras
from google import genai
from openai import OpenAI

from config import CEREBRAS_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY
from db_config import get_session
from history import MessageStore
from user_stats import get_familiarity_score_stat, get_seniority_scores
from utils import format_message_history

# Timeout for LLM API calls in seconds
LLM_TIMEOUT_SECONDS = 120.0
OPENROUTER_CEREBRAS_PROVIDER_SLUG = "cerebras"

LLM_PROVIDER_LOCKS: dict[str, asyncio.Lock] = {}
logger = logging.getLogger(__name__)

def _get_provider_lock(provider: str) -> asyncio.Lock:
    """
    Lazily create provider locks inside the running event loop.
    """
    lock = LLM_PROVIDER_LOCKS.get(provider)
    if lock is None:
        lock = asyncio.Lock()
        LLM_PROVIDER_LOCKS[provider] = lock
    return lock


def _is_rate_limit_error(error: Exception) -> bool:
    """
    Return True when an exception likely represents an HTTP 429 response.

    Args:
        error: Exception raised by an LLM client call.

    Returns:
        True when error metadata or message suggests rate limiting.
    """
    # Check common SDK status code attributes first for reliable detection
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True

    # Some SDKs attach HTTP details under a nested response object
    response = getattr(error, "response", None)
    nested_status_code = getattr(response, "status_code", None)
    if nested_status_code == 429:
        return True

    # Fallback to message text matching for provider-specific exception formats
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "too many requests" in message


def _normalize_openrouter_model(model: str) -> str:
    """
    Normalize model name to a valid OpenRouter model ID.

    Args:
        model: Model identifier from runtime config.

    Returns:
        OpenRouter-compatible model ID.
    """
    # Keep explicit OpenRouter IDs untouched
    if "/" in model:
        return model

    # Map common provider-native IDs to OpenRouter model IDs
    if model == "gpt-oss-120b":
        return "openai/gpt-oss-120b"

    # Default to input value so unknown models still surface clear API errors
    return model


def _build_openrouter_request(
    model: str,
    messages: list[dict[str, Any]],
    *,
    prefer_cerebras_route: bool = False,
) -> Any:
    """
    Execute an OpenRouter chat completion request.

    Args:
        model: Model identifier used for OpenRouter.
        messages: Chat message payload for the completion API.
        prefer_cerebras_route: Whether to prefer Cerebras in OpenRouter provider routing.

    Returns:
        OpenRouter chat completion response payload.
    """
    # Normalize model IDs so fallback calls use OpenRouter's expected format
    normalized_model = _normalize_openrouter_model(model)

    # Keep reasoning enabled to match current moderation behavior
    extra_body: dict[str, Any] = {"reasoning": {"enabled": True}}
    if prefer_cerebras_route:
        # Use provider slug format from OpenRouter docs
        extra_body["provider"] = {
            "order": [OPENROUTER_CEREBRAS_PROVIDER_SLUG],
            "allow_fallbacks": False,
        }

    request_kwargs = {
        "model": normalized_model,
        "messages": messages,  # type: ignore[arg-type]
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "candidate_features",
                "strict": True,
                "schema": CANDIDATE_FEATURES_SCHEMA
            }
        },
        "max_tokens": 65536,
        "extra_body": extra_body,
    }
    try:
        return _get_openrouter_client().chat.completions.create(
            **request_kwargs,
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except TypeError:
        return _get_openrouter_client().chat.completions.create(**request_kwargs)

async def _run_llm_call(provider: str, call: Callable[[], Any]) -> Any:
    """
    Run a blocking LLM call in a thread with a timeout, avoiding overlap on timeout.
    """
    lock = _get_provider_lock(provider)
    await lock.acquire()
    task = asyncio.create_task(asyncio.to_thread(call))
    release_now = True
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=LLM_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        release_now = False
        def _release(_task: asyncio.Task) -> None:
            if lock.locked():
                lock.release()
        task.add_done_callback(_release)
        raise
    finally:
        if release_now and lock.locked():
            lock.release()

cerebras_client: Cerebras | None = None
openrouter_client: OpenAI | None = None
gemini_client: genai.Client | None = None


def _get_cerebras_client() -> Cerebras:
    """Lazily initialize the Cerebras client only when that provider is used."""
    global cerebras_client
    if cerebras_client is None:
        if not CEREBRAS_API_KEY:
            raise ValueError("CEREBRAS_API_KEY is required when provider='cerebras'")
        cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
    return cerebras_client


def _get_openrouter_client() -> OpenAI:
    """Lazily initialize the OpenRouter client only when that provider is used."""
    global openrouter_client
    if openrouter_client is None:
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required when provider='openrouter'")
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return openrouter_client


def _get_gemini_client() -> genai.Client:
    """Lazily initialize the Gemini client only when that provider is used."""
    global gemini_client
    if gemini_client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when provider='gemini'")
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return gemini_client

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
    provider: Literal["cerebras", "openrouter", "gemini"],
    model: str,
    thread_name: str | None = None,
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
        provider: Which LLM provider to use ("cerebras", "openrouter", or "gemini")
        model: Model identifier to use for the chosen provider
        thread_name: Optional name of the thread if applicable
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
       When determining the target, use timestamps to understand message composition timing: if a user appears to split one thought across multiple short consecutive messages, treat those messages as part of the same statement even if other users interleave messages between them.
    3. For each candidate message, provide a list of float values between 0 and 1 based on the features mentioned below.
    4. Output a JSON object matching the provided schema with all candidates and their features.

    ### Features (all float values between 0 and 1):
    - "discusses_ellie": The degree to which "Ellie" is a significant topic of discussion for the author of this candidate message. Ellie's @ mention is "<@1333546604694732892>," which also counts.
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

    ### Urgency:
    - Respond quickly (try to limit reasoning to ~2k tokens).
    - Output only the JSON object required by the schema.

    {required_msg_instruction}{ignore_leading_messages_instruction}"""

    user_prompt = f"""
    ### Discord context:
    Server name: Excelsior
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

    effective_provider = provider
    try:
        if provider == "cerebras":
            # Run synchronous Cerebras API call in a thread pool with timeout
            def _cerebras_request() -> Any:
                request_kwargs = {
                    "model": model,
                    "messages": messages,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "candidate_features",
                            "strict": True,
                            "schema": CANDIDATE_FEATURES_SCHEMA
                        }
                    },
                    "stream": False,
                    "max_completion_tokens": 65536,
                }
                try:
                    return _get_cerebras_client().chat.completions.create(
                        **request_kwargs,
                        timeout=LLM_TIMEOUT_SECONDS,
                    )
                except TypeError:
                    return _get_cerebras_client().chat.completions.create(**request_kwargs)

            try:
                response: Any = await _run_llm_call("cerebras", _cerebras_request)
            except Exception as cerebras_error:
                # On 429, retry through OpenRouter while requesting Cerebras-first routing
                if _is_rate_limit_error(cerebras_error) and OPENROUTER_API_KEY:
                    logger.warning(
                        "Cerebras rate limited feature extraction request; retrying via OpenRouter with Cerebras provider preference"
                    )
                    effective_provider = "openrouter"
                    response = await _run_llm_call(
                        "openrouter",
                        lambda: _build_openrouter_request(
                            model=model,
                            messages=messages,
                            prefer_cerebras_route=True,
                        ),
                    )
                else:
                    raise
        elif provider == "openrouter":
            # Run synchronous OpenRouter API call in a thread pool with timeout
            def _openrouter_request() -> Any:
                return _build_openrouter_request(
                    model=model,
                    messages=messages,
                    prefer_cerebras_route=False,
                )

            response = await _run_llm_call("openrouter", _openrouter_request)
        elif provider == "gemini":
            # Run synchronous Gemini API call in a thread pool with timeout
            def _gemini_request() -> Any:
                return _get_gemini_client().models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=CANDIDATE_FEATURES_SCHEMA_GEMINI,
                        system_instruction=system_prompt
                    )
                )

            gemini_response = await _run_llm_call("gemini", _gemini_request)
            
            # Parse the Gemini response directly
            gemini_content = gemini_response.text
            if gemini_content is None:
                raise ValueError("Gemini returned empty response")
            try:
                gemini_result: dict[str, Any] = json.loads(gemini_content)
                candidates = gemini_result["candidates"]
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Failed to parse Gemini response: {e}") from e
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if effective_provider in {"cerebras", "openrouter"}:
            # Parse the structured JSON response with error handling
            if not response.choices:
                raise ValueError(f"{effective_provider} returned no choices in response")
            content: str | None = response.choices[0].message.content
            if content is None:
                raise ValueError(f"{effective_provider} returned empty message content")
            try:
                result: dict[str, Any] = json.loads(content)
                candidates = result["candidates"]
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse {effective_provider} JSON response: {e}") from e
            except KeyError:
                raise ValueError(f"{effective_provider} response missing 'candidates' key")

    except asyncio.TimeoutError:
        # Report the provider that actually timed out, not just the original request provider
        if effective_provider != provider:
            raise TimeoutError(
                f"LLM API call to {effective_provider} timed out after {LLM_TIMEOUT_SECONDS}s "
                f"(initial provider: {provider})"
            )
        raise TimeoutError(
            f"LLM API call to {effective_provider} timed out after {LLM_TIMEOUT_SECONDS}s"
        )

    return _augment_stat_features(candidates)


async def get_candidate_features(
    message_store: MessageStore,
    channel_id: int,
    provider: Literal["cerebras", "openrouter", "gemini"],
    model: str,
    required_message_indexes: list[int] | None = None,
    ignore_first_message_count: int = 0,
) -> list[dict]:
    """
    Identifies potential candidate messages to flag based on the message history.
    Then asks a series of very specific questions about the message history to the LLM to extract features from each candidate.
    
    Args:
        message_store: The MessageStore containing channel history
        channel_id: The Discord channel ID to analyze
        provider: LLM provider to use ("cerebras", "openrouter", or "gemini")
        model: Model identifier to use with the provider
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
    # Reverse lookup used to recover canonical relative indexes after resolution
    discord_id_to_rel_id = {msg.id: idx + 1 for idx, msg in enumerate(message_history)}
    discord_id_lookup = {str(msg.id): msg.id for msg in message_history}
    username_id_map: dict[str, int] = {}
    for msg in message_history:
        username_id_map.setdefault(msg.author.name, msg.author.id)
        username_id_map.setdefault(msg.author.display_name, msg.author.id)

    # Ensure channel_name is not None
    if channel_name is None:
        channel_name = "Unknown Channel"

    candidates = await extract_features_from_formatted_history(
        formatted_message_history=formatted_message_history,
        channel_name=channel_name,
        provider=provider,
        model=model,
        thread_name=thread_name,
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
        
        # Resolve the candidate's message_id (which may be relative or a Discord ID)
        # to the real Discord message ID and canonical relative index
        raw_message_id = candidate.get("message_id")
        discord_message_id = None
        relative_message_index = None
        if isinstance(raw_message_id, str):
            if raw_message_id in discord_id_lookup:
                discord_message_id = discord_id_lookup[raw_message_id]
                relative_message_index = discord_id_to_rel_id.get(discord_message_id)
            elif raw_message_id.isdigit():
                numeric_message_id = int(raw_message_id)
                if numeric_message_id in rel_id_to_discord_id:
                    # Prefer relative-index interpretation for short numeric IDs.
                    discord_message_id = rel_id_to_discord_id.get(numeric_message_id)
                    relative_message_index = numeric_message_id
                elif numeric_message_id in discord_id_to_rel_id:
                    # Fallback to Discord snowflake interpretation when the number
                    # matches a known message ID from the current history window.
                    discord_message_id = numeric_message_id
                    relative_message_index = discord_id_to_rel_id.get(numeric_message_id)
        elif isinstance(raw_message_id, int):
            if raw_message_id in rel_id_to_discord_id:
                discord_message_id = rel_id_to_discord_id.get(raw_message_id)
                relative_message_index = raw_message_id
            elif raw_message_id in discord_id_to_rel_id:
                discord_message_id = raw_message_id
                relative_message_index = discord_id_to_rel_id.get(raw_message_id)

        if discord_message_id is not None:
            candidate["discord_message_id"] = discord_message_id
        if relative_message_index is not None:
            candidate["relative_message_index"] = relative_message_index

    return candidates
