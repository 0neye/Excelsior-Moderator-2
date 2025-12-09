"""Continuous training module for automatic model retraining.

Provides functionality for retraining the moderation classifier when enough
new ratings have been collected. Supports two feature extraction modes:
- "existing_only": Use only messages with pre-extracted features (fast, no LLM cost)
- "extract_on_demand": Extract features for new messages via LLM (more complete)
"""

from pathlib import Path
from typing import Literal

import numpy as np

from config import (
    CONTINUOUS_TRAINING_FEATURE_MODE,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    get_logger,
)
from database import (
    FeatureExtractionRun,
    FlaggedMessage,
    FlaggedMessageRating,
    MessageFeatures,
)
from db_config import get_session, init_db
from ml import FEATURE_NAMES, create_classifier

logger = get_logger(__name__)

MODEL_SAVE_DIR = Path(__file__).parent / "models"


def load_all_features_from_db() -> dict[int, list[dict[str, float]]]:
    """
    Load features from the database, including runtime features and the most recent run.

    Loads:
    1. Features without an extraction_run_id (saved during normal bot operations)
    2. Features from the most recent extraction run (from bootstrapping)

    Runtime features take priority over bootstrapped features for the same message.

    Returns:
        Dictionary mapping message_id to list of feature dicts
    """
    init_db()
    session = get_session()

    try:
        features_by_message: dict[int, list[dict[str, float]]] = {}

        # Step 1: Load features from normal bot operations (no extraction_run_id)
        runtime_features = (
            session.query(MessageFeatures)
            .filter(MessageFeatures.extraction_run_id.is_(None))
            .order_by(MessageFeatures.message_id, MessageFeatures.run_index)
            .all()
        )

        for record in runtime_features:
            if record.message_id not in features_by_message:
                features_by_message[record.message_id] = []
            features_by_message[record.message_id].append(record.features)

        runtime_count = len(features_by_message)
        logger.info("Loaded features for %d messages from runtime (no run ID)", runtime_count)

        # Step 2: Get the most recent extraction run
        latest_run = (
            session.query(FeatureExtractionRun)
            .order_by(FeatureExtractionRun.created_at.desc())
            .first()
        )

        if latest_run:
            # Load features from the most recent run for messages we don't already have
            run_features = (
                session.query(MessageFeatures)
                .filter(MessageFeatures.extraction_run_id == latest_run.id)
                .order_by(MessageFeatures.message_id, MessageFeatures.run_index)
                .all()
            )

            run_messages_added: set[int] = set()
            for record in run_features:
                # Only add if we don't already have runtime features for this message
                if record.message_id not in features_by_message:
                    features_by_message[record.message_id] = []
                    run_messages_added.add(record.message_id)
                # Append features for messages from this run (may have multiple run_index values)
                if record.message_id in run_messages_added:
                    features_by_message[record.message_id].append(record.features)

            logger.info(
                "Loaded features for %d additional messages from run %d (%s)",
                len(run_messages_added),
                latest_run.id,
                latest_run.name or "(unnamed)",
            )

        logger.info("Total: loaded features for %d messages", len(features_by_message))
        return features_by_message

    finally:
        session.close()


def load_rated_messages_from_db() -> list[dict]:
    """
    Load all rated messages with their ratings from the database.

    Returns:
        List of dicts with message info and rating category
    """
    init_db()
    session = get_session()

    try:
        # Query flagged messages that have completed ratings
        results = (
            session.query(FlaggedMessage, FlaggedMessageRating)
            .join(
                FlaggedMessageRating,
                FlaggedMessage.message_id == FlaggedMessageRating.flagged_message_id,
            )
            .filter(FlaggedMessageRating.category.isnot(None))
            .all()
        )

        rated_messages = []
        for flagged_msg, rating in results:
            rated_messages.append(
                {
                    "message_id": flagged_msg.message_id,
                    "author_id": flagged_msg.author_id,
                    "category": rating.category.value if rating.category else "NA",
                }
            )

        logger.info("Loaded %d rated messages from database", len(rated_messages))
        return rated_messages

    finally:
        session.close()


def prepare_training_data_simple(
    rated_messages: list[dict],
    features_by_message: dict[int, list[dict[str, float]]],
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare feature matrix and labels from rated messages and their features.

    Args:
        rated_messages: List of rated message dicts with message_id and category
        features_by_message: Dict mapping message_id to feature lists
        feature_names: Ordered list of feature names to use (defaults to FEATURE_NAMES)

    Returns:
        Tuple of (feature matrix X, labels y, feature_names used)
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES.copy()

    X = []
    y = []

    for msg in rated_messages:
        msg_id = msg["message_id"]
        if msg_id not in features_by_message:
            continue

        feature_list = features_by_message[msg_id]
        for feature_dict in feature_list:
            # Extract features in consistent order
            feature_vector = [feature_dict.get(name, 0.0) for name in feature_names]
            X.append(feature_vector)
            y.append(msg["category"])

    X = np.array(X)
    y = np.array(y)

    logger.info(
        "Prepared training data: %d samples, %d features",
        X.shape[0] if len(X) > 0 else 0,
        len(feature_names),
    )

    return X, y, feature_names


async def retrain_model(
    feature_mode: Literal["existing_only", "extract_on_demand"] | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> bool:
    """
    Retrain the moderation classifier using current ratings and features.

    Args:
        feature_mode: How to handle feature extraction. Uses config default if None.
        llm_provider: LLM provider for on-demand extraction. Uses config default if None.
        llm_model: LLM model for on-demand extraction. Uses config default if None.

    Returns:
        True if retraining succeeded, False otherwise
    """
    # Use config defaults if not specified
    if feature_mode is None:
        feature_mode = CONTINUOUS_TRAINING_FEATURE_MODE  # type: ignore[assignment]
    if llm_provider is None:
        llm_provider = DEFAULT_LLM_PROVIDER
    if llm_model is None:
        llm_model = DEFAULT_LLM_MODEL

    logger.info("Starting model retraining (mode=%s)...", feature_mode)

    try:
        # Load rated messages from DB
        rated_messages = load_rated_messages_from_db()
        if not rated_messages:
            logger.warning("No rated messages found, skipping retraining")
            return False

        # Load or extract features based on mode
        if feature_mode == "existing_only":
            features_by_message = load_all_features_from_db()
        else:
            # Extract features on demand for messages without them
            features_by_message = await _extract_features_on_demand(
                rated_messages, llm_provider, llm_model
            )

        if not features_by_message:
            logger.warning("No features available, skipping retraining")
            return False

        # Prepare training data
        X, y, feature_names = prepare_training_data_simple(
            rated_messages, features_by_message
        )

        if len(X) == 0:
            logger.warning("No training samples after preparation, skipping retraining")
            return False

        # Need at least 2 samples per class for stratified split
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            logger.warning(
                "Need at least 2 classes for training, found %d", len(unique_classes)
            )
            return False

        min_samples = min(counts)
        if min_samples < 2:
            logger.warning(
                "Need at least 2 samples per class for stratified split, "
                "smallest class has %d samples",
                min_samples,
            )
            # Use all data for training without test split
            X_train, y_train = X, y
        else:
            # Use sklearn for train/test split
            from sklearn.model_selection import train_test_split

            # No test split because at this point we've already tested the model with the bootstrapping module
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.0, random_state=42, stratify=y
            )

        logger.info("Training with %d samples...", len(y_train))

        # Create and train classifier
        model = create_classifier("lightgbm")
        model.feature_names = feature_names
        model.fit(X_train, y_train)

        # Save model
        MODEL_SAVE_DIR.mkdir(exist_ok=True)
        model_path = MODEL_SAVE_DIR / "lightgbm_model.joblib"
        model.save(model_path)

        logger.info("Model retrained and saved to %s", model_path)
        return True

    except Exception as e:
        logger.error("Error during model retraining: %s", e, exc_info=True)
        return False


async def _extract_features_on_demand(
    rated_messages: list[dict],
    llm_provider: str,
    llm_model: str,
) -> dict[int, list[dict[str, float]]]:
    """
    Extract features for messages that don't have them yet.

    Loads existing features and extracts new ones for messages without features
    using the LLM provider.

    Args:
        rated_messages: List of rated message dicts
        llm_provider: LLM provider to use for extraction
        llm_model: LLM model to use for extraction

    Returns:
        Dict mapping message_id to feature lists
    """
    # Start with existing features
    features_by_message = load_all_features_from_db()

    # Find messages without features
    message_ids_with_features = set(features_by_message.keys())
    messages_needing_features = [
        msg for msg in rated_messages if msg["message_id"] not in message_ids_with_features
    ]

    if not messages_needing_features:
        logger.info("All rated messages have existing features")
        return features_by_message

    logger.info(
        "%d messages need feature extraction, importing bootstrapping module...",
        len(messages_needing_features),
    )

    # Import bootstrapping functions only when needed to avoid circular imports
    from bootstrapping import (
        extract_features,
        fetch_discord_context,
        load_rating_data,
    )

    # Load full rated message data needed for feature extraction
    full_rated_messages = load_rating_data()

    # Filter to just the ones needing features
    ids_needing = {msg["message_id"] for msg in messages_needing_features}
    messages_to_extract = [msg for msg in full_rated_messages if msg.message_id in ids_needing]

    if not messages_to_extract:
        logger.warning("Could not load full message data for feature extraction")
        return features_by_message

    # Fetch Discord context (uses cached data when available)
    messages_with_context = await fetch_discord_context(messages_to_extract)

    if not messages_with_context:
        logger.warning("No messages with context available for extraction")
        return features_by_message

    # Extract features
    messages_with_features, _ = await extract_features(
        messages_with_context,
        model=llm_model,
        provider=llm_provider,
        max_concurrent=5,
        auto_save=True,
    )

    # Add newly extracted features to our map
    for msg in messages_with_features:
        if msg.features:
            features_by_message[msg.message_id] = msg.features

    logger.info(
        "Extracted features for %d additional messages", len(messages_with_features)
    )

    return features_by_message

