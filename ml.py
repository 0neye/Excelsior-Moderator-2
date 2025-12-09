"""Machine learning models for moderation classification using LightGBM only."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Feature names matching the LLM extraction schema
FEATURE_NAMES = [
    "discusses_ellie",
    "familiarity_score",
    "tone_harshness_score",
    "positive_framing_score",
    "includes_positive_takeaways",
    "explains_why_score",
    "actionable_suggestion_score",
    "context_is_feedback_appropriate",
    "target_uncomfortableness_score",
    "is_part_of_discussion",
    "criticism_directed_at_image",
    "criticism_directed_at_statement",
    "criticism_directed_at_generality",
    "reciprocity_score",
    "solicited_score",
    "seniority_score_messages",
    "seniority_score_characters",
    "familiarity_score_stat",
]


class ModerationClassifier(ABC):
    """Abstract base class for moderation classifiers."""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        # Track the feature ordering used to fit the model
        self.feature_names: list[str] = FEATURE_NAMES.copy()

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier on feature matrix X and labels y.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted class labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        pass

    def save(self, path: str | Path) -> None:
        """
        Save the trained model to disk.

        Args:
            path: File path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        path = Path(path)
        model_data = self._get_model_data()
        model_data["label_encoder"] = self.label_encoder
        model_data["is_fitted"] = self.is_fitted
        model_data["feature_names"] = getattr(self, "feature_names", FEATURE_NAMES)
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ModerationClassifier":
        """
        Load a trained model from disk.

        Args:
            path: File path to load the model from

        Returns:
            Loaded classifier instance
        """
        path = Path(path)
        model_data = joblib.load(path)
        
        instance = cls()
        instance.label_encoder = model_data["label_encoder"]
        instance.is_fitted = model_data["is_fitted"]
        instance.feature_names = model_data.get("feature_names", FEATURE_NAMES)
        instance._set_model_data(model_data)
        
        logger.info(f"Model loaded from {path}")
        return instance

    @abstractmethod
    def _get_model_data(self) -> dict[str, Any]:
        """Get model-specific data for serialization."""
        pass

    @abstractmethod
    def _set_model_data(self, data: dict[str, Any]) -> None:
        """Set model-specific data from deserialization."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass


class LightGBMClassifier(ModerationClassifier):
    """LightGBM-based classifier for moderation."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 7,
        learning_rate: float = 0.1,
        num_leaves: int = 50,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
    ):
        """
        Initialize LightGBM classifier.

        Args:
            n_estimators: Number of boosting iterations
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            num_leaves: Maximum number of leaves in one tree
            class_weight: Weights for classes ('balanced' or dict)
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LightGBM classifier."""
        import lightgbm as lgb

        logger.info(f"Training LightGBM with {len(y)} samples...")

        y_encoded = self.label_encoder.fit_transform(y)
        classes = self.label_encoder.classes_
        n_classes = len(classes) if classes is not None else 0

        logger.info(f"Classes: {list(classes) if classes is not None else []}")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            class_weight=self.class_weight,
            random_state=self.random_state,
            objective="multiclass" if n_classes > 2 else "binary",
            verbose=-1,
        )

        self.model.fit(X, y_encoded)
        self.is_fitted = True
        logger.info("LightGBM training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        return np.array(self.model.predict_proba(X))

    def _get_model_data(self) -> dict[str, Any]:
        """Get LightGBM model data for serialization."""
        return {
            "model": self.model,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }

    def _set_model_data(self, data: dict[str, Any]) -> None:
        """Set LightGBM model data from deserialization."""
        self.model = data["model"]
        self.n_estimators = data["n_estimators"]
        self.max_depth = data["max_depth"]
        self.learning_rate = data["learning_rate"]
        self.num_leaves = data["num_leaves"]
        self.class_weight = data["class_weight"]
        self.random_state = data["random_state"]

    def get_feature_importance(self) -> dict[str, float]:
        """Get native LightGBM feature importance."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        importances = self.model.feature_importances_
        # Avoid division by zero if all importances are zero
        total = importances.sum()
        if total > 0:
            importances = importances / total
        feature_names = getattr(self, "feature_names", FEATURE_NAMES)

        return dict(zip(feature_names, importances))


def create_classifier(model_type: str = "lightgbm", **kwargs) -> ModerationClassifier:
    """
    Factory function to create a classifier.

    Args:
        model_type: Must be 'lightgbm' (kept for backward compatibility)
        **kwargs: Additional arguments passed to the classifier constructor

    Returns:
        Initialized classifier instance
    """
    model_type_lower = model_type.lower()
    if model_type_lower == "lightgbm":
        return LightGBMClassifier(**kwargs)
    raise ValueError("Unknown model type: only 'lightgbm' is supported")

def load_classifier(path: str | Path = "models/lightgbm_model.joblib") -> ModerationClassifier:
    """
    Load a classifier from disk.

    Args:
        model_type: The type of model to load
        path: The path to the model file

    Returns:
        Loaded classifier instance
    """
    return LightGBMClassifier.load(path)