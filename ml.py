"""Machine learning models for moderation classification.

Provides reusable classifier implementations for both bootstrapping and continuous training.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as SklearnMLP
from sklearn.preprocessing import LabelEncoder

# Configure logging
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
]


class ModerationClassifier(ABC):
    """Abstract base class for moderation classifiers."""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

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
        
        # Encode labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        classes = self.label_encoder.classes_
        n_classes = len(classes) if classes is not None else 0
        
        logger.info(f"Classes: {list(classes) if classes is not None else []}")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Create LightGBM classifier
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
        # Normalize to sum to 1
        importances = importances / importances.sum()
        
        return dict(zip(FEATURE_NAMES, importances))


class LogisticRegressionClassifier(ModerationClassifier):
    """Logistic regression classifier for moderation."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: str | dict | None = "balanced",
        multi_class: str = "auto",
        n_jobs: int | None = None,
        random_state: int | None = 42,
    ):
        """
        Initialize logistic regression classifier.

        Args:
            C: Inverse regularization strength (smaller is stronger)
            penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
            solver: Optimization solver compatible with the penalty choice
            max_iter: Maximum iterations for the solver to converge
            class_weight: Per-class weights or 'balanced' for inverse-frequency
            multi_class: Multi-class handling strategy ('auto', 'multinomial', 'ovr')
            n_jobs: Parallel jobs for fitting if supported by solver
            random_state: Seed for reproducible results when supported
        """
        super().__init__()
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the logistic regression classifier."""
        logger.info(f"Training logistic regression with {len(y)} samples...")

        # Encode labels to integers for sklearn
        y_encoded = self.label_encoder.fit_transform(y)
        classes = self.label_encoder.classes_
        n_classes = len(classes) if classes is not None else 0

        logger.info(f"Classes: {list(classes) if classes is not None else []}")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Create sklearn LogisticRegression model
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            multi_class=self.multi_class if n_classes > 2 else "auto",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        self.model.fit(X, y_encoded)
        self.is_fitted = True
        logger.info("Logistic regression training complete")

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
        """Get logistic regression model data for serialization."""
        return {
            "model": self.model,
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "multi_class": self.multi_class,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

    def _set_model_data(self, data: dict[str, Any]) -> None:
        """Set logistic regression model data from deserialization."""
        self.model = data["model"]
        self.C = data["C"]
        self.penalty = data["penalty"]
        self.solver = data["solver"]
        self.max_iter = data["max_iter"]
        self.class_weight = data["class_weight"]
        self.multi_class = data["multi_class"]
        self.n_jobs = data["n_jobs"]
        self.random_state = data["random_state"]

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance from absolute coefficient magnitudes.

        Uses the mean absolute coefficient across classes as a simple proxy.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        coefficients = np.abs(self.model.coef_)
        # Average importance across classes for multiclass settings
        importances = coefficients.mean(axis=0)
        importance_sum = importances.sum()
        if importance_sum == 0:
            # Avoid division by zero; return zeros if model produced zero weights
            normalized_importances = np.zeros_like(importances)
        else:
            normalized_importances = importances / importance_sum

        return dict(zip(FEATURE_NAMES, normalized_importances))


class MLPModerationClassifier(ModerationClassifier):
    """Multi-layer perceptron classifier for moderation."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        activation: str = "relu",
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        early_stopping: bool = True,
        random_state: int = 42,
        alpha: float = 0.01,
    ):
        """
        Initialize MLP classifier.

        Args:
            hidden_layer_sizes: Sizes of hidden layers
            activation: Activation function ('relu', 'tanh', 'logistic')
            learning_rate_init: Initial learning rate
            max_iter: Maximum number of iterations
            early_stopping: Whether to use early stopping
            random_state: Random seed for reproducibility
            alpha: L2 regularization strength
        """
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.alpha = alpha
        self.model: SklearnMLP | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the MLP classifier."""
        logger.info(f"Training MLP with {len(y)} samples...")
        
        # Encode labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        classes = self.label_encoder.classes_
        
        logger.info(f"Classes: {list(classes) if classes is not None else []}")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Create sklearn MLP classifier
        self.model = SklearnMLP(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
            alpha=self.alpha,
            verbose=True,
        )

        self.model.fit(X, y_encoded)
        self.is_fitted = True
        logger.info("MLP training complete")

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
        """Get MLP model data for serialization."""
        return {
            "model": self.model,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "early_stopping": self.early_stopping,
            "random_state": self.random_state,
            "alpha": self.alpha,
        }

    def _set_model_data(self, data: dict[str, Any]) -> None:
        """Set MLP model data from deserialization."""
        self.model = data["model"]
        self.hidden_layer_sizes = data["hidden_layer_sizes"]
        self.activation = data["activation"]
        self.learning_rate_init = data["learning_rate_init"]
        self.max_iter = data["max_iter"]
        self.early_stopping = data["early_stopping"]
        self.random_state = data["random_state"]
        self.alpha = data["alpha"]

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance via permutation importance.
        Note: This requires X_test and y_test to be passed, so we return
        the input layer weights as a proxy for feature importance.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Use absolute values of first layer weights as importance proxy
        first_layer_weights = np.abs(self.model.coefs_[0])
        importances = first_layer_weights.sum(axis=1)
        # Normalize to sum to 1
        importances = importances / importances.sum()
        
        return dict(zip(FEATURE_NAMES, importances))


def create_classifier(model_type: str = "lightgbm", **kwargs) -> ModerationClassifier:
    """
    Factory function to create a classifier by type.

    Args:
        model_type: Type of classifier ('lightgbm', 'mlp', or 'logistic')
        **kwargs: Additional arguments passed to the classifier constructor

    Returns:
        Initialized classifier instance
    """
    model_type_lower = model_type.lower()
    if model_type_lower == "lightgbm":
        return LightGBMClassifier(**kwargs)
    elif model_type_lower == "mlp":
        return MLPModerationClassifier(**kwargs)
    elif model_type_lower in {"logistic", "logistic_regression", "logreg"}:
        return LogisticRegressionClassifier(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Use 'lightgbm', 'mlp', or 'logistic'"
        )
