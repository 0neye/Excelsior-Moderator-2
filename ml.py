"""Machine learning models for moderation classification.

Provides reusable classifier implementations for both bootstrapping and continuous training.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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
        feature_names = getattr(self, "feature_names", FEATURE_NAMES)
        
        return dict(zip(feature_names, importances))


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
        feature_names = getattr(self, "feature_names", FEATURE_NAMES)

        return dict(zip(feature_names, normalized_importances))


class _MLPNetwork(nn.Module):
    """PyTorch neural network module for the MLP classifier."""

    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: tuple[int, ...],
        output_size: int,
        dropout_rate: float = 0.3,
        activation: str = "relu",
    ):
        """
        Initialize the MLP network architecture.

        Args:
            input_size: Number of input features
            hidden_layer_sizes: Tuple of hidden layer sizes
            output_size: Number of output classes
            dropout_rate: Dropout probability between layers
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
        """
        super().__init__()

        # Select activation function
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }
        activation_fn = activation_map.get(activation, nn.ReLU)

        # Build sequential layers dynamically
        layers: list[nn.Module] = []
        prev_size = input_size

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer (no activation - handled by loss function)
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class MLPModerationClassifier(ModerationClassifier):
    """Multi-layer perceptron classifier for moderation using PyTorch."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64, 32),
        activation: str = "relu",
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 200,
        early_stopping_patience: int = 15,
        random_state: int = 42,
        device: str | None = None,
    ):
        """
        Initialize PyTorch MLP classifier.

        Args:
            hidden_layer_sizes: Sizes of hidden layers (e.g., (128, 64, 32))
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
            dropout_rate: Dropout probability between layers (0.0-1.0)
            learning_rate: Adam optimizer learning rate
            weight_decay: L2 regularization strength
            batch_size: Training batch size
            max_epochs: Maximum number of training epochs
            early_stopping_patience: Epochs without improvement before stopping
            random_state: Random seed for reproducibility
            device: Device to train on ('cuda', 'cpu', or None for auto)
        """
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

        # Auto-select device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: _MLPNetwork | None = None
        self.scaler: StandardScaler = StandardScaler()
        self.n_features: int = 0
        self.n_classes: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the PyTorch MLP classifier."""
        logger.info(f"Training PyTorch MLP with {len(y)} samples on {self.device}...")

        # Set random seeds for reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Encode labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        classes = self.label_encoder.classes_
        self.n_classes = len(classes) if classes is not None else 0
        self.n_features = X.shape[1]

        logger.info(f"Classes: {list(classes) if classes is not None else []}")
        class_counts = dict(zip(*np.unique(y, return_counts=True)))
        logger.info(f"Class distribution: {class_counts}")

        # Standardize features (crucial for neural network performance)
        X_scaled = self.scaler.fit_transform(X)

        # Compute class weights for imbalanced data
        y_encoded_arr = np.asarray(y_encoded)
        class_weights = self._compute_class_weights(y_encoded_arr)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
            self.device
        )

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        # Create train/val split for early stopping (90/10)
        n_samples = len(X_tensor)
        indices = torch.randperm(n_samples)
        val_size = max(1, int(0.1 * n_samples))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
        X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]

        # Create data loaders (drop_last=True avoids single-sample batches that break BatchNorm)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # Initialize the network
        self.model = _MLPNetwork(
            input_size=self.n_features,
            hidden_layer_sizes=self.hidden_layer_sizes,
            output_size=self.n_classes,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        ).to(self.device)

        # Loss function with class weights and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Early stopping tracking
        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0

        # Training loop
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(X_batch)

            train_loss /= len(X_train)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                X_val_device = X_val.to(self.device)
                y_val_device = y_val.to(self.device)
                val_outputs = self.model(X_val_device)
                val_loss = criterion(val_outputs, y_val_device).item()

            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Log progress every 25 epochs
            if (epoch + 1) % 25 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted = True
        logger.info("PyTorch MLP training complete")

    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute balanced class weights inversely proportional to frequency."""
        class_counts = np.bincount(y)
        n_samples = len(y)
        n_classes = len(class_counts)
        # Inverse frequency weighting, scaled by number of classes
        weights = n_samples / (n_classes * class_counts)
        return weights.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Apply same standardization used during training
        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            # Argmax to get predicted class indices
            y_pred_encoded = outputs.argmax(dim=1).cpu().numpy()

        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Apply same standardization used during training
        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            # Softmax to convert logits to probabilities
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

    def _get_model_data(self) -> dict[str, Any]:
        """Get PyTorch MLP model data for serialization."""
        return {
            "model_state_dict": (
                self.model.state_dict() if self.model is not None else None
            ),
            "scaler": self.scaler,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "random_state": self.random_state,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
        }

    def _set_model_data(self, data: dict[str, Any]) -> None:
        """Set PyTorch MLP model data from deserialization."""
        self.scaler = data.get("scaler", StandardScaler())
        self.hidden_layer_sizes = data["hidden_layer_sizes"]
        self.activation = data["activation"]
        self.dropout_rate = data["dropout_rate"]
        self.learning_rate = data["learning_rate"]
        self.weight_decay = data["weight_decay"]
        self.batch_size = data["batch_size"]
        self.max_epochs = data["max_epochs"]
        self.early_stopping_patience = data["early_stopping_patience"]
        self.random_state = data["random_state"]
        self.n_features = data["n_features"]
        self.n_classes = data["n_classes"]

        # Rebuild the network architecture and load weights
        if data["model_state_dict"] is not None:
            self.model = _MLPNetwork(
                input_size=self.n_features,
                hidden_layer_sizes=self.hidden_layer_sizes,
                output_size=self.n_classes,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
            ).to(self.device)
            self.model.load_state_dict(data["model_state_dict"])
            self.model.eval()

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance from first layer weights.

        Uses absolute values of the first linear layer weights summed across
        all neurons as a proxy for feature importance.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        # Extract first linear layer weights
        first_layer = self.model.network[0]
        if isinstance(first_layer, nn.Linear):
            weights = first_layer.weight.detach().cpu().numpy()
            # Sum absolute weights across all output neurons
            importances = np.abs(weights).sum(axis=0)
            # Normalize to sum to 1
            importances = importances / importances.sum()
        else:
            # Fallback: equal importance if structure is unexpected
            importances = np.ones(self.n_features) / self.n_features

        feature_names = getattr(self, "feature_names", FEATURE_NAMES)
        return dict(zip(feature_names, importances))


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
