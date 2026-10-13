"""Classical Logistic Regression baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

from .config import MODELS_DIR, RANDOM_SEED, ensure_dirs
from .metrics import compute_metrics
from .utils import get_logger

logger = get_logger(__name__)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: str | Dict = "balanced",
) -> LogisticRegression:
    """Train Logistic Regression with class weighting for imbalance."""
    model = LogisticRegression(
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    logger.info(
        "LogisticRegression trained (classes=%s, coef shape=%s)",
        model.classes_,
        model.coef_.shape,
    )
    return model


def evaluate_classical(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Predict and compute metrics."""
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    return y_pred, metrics


def save_classical_model(
    model: LogisticRegression,
    metrics: Dict[str, Any],
    path: Path | None = None,
) -> Path:
    ensure_dirs()
    import joblib
    path = path or (MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(model, path)
    from .config import save_json
    save_json(
        {
            "model_type": "LogisticRegression",
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            "metrics_on_test": {
                k: v for k, v in metrics.items() if k != "confusion_matrix"
            },
            "confusion_matrix": metrics["confusion_matrix"],
        },
        MODELS_DIR / "classical_config.json",
    )
    logger.info("Classical model saved to %s", path)
    return path
