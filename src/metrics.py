"""Classification metrics focused on Abnormal-class recall."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from .utils import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: int = 1,
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall (Abnormal), F1 and confusion matrix."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(
        precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    )
    rec = float(
        recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    )
    f1 = float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    metrics = {
        "accuracy": acc,
        "precision_abnormal": prec,
        "recall_abnormal": rec,
        "f1_abnormal": f1,
        "confusion_matrix": cm,
        "n_samples": int(len(y_true)),
        "n_abnormal": int((y_true == positive_label).sum()),
    }
    logger.info(
        "Metrics – Acc: %.4f | Prec: %.4f | Recall(Abn): %.4f | F1: %.4f",
        acc, prec, rec, f1,
    )
    return metrics


def metrics_to_row(name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten metrics dict for CSV export."""
    cm = metrics["confusion_matrix"]
    return {
        "model": name,
        "accuracy": metrics["accuracy"],
        "precision_abnormal": metrics["precision_abnormal"],
        "recall_abnormal": metrics["recall_abnormal"],
        "f1_abnormal": metrics["f1_abnormal"],
        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1],
        "n_samples": metrics["n_samples"],
    }
