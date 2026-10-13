"""Tests for metric calculations."""

import numpy as np
from src.metrics import compute_metrics


def test_perfect_prediction():
    y = np.array([0, 0, 1, 1, 1])
    pred = y.copy()
    m = compute_metrics(y, pred)
    assert m["accuracy"] == 1.0
    assert m["recall_abnormal"] == 1.0
    assert m["precision_abnormal"] == 1.0


def test_all_normal_pred():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0])
    m = compute_metrics(y, pred)
    assert m["recall_abnormal"] == 0.0
    assert m["accuracy"] == 0.5
