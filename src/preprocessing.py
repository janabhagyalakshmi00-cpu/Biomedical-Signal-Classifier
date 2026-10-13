"""Preprocessing utilities – scaling fitted on train only."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED, TEST_SIZE
from .utils import get_logger

logger = get_logger(__name__)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split with fixed seed.

    Note: synthetic samples are independent; for real multi-patient data
    a GroupShuffleSplit would be preferred to avoid patient leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    logger.info(
        "Split: train=%d (abn=%d), test=%d (abn=%d)",
        len(y_train),
        int((y_train == 1).sum()),
        len(y_test),
        int((y_test == 1).sum()),
    )
    return X_train, X_test, y_train, y_test


def fit_scaler(X_train: np.ndarray) -> MinMaxScaler:
    """Fit MinMaxScaler to [0, π] range suitable for angle encoding."""
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    scaler.fit(X_train)
    return scaler


def transform_features(
    scaler: MinMaxScaler,
    X: np.ndarray,
) -> np.ndarray:
    """Apply already-fitted scaler."""
    return scaler.transform(X)


def preprocess_pipeline(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Full reproducible preprocessing: split then scale (fit on train only)."""
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler = fit_scaler(X_train)
    X_train_s = transform_features(scaler, X_train)
    X_test_s = transform_features(scaler, X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler
