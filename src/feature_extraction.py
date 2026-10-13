"""Feature extraction helpers.

For the synthetic demonstration the features are generated directly.
This module documents the intended real-data pipeline and provides
utility functions that would be used with actual ECG waveforms.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .data_loader import FEATURE_NAMES
from .utils import get_logger

logger = get_logger(__name__)


def get_feature_names(n: int = 6) -> List[str]:
    return FEATURE_NAMES[:n]


def extract_features_from_waveform(
    signal: np.ndarray,
    fs: float = 360.0,
) -> np.ndarray:
    """Placeholder for real ECG feature extraction.

    In a production pipeline this would compute RR statistics, QRS amplitude,
    spectral entropy, etc.  For the synthetic mode this function is not called.
    """
    # Minimal deterministic stub so the import path exists
    if signal.ndim != 1:
        raise ValueError("Expected 1-D waveform")
    # Return zeros of correct length – never used in synthetic path
    return np.zeros(len(FEATURE_NAMES), dtype=np.float64)


def validate_feature_matrix(X: np.ndarray, expected_dim: int = 6) -> None:
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix, got shape {X.shape}")
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} features, got {X.shape[1]}"
        )
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Feature matrix contains NaN or Inf")
    logger.info("Feature matrix validated: shape=%s", X.shape)
