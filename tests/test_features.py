"""Tests for feature extraction utilities."""

import numpy as np
import pytest

from src.feature_extraction import get_feature_names, validate_feature_matrix
from src.data_loader import generate_synthetic_ecg_features
from src.config import N_FEATURES


def test_feature_names_length():
    names = get_feature_names(N_FEATURES)
    assert len(names) == N_FEATURES


def test_validate_good_matrix():
    X, _, _ = generate_synthetic_ecg_features(n_samples=50)
    validate_feature_matrix(X, expected_dim=N_FEATURES)


def test_validate_bad_dim():
    X = np.random.randn(10, 3)
    with pytest.raises(ValueError):
        validate_feature_matrix(X, expected_dim=N_FEATURES)
