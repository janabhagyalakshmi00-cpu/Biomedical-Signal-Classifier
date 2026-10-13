"""Tests for preprocessing and splitting."""

import numpy as np
import pytest

from src.preprocessing import split_data, fit_scaler, transform_features, preprocess_pipeline
from src.data_loader import generate_synthetic_ecg_features
from src.config import RANDOM_SEED, N_FEATURES


def test_split_stratified():
    X, y, _ = generate_synthetic_ecg_features(n_samples=200)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    assert len(X_tr) + len(X_te) == 200
    # Stratification roughly preserves ratio
    ratio_train = y_tr.mean()
    ratio_test = y_te.mean()
    assert abs(ratio_train - ratio_test) < 0.15


def test_scaler_range():
    X, y, _ = generate_synthetic_ecg_features(n_samples=100)
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    scaler = fit_scaler(X_tr)
    X_tr_s = transform_features(scaler, X_tr)
    assert X_tr_s.min() >= -1e-6
    assert X_tr_s.max() <= np.pi + 1e-6


def test_preprocess_pipeline_shapes():
    X, y, _ = generate_synthetic_ecg_features(n_samples=120)
    X_tr, X_te, y_tr, y_te, scaler = preprocess_pipeline(X, y)
    assert X_tr.shape[1] == N_FEATURES
    assert X_te.shape[1] == N_FEATURES
    assert len(y_tr) == X_tr.shape[0]
