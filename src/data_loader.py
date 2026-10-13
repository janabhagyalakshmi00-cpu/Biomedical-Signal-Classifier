"""Dataset loading – synthetic ECG-like features (deterministic).

This project uses a synthetic demonstration dataset that mimics typical
numerical features extracted from short ECG segments.  It is clearly
labelled as synthetic and must never be described as real patient data.

Real MIT-BIH Arrhythmia Database support can be added later with the
wfdb library; for Phase-1 reproducibility we rely on the synthetic set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .config import (
    N_SAMPLES,
    N_FEATURES,
    PROCESSED_DIR,
    SAMPLE_DIR,
    RANDOM_SEED,
    ensure_dirs,
)
from .utils import set_seed, get_logger

logger = get_logger(__name__)

FEATURE_NAMES = [
    "rr_mean",          # mean RR interval (normalized)
    "rr_std",           # std of RR intervals
    "qrs_amp",          # QRS amplitude
    "t_wave_amp",       # T-wave amplitude
    "spectral_entropy", # spectral entropy of the segment
    "hrv_rmssd",        # RMSSD heart-rate variability proxy
]


def generate_synthetic_ecg_features(
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate deterministic synthetic feature matrix and binary labels.

    Class 0 = Normal, Class 1 = Abnormal.
    Features are scaled roughly into [0, π] range later for angle encoding.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # Base features drawn from slightly different distributions per class
    X = np.zeros((n_samples, n_features), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.int64)

    # Roughly balanced but with mild imbalance (~40% abnormal)
    n_abnormal = int(0.40 * n_samples)
    y[:n_abnormal] = 1
    rng.shuffle(y)

    # Base distributions (only first n_features columns are filled)
    normal_means = [0.85, 0.05, 1.10, 0.35, 0.55, 0.025]
    normal_stds  = [0.08, 0.02, 0.15, 0.08, 0.10, 0.008]
    abn_means    = [0.65, 0.18, 0.70, 0.15, 0.85, 0.09]
    abn_stds     = [0.18, 0.07, 0.25, 0.12, 0.15, 0.04]

    for i in range(n_samples):
        means = normal_means if y[i] == 0 else abn_means
        stds  = normal_stds  if y[i] == 0 else abn_stds
        for j in range(n_features):
            X[i, j] = rng.normal(means[j], stds[j])

    # Clip extreme outliers
    X = np.clip(X, 0.0, 2.5)
    names = FEATURE_NAMES[:n_features]
    return X, y, names


def load_dataset(
    use_synthetic: bool = True,
    n_samples: int = N_SAMPLES,
) -> Tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load (or generate) the dataset.

    Returns
    -------
    df : DataFrame with feature columns + label
    y  : label array
    feature_names : list of feature column names
    """
    ensure_dirs()
    if not use_synthetic:
        raise NotImplementedError(
            "Real MIT-BIH download is not enabled in this demonstration build. "
            "Set use_synthetic=True (default)."
        )

    X, y, names = generate_synthetic_ecg_features(n_samples=n_samples)
    df = pd.DataFrame(X, columns=names)
    df["label"] = y
    df["sample_id"] = [f"syn_{i:04d}" for i in range(len(df))]

    # Persist raw synthetic data
    raw_path = SAMPLE_DIR / "synthetic_ecg_features.csv"
    df.to_csv(raw_path, index=False)
    logger.info("Synthetic dataset saved to %s (%d samples)", raw_path, len(df))
    return df, y, names


def save_processed(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save train/test splits and feature definition."""
    ensure_dirs()
    np.savez(
        PROCESSED_DIR / "train_test_split.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=np.array(feature_names),
    )
    meta = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "class_distribution_train": {
            "normal": int((y_train == 0).sum()),
            "abnormal": int((y_train == 1).sum()),
        },
        "class_distribution_test": {
            "normal": int((y_test == 0).sum()),
            "abnormal": int((y_test == 1).sum()),
        },
        "synthetic": True,
        "random_seed": RANDOM_SEED,
    }
    from .config import save_json
    save_json(meta, PROCESSED_DIR / "dataset_meta.json")
    logger.info("Processed splits saved to %s", PROCESSED_DIR)
