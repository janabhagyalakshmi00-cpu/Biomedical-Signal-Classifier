"""Dataset loading – Phase 1 Master Guide synthetic configuration.

Official specification (Master Guide page 5):
    make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        random_state=42
    )

scikit-learn default n_redundant=2 makes the call invalid
(3 informative + 2 redundant > 4 features).  We therefore set
n_redundant=0 (and n_repeated=0) explicitly.  This assumption is
documented in data/README.md, report.md and docs/methodology.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from .config import (
    N_SAMPLES,
    N_FEATURES,
    N_INFORMATIVE,
    N_REDUNDANT,
    PROCESSED_DIR,
    SAMPLE_DIR,
    RANDOM_SEED,
    ensure_dirs,
)
from .utils import set_seed, get_logger

logger = get_logger(__name__)

FEATURE_NAMES = [
    "feat_0",
    "feat_1",
    "feat_2",
    "feat_3",
]


def generate_synthetic_features(
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    n_informative: int = N_INFORMATIVE,
    n_redundant: int = N_REDUNDANT,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate the Master-Guide synthetic dataset with the necessary fix.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)  – 0 = Normal, 1 = Abnormal
    feature_names : list of length n_features
    """
    set_seed(seed)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        random_state=seed,
    )
    names = FEATURE_NAMES[:n_features]
    logger.info(
        "make_classification: n_samples=%d, n_features=%d, "
        "n_informative=%d, n_redundant=%d, seed=%d",
        n_samples, n_features, n_informative, n_redundant, seed,
    )
    return X.astype(np.float64), y.astype(np.int64), names


# Keep old name as alias so existing imports continue to work
generate_synthetic_ecg_features = generate_synthetic_features


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
            "Real MIT-BIH download is not enabled in this Phase-1 build. "
            "Set use_synthetic=True (default)."
        )

    X, y, names = generate_synthetic_features(n_samples=n_samples)
    df = pd.DataFrame(X, columns=names)
    df["label"] = y
    df["sample_id"] = [f"syn_{i:04d}" for i in range(len(df))]

    # Persist raw synthetic data
    raw_path = SAMPLE_DIR / "synthetic_features.csv"
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
        "source": "sklearn.datasets.make_classification",
        "make_classification_params": {
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "n_informative": N_INFORMATIVE,
            "n_redundant": N_REDUNDANT,
            "n_repeated": 0,
            "random_state": RANDOM_SEED,
        },
        "note": (
            "Master Guide specifies n_informative=3 with n_features=4. "
            "Default n_redundant=2 is invalid; n_redundant=0 is the documented fix."
        ),
        "random_seed": RANDOM_SEED,
    }
    from .config import save_json
    save_json(meta, PROCESSED_DIR / "dataset_meta.json")
    logger.info("Processed splits saved to %s", PROCESSED_DIR)
