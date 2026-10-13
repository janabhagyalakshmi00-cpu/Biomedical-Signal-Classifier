"""Utility helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import LOGS_DIR, RANDOM_SEED, ensure_dirs


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set global random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


def get_logger(name: str = "psq1") -> logging.Logger:
    """Configure and return a logger that writes to console and file."""
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(LOGS_DIR / "experiment.log", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)
