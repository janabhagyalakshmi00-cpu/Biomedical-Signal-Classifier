"""Processor configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

from qiskit.transpiler import CouplingMap

from .config import PROCESSORS_DIR, load_json, save_json
from .utils import get_logger

logger = get_logger(__name__)


def load_processor(name: str) -> Dict[str, Any]:
    """Load processor JSON (A or B)."""
    path = PROCESSORS_DIR / f"processor_{name.upper()}.json"
    if not path.exists():
        # fallback lower-case
        path = PROCESSORS_DIR / f"processor_{name}.json"
    cfg = load_json(path)
    validate_processor(cfg)
    return cfg


def validate_processor(cfg: Dict[str, Any]) -> None:
    """Validate qubit budget and coupling edges."""
    n = cfg["physical_qubit_budget"]
    edges = cfg["coupling_map"]
    for e in edges:
        if len(e) != 2:
            raise ValueError(f"Invalid edge {e}")
        a, b = e
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"Edge {e} outside qubit range [0, {n})")
        if a == b:
            raise ValueError(f"Self-loop not allowed: {e}")
    logger.info(
        "Processor %s validated: %d qubits, %d edges",
        cfg.get("name"),
        n,
        len(edges),
    )


def get_coupling_map(cfg: Dict[str, Any]) -> CouplingMap:
    """Return a Qiskit CouplingMap from config."""
    return CouplingMap(cfg["coupling_map"])


def list_processors() -> List[str]:
    return ["A", "B"]
