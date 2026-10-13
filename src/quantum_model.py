"""Hybrid Variational Quantum Classifier (VQC).

Uses angle encoding (via ZZFeatureMap / PauliFeatureMap style) and a
RealAmplitudes ansatz.  Training is performed with a classical optimizer
(COBYLA) on the Qiskit Machine Learning VQC interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA as ML_COBYLA  # fallback

from .config import (
    N_QUBITS,
    MAX_ITER,
    RANDOM_SEED,
    MODELS_DIR,
    FIGURES_DIR,
    ensure_dirs,
)
from .metrics import compute_metrics
from .utils import get_logger, set_seed

logger = get_logger(__name__)


def build_feature_map(n_qubits: int = N_QUBITS, reps: int = 1):
    """Angle-encoding style feature map (ZZFeatureMap uses RZ + ZZ)."""
    return ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement="linear")


def build_ansatz(n_qubits: int = N_QUBITS, reps: int = 2):
    """Compact variational ansatz with trainable parameters + CX entangling."""
    return RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement="linear")


def create_vqc(
    n_qubits: int = N_QUBITS,
    max_iter: int = MAX_ITER,
    seed: int = RANDOM_SEED,
) -> VQC:
    """Construct a trainable VQC instance."""
    set_seed(seed)
    feature_map = build_feature_map(n_qubits)
    ansatz = build_ansatz(n_qubits)

    # Prefer the optimizer from qiskit_algorithms
    optimizer = COBYLA(maxiter=max_iter)

    # StatevectorSampler is ideal (exact) simulator – fast for small circuits
    sampler = StatevectorSampler(seed=seed)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
    )
    logger.info(
        "VQC created: qubits=%d, feature_map_params=%d, ansatz_params=%d",
        n_qubits,
        feature_map.num_parameters,
        ansatz.num_parameters,
    )
    return vqc


def train_vqc(
    vqc: VQC,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[VQC, List[float]]:
    """Train the VQC and return the fitted model + loss history if available."""
    logger.info("Starting VQC training (%d samples) ...", len(y_train))
    # VQC expects labels as 0/1 integers
    vqc.fit(X_train, y_train)

    # Attempt to recover objective values for convergence plot
    loss_history: List[float] = []
    if hasattr(vqc, "fit_result") and vqc.fit_result is not None:
        # COBYLA / SciPy style
        try:
            loss_history = list(vqc.fit_result.fun_vals) if hasattr(vqc.fit_result, "fun_vals") else []
        except Exception:
            pass
    if not loss_history and hasattr(vqc.optimizer, "callback"):
        # Some versions store history via callback
        pass

    logger.info("VQC training finished.")
    return vqc, loss_history


def evaluate_vqc(
    vqc: VQC,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Predict class labels and compute metrics."""
    y_pred = vqc.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    return y_pred, metrics


def save_vqc_parameters(
    vqc: VQC,
    path: Path | None = None,
) -> Path:
    """Persist the trained weights."""
    ensure_dirs()
    path = path or (MODELS_DIR / "vqc_weights.npy")
    weights = vqc.weights
    np.save(path, weights)
    from .config import save_json
    save_json(
        {
            "model_type": "VQC",
            "n_qubits": N_QUBITS,
            "feature_map": "ZZFeatureMap(reps=1, entanglement=linear)",
            "ansatz": "RealAmplitudes(reps=2, entanglement=linear)",
            "optimizer": "COBYLA",
            "max_iter": MAX_ITER,
            "n_parameters": int(len(weights)),
            "weights_path": str(path.name),
        },
        MODELS_DIR / "vqc_config.json",
    )
    logger.info("VQC weights saved to %s (dim=%d)", path, len(weights))
    return path


def get_logical_circuit(vqc: VQC, x_sample: np.ndarray):
    """Return the bound quantum circuit for a single sample (for transpilation)."""
    # Compose feature map + ansatz with assigned parameters
    # VQC stores the circuit internally; we rebuild for clarity
    feature_map = build_feature_map()
    ansatz = build_ansatz()
    # Bind feature values
    bound_fm = feature_map.assign_parameters(x_sample)
    # Bind trained weights
    bound_ansatz = ansatz.assign_parameters(vqc.weights)
    circuit = bound_fm.compose(bound_ansatz)
    circuit.measure_all()
    return circuit
