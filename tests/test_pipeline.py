"""End-to-end smoke test (lightweight)."""

import numpy as np
import pytest

from src.data_loader import generate_synthetic_ecg_features
from src.preprocessing import preprocess_pipeline
from src.classical_model import train_logistic_regression, evaluate_classical
from src.quantum_model import create_vqc, build_feature_map, build_ansatz
from src.config import N_FEATURES, N_QUBITS


def test_classical_pipeline_smoke():
    X, y, _ = generate_synthetic_ecg_features(n_samples=80)
    X_tr, X_te, y_tr, y_te, _ = preprocess_pipeline(X, y)
    model = train_logistic_regression(X_tr, y_tr)
    y_pred, metrics = evaluate_classical(model, X_te, y_te)
    assert "recall_abnormal" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_vqc_construction():
    vqc = create_vqc(n_qubits=N_QUBITS, max_iter=2)
    assert vqc is not None
    fm = build_feature_map(N_QUBITS)
    ans = build_ansatz(N_QUBITS)
    assert fm.num_qubits == N_QUBITS
    assert ans.num_qubits == N_QUBITS
