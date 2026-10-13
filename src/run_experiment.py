"""End-to-end experiment runner (produces all tables, figures, predictions)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ensure_dirs,
    TABLES_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    N_SAMPLES,
    N_QUBITS,
    MAX_ITER,
    RANDOM_SEED,
    save_json,
)
from .utils import set_seed, get_logger
from .data_loader import load_dataset, save_processed
from .preprocessing import preprocess_pipeline
from .feature_extraction import validate_feature_matrix
from .classical_model import (
    train_logistic_regression,
    evaluate_classical,
    save_classical_model,
)
from .quantum_model import (
    create_vqc,
    train_vqc,
    evaluate_vqc,
    save_vqc_parameters,
    get_logical_circuit,
)
from .metrics import metrics_to_row
from .processors import load_processor
from .benchmark import measure_circuit_resources
from .visualization import (
    plot_coupling_map,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_resource_comparison,
    plot_training_convergence,
)

logger = get_logger("experiment")


def main() -> None:
    set_seed(RANDOM_SEED)
    ensure_dirs()
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    logger.info("=== Loading synthetic dataset ===")
    df, y, feature_names = load_dataset(use_synthetic=True, n_samples=N_SAMPLES)
    X = df[feature_names].values
    validate_feature_matrix(X, expected_dim=len(feature_names))

    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(X, y)
    save_processed(X_train, X_test, y_train, y_test, feature_names)

    # ------------------------------------------------------------------
    # 2. Classical baseline
    # ------------------------------------------------------------------
    logger.info("=== Classical Logistic Regression ===")
    clf = train_logistic_regression(X_train, y_train)
    y_pred_clf, metrics_clf = evaluate_classical(clf, X_test, y_test)
    save_classical_model(clf, metrics_clf)

    # ------------------------------------------------------------------
    # 3. VQC
    # ------------------------------------------------------------------
    logger.info("=== Variational Quantum Classifier ===")
    vqc = create_vqc(n_qubits=N_QUBITS, max_iter=MAX_ITER)
    vqc, loss_history = train_vqc(vqc, X_train, y_train)
    y_pred_vqc, metrics_vqc = evaluate_vqc(vqc, X_test, y_test)
    save_vqc_parameters(vqc)

    # ------------------------------------------------------------------
    # 4. Logical circuit for resource analysis
    # ------------------------------------------------------------------
    # Use a representative sample (first test vector)
    sample = X_test[0]
    logical_qc = get_logical_circuit(vqc, sample)
    logger.info("Logical circuit: %d qubits, depth %d", logical_qc.num_qubits, logical_qc.depth())

    # ------------------------------------------------------------------
    # 5. Processor A / B resource measurement
    # ------------------------------------------------------------------
    logger.info("=== Processor A/B transpilation ===")
    res_A = measure_circuit_resources(logical_qc, "A")
    res_B = measure_circuit_resources(logical_qc, "B")

    # Coupling-map figures
    cfg_A = load_processor("A")
    cfg_B = load_processor("B")
    plot_coupling_map(
        cfg_A["coupling_map"],
        cfg_A["physical_qubit_budget"],
        "Processor A Coupling Map (5 qubits)",
        "processor_A_coupling.png",
    )
    plot_coupling_map(
        cfg_B["coupling_map"],
        cfg_B["physical_qubit_budget"],
        "Processor B Coupling Map (9 qubits)",
        "processor_B_coupling.png",
    )

    # ------------------------------------------------------------------
    # 6. Tables & predictions
    # ------------------------------------------------------------------
    rows = [
        metrics_to_row("LogisticRegression", metrics_clf),
        metrics_to_row("VQC_ideal", metrics_vqc),
    ]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(TABLES_DIR / "classification_metrics.csv", index=False)

    resources_df = pd.DataFrame([res_A, res_B])
    resources_df.to_csv(TABLES_DIR / "circuit_resources.csv", index=False)

    arch_df = pd.DataFrame(
        [
            {
                "processor": "A",
                "physical_budget": res_A["physical_qubit_budget"],
                "depth": res_A["depth"],
                "cx_count": res_A["cx_count"],
                "swap_count": res_A["swap_count"],
                "logical_qubits": res_A["logical_qubits"],
            },
            {
                "processor": "B",
                "physical_budget": res_B["physical_qubit_budget"],
                "depth": res_B["depth"],
                "cx_count": res_B["cx_count"],
                "swap_count": res_B["swap_count"],
                "logical_qubits": res_B["logical_qubits"],
            },
        ]
    )
    arch_df.to_csv(TABLES_DIR / "architecture_comparison.csv", index=False)

    # Predictions
    pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred_classical": y_pred_clf,
            "y_pred_vqc": y_pred_vqc,
        }
    )
    pred_df.to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=False)

    # Experiment config snapshot
    exp_cfg = {
        "random_seed": RANDOM_SEED,
        "n_samples": N_SAMPLES,
        "n_features": len(feature_names),
        "n_qubits": N_QUBITS,
        "max_iter": MAX_ITER,
        "feature_names": feature_names,
        "synthetic": True,
        "primary_metric": "recall_abnormal",
    }
    save_json(exp_cfg, TABLES_DIR / "experiment_config.json")
    save_json(exp_cfg, MODELS_DIR / "model_config.json")

    # ------------------------------------------------------------------
    # 7. Figures
    # ------------------------------------------------------------------
    plot_confusion_matrix(
        metrics_clf["confusion_matrix"],
        "Classical Logistic Regression",
        "cm_classical.png",
    )
    plot_confusion_matrix(
        metrics_vqc["confusion_matrix"],
        "VQC (ideal)",
        "cm_vqc.png",
    )
    plot_metrics_comparison(
        {
            "LogisticRegression": metrics_clf,
            "VQC_ideal": metrics_vqc,
        }
    )
    plot_resource_comparison(
        {"Processor A": res_A, "Processor B": res_B},
        "depth",
        "Transpiled Circuit Depth",
        "circuit_depth_comparison.png",
    )
    plot_resource_comparison(
        {"Processor A": res_A, "Processor B": res_B},
        "cx_count",
        "CX Gate Count",
        "cx_count_comparison.png",
    )
    plot_resource_comparison(
        {"Processor A": res_A, "Processor B": res_B},
        "swap_count",
        "SWAP Count",
        "swap_count_comparison.png",
    )
    plot_training_convergence(loss_history)

    elapsed = time.time() - t0
    logger.info("Experiment finished in %.1f s", elapsed)
    logger.info("Classical recall_abnormal = %.4f", metrics_clf["recall_abnormal"])
    logger.info("VQC      recall_abnormal = %.4f", metrics_vqc["recall_abnormal"])
    logger.info("Processor A depth=%d CX=%d SWAP=%d", res_A["depth"], res_A["cx_count"], res_A["swap_count"])
    logger.info("Processor B depth=%d CX=%d SWAP=%d", res_B["depth"], res_B["cx_count"], res_B["swap_count"])


if __name__ == "__main__":
    main()
