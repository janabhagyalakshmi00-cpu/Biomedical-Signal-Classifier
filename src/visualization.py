"""Plotting utilities for results and coupling maps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from .config import FIGURES_DIR, ensure_dirs
from .utils import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


def plot_coupling_map(
    edges: List[List[int]],
    n_qubits: int,
    title: str,
    filename: str,
) -> Path:
    ensure_dirs()
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=800,
        font_size=12,
        font_weight="bold",
        edge_color="gray",
        width=2,
        ax=ax,
    )
    ax.set_title(title)
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_confusion_matrix(
    cm: List[List[int]],
    title: str,
    filename: str,
    labels: List[str] = None,
) -> Path:
    ensure_dirs()
    if labels is None:
        labels = ["Normal", "Abnormal"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    filename: str = "model_performance_comparison.png",
) -> Path:
    ensure_dirs()
    models = list(metrics_dict.keys())
    keys = ["accuracy", "precision_abnormal", "recall_abnormal", "f1_abnormal"]
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        vals = [metrics_dict[m].get(k, 0) for k in keys]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Accuracy", "Precision (Abn)", "Recall (Abn)", "F1 (Abn)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_resource_comparison(
    resources: Dict[str, Dict[str, int]],
    metric: str,
    title: str,
    filename: str,
) -> Path:
    ensure_dirs()
    names = list(resources.keys())
    vals = [resources[n].get(metric, 0) for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, color=["#4c72b0", "#dd8452"])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_training_convergence(
    loss_history: List[float],
    filename: str = "training_convergence.png",
) -> Optional[Path]:
    if not loss_history:
        logger.warning("No loss history available – skipping convergence plot")
        return None
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history, marker="o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective / Loss")
    ax.set_title("VQC Training Convergence")
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path
