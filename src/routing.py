"""Transpilation and resource counting for Processor A / B."""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from .config import TRANSPILATION_SEED, OPTIMIZATION_LEVEL
from .utils import get_logger

logger = get_logger(__name__)


def count_ops(circuit: QuantumCircuit) -> Dict[str, int]:
    """Count CX, SWAP, single-qubit gates, depth, etc."""
    counts = circuit.count_ops()
    cx = counts.get("cx", 0) + counts.get("cnot", 0)
    swap = counts.get("swap", 0)
    # Approximate single-qubit as everything else that is 1-qubit
    single = 0
    for name, cnt in counts.items():
        if name in ("cx", "cnot", "swap", "measure", "barrier"):
            continue
        # Heuristic: most remaining are single-qubit
        single += cnt
    return {
        "depth": circuit.depth(),
        "cx_count": int(cx),
        "swap_count": int(swap),
        "single_qubit_count": int(single),
        "num_qubits": circuit.num_qubits,
        "size": circuit.size(),
    }


def transpile_for_processor(
    circuit: QuantumCircuit,
    coupling_map: CouplingMap,
    seed: int = TRANSPILATION_SEED,
    optimization_level: int = OPTIMIZATION_LEVEL,
) -> Tuple[QuantumCircuit, Dict[str, int]]:
    """Transpile logical circuit onto a target coupling map and count resources."""
    tqc = transpile(
        circuit,
        coupling_map=coupling_map,
        seed_transpiler=seed,
        optimization_level=optimization_level,
        layout_method="sabre",
        routing_method="sabre",
    )
    resources = count_ops(tqc)
    logger.info(
        "Transpiled: depth=%d, CX=%d, SWAP=%d, qubits=%d",
        resources["depth"],
        resources["cx_count"],
        resources["swap_count"],
        resources["num_qubits"],
    )
    return tqc, resources
