"""Processor A/B benchmarking utilities."""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from .config import NOISE_CONFIG, SHOTS
from .processors import load_processor, get_coupling_map
from .routing import transpile_for_processor, count_ops
from .utils import get_logger

logger = get_logger(__name__)


def build_noise_model(
    p1: float = NOISE_CONFIG["single_qubit_depolarizing"],
    p2: float = NOISE_CONFIG["cx_depolarizing"],
    readout: float = NOISE_CONFIG["readout_error"],
) -> NoiseModel:
    """Construct a simple depolarizing + readout noise model."""
    nm = NoiseModel()
    err1 = depolarizing_error(p1, 1)
    err2 = depolarizing_error(p2, 2)
    nm.add_all_qubit_quantum_error(err1, ["rx", "ry", "rz", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "u", "u1", "u2", "u3", "id"])
    nm.add_all_qubit_quantum_error(err2, ["cx", "cnot"])
    # Symmetric readout error
    ro = ReadoutError([[1 - readout, readout], [readout, 1 - readout]])
    nm.add_all_qubit_readout_error(ro)
    return nm


def measure_circuit_resources(
    logical_circuit: QuantumCircuit,
    processor_name: str,
) -> Dict[str, Any]:
    """Transpile and return resource dictionary for one processor."""
    cfg = load_processor(processor_name)
    cmap = get_coupling_map(cfg)
    tqc, resources = transpile_for_processor(logical_circuit, cmap)
    resources["processor"] = cfg["name"]
    resources["physical_qubit_budget"] = cfg["physical_qubit_budget"]
    resources["logical_qubits"] = logical_circuit.num_qubits
    resources["coupling_edges"] = len(cfg["coupling_map"])
    return resources


def run_ideal_simulation(
    circuit: QuantumCircuit,
    shots: int = SHOTS,
) -> Dict[str, int]:
    """Execute on ideal Aer simulator and return counts."""
    sim = AerSimulator(method="statevector")
    # For statevector we can use shots=None but keep interface uniform
    job = sim.run(circuit, shots=shots)
    result = job.result()
    return dict(result.get_counts())


def run_noisy_simulation(
    circuit: QuantumCircuit,
    noise_model: Optional[NoiseModel] = None,
    shots: int = SHOTS,
) -> Dict[str, int]:
    """Execute on noisy Aer simulator."""
    if noise_model is None:
        noise_model = build_noise_model()
    sim = AerSimulator(noise_model=noise_model)
    job = sim.run(circuit, shots=shots)
    result = job.result()
    return dict(result.get_counts())
