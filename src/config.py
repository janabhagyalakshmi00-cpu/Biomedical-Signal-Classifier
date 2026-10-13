"""Global configuration for PS-Q1 Biomedical Signal Classifier."""

from pathlib import Path
import json

# Project root (relative to this file: src/config.py -> parent.parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Random seed for reproducibility
RANDOM_SEED = 42

# Data settings
N_SAMPLES = 400  # Approximate total samples (200-500 range)
N_FEATURES = 4   # 4-8 numerical features (chosen so logical circuit fits Processor A budget of 5)
TEST_SIZE = 0.25
N_QUBITS = 4     # Matches feature count for angle encoding; fits both Processor A (5) and B (9)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LOGS_DIR = RESULTS_DIR / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSORS_DIR = PROJECT_ROOT / "processors"

# Quantum training
MAX_ITER = 40
OPTIMIZER_NAME = "COBYLA"
SHOTS = 4096

# Noise model (Phase 1 reference values)
NOISE_CONFIG = {
    "single_qubit_depolarizing": 0.001,
    "cx_depolarizing": 0.005,
    "readout_error": 0.01,
    "shots": 4096,
}

# Processor budgets
PROCESSOR_A_QUBITS = 5
PROCESSOR_B_QUBITS = 9

# Transpilation
TRANSPILATION_SEED = 42
OPTIMIZATION_LEVEL = 1

def ensure_dirs():
    """Create all required directories if missing."""
    for d in [
        RAW_DIR, PROCESSED_DIR, SAMPLE_DIR,
        TABLES_DIR, FIGURES_DIR, PREDICTIONS_DIR, LOGS_DIR,
        MODELS_DIR, PROCESSORS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
