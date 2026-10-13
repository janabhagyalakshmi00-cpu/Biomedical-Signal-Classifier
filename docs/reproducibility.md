# Reproducibility Instructions

## Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.12 and the package versions listed in `requirements.txt`.

## Fixed Seeds

- Global NumPy / random seed: 42
- Train/test split: 42
- Transpiler seed: 42
- StatevectorSampler seed: 42

## Execution Order

1. `pytest tests/ -q` – unit & smoke tests
2. `jupyter notebook main.ipynb` – full end-to-end experiment
   (or run all cells programmatically)

All paths inside the notebook and source modules are relative to the project root.

## Synthetic Data

The dataset is generated on-the-fly with a fixed RNG seed. No external download is required. The generated CSV and NPZ files are written under `data/` for inspection.

## Determinism Notes

COBYLA is deterministic given the same starting point and objective evaluations. Statevector simulation is exact. Transpilation with a fixed seed yields identical circuits on the same Qiskit version.

Slight numerical differences may appear across Qiskit minor versions because of internal changes in the optimizer or transpiler heuristics; the qualitative conclusions remain stable.
