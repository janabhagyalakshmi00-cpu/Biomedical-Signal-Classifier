# PS-Q1 Biomedical Signal Classifier

**Qiskit Fall Fest 2026 – Phase 1**

Hybrid quantum-classical Variational Quantum Classifier (VQC) for binary classification of short biomedical signal segments (Normal vs Abnormal), benchmarked against a classical Logistic Regression baseline and evaluated on two reference processor architectures.

## Problem Statement (PS-Q1)

Build a hybrid VQC that classifies short ECG/EEG-style signal segments into **Normal** or **Abnormal**. Compare against Logistic Regression using identical features and the same train/test split. Primary metric: **recall of the Abnormal class**. Benchmark the trained logical circuit on Processor A (5 qubits) and Processor B (9 qubits), reporting circuit depth, CX count, SWAP count and classification metrics.

## Dataset

- **Synthetic demonstration dataset** (deterministic, seed = 42).
- 400 samples, 4 numerical features inspired by ECG descriptors (RR mean/std, QRS amplitude, T-wave amplitude).
- Labels: 0 = Normal, 1 = Abnormal (~40 % abnormal).
- **Never described as real patient data.**
- Real MIT-BIH support can be added later; this build is fully reproducible offline.

See `data/README.md` for details.

## Methodology (short)

1. Generate / load features → stratified split (seed 42) → MinMax scale to [0, π] (fit on train only).
2. Classical baseline: Logistic Regression (`class_weight="balanced"`).
3. VQC: ZZFeatureMap (angle encoding) + RealAmplitudes ansatz (CX entanglement), trained with COBYLA on StatevectorSampler.
4. Transpile the identical logical circuit onto Processor A and Processor B (seed 42, opt-level 1).
5. Report ideal metrics + resource counts. Optional noisy simulation uses Phase-1 reference noise rates.

Full details: `docs/methodology.md`, `docs/architecture_analysis.md`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execution

```bash
# Unit & smoke tests
PYTHONPATH=. pytest tests/ -q

# Full experiment (writes tables, figures, models)
PYTHONPATH=. python -m src.run_experiment

# Interactive notebook
jupyter notebook main.ipynb
```

## Project Structure

```
PS-Q1-Biomedical-Signal-Classifier/
├── README.md
├── main.ipynb
├── requirements.txt
├── report.md
├── LICENSE
├── .gitignore
├── data/
├── processors/
│   ├── processor_A.json
│   └── processor_B.json
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── classical_model.py
│   ├── quantum_model.py
│   ├── processors.py
│   ├── routing.py
│   ├── benchmark.py
│   ├── metrics.py
│   ├── visualization.py
│   └── run_experiment.py
├── tests/
├── results/
│   ├── tables/
│   ├── figures/
│   └── predictions/
├── models/
└── docs/
```

## Key Results (from last successful run)

| Model                  | Accuracy | Precision (Abn) | Recall (Abn) | F1 (Abn) |
|------------------------|----------|-----------------|--------------|----------|
| Logistic Regression    | 0.990    | 1.000           | **0.975**    | 0.987    |
| VQC (ideal)            | 0.730    | 0.710           | **0.550**    | 0.620    |

| Processor | Physical budget | Depth | CX | SWAP |
|-----------|-----------------|-------|----|------|
| A         | 5               | 3     | 0  | 0    |
| B         | 9               | 3     | 0  | 0    |

(Actual numbers written to `results/tables/`. With the compact 4-qubit logical circuit both architectures incur zero routing SWAPs under the chosen transpiler settings.)

## Limitations

- Synthetic features only – no clinical validity claimed.
- VQC trained for a modest number of iterations; performance gap vs classical is expected for this small feature set and limited ansatz depth.
- Noise model is a simple depolarizing + readout approximation; real-device noise is more complex.
- Processor A/B comparison uses the same logical parameters (no per-architecture re-training).

## References

- Qiskit documentation & Qiskit Machine Learning
- MIT-BIH Arrhythmia Database (PhysioNet) – for future real-data extension
- QFF 2026 Phase 1 Master Challenge Guide (processor reference configurations)

## License

MIT – see `LICENSE`.
