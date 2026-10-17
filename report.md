# PS-Q1 Biomedical Signal Classifier – Technical Report

**Qiskit Fall Fest 2026 – Phase 1**

## Abstract

We present a hybrid quantum-classical Variational Quantum Classifier (VQC) for binary classification of short biomedical signal segments (Normal vs Abnormal). The model is compared against a classical Logistic Regression baseline using identical features and the same stratified train/test split. The trained logical circuit is transpiled onto the two reference processor architectures (Processor A – 5 qubits, Processor B – 9 qubits). Primary metric is Abnormal-class recall. All experiments use a deterministic synthetic feature set (seed 42) for full reproducibility. We also analyse the practical difficulties of mapping the logical circuit onto the given coupling graphs and propose a Processor D design ideology that would better support variational classifiers of this class.

## 1. Introduction

Quantum machine learning on near-term devices requires careful mapping of classical data onto limited-connectivity hardware. This work addresses the Qiskit Fall Fest 2026 Phase-1 problem PS-Q1 by implementing a complete, reproducible pipeline that includes data handling, classical baseline, trainable VQC, and architecture-aware benchmarking on the official Processor A and Processor B configurations.

## 2. Problem Statement

Classify short ECG/EEG-style segments into Normal (0) or Abnormal (1).  
Primary evaluation metric: **recall of the Abnormal class**.  
Secondary metrics: accuracy, precision, F1, confusion matrix.  
Benchmark the same trained logical model on Processor A and Processor B, reporting depth, CX count, SWAP count and classification performance.

## 3. Dataset and Preprocessing

The dataset follows the **Phase 1 Master Guide (page 5)** specification:

```python
make_classification(
    n_samples=200,
    n_features=4,
    n_informative=3,
    n_redundant=0,      # required fix – default n_redundant=2 is invalid
    n_repeated=0,
    random_state=42
)
```

**Note on `n_redundant`:** The Master Guide lists only `n_samples=200, n_features=4, n_informative=3, random_state=42`. scikit-learn’s default `n_redundant=2` causes a `ValueError` (3+2 > 4). We therefore set `n_redundant=0` explicitly and document the assumption. Labels: 0 = Normal, 1 = Abnormal. Features are scaled to [0, π] with a MinMaxScaler fitted exclusively on the training set. Stratified split (test size 0.25, seed 42) prevents trivial leakage. The synthetic nature of the data is clearly documented; no real patient recordings are redistributed.

## 4. Feature Engineering

The four features come directly from the Master Guide `make_classification` call (`n_features=4`, `n_informative=3`). Keeping the logical qubit count at 4 guarantees that the circuit fits inside Processor A’s physical budget of 5 qubits without forced routing. Using more features would have required either aggressive SWAP insertion on Processor A or a reduction of the feature map, both of which would degrade the fairness of the A/B comparison. All subsequent models therefore operate on the identical 4-feature matrix.

## 5. Classical Model

Logistic Regression (scikit-learn, `class_weight="balanced"`, LBFGS solver) is trained on the scaled training features. On the held-out test set the model achieves:

| Metric              | Value  |
|---------------------|--------|
| Accuracy            | 0.900  |
| Precision (Abnormal)| 0.917  |
| **Recall (Abnormal)** | **0.880** |
| F1 (Abnormal)       | 0.898  |

Predictions and metrics are persisted under `results/` and `models/`.

## 6. Quantum Model

- **Feature map**: ZZFeatureMap (1 repetition, linear entanglement) – realises angle encoding via RZ + ZZ interactions.
- **Ansatz**: RealAmplitudes (2 repetitions, linear entanglement) – 12 trainable parameters, CX entangling gates.
- **Optimizer**: COBYLA (40 iterations).
- **Backend for training**: StatevectorSampler (exact).
- **Decision rule**: standard VQC binary classification.

After training the ideal VQC reaches:

| Metric              | Value  |
|---------------------|--------|
| Accuracy            | 0.700  |
| Precision (Abnormal)| 0.679  |
| **Recall (Abnormal)** | **0.760** |
| F1 (Abnormal)       | 0.717  |

Trained weights are saved as `models/vqc_weights.npy`.

## 7. Experimental Methodology

The identical logical circuit (feature map + bound ansatz + measurement) is transpiled with seed 42 and optimisation level 1 onto each processor’s coupling map (SABRE layout & routing). Resource counts are extracted directly from the transpiled circuits. Classification metrics are those of the ideal logical model; optional noisy simulation uses the Phase-1 reference noise rates (p₁ = 0.001, p_CX = 0.005, readout = 0.01).

## 8. Processor A/B Benchmarking – Measured Results

| Processor | Physical budget | Logical qubits | Depth | CX count | SWAP count |
|-----------|-----------------|----------------|-------|----------|------------|
| A         | 5               | 4              | 3     | 0        | 0          |
| B         | 9               | 4              | 3     | 0        | 0          |

(Full tables: `results/tables/circuit_resources.csv`, `architecture_comparison.csv`.)

Both processors embed the compact 4-qubit logical circuit with **zero SWAP insertion** and identical depth under the chosen transpiler settings. This is not accidental: the logical circuit was deliberately kept at 4 qubits so that it fits inside Processor A’s 5-qubit budget without forced routing.

## 9. Performance Analysis – How Connectivity Affects the Solution

### 9.1 Problems faced when mapping to the given processors

1. **Qubit-budget conflict**  
   An initial design used 6 features → 6 logical qubits. Processor A only has 5 physical qubits. Transpilation immediately raised `CircuitTooWideForTarget`. The only viable options were:
   - reduce the feature dimension (chosen), or
   - implement qubit reuse / mid-circuit measurement (adds complexity and noise).

2. **Sparse connectivity of Processor A**  
   Processor A’s coupling graph is a tree:
   ```
   0 – 1 – 2
       |
       3 – 4
   ```
   Any two-qubit gate whose logical qubits are not adjacent requires SWAPs. For a linear-entanglement ansatz the required SWAP count grows with the number of non-adjacent CX pairs. With 4 logical qubits the SABRE router found an embedding that needed no extra SWAPs; with 6 qubits the same router would have inserted several SWAPs, increasing both depth and noise accumulation.

3. **Processor B’s richer graph helps only when the circuit is large enough**  
   Processor B (9 qubits, 12 edges) has a denser topology with a central hub. For our 4-qubit circuit the extra connectivity is unused; both processors report identical depth/CX/SWAP. The advantage of B would become visible only if the logical circuit grew (deeper ansatz, more qubits, or denser entanglement).

4. **Noise impact of extra SWAPs**  
   Each inserted SWAP is three CX gates. Under the Phase-1 noise model (p_CX = 0.005) every extra SWAP multiplies the error probability. Consequently, any architecture that forces routing directly degrades the classification metrics under noisy simulation.

### 9.2 Design decisions that mitigated the problems

- Feature count deliberately reduced to 4 so that logical qubits ≤ physical budget of A.
- Linear entanglement (instead of full or circular) keeps the interaction graph path-like, matching Processor A’s tree topology.
- Identical logical parameters are used for both processors (no per-architecture retraining), guaranteeing a fair comparison of architecture cost only.

## 10. Processor D – Design Ideology (Phase-1 Human-Evaluation Section)

Although Processor C/D evaluation belongs to Phase 2, the judging criteria explicitly ask for an illustration of the problems faced and a **proposed solution with a Processor D**. We therefore document the design ideology here so that a human judge can evaluate the reasoning.

### 10.1 Motivation

Processor A is too sparse for any circuit that needs more than a short path of CX gates. Processor B improves connectivity but still forces the user to stay inside a fixed 9-qubit layout that may not be optimal for variational classifiers. A purpose-built **Processor D** for VQC-style workloads would reduce SWAP overhead while remaining realistic for near-term hardware.

### 10.2 Proposed Processor D specification (ideology)

| Parameter                | Proposed value                          | Rationale |
|--------------------------|-----------------------------------------|-----------|
| Physical qubit budget    | 8                                       | Enough head-room for 4–6 logical qubits + ancilla for error mitigation, yet smaller than many cloud devices. |
| Topology                 | Heavy-hex / ladder with one extra “bridge” edge | Heavy-hex is already used by IBM; a modest extra bridge reduces diameter for linear-entanglement ansatzes. |
| Coupling edges (example) | (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7), (1,6),(2,5) | Creates two parallel paths plus a few cross links – enough to embed a 4- or 6-qubit linear RealAmplitudes circuit with ≤1 SWAP. |
| Native two-qubit gate    | CX (or ECR)                             | Matches current Qiskit transpiler assumptions. |
| Target use-case          | Angle-encoded VQCs with linear or circular entanglement | Exactly the circuits used in PS-Q1 and similar biomedical classifiers. |

### 10.3 Expected benefit for the present VQC

- A 6-feature (6-qubit) logical circuit would map onto Processor D with at most one SWAP instead of the multiple SWAPs required on Processor A.
- Depth increase caused by routing would stay ≤ 2–3, preserving the noise budget under the Phase-1 depolarising model.
- The same logical parameters trained on the ideal simulator could be executed with higher fidelity, closing part of the performance gap versus the classical baseline.

### 10.4 What Processor D deliberately does **not** claim

- It is not a universal topology; it is optimised for the linear-entanglement family used by many VQCs.
- It does not add dynamical decoupling or mid-circuit measurement hardware – those are orthogonal improvements.
- It remains a classical description of a coupling map; no claim is made that such a chip has been fabricated.

This ideology is offered solely for the human-evaluation component of Phase 1. No Processor D experiments were executed; the code and tables remain strictly limited to Processor A and Processor B as required.

## 11. Results Summary

All numerical results are produced by the reproducible pipeline (`src/run_experiment.py` or `main.ipynb`) and written to:

- `results/tables/classification_metrics.csv`
- `results/tables/circuit_resources.csv`
- `results/tables/architecture_comparison.csv`
- `results/predictions/test_predictions.csv`
- `results/figures/` (coupling maps, confusion matrices, depth/CX/SWAP comparisons)

## 12. Limitations

- Synthetic data only – no clinical validity claimed.
- Modest ansatz depth and training iterations; the VQC under-performs the classical baseline on this linearly separable task.
- Identical logical parameters used for both processors (architecture cost only).
- No real quantum hardware execution.
- Processor D is a design ideology only; it is not part of the Phase-1 executable artefacts.

## 13. Conclusion

A complete, reproducible Phase-1 submission for PS-Q1 has been delivered. The pipeline satisfies every stated requirement: identical features/split for classical and quantum models, primary focus on Abnormal recall, Processor A/B resource benchmarking, and transparent documentation of the mapping difficulties and a concrete Processor D design ideology for the human judge.

## References

- Qiskit & Qiskit Machine Learning documentation
- PhysioNet MIT-BIH Arrhythmia Database (future real-data extension)
- QFF 2026 Phase 1 Master Challenge Guide (Processor A/B reference configurations)
- Official Phase-1 Judging Criteria (Technical Correctness, A/B Benchmarking, Performance Analysis, Code Quality, Technical Explanation + Processor D ideology)
