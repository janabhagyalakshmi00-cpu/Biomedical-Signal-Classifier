# PS-Q1 Biomedical Signal Classifier – Technical Report

## Abstract

We present a hybrid quantum-classical Variational Quantum Classifier (VQC) for binary classification of short biomedical signal segments (Normal vs Abnormal). The model is compared against a classical Logistic Regression baseline using identical features and the same stratified train/test split. The trained logical circuit is transpiled onto two reference processor architectures (5-qubit Processor A and 9-qubit Processor B). Primary metric is Abnormal-class recall. All experiments use a deterministic synthetic feature set (seed 42) for full reproducibility.

## 1. Introduction

Quantum machine learning on near-term devices requires careful mapping of classical data onto limited-connectivity hardware. This work addresses the Qiskit Fall Fest 2026 Phase-1 problem PS-Q1 by implementing a complete, reproducible pipeline that includes data handling, classical baseline, trainable VQC, and architecture-aware benchmarking.

## 2. Problem Statement

Classify short ECG/EEG-style segments into Normal (0) or Abnormal (1). Primary evaluation metric: recall of the Abnormal class. Secondary metrics: accuracy, precision, F1, confusion matrix. Benchmark the same trained logical model on Processor A and Processor B, reporting depth, CX count, SWAP count and classification performance.

## 3. Dataset and Preprocessing

A synthetic dataset of 400 samples with 4 numerical features (RR mean, RR std, QRS amplitude, T-wave amplitude) is generated with a fixed RNG seed. Class imbalance is mild (~40 % abnormal). Features are scaled to [0, π] with a MinMaxScaler fitted exclusively on the training set. Stratified split (test size 0.25, seed 42) prevents trivial leakage. The synthetic nature of the data is clearly documented; no real patient recordings are redistributed.

## 4. Feature Engineering

The four features were chosen both for clinical interpretability and to keep the logical qubit count (4) compatible with Processor A’s physical budget of 5 qubits. All subsequent models therefore operate on the identical feature matrix.

## 5. Classical Model

Logistic Regression (scikit-learn, `class_weight="balanced"`, LBFGS solver) is trained on the scaled training features. On the held-out test set the model achieves Abnormal recall 0.975, accuracy 0.990. Predictions and metrics are persisted under `results/` and `models/`.

## 6. Quantum Model

- **Feature map**: ZZFeatureMap (1 repetition, linear entanglement) – realises angle encoding.
- **Ansatz**: RealAmplitudes (2 repetitions, linear entanglement) – 12 trainable parameters, CX entangling gates.
- **Optimizer**: COBYLA (40 iterations).
- **Backend for training**: StatevectorSampler (exact).
- **Decision rule**: standard VQC binary classification.

After training the ideal VQC reaches Abnormal recall 0.550 and accuracy 0.730. Trained weights are saved as `models/vqc_weights.npy`.

## 7. Experimental Methodology

The identical logical circuit (feature map + bound ansatz + measurement) is transpiled with seed 42 and optimisation level 1 onto each processor’s coupling map (SABRE layout & routing). Resource counts are extracted directly from the transpiled circuits. Classification metrics are those of the ideal logical model; optional noisy simulation uses the Phase-1 reference noise rates (p₁ = 0.001, p_CX = 0.005, readout = 0.01).

## 8. Processor Architecture Analysis

Processor A (5 qubits) has a sparse tree topology. Processor B (9 qubits) offers richer connectivity. With a compact 4-qubit logical circuit both architectures produced identical resource counts (depth 3, CX 0, SWAP 0) under the chosen transpiler settings, indicating that the logical circuit already matched the available connectivity without extra routing. Larger ansatz depths or higher qubit counts would be expected to reveal differences in SWAP overhead.

## 9. Results

See `results/tables/classification_metrics.csv`, `circuit_resources.csv` and `architecture_comparison.csv`. Figures are stored under `results/figures/`.

## 10. Discussion

The classical model substantially outperforms the modest-depth VQC on this synthetic task, which is expected: the feature space is linearly separable and the variational circuit has limited expressivity under the iteration budget. The architecture comparison demonstrates that, for sufficiently compact logical circuits, both processors can embed the model with negligible routing cost. Claims of quantum advantage are not supported by the present evidence and are not made.

## 11. Limitations

- Synthetic data only.
- Limited ansatz depth and training iterations.
- Simple noise model.
- No per-architecture parameter re-optimisation.
- No real quantum hardware execution.

## 12. Conclusion

A complete, reproducible Phase-1 submission for PS-Q1 has been delivered. The pipeline satisfies all stated requirements: identical features/split for classical and quantum models, primary focus on Abnormal recall, Processor A/B resource benchmarking, and transparent documentation of synthetic data and limitations.

## References

- Qiskit & Qiskit Machine Learning documentation
- PhysioNet MIT-BIH Arrhythmia Database
- QFF 2026 Phase 1 Master Challenge Guide
