# Methodology

## Overview

We implement a hybrid quantum-classical Variational Quantum Classifier (VQC) for binary classification of short biomedical signal segments (Normal vs Abnormal). The same numerical features and the identical train/test split are used for both the classical Logistic Regression baseline and the VQC.

## Data

A deterministic synthetic feature matrix (seed = 42) with 6 clinically-motivated numerical descriptors is generated. Features are scaled to the interval [0, π] using a MinMaxScaler fitted exclusively on the training set, making them suitable for angle encoding.

## Classical Baseline

Scikit-learn Logistic Regression with `class_weight="balanced"` to mitigate mild class imbalance. Metrics focus on Abnormal-class recall (primary), accuracy, precision and F1.

## Quantum Model

- **Encoding**: ZZFeatureMap (linear entanglement, 1 repetition) – implements angle encoding via RZ rotations and ZZ interactions.
- **Ansatz**: RealAmplitudes (2 repetitions, linear entanglement) – trainable single-qubit rotations + CX gates.
- **Measurement**: Computational-basis measurement; class prediction obtained from the VQC decision rule.
- **Optimizer**: COBYLA (maxiter = 80).
- **Simulator**: StatevectorSampler (exact, ideal) for training; AerSimulator for optional noisy evaluation.

## Processor Benchmarking

The identical trained logical circuit is transpiled (seed = 42, optimization_level = 1, SABRE layout/routing) onto Processor A (5 qubits) and Processor B (9 qubits). Resource metrics (depth, CX count, SWAP count) are extracted from the transpiled circuits. Classification metrics are reported for the ideal logical model; noisy simulation uses the Phase-1 reference depolarizing and readout probabilities.

## Primary Metric

Recall of the Abnormal class. All other standard metrics are also reported together with confusion matrices.
