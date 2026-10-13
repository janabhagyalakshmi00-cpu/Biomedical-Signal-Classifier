# Data Directory

## Synthetic Demonstration Dataset

This project ships with a **deterministic synthetic dataset** that mimics numerical features commonly extracted from short ECG segments (RR statistics, QRS amplitude, spectral entropy, HRV proxy, etc.).

- **Never** describe these samples as real patient recordings.
- Labels: 0 = Normal, 1 = Abnormal.
- Generated with fixed seed 42 for full reproducibility.
- Approximately 400 samples, ~40 % abnormal class.

File location after first run:

- `sample/synthetic_ecg_features.csv`
- `processed/train_test_split.npz`
- `processed/dataset_meta.json`

## Real Dataset Option (MIT-BIH Arrhythmia Database)

The MIT-BIH Arrhythmia Database is publicly available under PhysioNet:

- Source: https://physionet.org/content/mitdb/1.0.0/
- License: Open Data Commons Attribution License v1.0
- Typical use: extract 1–2 s windows around annotated beats, map beat annotations to Normal / Abnormal binary labels.

To enable real-data mode one would:

1. Install `wfdb`
2. Download selected records (e.g. 100, 101, 103, …)
3. Implement windowing + feature extraction in `src/feature_extraction.py`

The current Phase-1 demonstration deliberately uses synthetic data so that every reviewer can reproduce results without network access or large downloads.
