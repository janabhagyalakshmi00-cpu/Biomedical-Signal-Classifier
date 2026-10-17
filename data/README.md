# Data Directory

## Synthetic Dataset – Phase 1 Master Guide Configuration

The dataset follows the official specification from the **Phase 1 Master Guide (page 5)**:

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=200,
    n_features=4,
    n_informative=3,
    n_redundant=0,      # explicit fix (see note below)
    n_repeated=0,
    random_state=42
)
```

### Important note on `n_redundant`

The Master Guide lists only:
```
n_samples=200, n_features=4, n_informative=3, random_state=42
```

scikit-learn’s **default** `n_redundant=2` makes the call invalid  
(3 informative + 2 redundant = 5 > 4 features) and raises `ValueError`.

We therefore set `n_redundant=0` (and `n_repeated=0`) explicitly.  
This assumption is documented in `report.md` and `docs/methodology.md`.

- Labels: 0 = Normal, 1 = Abnormal
- Generated with fixed seed 42 for full reproducibility
- Never described as real patient data

### Files produced after first run

- `sample/synthetic_features.csv`
- `processed/train_test_split.npz`
- `processed/dataset_meta.json`

## Real Dataset Option (future)

The MIT-BIH Arrhythmia Database (PhysioNet) can be added later with the `wfdb` library.  
Phase-1 evaluation uses only the Master-Guide synthetic configuration above.
