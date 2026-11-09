"""Data loading and controlled drift injection."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ELEC2_COLS = ["date", "day", "period", "nswprice", "nswdemand",
              "vicprice", "vicdemand", "transfer"]
FEATURES = ELEC2_COLS[1:]  # drop the date index to avoid temporal leakage


def load_elec2():
    """Elec2 (Harries 1999): 45,312 half-hourly NSW market records, label = price UP(1)/DOWN(0)."""
    X = pd.read_csv(DATA_DIR / "elec2_data.dat", sep=r"\s+", header=None, names=ELEC2_COLS)
    y = pd.read_csv(DATA_DIR / "elec2_label.dat", header=None)[0].to_numpy()
    return X[FEATURES].astype(float), y


def make_base(seed, n=14000, n_features=8, n_informative=4):
    """Synthetic tabular task with known informative (0..3) and pure-noise (4..7) features."""
    X, y = make_classification(n_samples=n, n_features=n_features, n_informative=n_informative,
                               n_redundant=0, n_repeated=0, shuffle=False, class_sep=1.2,
                               flip_y=0.02, random_state=seed)
    perm = np.random.default_rng(seed).permutation(n)  # shuffle rows only (keep column roles)
    X, y = X[perm], y[perm]
    cols = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y, cols[:n_informative], cols[n_informative:]


def inject(X, y, kind, cols, magnitude, rng):
    """Return a drifted copy of a window.
    benign  : shift pure-noise features (model output should not change)
    harmful : shift informative features (covariate shift that hurts accuracy)
    concept : keep P(X), flip a share of labels (P(y|X) changes; invisible to label-free tests)
    """
    X = X.copy(); y = y.copy()
    if kind in ("benign", "harmful"):
        for c in cols:
            X[c] = X[c] + magnitude * X[c].std()
    elif kind == "concept":
        flip = rng.random(len(y)) < magnitude
        y[flip] = 1 - y[flip]
    return X, y
