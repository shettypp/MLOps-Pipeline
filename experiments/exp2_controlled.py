"""Exp 2 - Controlled drift: false alarms vs detection, with known ground truth.

For each seed: AutoML model on clean data -> 30-window stream (300 rows each) with drift
injected from window 15 onward. Scenarios: none, benign (noise features shifted),
harmful (informative features shifted), concept (30% labels flipped, P(X) unchanged).
"""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import make_base, inject
from src.automl import search, predict_proba
from src.drift import KSDetector, PSIDetector, MultiSignalDetector, ErrorRateDetector

OUT = Path(__file__).resolve().parent.parent / "results"; OUT.mkdir(exist_ok=True)
SEEDS = range(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
W, N_WIN, DRIFT_AT = 300, 30, 15
SCEN = {"none": ("none", None), "benign-1.0": ("benign", 1.0), "benign-2.0": ("benign", 2.0),
        "harmful-0.3": ("harmful", 0.3), "harmful-0.6": ("harmful", 0.6),
        "harmful-1.0": ("harmful", 1.0), "concept-0.3": ("concept", 0.30)}


def detectors():
    return {"KS": KSDetector(), "PSI": PSIDetector(), "ErrorRate (labels)": ErrorRateDetector(),
            "Multi-signal": MultiSignalDetector(),
            "MS no-persistence": MultiSignalDetector(persist=1),
            "MS 1-vote (OR)": MultiSignalDetector(votes=1)}


rows = []
for seed in SEEDS:
    X, y, inf, noise = make_base(seed)
    tr, ref = slice(0, 3000), slice(3000, 4000)
    m = search(X.iloc[tr], y[tr], budget=8, seed=seed)
    model = m["model"]
    p_ref = predict_proba(model, X.iloc[ref]); acc_ref = accuracy_score(y[ref], p_ref > .5)
    rng = np.random.default_rng(seed)
    for scen, (kind, mag) in SCEN.items():
        cols = noise if kind == "benign" else inf[:2]
        dets = {k: d.fit(X.iloc[ref], p_ref, acc_ref) for k, d in detectors().items()}
        alarms = {k: [] for k in dets}; accs = []
        for w in range(N_WIN):
            s = slice(4000 + w * W, 4000 + (w + 1) * W)
            Xw, yw = X.iloc[s], y[s]
            if scen != "none" and w >= DRIFT_AT:
                Xw, yw = inject(Xw, yw, kind, cols, mag, rng)
            p = predict_proba(model, Xw); acc = accuracy_score(yw, p > .5); accs.append(acc)
            for k, d in dets.items():
                alarms[k].append(bool(d.check(Xw, p, acc)["alarm"]))
        for k, a in alarms.items():
            pre, post = a[:DRIFT_AT], a[DRIFT_AT:]
            first = next((i for i, v in enumerate(post) if v), None)
            rows.append({"seed": seed, "scenario": scen, "detector": k, "estimator": m["estimator"],
                         "pre_alarm_rate": float(np.mean(pre)), "post_alarm_rate": float(np.mean(post)),
                         "detected": first is not None, "delay": first,
                         "acc_pre": float(np.mean(accs[:DRIFT_AT])), "acc_post": float(np.mean(accs[DRIFT_AT:]))})
    print(f"seed {seed} done ({m['estimator']}, ref acc {acc_ref:.3f})", flush=True)

(OUT / "exp2_raw.json").write_text(json.dumps(rows, indent=1))
print("saved", len(rows))
