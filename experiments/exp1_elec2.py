"""Exp 1 - Real-world drift: retraining strategies on the Elec2 stream.

Initial AutoML model on rows 0-2999, detector reference on 3000-3999 (held out).
Remaining 41,312 rows replayed in windows of 500 (test-then-train). Labels arrive
with a one-window delay, so a retrain triggered after window t trains on the 3,000
most recent labelled rows ending at window t-1 and serves from window t+1.
Every strategy uses the same warm-started AutoML retrain + deployment gate.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_elec2
from src.automl import search, predict_proba, retrain_step
from src.drift import KSDetector, PSIDetector, MultiSignalDetector, ErrorRateDetector

OUT = Path(__file__).resolve().parent.parent / "results"
W, START, TRAIN, COOLDOWN, BUDGET = 500, 4000, 3000, 2, 6
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

X, y = load_elec2()
init = search(X.iloc[:3000], y[:3000], budget=30, seed=SEED)
p_ref0 = predict_proba(init["model"], X.iloc[3000:4000])
acc_ref0 = accuracy_score(y[3000:4000], p_ref0 > .5)
print("initial AutoML:", init["estimator"], f"holdout acc {acc_ref0:.3f}", flush=True)

STRATS = {
    "No retraining": None, "Fixed (every 4 win)": ("fixed", 4), "Fixed (every 10 win)": ("fixed", 10),
    "KS-triggered": ("det", KSDetector), "PSI-triggered": ("det", PSIDetector),
    "Multi-signal (ours)": ("det", MultiSignalDetector),
    "ErrorRate (labels)": ("det", ErrorRateDetector),
}
n_win = (len(X) - START) // W
results, traces = [], {}
for name, spec in STRATS.items():
    t0 = time.time()
    model, warm = init["model"], init["warm"]
    det = spec[1]().fit(X.iloc[3000:4000], p_ref0, acc_ref0) if spec and spec[0] == "det" else None
    accs, retrains, promoted, cool, rt_s, alarms = [], [], 0, 0, 0.0, []
    for w in range(n_win):
        s, e = START + w * W, START + (w + 1) * W
        Xw, yw = X.iloc[s:e], y[s:e]
        p = predict_proba(model, Xw); acc = accuracy_score(yw, p > .5); accs.append(acc)
        if spec is None:
            continue
        if spec[0] == "fixed":
            fire = (w + 1) % spec[1] == 0
        else:
            res = det.check(Xw, p, acc)
            fire = res["alarm"] and cool == 0
            cool = max(0, cool - 1)
        alarms.append(bool(fire))
        if fire and s - TRAIN >= 0:
            lab_end = s  # one-window label delay
            Xt, yt = X.iloc[lab_end - TRAIN:lab_end], y[lab_end - TRAIN:lab_end]
            r = retrain_step(Xt, yt, model, warm=warm, budget=BUDGET)
            rt_s += r["train_s"]; retrains.append(w)
            if r["promoted"]:
                promoted += 1; model, warm = r["model"], r["warm"]
                if det is not None:
                    det.fit(r["ref_X"], r["ref_p"], r["acc_new"])
            cool = COOLDOWN
    results.append({"strategy": name, "mean_acc": float(np.mean(accs)), "min_acc": float(np.min(accs)),
                    "retrains": len(retrains), "promoted": promoted, "retrain_cpu_s": rt_s,
                    "wall_s": time.time() - t0})
    traces[name] = {"acc": accs, "retrain_at": retrains}
    print(results[-1], flush=True)

(OUT / f"exp1_results_s{SEED}.json").write_text(json.dumps({"results": results, "traces": traces,
    "initial": {"estimator": init["estimator"], "holdout_acc": acc_ref0}, "n_windows": n_win}, indent=1))
