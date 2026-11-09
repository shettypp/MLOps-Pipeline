"""CLI used by CI/CD and locally.

  python -m src.pipeline train   --budget 30      initial AutoML on Elec2 rows 0-3999, publish v1
  python -m src.pipeline retrain --budget 8       drift-triggered retrain from logged traffic
  python -m src.pipeline monitor                  run one drift-job pass
Store: S3 if MLOPS_BUCKET is set, else ./store (or MLOPS_STORE).
"""
import argparse
import json

from src.automl import search, predict_proba
from src.data import load_elec2, FEATURES
from src.drift import MultiSignalDetector
from src.monitor import DriftJob, retrain_from_store
from src.registry import open_store


def train(store, budget):
    X, y = load_elec2()
    m = search(X.iloc[:3000], y[:3000], budget=budget)
    ref_X = X.iloc[3000:4000]
    v = store.publish(m["model"],  # FLAML object keeps its preprocessing (drops constant cols)
                      {"features": FEATURES, "estimator": m["estimator"], "val_acc": m["val_acc"], "warm": m["warm"]},
                      ref={"X": ref_X.to_dict("list"), "p": predict_proba(m["model"], ref_X).tolist()})
    return {"version": v, "estimator": m["estimator"], "val_acc": m["val_acc"], "seconds": m["train_s"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "retrain", "monitor"])
    ap.add_argument("--budget", type=int, default=None)
    a = ap.parse_args()
    s = open_store()
    if a.cmd == "train":
        out = train(s, a.budget or 30)
    elif a.cmd == "retrain":
        out = retrain_from_store(s, budget=a.budget or 8)
    else:
        out = DriftJob(s, MultiSignalDetector()).run_once()
    print(json.dumps(out, default=str))
