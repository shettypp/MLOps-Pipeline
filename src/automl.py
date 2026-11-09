"""AutoML model selection (FLAML, CASH search) with warm-started retraining."""
import time
import warnings
import numpy as np
from flaml import AutoML

warnings.filterwarnings("ignore")
ESTIMATORS = ["lgbm", "rf", "extra_tree", "xgboost", "lrl1"]


def search(X, y, budget=30, seed=0, warm=None):
    """Run a CASH search. `warm` = previous best configs (warm start for retraining)."""
    t0 = time.time()
    a = AutoML()
    a.fit(X, y, task="classification", metric="accuracy", time_budget=budget,
          estimator_list=ESTIMATORS, eval_method="holdout", split_ratio=0.2,
          seed=seed, verbose=0, starting_points=warm or "static", n_jobs=2)
    return {"model": a, "estimator": a.best_estimator, "config": a.best_config,
            "warm": a.best_config_per_estimator, "val_acc": 1 - a.best_loss,
            "train_s": time.time() - t0}


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


def retrain_step(X, y, current_model, warm=None, budget=8, holdout=0.2):
    """Warm-started AutoML on fresh labelled data + deployment gate.
    Search on the older 80%, gate against the current model on the newest 20%
    (time-ordered), and if the candidate is at least as good, refit the winning
    configuration on all fresh data."""
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score
    t0 = time.time()
    cut = int(len(X) * (1 - holdout))
    cand = search(X.iloc[:cut], y[:cut], budget=budget, warm=warm)
    acc_new = accuracy_score(y[cut:], cand["model"].predict(X.iloc[cut:]))
    acc_old = accuracy_score(y[cut:], current_model.predict(X.iloc[cut:]))
    out = {"promoted": bool(acc_new >= acc_old), "acc_new": float(acc_new), "acc_old": float(acc_old),
           "estimator": cand["estimator"], "warm": cand["warm"], "model": current_model,
           # out-of-sample reference for the drift detector (refit model would be overconfident in-sample)
           "ref_X": X.iloc[cut:], "ref_p": cand["model"].predict_proba(X.iloc[cut:])[:, 1]}
    if out["promoted"]:
        out["model"] = clone(cand["model"].model.estimator).fit(X, y)
    out["train_s"] = time.time() - t0
    return out
