"""Drift job + closed-loop retraining.

DriftJob.run_once() keeps no memory between calls (its checkpoint lives in the store),
so the same code runs as a local loop, a cron job, or the scheduled AWS Lambda:
  read new inference logs -> score each full window with the detector ->
  on alarm call `on_alarm(upto_id)`:
    local : retrain_from_store() in-process
    cloud : GitHub repository_dispatch -> Actions workflow runs retrain_from_store()
"""
import time
import numpy as np
import pandas as pd

from src.automl import retrain_step


def retrain_from_store(store, train_size=3000, budget=8):
    """Warm-started AutoML on the freshest labelled traffic, gated, published to the registry."""
    t0 = time.time()
    model, meta = store.load()
    f, labels = meta["features"], store.read_labels()
    recs = [r for r in store.read_logs() if r["id"] in labels][-train_size:]
    X = pd.DataFrame([r["x"] for r in recs])[f]
    y = np.array([labels[r["id"]] for r in recs])
    r = retrain_step(X, y, model, warm=meta.get("warm"), budget=budget)
    ev = {"type": "retrain", "n_train": len(y), "acc_new": r["acc_new"], "acc_old": r["acc_old"],
          "estimator": r["estimator"], "promoted": r["promoted"], "version": meta["version"]}
    if r["promoted"]:
        ev["version"] = store.publish(
            r["model"], {"features": f, "estimator": r["estimator"], "val_acc": r["acc_new"], "warm": r["warm"]},
            ref={"X": r["ref_X"].to_dict("list"), "p": r["ref_p"].tolist()})
    ev["seconds"] = time.time() - t0
    return ev


class DriftJob:
    def __init__(self, store, detector, window=500, cooldown=2, on_alarm=None):
        self.s, self.det, self.window, self.cooldown = store, detector, window, cooldown
        self.on_alarm = on_alarm or (lambda upto: retrain_from_store(self.s))
        self.events = []

    def run_once(self):
        st = self.s.get_state("monitor") or {"last_id": -1, "cool": 0, "streak": 0, "ref_version": 0}
        v = self.s.current_version()
        _, meta = self.s.load(v); ref = self.s.load_ref(v); f = meta["features"]
        self.det.fit(pd.DataFrame(ref["X"])[f], np.array(ref["p"]), meta.get("val_acc"))
        if st["ref_version"] != v:          # new model -> new reference, fresh persistence count
            st.update(ref_version=v, streak=0)
        self.det._streak = st["streak"]
        new = self.s.read_logs(st["last_id"])
        fired = []
        for i in range(len(new) // self.window):   # leftover partial window is re-read next run
            win = new[i * self.window:(i + 1) * self.window]
            res = self.det.check(pd.DataFrame([r["x"] for r in win])[f], np.array([r["p"] for r in win]))
            alarm = bool(res["alarm"]) and st["cool"] == 0
            st["cool"] = max(0, st["cool"] - 1)
            st["last_id"], st["streak"] = win[-1]["id"], self.det._streak
            self.events.append({"t": time.time(), "last_id": st["last_id"], "alarm": alarm,
                                **{k: float(x) for k, x in res.items() if k != "alarm"}})
            if alarm:
                st.update(cool=self.cooldown, streak=0)
                self.s.put_state("monitor", st)
                fired.append(self.on_alarm(st["last_id"]))
                return fired                  # reference may have changed; next run re-syncs
        self.s.put_state("monitor", st)
        return fired
