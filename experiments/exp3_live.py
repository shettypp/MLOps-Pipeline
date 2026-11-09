"""Exp 3 - Live service: the full closed loop over HTTP.

Starts the FastAPI service under uvicorn, publishes the initial AutoML model (v1),
then replays Elec2 traffic (rows 4000+) as real HTTP requests in batches of 50.
Ground-truth labels are posted to /feedback one window late. The drift job runs
every 500 requests (as the scheduled Lambda would); on alarm it retrains in-process,
publishes to the registry and the running API hot-swaps without restart.
Measures request latency, alarm->serving time, versions deployed, and live accuracy.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_elec2
from src.drift import MultiSignalDetector
from src.monitor import DriftJob, retrain_from_store
from src.pipeline import train
from src.registry import Store

N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
BATCH, WINDOW, PORT = 50, 500, 8765
store_dir = ROOT / "results" / "live_store"; shutil.rmtree(store_dir, ignore_errors=True)
store = Store(store_dir)
init = train(store, budget=30)
print("v1:", init, flush=True)

env = {**os.environ, "MLOPS_STORE": str(store_dir), "REFRESH_S": "0.5", "PYTHONPATH": str(ROOT)}
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app", "--port", str(PORT), "--log-level", "warning"],
                       cwd=ROOT, env=env)
c = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=30)
for _ in range(50):
    try:
        c.get("/health"); break
    except httpx.TransportError:
        time.sleep(0.2)

X, y = load_elec2()
retrains = []
def on_alarm(upto):
    t_called = time.time()
    e = retrain_from_store(store, budget=6)
    retrains.append({**e, "detect_s": t_called - t_alarm[0], "t_published": time.time()})
    return e


job = DriftJob(store, MultiSignalDetector(), window=WINDOW, on_alarm=on_alarm)
t_alarm = [0.0]
lat, served, pending = [], [], []
t_start = time.time()
for s in range(4000, 4000 + N_ROWS, BATCH):
    rows = X.iloc[s:s + BATCH]
    t0 = time.perf_counter()
    r = c.post("/predict", json={"rows": rows.to_dict("records")}).json()
    lat.append((time.perf_counter() - t0) * 1000)
    served += [(i, p, r["model_version"], int(t)) for i, p, t in zip(r["ids"], r["proba"], y[s:s + BATCH])]
    pending.append([{"id": i, "y": int(t)} for i, t in zip(r["ids"], y[s:s + BATCH])])
    if len(pending) > WINDOW // BATCH:          # labels arrive one window late
        c.post("/feedback", json={"labels": pending.pop(0)})
    if (s - 4000 + BATCH) % WINDOW == 0:        # scheduled drift job
        t_alarm[0] = time.time()
        for e in job.run_once():
            v_new = e["version"]
            while c.get("/health").json()["model_version"] != v_new:
                time.sleep(0.05)
            retrains[-1]["alarm_to_serving_s"] = time.time() - t_alarm[0]
            retrains[-1]["swap_s"] = time.time() - retrains[-1]["t_published"]
            retrains[-1]["at_row"] = s
            print("retrain:", {k: retrains[-1][k] for k in ("version", "promoted", "acc_old", "acc_new", "seconds", "swap_s", "alarm_to_serving_s")}, flush=True)
srv.terminate()

acc = np.array([(p > .5) == t for _, p, _, t in served])
win_acc = [float(acc[i:i + WINDOW].mean()) for i in range(0, len(acc), WINDOW)]
out = {"rows": len(served), "requests": len(lat), "batch": BATCH, "wall_s": time.time() - t_start,
       "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                      "p99": float(np.percentile(lat, 99))},
       "live_acc": float(acc.mean()), "window_acc": win_acc, "versions_served": sorted({v for _, _, v, _ in served}),
       "retrains": retrains, "drift_events": job.events, "initial": init}
(ROOT / "results" / "exp3_live.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k not in ("window_acc", "drift_events", "retrains")}, default=str))
