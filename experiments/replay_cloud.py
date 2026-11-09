"""Send Elec2 traffic to the deployed AWS API (demo for the cloud deployment).

  python experiments/replay_cloud.py <ApiUrl> [rows] [start_row]

Posts rows in batches of 50 to /predict, then posts the true labels to /feedback,
so the drift Lambda has inference logs and labels to work with. Afterwards, trigger
the drift job (or wait for the hourly schedule) and check /health for a new version.
"""
import sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_elec2

url = sys.argv[1].rstrip("/")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
start = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
X, y = load_elec2()
c = httpx.Client(base_url=url, timeout=60)
print("health:", c.get("/health").json())
versions = set()
for s in range(start, start + n, 50):
    r = c.post("/predict", json={"rows": X.iloc[s:s + 50].to_dict("records")}).json()
    versions.add(r["model_version"])
    c.post("/feedback", json={"labels": [{"id": i, "y": int(t)} for i, t in zip(r["ids"], y[s:s + 50])]})
    if (s - start) % 500 == 0:
        print(f"sent rows {start}..{s + 50}", flush=True)
print(f"done: {n} rows, model versions seen {sorted(versions)}")
