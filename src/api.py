"""Prediction service (FastAPI). Runs locally under uvicorn, or on AWS Lambda via Mangum.

POST /predict   {"rows":[{feature: value, ...}, ...]}  -> probabilities, ids, model version
POST /feedback  {"labels":[{"id": .., "y": ..}]}        delayed ground truth
GET  /health    liveness + serving model version
Models are data, not code: the service re-reads the registry pointer at most every
REFRESH_S seconds and hot-swaps to a newer version without a redeploy.
"""
import os
import threading
import time
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.registry import open_store, new_ids

STORE = open_store()
REFRESH_S = float(os.environ.get("REFRESH_S", "0.5"))
app = FastAPI(title="MLOps prediction API")
_state = {"v": 0, "model": None, "features": None, "checked": 0.0}
_lock = threading.Lock()


def _refresh(force=False):
    now = time.time()
    if not force and now - _state["checked"] < REFRESH_S:
        return
    _state["checked"] = now
    v = STORE.current_version()
    if v and v != _state["v"]:
        model, meta = STORE.load(v)
        with _lock:
            _state.update(v=v, model=model, features=meta["features"])


class PredictIn(BaseModel):
    rows: list[dict]
    ids: list[int] | None = None


class FeedbackIn(BaseModel):
    labels: list[dict]


@app.get("/health")
def health():
    _refresh(force=True)
    return {"status": "ok", "model_version": _state["v"]}


@app.post("/predict")
def predict(req: PredictIn):
    _refresh()
    with _lock:
        model, feats, v = _state["model"], _state["features"], _state["v"]
    X = pd.DataFrame(req.rows)[feats]
    proba = model.predict_proba(X)[:, 1].tolist()
    ids = req.ids or new_ids(len(proba))
    STORE.log([{"id": i, "x": r, "p": p, "v": v, "ts": time.time()}
               for i, r, p in zip(ids, req.rows, proba)], part=f"v{v}")
    return {"model_version": v, "proba": proba, "ids": ids}


@app.post("/feedback")
def feedback(req: FeedbackIn):
    STORE.add_labels(req.labels)
    return {"received": len(req.labels)}
