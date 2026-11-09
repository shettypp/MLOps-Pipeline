import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.automl import search, predict_proba
from src.data import make_base, inject
from src.drift import MultiSignalDetector, KSDetector, psi
from src.registry import Store


def test_psi_zero_for_same_distribution():
    r = np.random.default_rng(0).normal(size=5000)
    assert psi(r[:2500], r[2500:]) < 0.02


@pytest.fixture(scope="module")
def fitted():
    X, y, inf, noise = make_base(1, n=6000)
    m = search(X.iloc[:2500], y[:2500], budget=3)
    p_ref = predict_proba(m["model"], X.iloc[2500:3500])
    return X, y, inf, noise, m, p_ref


def _run(det, X, m, cols, kind, n=4):
    rng = np.random.default_rng(0); out = []
    for w in range(n):
        Xw, _ = inject(X.iloc[3500 + w * 500:4000 + w * 500], np.zeros(500, int), kind, cols, 1.0, rng)
        out.append(det.check(Xw, predict_proba(m["model"], Xw))["alarm"])
    return out


def test_multisignal_ignores_benign_but_ks_does_not(fitted):
    X, y, inf, noise, m, p_ref = fitted
    assert not any(_run(MultiSignalDetector().fit(X.iloc[2500:3500], p_ref), X, m, noise, "benign"))
    assert all(_run(KSDetector().fit(X.iloc[2500:3500], p_ref), X, m, noise, "benign"))


def test_multisignal_catches_harmful(fitted):
    X, y, inf, noise, m, p_ref = fitted
    assert any(_run(MultiSignalDetector().fit(X.iloc[2500:3500], p_ref), X, m, inf[:2], "harmful"))


def test_s3_store_roundtrip():
    boto3 = pytest.importorskip("boto3"); moto = pytest.importorskip("moto")
    from src.registry import S3Store
    with moto.mock_aws():
        c = boto3.client("s3", region_name="us-east-1"); c.create_bucket(Bucket="mlops-test-bucket")
        s = S3Store("mlops-test-bucket", client=c)
        assert s.current_version() == 0
        assert s.publish({"m": 1}, {"features": ["a"]}, ref={"X": {"a": [1]}, "p": [0.5]}) == 1
        s.log([{"id": 1, "x": {"a": 1}, "p": .5}], "v1"); s.add_labels([{"id": 1, "y": 0}])
        assert s.load()[0] == {"m": 1} and s.read_logs()[0]["id"] == 1 and s.read_labels() == {1: 0}


def test_closed_loop_api_drift_retrain_hotswap(tmp_path, monkeypatch, fitted):
    """API serves v1 -> harmful drift in traffic -> drift job alarms -> retrain -> API serves v2."""
    X, y, inf, noise, m, p_ref = fitted
    store = Store(tmp_path)
    store.publish(m["model"].model.estimator, {"features": list(X.columns), "warm": m["warm"], "val_acc": .9},
                  ref={"X": X.iloc[2500:3500].to_dict("list"), "p": p_ref.tolist()})
    import src.api as api
    monkeypatch.setattr(api, "STORE", store); monkeypatch.setattr(api, "REFRESH_S", 0.0)
    api._state.update(v=0, checked=0.0)
    client = TestClient(api.app)
    assert client.get("/health").json()["model_version"] == 1
    from src.monitor import DriftJob, retrain_from_store
    job = DriftJob(store, MultiSignalDetector(), window=500,
                   on_alarm=lambda upto: retrain_from_store(store, budget=2))
    rng = np.random.default_rng(0); fired = []
    for w in range(5):
        Xw, yw = inject(X.iloc[3500 + w * 500:4000 + w * 500], y[3500 + w * 500:4000 + w * 500],
                        "harmful", inf[:2], 1.5, rng)
        r = client.post("/predict", json={"rows": Xw.to_dict("records")}).json()
        client.post("/feedback", json={"labels": [{"id": i, "y": int(t)} for i, t in zip(r["ids"], yw)]})
        fired += job.run_once()
    assert fired and fired[0]["type"] == "retrain"
    assert fired[0]["promoted"] and client.get("/health").json()["model_version"] == 2
