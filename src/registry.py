"""Versioned model registry, inference-log store, label store and job state.

Two interchangeable back ends with one interface:
  Store(root)      local directory (development, tests, experiments)
  S3Store(bucket)  Amazon S3 (cloud deployment; Lambda + GitHub Actions)
Layout (same in both):
  models/CURRENT                 -> "7"   (atomic pointer to the serving version)
  models/v7/model.pkl, meta.json, ref.json
  logs/<partition>/<object>.jsonl  inference records {id, x, p, v, ts}
  labels/<object>.jsonl            delayed ground truth {id, y}
  state/monitor.json               drift-job checkpoint
"""
import io
import json
import os
import pickle
import time
import uuid
from pathlib import Path


def new_ids(n):
    """Globally ordered ids that stay unique across concurrent Lambda containers."""
    base = time.time_ns() // 1000 * 1000
    return [base + i for i in range(n)]


class _Common:
    def publish(self, model, meta, ref=None):
        v = self.current_version() + 1
        p = f"models/v{v}/"
        self._put(p + "model.pkl", pickle.dumps(model))
        self._put(p + "meta.json", json.dumps({**meta, "version": v, "ts": time.time()}, default=str).encode())
        if ref is not None:
            self._put(p + "ref.json", json.dumps(ref).encode())
        self._put("models/CURRENT", str(v).encode())  # pointer written last => readers never see half a version
        return v

    def current_version(self):
        b = self._get("models/CURRENT")
        return int(b) if b else 0

    def load(self, v=None):
        v = v or self.current_version()
        return (pickle.loads(self._get(f"models/v{v}/model.pkl")),
                json.loads(self._get(f"models/v{v}/meta.json")))

    def load_ref(self, v=None):
        b = self._get(f"models/v{v or self.current_version()}/ref.json")
        return json.loads(b) if b else None

    def log(self, records, part):
        self._put(f"logs/{part}/{uuid.uuid4().hex}.jsonl",
                  "".join(json.dumps(r) + "\n" for r in records).encode())

    def read_logs(self, since_id=-1):
        out = [r for k in self._list("logs/") for r in self._jsonl(k) if r["id"] > since_id]
        return sorted(out, key=lambda r: r["id"])

    def add_labels(self, labels):
        self._put(f"labels/{uuid.uuid4().hex}.jsonl", "".join(json.dumps(r) + "\n" for r in labels).encode())

    def read_labels(self):
        return {r["id"]: r["y"] for k in self._list("labels/") for r in self._jsonl(k)}

    def get_state(self, name):
        b = self._get(f"state/{name}.json")
        return json.loads(b) if b else {}

    def put_state(self, name, d):
        self._put(f"state/{name}.json", json.dumps(d).encode())

    def _jsonl(self, key):
        return [json.loads(l) for l in self._get(key).decode().splitlines() if l]


class Store(_Common):
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def _put(self, key, data):
        f = self.root / key; f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f.suffix + ".tmp"); tmp.write_bytes(data); tmp.replace(f)

    def _get(self, key):
        f = self.root / key
        return f.read_bytes() if f.exists() else None

    def _list(self, prefix):
        d = self.root / prefix
        return sorted(str(p.relative_to(self.root)) for p in d.rglob("*.jsonl")) if d.exists() else []


class S3Store(_Common):
    def __init__(self, bucket, client=None):
        import boto3
        self.bucket, self.s3 = bucket, client or boto3.client("s3")

    def _put(self, key, data):
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)

    def _get(self, key):
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def _list(self, prefix):
        keys, tok = [], {}
        while True:
            r = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, **tok)
            keys += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return keys
            tok = {"ContinuationToken": r["NextContinuationToken"]}


def open_store():
    """MLOPS_BUCKET set -> S3 (cloud); otherwise local directory MLOPS_STORE (default ./store)."""
    b = os.environ.get("MLOPS_BUCKET")
    return S3Store(b) if b else Store(os.environ.get("MLOPS_STORE", "store"))
