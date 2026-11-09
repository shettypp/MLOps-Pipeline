"""AWS Lambda entry points (same container image, two handlers).

api_handler   : API Gateway HTTP API -> FastAPI app (via Mangum)
drift_handler : EventBridge schedule -> DriftJob.run_once(); publishes CloudWatch
                metrics and, on alarm, retrains:
                  GITHUB_REPO empty (default) -> inside this Lambda (warm AutoML + gate -> S3)
                  GITHUB_REPO set             -> GitHub repository_dispatch -> Actions retrain
"""
import json
import os
import urllib.request

import boto3
from mangum import Mangum

from src.api import app
from src.drift import MultiSignalDetector
from src.monitor import DriftJob, retrain_from_store
from src.registry import open_store

api_handler = Mangum(app, lifespan="off")
cw = boto3.client("cloudwatch")


def _dispatch_retrain(upto_id):
    token = boto3.client("ssm").get_parameter(Name=os.environ["GITHUB_TOKEN_PARAM"],
                                              WithDecryption=True)["Parameter"]["Value"]
    req = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPO']}/dispatches",
        data=json.dumps({"event_type": "drift-retrain", "client_payload": {"upto_id": upto_id}}).encode(),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, method="POST")
    urllib.request.urlopen(req, timeout=10)
    return {"dispatched": True, "upto_id": upto_id}


def drift_handler(event, context):
    """Scheduled run: event = {} (drift check). Console test event {"action": "train"} seeds model v1."""
    store = open_store()
    if (event or {}).get("action") == "train":
        from src.pipeline import train
        return train(store, budget=int(event.get("budget", 30)))
    inline = not os.environ.get("GITHUB_REPO")
    job = DriftJob(store, MultiSignalDetector(), window=int(os.environ.get("WINDOW", 500)),
                   on_alarm=(lambda upto: retrain_from_store(store, budget=int(os.environ.get("RETRAIN_BUDGET", 8))))
                   if inline else _dispatch_retrain)
    fired = job.run_once()
    data = []
    for e in job.events:
        data += [{"MetricName": "DriftAlarm", "Value": float(e["alarm"])},
                 {"MetricName": "DriftVotes", "Value": e.get("votes", 0.0)},
                 {"MetricName": "PredictionPSI", "Value": e.get("pred_psi", 0.0)}]
    if data:
        cw.put_metric_data(Namespace="MLOpsPipeline", MetricData=data[:1000])
    return json.loads(json.dumps({"windows": len(job.events), "alarms": fired}, default=str))
