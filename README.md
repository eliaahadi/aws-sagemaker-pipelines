# Local SageMaker-Style MLOps: train → register → approve → deploy

This repo simulates an AWS SageMaker Pipelines flow 100% locally and free:
- **Train** a model and produce a `model.tar.gz` artifact (SageMaker-style).
- **Register** the trained model into a local Model Registry (versions + metadata).
- **Approve/Promote** a version to `Production`.
- **Deploy** a real-time endpoint (FastAPI) that mimics SageMaker's `/invocations`.

## Stack
- Python 3.11+
- scikit-learn, pandas, joblib
- FastAPI + Uvicorn
- No AWS account required. No Docker required (optional Dockerfile provided for the server).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Train (writes artifacts to ./artifacts/latest/)
make train

# 2) Register trained model as a new version in local registry
make register

# 3) Approve/promote a version to Production (default = latest)
make approve

# 4) Deploy real-time server (local SageMaker-like endpoint)
make deploy

# 5) Invoke endpoint with sample payload
make invoke