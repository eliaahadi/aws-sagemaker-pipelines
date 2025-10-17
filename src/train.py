import tarfile
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

from common import ARTIFACTS_DIR, now_iso

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load sample data (Iris) locally
    iris = datasets.load_iris(as_frame=True)
    X = iris.data.values  # (n, 4)
    y = iris.target.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2) Define model pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    # 3) Train
    pipe.fit(X_train, y_train)

    # 4) Evaluate
    preds = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    metrics = {
        "accuracy": acc,
        "dataset": "Iris",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "trained_at": now_iso(),
    }

    # 5) Save model and an inference entrypoint like SageMaker expects
    model_dir = ARTIFACTS_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, model_dir / "model.joblib")

    # In SageMaker containers, an `inference.py` typically defines input→predict→output.
    (model_dir / "inference.py").write_text(
        """import joblib, numpy as np
_model = None
def model_fn(model_dir):
    global _model
    if _model is None:
        _model = joblib.load(f"{model_dir}/model.joblib")
    return _model

def predict_fn(data, model):
    # data: np.ndarray of shape (n,4)
    return model.predict(data).tolist()
"""
    )

    # Package as model.tar.gz to mimic SageMaker artifact layout
    tar_path = ARTIFACTS_DIR / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_dir, arcname=".")

    # 6) Save a simple model card (metadata)
    (ARTIFACTS_DIR / "model_card.json").write_text(json.dumps({
        "model_name": "iris-logreg",
        "framework": "scikit-learn",
        "task": "multiclass_classification",
        "metrics": metrics,
        "artifacts": {
            "model_tar": "model.tar.gz"
        }
    }, indent=2))

    print(f"Saved: {tar_path}")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()