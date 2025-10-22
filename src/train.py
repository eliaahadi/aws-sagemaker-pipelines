# src/train.py
from __future__ import annotations
import argparse, json, tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score
)

from common import ARTIFACTS_DIR, now_iso

def load_iris():
    from sklearn import datasets
    iris = datasets.load_iris(as_frame=True)
    X, y = iris.data, iris.target
    return X, y, {"task":"multiclass", "label_name":"target"}

def ensure_telco_csv():
    path = Path("data/telco_churn.csv")
    if not path.exists():
        # generate on the fly
        try:
            # normal case: project root is on sys.path
            from data.generate_telco import generate_telco
        except ModuleNotFoundError:
            # when running `python src/train.py`, Python adds 'src' to sys.path
            # which prevents the top-level `data` package from being found. Add
            # the repository root to sys.path and retry the import.
            import sys
            repo_root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(repo_root))
            from data.generate_telco import generate_telco

        path.parent.mkdir(parents=True, exist_ok=True)
        generate_telco().to_csv(path, index=False)
    return path

def load_telco():
    path = ensure_telco_csv()
    df = pd.read_csv(path)
    y = df["churn"].astype(int)
    X = df.drop(columns=["churn"])
    return X, y, {"task":"binary", "label_name":"churn"}

def build_telco_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = ["tenure_months", "monthly_charges", "total_charges"]
    cat_cols = [
        "contract","internet_service","online_security","tech_support",
        "paperless_billing","payment_method"
    ]
    bool_cols = ["is_senior","has_partner","has_dependents"]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("bool", "passthrough", bool_cols),
    ])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def write_inference_py(model_dir: Path):
    (model_dir / "inference.py").write_text(
        """# inference.py
import joblib, numpy as np, pandas as pd
_model = None
def model_fn(model_dir):
    global _model
    if _model is None:
        _model = joblib.load(f"{model_dir}/model.joblib")
    return _model

def _to_X(data):
    # Accept list-of-lists or list-of-dicts
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return pd.DataFrame(data)
    return np.array(data, dtype=float)

def predict_fn(data, model):
    X = _to_X(data)
    return model.predict(X).tolist()

def predict_proba_fn(data, model):
    X = _to_X(data)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # If binary, return positive-class probability as 1D list
        if proba.shape[1] == 2:
            return proba[:, 1].tolist()
        return proba.tolist()
    raise RuntimeError("Model has no predict_proba()")
"""
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["telco","iris"], default="telco")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = ARTIFACTS_DIR / "model"; model_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "telco":
        X, y, meta = load_telco()
        pipe = build_telco_pipeline(X)
    else:
        X, y, meta = load_iris()
        # simple numeric-only scaling for iris
        pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if meta["task"]=="binary" else None
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    if meta["task"] == "binary":
        y_proba = pipe.predict_proba(X_test)[:, 1]
        f1 = float(f1_score(y_test, (y_proba >= 0.5).astype(int)))
        roc = float(roc_auc_score(y_test, y_proba))
        pr  = float(average_precision_score(y_test, y_proba))
    else:
        # basic proxies for multiclass
        y_proba = None
        f1 = float(f1_score(y_test, y_pred, average="macro"))
        roc = None; pr = None

    metrics = {
        "dataset": args.dataset,
        "task": meta["task"],
        "trained_at": now_iso(),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "accuracy": acc,
        "f1@0.5": f1,
        "roc_auc": roc, "pr_auc": pr,
    }

    # Save model
    joblib.dump(pipe, model_dir / "model.joblib")
    write_inference_py(model_dir)

    # Package to model.tar.gz (SageMaker style)
    tar_path = ARTIFACTS_DIR / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_dir, arcname=".")

    # Save card + eval cache
    (ARTIFACTS_DIR / "model_card.json").write_text(json.dumps({
        "model_name": f"{args.dataset}-logreg",
        "framework": "scikit-learn",
        "task": meta["task"],
        "metrics": metrics,
        "features": list(X.columns) if isinstance(X, pd.DataFrame) else None
    }, indent=2))

    if meta["task"] == "binary":
        # store evaluation cache for dashboard: y_true + proba
        np.savez(ARTIFACTS_DIR / "eval_cache.npz",
                 y_true=np.asarray(y_test), y_proba=np.asarray(y_proba))

    print("Metrics:", metrics)
    print(f"Wrote {tar_path}")

if __name__ == "__main__":
    main()