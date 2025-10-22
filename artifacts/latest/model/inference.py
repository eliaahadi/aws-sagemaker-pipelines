# inference.py
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
