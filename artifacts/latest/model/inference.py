import joblib, numpy as np
_model = None
def model_fn(model_dir):
    global _model
    if _model is None:
        _model = joblib.load(f"{model_dir}/model.joblib")
    return _model

def predict_fn(data, model):
    # data: np.ndarray of shape (n,4)
    return model.predict(data).tolist()
