from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from .model_loader import load_production_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Local SageMaker-like Endpoint")

# allow your Streamlit app to call the API from another domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict later to your Streamlit URL
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
INFER = None

class InvokeRequest(BaseModel):
    instances: list[Any]

@app.on_event("startup")
def _load():
    global MODEL, INFER
    MODEL, INFER = load_production_model()

@app.get("/ping")
def ping():
    return {"status": "pong"}

@app.post("/invocations")
def invocations(payload: InvokeRequest):
    try:
        preds = INFER.predict_fn(payload.instances, MODEL)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/invocations_proba")
def invocations_proba(payload: InvokeRequest):
    try:
        probs = INFER.predict_proba_fn(payload.instances, MODEL)
        return {"probabilities": probs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))