import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
ENDPOINT_URL = os.getenv("ENDPOINT_URL", "http://127.0.0.1:8000/invocations")
PING_URL = os.getenv("PING_URL", "http://127.0.0.1:8000/ping")

st.set_page_config(page_title="Local SageMaker-style Demo", layout="centered")

st.title("🔬 Local SageMaker-style MLOps Demo")
st.caption("Train → Register → Approve → Deploy (FastAPI) • Streamlit client UI")

# Health check
try:
    r = requests.get(PING_URL, timeout=3)
    ok = r.ok
except Exception:
    ok = False

st.info(f"Endpoint: {ENDPOINT_URL}")
if ok:
    st.success("Endpoint is healthy ✅")
else:
    st.warning("Endpoint is not reachable ⚠️. Start it with `make deploy`.")

tabs = st.tabs(["Single prediction", "Batch (CSV)", "About"])

with tabs[0]:
    st.subheader("Single prediction (Iris features)")
    c1, c2 = st.columns(2)
    with c1:
        sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.1, 0.1)
        sepal_width  = st.number_input("Sepal width (cm)",  0.0, 10.0, 3.5, 0.1)
    with c2:
        petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, 1.4, 0.1)
        petal_width  = st.number_input("Petal width (cm)",  0.0, 10.0, 0.2, 0.1)

    if st.button("Predict"):
        payload = {"instances": [[sepal_length, sepal_width, petal_length, petal_width]]}
        try:
            resp = requests.post(ENDPOINT_URL, json=payload, timeout=5)
            resp.raise_for_status()
            preds = resp.json().get("predictions", [])
            mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
            label = mapping.get(int(preds[0]), str(preds[0]))
            st.success(f"Prediction: **{label}**  (raw={preds[0]})")
            with st.expander("Raw response"):
                st.code(json.dumps(resp.json(), indent=2))
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[1]:
    st.subheader("Batch predictions (CSV upload)")
    st.caption("Upload CSV with columns: sepal_length,sepal_width,petal_length,petal_width")
    file = st.file_uploader("Choose CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Preview", df.head(10))
        if st.button("Run batch predict"):
            try:
                arr = df.values.tolist()
                payload = {"instances": arr}
                resp = requests.post(ENDPOINT_URL, json=payload, timeout=30)
                resp.raise_for_status()
                preds = resp.json().get("predictions", [])
                out = df.copy()
                out["prediction"] = preds
                st.success("Done")
                st.dataframe(out.head(20), use_container_width=True)
                st.download_button("Download results CSV", out.to_csv(index=False), "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")

with tabs[2]:
    st.markdown("""
**How it works**

- This app calls the FastAPI endpoint’s `/invocations` to perform predictions.
- The FastAPI service always loads the **Production** model from the local registry (`model_registry/production`).
- Aligns with SageMaker concepts: Pipelines → Model Registry → Approved model → Real-time Endpoint.

**Run locally**
```bash
# Terminal 1: start the endpoint
make deploy

# Terminal 2: run the UI (defaults to localhost endpoints)
streamlit run streamlit_app.py""")