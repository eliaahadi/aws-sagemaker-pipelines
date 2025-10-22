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

 
# --- New: Metrics & threshold tab ---
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
    st.subheader("Metrics & threshold (Telco churn)")
    st.caption("Use local eval cache if present; otherwise upload a labeled CSV with a 'churn' column.")

    import os, numpy as np, pandas as pd
    from sklearn.metrics import (
        confusion_matrix, roc_auc_score, average_precision_score,
        precision_recall_fscore_support
    )

    # Try local eval cache first
    cache_path = "artifacts/latest/eval_cache.npz"
    have_cache = os.path.exists(cache_path)

    label_col = st.text_input("Label column (for uploaded CSV)", value="churn")
    cost_fp = st.number_input("Cost of False Positive (contacting a happy customer)", 0.0, 1000.0, 5.0, 0.5)
    cost_fn = st.number_input("Cost of False Negative (losing a churning customer)", 0.0, 10000.0, 100.0, 1.0)

    def eval_report(y_true, y_proba, thr):
        y_hat = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        roc = roc_auc_score(y_true, y_proba)
        pr  = average_precision_score(y_true, y_proba)
        exp_cost = fp*cost_fp + fn*cost_fn
        return dict(threshold=float(thr), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                    precision=float(prec), recall=float(rec), f1=float(f1),
                    roc_auc=float(roc), pr_auc=float(pr), expected_cost=float(exp_cost))

    if have_cache:
        data = np.load(cache_path)
        y_true = data["y_true"]; y_proba = data["y_proba"]
        st.success(f"Loaded local eval cache with {len(y_true)} rows.")
    else:
        st.warning("No local eval cache found. Upload a CSV with features + label to compute metrics using the live endpoint.")
        file = st.file_uploader("Upload labeled CSV (must include the label column)", type=["csv"])
        if file:
            df = pd.read_csv(file)
            if label_col not in df.columns:
                st.error(f"Column '{label_col}' not found.")
                st.stop()
            y_true = df[label_col].astype(int).values
            X = df.drop(columns=[label_col])
            # Call /invocations_proba with list-of-dicts
            records = X.to_dict(orient="records")

            # Chunk to avoid oversized payloads
            probs = []
            for i in range(0, len(records), 200):
                chunk = records[i:i+200]
                resp = requests.post(os.getenv("ENDPOINT_URL", "http://127.0.0.1:8000/invocations_proba"),
                                     json={"instances": chunk}, timeout=60)
                resp.raise_for_status()
                probs.extend(resp.json()["probabilities"])
            y_proba = np.array(probs)
            st.success(f"Scored {len(y_proba)} rows via endpoint.")
        else:
            st.stop()

    thr = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
    rep = eval_report(y_true, y_proba, thr)
    st.write(f"**ROC AUC:** {rep['roc_auc']:.3f} • **PR AUC:** {rep['pr_auc']:.3f}")
    st.write(f"**F1@{thr:.2f}:** {rep['f1']:.3f} • **Precision:** {rep['precision']:.3f} • **Recall:** {rep['recall']:.3f}")
    st.write(f"TP={rep['tp']} | FP={rep['fp']} | TN={rep['tn']} | FN={rep['fn']}")
    st.write(f"**Expected cost:** {rep['expected_cost']:.2f} (FP×{cost_fp} + FN×{cost_fn})")

    if st.button("Find cost-optimal threshold on this data"):
        cand = np.linspace(0.01, 0.99, 99)
        reports = [eval_report(y_true, y_proba, t) for t in cand]
        best = min(reports, key=lambda r: r["expected_cost"])
        st.success(f"Best threshold ≈ **{best['threshold']:.2f}** with expected cost **{best['expected_cost']:.2f}**")