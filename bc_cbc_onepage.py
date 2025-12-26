# bc_cbc_onepage.py
# BC CBC Streamlit app (CSV upload → prediction)

import os
import pandas as pd
import numpy as np
import streamlit as st

try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="BC CBC", layout="wide")
st.title("BC CBC — CBC-based Prediction")
st.caption("Upload a CBC CSV file and generate predictions.")

# ------------------ Model config ------------------
DEFAULT_MODEL = os.path.join("models", "bc_cbc_model.pkl")
CBC_MODEL_PATH = os.getenv("CBC_MODEL_PATH", DEFAULT_MODEL)

with st.expander("Model configuration", expanded=False):
    st.write("Model path:", CBC_MODEL_PATH)
    st.write("Model exists:", os.path.exists(CBC_MODEL_PATH))


@st.cache_resource(show_spinner=False)
def load_cbc_model(path: str):
    if joblib is None:
        raise RuntimeError("joblib is not installed. Add it to requirements.txt.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CBC model not found: {path}")
    return joblib.load(path)


uploaded = st.file_uploader("Upload CBC CSV", type=["csv", "tsv"])

if uploaded:
    try:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, sep="\t")
    except Exception as e:
        st.error("Could not read file as CSV/TSV.")
        st.exception(e)
        st.stop()

    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    X = df.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    st.write(f"Numeric feature columns detected: {X.shape[1]}")

    run_btn = st.button("🔎 Run CBC Prediction", type="primary")

    if run_btn:
        if os.path.exists(CBC_MODEL_PATH):
            try:
                model = load_cbc_model(CBC_MODEL_PATH)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    pred_idx = proba.argmax(axis=1)
                    classes = getattr(model, "classes_", None)
                    preds = [classes[i] if classes is not None else i for i in pred_idx]

                    out = df.copy()
                    out["CBC_Predicted_Label"] = preds

                    if classes is not None:
                        for j, cls in enumerate(classes):
                            out[f"P_{cls}"] = proba[:, j]
                else:
                    preds = model.predict(X)
                    out = df.copy()
                    out["CBC_Predicted_Label"] = preds

            except Exception as e:
                st.error("CBC model prediction failed.")
                st.exception(e)
                st.stop()
        else:
            st.warning("CBC model not found — running demo logic.")
            score = X.sum(axis=1) if X.shape[1] else np.zeros(len(df))
            threshold = float(np.median(score)) if len(score) else 0.0
            preds = np.where(score >= threshold, "HighRisk", "LowRisk")

            out = df.copy()
            out["CBC_Predicted_Label"] = preds

        st.subheader("Results")
        st.dataframe(out.head(200), use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CBC Predictions CSV",
            data=csv_bytes,
            file_name="bc_cbc_predictions.csv",
            mime="text/csv",
        )
