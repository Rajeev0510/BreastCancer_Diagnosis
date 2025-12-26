# bc_cbc_onepage.py
# BC CBC Streamlit app — form inputs (no file upload)
# Fix: supports joblib bundles that load as dict (model + metadata)

import os
import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="BC CBC", layout="wide")
st.title("BC CBC — CBC-based Prediction")
st.caption("Enter CBC markers manually and click Predict.")

# ------------------ Model config ------------------
DEFAULT_MODEL = os.path.join("models", "bc_cbc_model.pkl")
CBC_MODEL_PATH = os.getenv("CBC_MODEL_PATH", DEFAULT_MODEL)

with st.expander("Model configuration", expanded=False):
    st.write("Model path:", CBC_MODEL_PATH)
    st.write("Model exists:", os.path.exists(CBC_MODEL_PATH))


@st.cache_resource(show_spinner=False)
def load_cbc_bundle(path: str):
    """
    Returns (model, feature_names_or_None)

    - If joblib contains sklearn model/pipeline directly: returns it.
    - If joblib contains dict bundle: extracts model from common keys and optionally feature order.
    """
    if joblib is None:
        raise RuntimeError("joblib is not installed. Add `joblib` to requirements.txt.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CBC model not found: {path}")

    obj = joblib.load(path)

    # If a bundle dict was saved: {"model": ..., "feature_names": [...], ...}
    if isinstance(obj, dict):
        model = None
        for k in ["pipeline", "model", "estimator", "clf", "classifier"]:
            if k in obj:
                model = obj[k]
                break
        if model is None:
            raise ValueError(
                f"Loaded object is a dict but no model key found. Keys: {list(obj.keys())}"
            )

        feature_names = obj.get("feature_names") or obj.get("features") or obj.get("columns")
        return model, feature_names

    # Otherwise it's likely a sklearn estimator/pipeline
    return obj, None


# ------------------ Input schema ------------------
# IMPORTANT: feature keys MUST match your model training feature names if you use feature order.
FIELDS = [
    ("WBC (10^3/uL)", "WBC", 7.0, 0.0, 200.0, 0.1),
    ("RBC (10^6/uL)", "RBC", 4.6, 0.0, 20.0, 0.1),
    ("HGB (g/dL)", "HGB", 13.2, 0.0, 30.0, 0.1),
    ("HCT (%)", "HCT", 40.0, 0.0, 80.0, 0.1),
    ("MCV (fL)", "MCV", 88.0, 40.0, 140.0, 0.1),
    ("MCH (pg)", "MCH", 29.0, 10.0, 50.0, 0.1),
    ("MCHC (g/dL)", "MCHC", 33.0, 10.0, 50.0, 0.1),

    ("Platelets (10^3/uL)", "PLT", 250.0, 0.0, 2000.0, 1.0),
    ("Neutrophils (%)", "Neutrophils", 58.0, 0.0, 100.0, 0.1),
    ("Lymphocytes (%)", "Lymphocytes", 30.0, 0.0, 100.0, 0.1),
    ("Monocytes (%)", "Monocytes", 7.0, 0.0, 100.0, 0.1),
    ("Eosinophils (%)", "Eosinophils", 2.5, 0.0, 100.0, 0.1),
    ("Basophils (%)", "Basophils", 0.5, 0.0, 100.0, 0.1),

    ("CRP (mg/L)", "CRP", 2.0, 0.0, 500.0, 0.1),
]

half = int(np.ceil(len(FIELDS) / 2))
left_fields = FIELDS[:half]
right_fields = FIELDS[half:]


# ------------------ UI: Inputs ------------------
with st.expander("Input CBC markers", expanded=True):
    colL, colR = st.columns(2, gap="large")
    values = {}

    with colL:
        for label, key, default, vmin, vmax, step in left_fields:
            values[key] = st.number_input(
                label,
                value=float(default),
                min_value=float(vmin),
                max_value=float(vmax),
                step=float(step),
                format="%.3f",
                key=f"cbc_{key}",
            )

    with colR:
        for label, key, default, vmin, vmax, step in right_fields:
            values[key] = st.number_input(
                label,
                value=float(default),
                min_value=float(vmin),
                max_value=float(vmax),
                step=float(step),
                format="%.3f",
                key=f"cbc_{key}",
            )

X = pd.DataFrame([values])

with st.expander("Show model input row", expanded=False):
    st.dataframe(X, use_container_width=True)

predict_btn = st.button("Predict", type="primary")


# ------------------ Prediction ------------------
if predict_btn:
    if not os.path.exists(CBC_MODEL_PATH):
        st.error(
            "CBC model not found. Add it to the repo at "
            "`models/bc_cbc_model.pkl` (or set CBC_MODEL_PATH env var)."
        )
        st.stop()

    try:
        model, feature_names = load_cbc_bundle(CBC_MODEL_PATH)

        # If the bundle includes a preferred feature order, align to it
        if feature_names is not None:
            cols = [str(c) for c in list(feature_names)]
            missing = [c for c in cols if c not in X.columns]
            if missing:
                st.warning(
                    "Your model expects some features that are not present in the form. "
                    f"Missing: {missing}. They will be filled with 0.0."
                )
            X_in = X.reindex(columns=cols, fill_value=0.0)
        else:
            X_in = X

        # Also allow overriding order via env var (optional)
        env_order = os.getenv("CBC_FEATURE_ORDER", "").strip()
        if env_order:
            cols = [c.strip() for c in env_order.split(",") if c.strip()]
            X_in = X_in.reindex(columns=cols, fill_value=0.0)

        # Predict + show probs
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_in)[0]
            classes = getattr(model, "classes_", None)
            if classes is None:
                classes = np.array([f"class_{i}" for i in range(len(proba))])

            pred_idx = int(np.argmax(proba))
            pred_label = str(classes[pred_idx])

            st.header("Prediction")
            st.subheader(pred_label)

            st.header("Probabilities")
            prob_df = pd.DataFrame({"Class": classes, "Probability": proba})
            st.table(prob_df)

        else:
            pred = model.predict(X_in)[0]
            st.header("Prediction")
            st.subheader(str(pred))
            st.info("This model does not provide predict_proba(), so probabilities are not available.")

    except Exception as e:
        st.error("CBC prediction failed.")
        st.exception(e)
