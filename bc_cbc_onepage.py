# bc_cbc_onepage.py
# BC CBC Streamlit app — form inputs (no file upload)
# Fixes:
#  1) StreamlitDuplicateElementKey (namespaced keys)
#  2) joblib bundle saved as dict -> unwrap actual model/pipeline
#  3) aligns to model's feature_names if present

import os
import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except Exception:
    joblib = None

APP_NS = "bc_cbc"  # namespace for Streamlit keys

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

    Supports:
    - sklearn estimator/pipeline saved directly
    - dict bundles: {"model"/"pipeline"/...: estimator, "feature_names": [...]}
    """
    if joblib is None:
        raise RuntimeError("joblib is not installed. Add `joblib` to requirements.txt.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CBC model not found: {path}")

    obj = joblib.load(path)

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

    return obj, None


# ------------------ Input schema ------------------
# Keep these as-is for now; if your model expects different names, we'll align via feature_names/order.
FIELDS = [
    ("WBC (10^3/uL)", "WBC", 7.0, 0.0, 200.0, 0.1),
    ("RBC (10^6/uL)", "RBC", 4.6, 0.0, 20.0, 0.1),
    ("HGB (g/dL)", "HGB", 13.2, 0.0, 30.0, 0.1),
    ("HCT (%)", "HCT", 40.0, 0.0, 80.0, 0.1),
    ("MCV (fL)", "MCV", 88.0, 40.0, 140.0, 0.1),
    ("MCH (pg)", "MCH", 29.0, 10.0, 50.0, 0.1),
    ("MCHC (g/dL)", "MCHC", 33.0, 10.0, 50.0, 0.1),
    ("RDW (%)", "RDW", 13.0, 0.0, 40.0, 0.1),

    ("Platelets (10^3/uL)", "PLT", 250.0, 0.0, 2000.0, 1.0),

    ("Neutrophils (%)", "NEUT_PCT", 58.0, 0.0, 100.0, 0.1),
    ("Lymphocytes (%)", "LYMPH_PCT", 30.0, 0.0, 100.0, 0.1),
    ("Monocytes (%)", "MONO_PCT", 7.0, 0.0, 100.0, 0.1),
    ("Eosinophils (%)", "EOS_PCT", 2.5, 0.0, 100.0, 0.1),
    ("Basophils (%)", "BASO_PCT", 0.5, 0.0, 100.0, 0.1),

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
        for label, feat, default, vmin, vmax, step in left_fields:
            values[feat] = st.number_input(
                label,
                value=float(default),
                min_value=float(vmin),
                max_value=float(vmax),
                step=float(step),
                format="%.3f",
                key=f"{APP_NS}_{feat}",  # ✅ unique key
            )

    with colR:
        for label, feat, default, vmin, vmax, step in right_fields:
            values[feat] = st.number_input(
                label,
                value=float(default),
                min_value=float(vmin),
                max_value=float(vmax),
                step=float(step),
                format="%.3f",
                key=f"{APP_NS}_{feat}",  # ✅ unique key
            )

# Build model input row
X = pd.DataFrame([values])

# Derived features (expected by your model earlier)
eps = 1e-9
if "NEUT_PCT" in X.columns and "LYMPH_PCT" in X.columns and "PLT" in X.columns:
    neut = X["NEUT_PCT"].astype(float)
    lymph = X["LYMPH_PCT"].astype(float)
    plt = X["PLT"].astype(float)
    X["NLR"] = neut / (lymph + eps)
    X["PLR"] = plt / (lymph + eps)
    X["SII"] = (plt * neut) / (lymph + eps)

with st.expander("Show model input row", expanded=False):
    st.dataframe(X, use_container_width=True)

predict = st.button("Predict", type="primary", key=f"{APP_NS}_predict")


# ------------------ Prediction ------------------
if predict:
    if not os.path.exists(CBC_MODEL_PATH):
        st.error("CBC model not found. Add `models/bc_cbc_model.pkl` (or set CBC_MODEL_PATH).")
        st.stop()

    try:
        model, feature_names = load_cbc_bundle(CBC_MODEL_PATH)

        # Align columns to what the model expects
        X_in = X.copy()

        # 1) bundle-provided order (best)
        if feature_names is not None:
            cols = [str(c) for c in list(feature_names)]
            missing = [c for c in cols if c not in X_in.columns]
            if missing:
                st.warning(f"Missing features filled with 0.0: {missing}")
            X_in = X_in.reindex(columns=cols, fill_value=0.0)

        # 2) env override order (optional)
        env_order = os.getenv("CBC_FEATURE_ORDER", "").strip()
        if env_order:
            cols = [c.strip() for c in env_order.split(",") if c.strip()]
            X_in = X_in.reindex(columns=cols, fill_value=0.0)

        # Predict
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
            st.table(pd.DataFrame({"Class": classes, "Probability": proba}))

        else:
            pred = model.predict(X_in)[0]
            st.header("Prediction")
            st.subheader(str(pred))
            st.info("This model does not provide predict_proba(), so probabilities are not available.")

    except Exception as e:
        st.error("CBC prediction failed.")
        st.exception(e)
