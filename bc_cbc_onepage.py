# bc_cbc_onepage.py
# BC CBC Streamlit app — form inputs (no file upload)

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
def load_cbc_model(path: str):
    if joblib is None:
        raise RuntimeError("joblib is not installed. Add `joblib` to requirements.txt.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CBC model not found: {path}")
    return joblib.load(path)

# ------------------ Input schema ------------------
# Edit/extend this list to match your model feature names EXACTLY.
FIELDS = [
    # left column
    ("WBC (10^3/uL)", "WBC", 7.0, 0.0, 200.0, 0.1),
    ("RBC (10^6/uL)", "RBC", 4.6, 0.0, 20.0, 0.1),
    ("HGB (g/dL)", "HGB", 13.2, 0.0, 30.0, 0.1),
    ("HCT (%)", "HCT", 40.0, 0.0, 80.0, 0.1),
    ("MCV (fL)", "MCV", 88.0, 40.0, 140.0, 0.1),
    ("MCH (pg)", "MCH", 29.0, 10.0, 50.0, 0.1),
    # right column
    ("Neutrophils (%)", "Neutrophils", 58.0, 0.0, 100.0, 0.1),
    ("Lymphocytes (%)", "Lymphocytes", 30.0, 0.0, 100.0, 0.1),
    ("Monocytes (%)", "Monocytes", 7.0, 0.0, 100.0, 0.1),
    ("Eosinophils (%)", "Eosinophils", 2.5, 0.0, 100.0, 0.1),
    ("Basophils (%)", "Basophils", 0.5, 0.0, 100.0, 0.1),
    ("CRP (mg/L)", "CRP", 2.0, 0.0, 500.0, 0.1),
]

# Split into two columns for UI (first half left, second half right)
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

# Build a 1-row dataframe (model input)
X = pd.DataFrame([values])

# Optional: show what will be fed into the model
with st.expander("Show model input row", expanded=False):
    st.dataframe(X, use_container_width=True)

predict = st.button("Predict", type="primary")

# ------------------ Prediction ------------------
if predict:
    # If model exists, run it. Otherwise run demo logic.
    if os.path.exists(CBC_MODEL_PATH):
        try:
            model = load_cbc_model(CBC_MODEL_PATH)

            # Ensure same column order as training if your model expects it
            # If you trained on a specific feature order, set FEATURE_ORDER env var or hardcode it here.
            feature_order = os.getenv("CBC_FEATURE_ORDER", "")
            if feature_order.strip():
                cols = [c.strip() for c in feature_order.split(",") if c.strip()]
                missing = [c for c in cols if c not in X.columns]
                if missing:
                    st.error(f"Missing required features for model: {missing}")
                    st.stop()
                X_in = X[cols]
            else:
                X_in = X

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_in)[0]
                classes = getattr(model, "classes_", np.array([str(i) for i in range(len(proba))]))
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
                st.info("This model does not expose predict_proba(), so probabilities are not available.")

        except Exception as e:
            st.error("CBC prediction failed.")
            st.exception(e)

    else:
        # Demo: simple scoring for UI testing
        st.warning("CBC model not found — showing DEMO prediction. Add models/bc_cbc_model.pkl for real output.")
        score = float(X.sum(axis=1).iloc[0])
        # Fake 3-class probabilities from a score
        p_normal = 1 / (1 + np.exp((score - 250) / 20))
        p_malignant = 1 - p_normal
        p_benign = max(0.0, 1.0 - (p_normal + p_malignant))
        probs = np.array([p_normal, p_benign, p_malignant])
        probs = probs / probs.sum()

        classes = np.array(["Normal", "Benign", "Malignant"])
        pred_label = str(classes[int(np.argmax(probs))])

        st.header("Prediction")
        st.subheader(pred_label)

        st.header("Probabilities")
        st.table(pd.DataFrame({"Class": classes, "Probability": probs}))
