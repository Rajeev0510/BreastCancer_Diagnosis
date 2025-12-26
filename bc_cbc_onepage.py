{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # bc_cbc_onepage.py\
# BC_CBC Streamlit app (CSV upload \uc0\u8594  prediction)\
\
import os\
import io\
import pandas as pd\
import numpy as np\
import streamlit as st\
\
try:\
    import joblib\
except Exception:\
    joblib = None\
\
st.set_page_config(page_title="BC CBC", layout="wide")\
st.title("BC CBC \'97 CBC-based Prediction")\
st.caption("Upload a CBC CSV and generate predictions (model optional).")\
\
# ---- Model config ----\
DEFAULT_MODEL = os.path.join("models", "bc_cbc_model.pkl")\
CBC_MODEL_PATH = os.getenv("CBC_MODEL_PATH", DEFAULT_MODEL)\
\
with st.expander("Model configuration", expanded=False):\
    st.write("Model path:", CBC_MODEL_PATH)\
    st.write("Model exists:", os.path.exists(CBC_MODEL_PATH))\
\
@st.cache_resource(show_spinner=False)\
def load_cbc_model(path: str):\
    if joblib is None:\
        raise RuntimeError("joblib is not installed. Add `joblib` to requirements.txt.")\
    if not os.path.exists(path):\
        raise FileNotFoundError(f"Missing CBC model: \{path\}")\
    return joblib.load(path)\
\
uploaded = st.file_uploader("Upload CBC CSV", type=["csv", "tsv"])\
\
if uploaded:\
    # Read CSV / TSV\
    try:\
        try:\
            df = pd.read_csv(uploaded)\
        except Exception:\
            uploaded.seek(0)\
            df = pd.read_csv(uploaded, sep="\\t")\
    except Exception as e:\
        st.error("Could not read file as CSV/TSV.")\
        st.exception(e)\
        st.stop()\
\
    st.subheader("Preview")\
    st.dataframe(df.head(50), use_container_width=True)\
\
    st.divider()\
\
    # Numeric features only (adjust to your needs)\
    X = df.select_dtypes(include=np.number).copy()\
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)\
\
    st.write(f"Numeric feature columns detected: **\{X.shape[1]\}**")\
\
    run_btn = st.button("\uc0\u55357 \u56590  Run CBC Prediction", type="primary", disabled=(len(df) == 0))\
\
    if run_btn:\
        # If you haven't added a real model yet, this demo logic will run\
        use_demo = not os.path.exists(CBC_MODEL_PATH)\
\
        if use_demo:\
            st.warning("CBC model not found \'97 running DEMO logic. Add models/bc_cbc_model.pkl to enable real predictions.")\
            score = X.sum(axis=1) if X.shape[1] else pd.Series([0]*len(df))\
            thr = float(score.median()) if len(score) else 0.0\
            pred = np.where(score >= thr, "HighRisk", "LowRisk")\
            out = df.copy()\
            out["CBC_Predicted_Label"] = pred\
        else:\
            try:\
                model = load_cbc_model(CBC_MODEL_PATH)\
                if hasattr(model, "predict_proba"):\
                    proba = model.predict_proba(X)\
                    pred_idx = proba.argmax(axis=1)\
                    classes = getattr(model, "classes_", None)\
                    if classes is not None:\
                        pred = [classes[i] for i in pred_idx]\
                    else:\
                        pred = pred_idx\
                    out = df.copy()\
                    out["CBC_Predicted_Label"] = pred\
                    if classes is not None:\
                        for j, cls in enumerate(classes):\
                            out[f"P_\{cls\}"] = proba[:, j]\
                else:\
                    y = model.predict(X)\
                    out = df.copy()\
                    out["CBC_Prediction"] = y\
            except Exception as e:\
                st.error("CBC model prediction failed.")\
                st.exception(e)\
                st.stop()\
\
        st.subheader("Results")\
        st.dataframe(out.head(200), use_container_width=True)\
\
        csv_bytes = out.to_csv(index=False).encode("utf-8")\
        st.download_button(\
            "\uc0\u11015 \u65039  Download CBC Predictions CSV",\
            data=csv_bytes,\
            file_name="bc_cbc_predictions.csv",\
            mime="text/csv",\
        )}