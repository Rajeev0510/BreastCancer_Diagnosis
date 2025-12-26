# app.py
# BC_ultrasound Streamlit app (Cloud-safe)
#
# Key fixes for Streamlit Cloud:
# - No hard-coded local Mac paths
# - Model weights loaded from repo-relative path or env var
# - Clear error message if weights missing (instead of crashing)
# - Model is cached so it loads once per session

import os
from datetime import datetime
from typing import List, Dict

import pandas as pd
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
from torchvision import models, transforms


# -------------------- CONFIG --------------------
CLASSES = ["benign", "malignant", "normal"]
IMG_SIZE = 224

# ✅ Cloud-safe default path (put your weights here in the GitHub repo)
# Repo layout recommendation:
#   breastcancer_diagnosis/
#     app.py
#     models/
#       best_model.pth
DEFAULT_WEIGHTS = os.path.join("models", "best_model.pth")

# Allow override via Streamlit Cloud env var:
#   BEST_WEIGHTS=models/best_model.pth   (or an absolute path inside the container)
BEST_WEIGHTS = os.getenv("BEST_WEIGHTS", DEFAULT_WEIGHTS)

DEVICE = "cpu"  # Streamlit Community Cloud is CPU; keep it deterministic


# -------------------- MODEL --------------------
def build_model(num_classes: int = 3) -> nn.Module:
    """Create a simple ResNet18 classifier head."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> nn.Module:
    """Load model weights (cached). Raises FileNotFoundError if missing."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}. "
            f"Add it to the repo (e.g., {DEFAULT_WEIGHTS}) or set BEST_WEIGHTS env var."
        )

    model = build_model(num_classes=len(CLASSES))
    state = torch.load(weights_path, map_location=DEVICE)

    # Support both {state_dict: ...} and direct state dict formats
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Sometimes training saves keys like 'module.' prefix
    cleaned = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # If your training used normalization, uncomment and match means/std:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_images(model: nn.Module, images: List[Image.Image], filenames: List[str]) -> List[Dict]:
    """Run inference on a list of PIL images."""
    tensors = [TF(im.convert("RGB")) for im in images]
    batch = torch.stack(tensors).to(DEVICE)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    rows = []
    for i, fn in enumerate(filenames):
        p = probs[i]
        pred_idx = int(p.argmax())
        pred_class = CLASSES[pred_idx]
        confidence = float(p[pred_idx])

        row = {
            "file": fn,
            "pred_class": pred_class,
            "confidence": confidence,
        }
        for j, cls in enumerate(CLASSES):
            row[cls] = float(p[j])
        rows.append(row)
    return rows


# -------------------- UI --------------------
st.set_page_config(page_title="BC Ultrasound", layout="wide")
st.title("BC Ultrasound — Image Prediction")
st.caption("Upload ultrasound images and get predicted class + probabilities.")

with st.expander("Model configuration", expanded=False):
    st.write("Weights path:", BEST_WEIGHTS)
    st.write("Weights exists:", os.path.exists(BEST_WEIGHTS))
    st.write("Device:", DEVICE)

# Session state for accumulating predictions
if "session_preds" not in st.session_state:
    st.session_state.session_preds = []

colA, colB = st.columns([1, 1])

with colA:
    uploaded = st.file_uploader(
        "Upload ultrasound images (PNG/JPG).",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

with colB:
    if st.button("🧹 Reset results", type="secondary"):
        st.session_state.session_preds = []
        st.success("Cleared prediction table.")

run_btn = st.button("🔎 Run prediction", type="primary", disabled=not uploaded)

# Try to load model (graceful error)
model = None
model_error = None
try:
    model = load_model(BEST_WEIGHTS)
except Exception as e:
    model_error = e

if run_btn:
    if model_error is not None:
        st.error("Model could not be loaded.")
        st.exception(model_error)
        st.stop()

    # Read images
    images = []
    names = []
    for f in uploaded:
        try:
            im = Image.open(f).convert("RGB")
            images.append(im)
            names.append(f.name)
        except Exception as e:
            st.warning(f"Skipping {getattr(f, 'name', 'file')}: {e}")

    if not images:
        st.warning("No valid images to run.")
        st.stop()

    with st.spinner("Running inference..."):
        rows = predict_images(model, images, names)
        st.session_state.session_preds.extend(rows)

# Show results table + download
if st.session_state.session_preds:
    df_all = pd.DataFrame(st.session_state.session_preds)

    # Ensure consistent column ordering
    ordered_cols = ["file", "pred_class", "confidence"] + CLASSES
    for c in ordered_cols:
        if c not in df_all.columns:
            df_all[c] = None
    df_all = df_all[ordered_cols]

    st.subheader("Predictions")
    st.dataframe(df_all, use_container_width=True, height=420)

    csv_bytes = df_all.to_csv(index=False).encode("utf-8")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name=f"bc_ultrasound_predictions_{stamp}.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions yet — upload images and click **Run prediction**.")
