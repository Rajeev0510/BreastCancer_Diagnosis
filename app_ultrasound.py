# app.py
import os
from datetime import datetime
import pandas as pd
from PIL import Image

import streamlit as st

import torch
import torch.nn as nn
from torchvision import models, transforms

# -------------------- CONFIG --------------------
CLASSES   = ["benign", "malignant", "normal"]
IMG_SIZE  = 224
BEST_WEIGHTS = "/Users/rajeevkumar2/Documents/breast_ultrasound/ultrasound/output_data/best_model.pth"

# -------------------- DEVICE --------------------
device = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))

# -------------------- TRANSFORMS ----------------
infer_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# -------------------- MODEL ---------------------
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    state = torch.load(weights_path, map_location="cpu")
    m.load_state_dict(state)
    m = m.to(device).eval()
    return m

def predict_probs(mdl, pil_img: Image.Image):
    x = infer_tfms(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(mdl(x), dim=1)[0].cpu().numpy()
    return {CLASSES[i]: float(p) for i, p in enumerate(probs)}

# -------------------- UI ------------------------
st.set_page_config(page_title="Ultrasound Classifier", layout="wide")
st.title("🩺 Breast Ultrasound Classifier")
st.caption(f"Device: **{device}**")

# load model silently
if not os.path.isfile(BEST_WEIGHTS):
    st.error(f"Model weights not found:\n{BEST_WEIGHTS}")
    st.stop()
model = load_model(BEST_WEIGHTS)

# session results
if "session_preds" not in st.session_state:
    st.session_state.session_preds = []

# controls
col_left, col_right = st.columns([3,1], vertical_alignment="center")
with col_left:
    uploads = st.file_uploader(
        "Upload ultrasound images (png/jpg/jpeg/bmp/tif/tiff). You can select multiple.",
        type=["png","jpg","jpeg","bmp","tif","tiff"],
        accept_multiple_files=True
    )
with col_right:
    if st.button("Reset table", use_container_width=True, type="secondary"):
        st.session_state.session_preds = []
        st.rerun()

# process uploads
if uploads:
    for up in uploads:
        try:
            if "_mask" in up.name.lower():
                st.info(f"Skipped mask: {up.name}")
                continue
            pil = Image.open(up).convert("RGB")
            probs = predict_probs(model, pil)

            # choose class with highest prob
            pred_class = max(probs, key=probs.get)
            confidence = probs[pred_class]

            row = {
                "file": up.name,
                "pred_class": pred_class,
                "confidence": confidence,
                "benign": probs["benign"],
                "malignant": probs["malignant"],
                "normal": probs["normal"],
            }
            st.session_state.session_preds.append(row)
        except Exception as e:
            st.warning(f"Skipping {up.name}: {e}")

# predictions table + download
st.subheader("Predictions")
if st.session_state.session_preds:
    df_all = pd.DataFrame(st.session_state.session_preds)
    # put pred_class + confidence right after filename
    df_all = df_all[["file","pred_class","confidence","benign","malignant","normal"]]

    with st.expander("Preview (first 200 rows)", expanded=True):
        st.dataframe(df_all.head(200), use_container_width=True)

    csv_bytes = df_all.to_csv(index=False).encode("utf-8")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("⬇️ Download CSV",
                       data=csv_bytes,
                       file_name=f"predictions_{stamp}.csv",
                       mime="text/csv")
else:
    st.info("No predictions yet — upload images above.")