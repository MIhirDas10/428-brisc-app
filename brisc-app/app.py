import streamlit as st
import torch
import os, hashlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import UNetMultiTask
from huggingface_hub import hf_hub_download



# ---- import your model classes here ----
# from your_model_file import UNetMultiTask, AttentionUNetSeg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256

# ---------- Helpers (same logic you used) ----------
def load_any_image_for_model(pil_img, image_size=256, expect_channels=1):
    img = pil_img.convert("RGB").resize((image_size, image_size))
    arr = np.array(img).astype(np.float32)

    if expect_channels == 1:
        arr = arr.mean(axis=2, keepdims=True)  # H,W,1

    arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    x = torch.from_numpy(arr).unsqueeze(0)  # 1,C,H,W
    return x, img

@torch.no_grad()
def predict_pil(pil_img, model, expect_channels=1, seg_thresh=0.5):
    model.eval()
    x, img_vis = load_any_image_for_model(pil_img, IMAGE_SIZE, expect_channels)
    x = x.to(DEVICE)

    out = model(x)

    # multitask: (seg, cls)
    if isinstance(out, (tuple, list)):
        seg_logits = out[0]
        cls_logits = out[1]
    else:
        seg_logits = out
        cls_logits = None

    if seg_logits.ndim == 3:
        seg_logits = seg_logits.unsqueeze(1)

    seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
    seg_mask = (seg_prob >= seg_thresh).astype(np.uint8)

    cls_prob = None
    if cls_logits is not None:
        cls_prob = torch.sigmoid(cls_logits).view(-1)[0].item()

    return img_vis, seg_prob, seg_mask, cls_prob

def overlay_mask(img_vis, seg_mask):
    base = np.array(img_vis.convert("RGB"))
    overlay = base.copy()
    overlay[seg_mask == 1] = (255, 0, 0)
    blended = (0.6 * base + 0.4 * overlay).astype(np.uint8)
    return blended

# -------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- Load model once ----------
@st.cache_resource
def load_model():
    weights_path = hf_hub_download(
        repo_id="MihirDas/brisc-unet",
        filename="unet_multitask_best.pth",
        revision="main"
    )

    # Print what got downloaded (debug)
    st.sidebar.write("HF weights MB:", round(os.path.getsize(weights_path)/1024/1024, 2))
    st.sidebar.write("HF sha16:", sha256_file(weights_path)[:16])

    model = UNetMultiTask(n_channels=1, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


model = load_model()

# ---------- UI ----------
st.title("Brain Tumor Segmentation + Classification Demo")

uploaded = st.file_uploader("Upload an image (.png/.jpg)", type=["png", "jpg", "jpeg"])
seg_thresh = st.slider("Segmentation threshold", 0.05, 0.95, 0.50, 0.05)

if uploaded is not None:
    pil = Image.open(uploaded)
    st.image(pil, caption="Input image", use_container_width=True)

    img_vis, seg_prob, seg_mask, cls_prob = predict_pil(pil, model, expect_channels=1, seg_thresh=seg_thresh)

    st.subheader("Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_vis, caption="Resized input", use_container_width=True)

    with col2:
        st.image(seg_prob, caption="Segmentation probability", use_container_width=True, clamp=True)

    with col3:
        st.image(overlay_mask(img_vis, seg_mask), caption="Mask overlay", use_container_width=True)
