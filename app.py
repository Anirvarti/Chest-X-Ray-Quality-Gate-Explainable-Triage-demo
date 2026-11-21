import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from PIL import Image

st.set_page_config(page_title="AIIMS Demo: X-Ray Quality Gate & Triage", layout="wide")
st.title("Chest X-Ray Quality Gate & Explainable Triage")
st.caption("Research & Workflow Prototype — Not for Autonomous Clinical Diagnosis")

@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model

def apply_clahe(gray_img, clip_limit, grid_size):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(gray_img)

def compute_gradcam(model, img_tensor, target_idx):
    model.zero_grad()
    feats = model.features(img_tensor)
    feats.retain_grad()
    out = F.relu(feats)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    logits = model.classifier(out)
    
    # Backprop for the selected pathology
    score = logits[0, target_idx]
    score.backward()
    
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(dim=1)).squeeze().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, torch.sigmoid(logits).detach().numpy()[0]

# --- Sidebar Controls ---
st.sidebar.header("1. Upload Chest X-Ray")
uploaded_file = st.sidebar.file_uploader("Upload PNG / JPG", type=["png", "jpg", "jpeg"])

st.sidebar.header("2. Quality Gate (CLAHE)")
use_clahe = st.sidebar.checkbox("Enable CLAHE Contrast Enhancement", value=True)
clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 6.0, 2.5, 0.5)
grid_size = st.sidebar.slider("Tile Grid Size", 4, 16, 8, 2)

if uploaded_file is not None:
    # Load as grayscale
    pil_img = Image.open(uploaded_file).convert("L")
    orig_gray = np.array(pil_img)

    # Quality check metrics
    mean_brightness = np.mean(orig_gray)
    contrast_std = np.std(orig_gray)
    
    # Apply deterministic CLAHE
    enhanced_gray = apply_clahe(orig_gray, clip_limit, grid_size) if use_clahe else orig_gray

    # Show Quality Gate Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Mean Brightness (0–255)", f"{mean_brightness:.1f}")
    col_m2.metric("Contrast (Std Dev)", f"{contrast_std:.1f}")
    quality_status = "Low Contrast / Dark" if (mean_brightness < 70 or contrast_std < 40) else "Acceptable Exposure"
    col_m3.metric("Exposure Check", quality_status)

    # Prepare tensor for TorchXRayVision (224x224, normalized to [-1024, 1024])
    resized = cv2.resize(enhanced_gray, (224, 224))
    norm_img = xrv.datasets.normalize(resized, 255)
    img_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float()

    with st.spinner("Running TorchXRayVision Triage & Grad-CAM..."):
        model = load_model()
        with torch.no_grad():
            raw_preds = model(img_tensor)[0].numpy()

    # Rank top 5 pathologies
    pathologies = model.pathologies
    ranked = sorted(zip(pathologies, raw_preds), key=lambda x: x[1], reverse=True)
    top_names = [p[0] for p in ranked[:6]]

    st.sidebar.header("3. Explainability (Grad-CAM)")
    selected_pathology = st.sidebar.selectbox("Inspect Heatmap For:", top_names)
    target_idx = pathologies.index(selected_pathology)

    cam, _ = compute_gradcam(model, img_tensor, target_idx)
    cam_resized = cv2.resize(cam, (orig_gray.shape[1], orig_gray.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    base_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(base_rgb, 0.6, heatmap_rgb, 0.4, 0)

    # --- Visual Display ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Original X-Ray")
        st.image(orig_gray, width='stretch')
    with c2:
        st.subheader("CLAHE Enhanced (No AI)")
        st.image(enhanced_gray, width='stretch')
    with c3:
        st.subheader(f"Grad-CAM: {selected_pathology}")
        st.image(overlay, width='stretch')

    st.subheader("Top Triage Flags (Confidence Scores)")
    for name, score in ranked[:6]:
        st.progress(float(min(max(score, 0.0), 1.0)), text=f"{name}: {score*100:.1f}%")
else:
    st.info("Upload a Chest X-Ray image from the left sidebar to begin the live demo.")