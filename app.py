import os
import streamlit as st
from PIL import Image

from inference import InferenceConfig, load_model, load_temperature, predict_single


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide"
)

st.markdown(
    """
# 👁️ Diabetic Retinopathy Detection System
### AI-based screening for **Referable Diabetic Retinopathy (diagnosis ≥ 2)**
Upload a retinal fundus image to estimate a **calibrated DR probability** and get an uncertainty-aware triage decision.
"""
)

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = os.path.join("models", "best_model.pt")
CALIB_PATH = os.path.join("models", "calibration.json")

# ----------------------------
# Sidebar settings
# ----------------------------
st.sidebar.header("Settings")
img_size = st.sidebar.selectbox("Input image size", [224, 256, 300], index=0)

# Optional: keep thresholds editable from sidebar
low_thresh = st.sidebar.slider("Low triage threshold", 0.0, 1.0, 0.30, 0.01)
high_thresh = st.sidebar.slider("High triage threshold", 0.0, 1.0, 0.70, 0.01)

cfg = InferenceConfig(img_size=int(img_size), device="cpu")


# ----------------------------
# Load model once (cached)
# ----------------------------
@st.cache_resource
def init_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    model = load_model(MODEL_PATH, cfg)
    temperature = load_temperature(CALIB_PATH)
    return model, temperature


try:
    model, T = init_model()
    st.caption(f"✅ Loaded model: {cfg.backbone_name} | Device: {cfg.device} | Temperature T: {T:.3f}")
except Exception as e:
    st.error("❌ Failed to load model or calibration.")
    st.exception(e)
    st.stop()


# ----------------------------
# Upload
# ----------------------------
uploaded = st.file_uploader(
    "Upload fundus image (jpg/jpeg/png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("⬆️ Upload an image to run prediction.")
    st.stop()

img = Image.open(uploaded)

# ----------------------------
# Predict
# ----------------------------
out = predict_single(model, img, cfg, temperature=T)
prob = float(out["calibrated_probability"])
raw_logit = float(out["raw_logit"])

# ----------------------------
# Layout: image + result
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Uploaded Image")
    st.image(img, caption="Fundus image", use_container_width=True)

with col2:
    st.subheader("Prediction Result")

    st.metric("Raw Logit", f"{raw_logit:.4f}")
    st.metric("Temperature (T)", f"{T:.3f}")
    st.metric("Calibrated Probability", f"{prob:.4f}")

    st.write("Risk level")
    st.progress(min(max(prob, 0.0), 1.0))

    st.write("### Uncertainty-aware Triage Decision")

    if prob < low_thresh:
        st.success(
            f"✅ Non-referable DR\n\n"
            f"Calibrated probability = {prob:.4f} < {low_thresh:.2f}"
        )
    elif prob > high_thresh:
        st.error(
            f"⚠️ Referable DR (Positive)\n\n"
            f"Calibrated probability = {prob:.4f} > {high_thresh:.2f}"
        )
    else:
        st.warning(
            f"🟡 Uncertain Case – Needs Expert Review\n\n"
            f"Calibrated probability = {prob:.4f} is between {low_thresh:.2f} and {high_thresh:.2f}"
        )

    with st.expander("Show calculation details"):
        st.code(
            f"raw_logit = {raw_logit:.6f}\n"
            f"T = {T:.6f}\n"
            f"calibrated_logit = raw_logit / T = {raw_logit / T:.6f}\n"
            f"prob = sigmoid(calibrated_logit) = {prob:.6f}\n"
            f"low_thresh = {low_thresh:.2f}\n"
            f"high_thresh = {high_thresh:.2f}"
        )

# ----------------------------
# Disclaimer
# ----------------------------
st.markdown("---")
st.caption("⚠️ Research prototype only — not a medical device. Do not use for clinical diagnosis.")
