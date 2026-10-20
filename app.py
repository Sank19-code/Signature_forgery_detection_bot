"""
Streamlit Web App: EfficientNet-B2 Signature Forgery Detection
Run locally with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pipeline

st.set_page_config(
    page_title="Signature Forgery Detection",
    page_icon="✍️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1.2rem;
        border-radius: 12px;
        background: #F3F4F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .status-genuine {
        color: #059669;
        font-weight: 700;
        font-size: 1.5rem;
    }
    .status-forged {
        color: #DC2626;
        font-weight: 700;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">✍️ Signature Forgery Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classify signatures as <b>Genuine</b> or <b>Forged</b> using Digital Image Processing (DIP) & EfficientNet-B2.</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    model_path = st.text_input("Model Checkpoint Path", "models/efficientnet_b2_signature.pt")
    custom_threshold = st.slider("Classification Threshold (Forged Probability)", 0.0, 1.0, 0.50, 0.01)
    use_calibrated = st.checkbox("Use Checkpoint Calibrated Threshold", value=True)
    st.markdown("---")
    st.markdown("### 📌 About the Model")
    st.markdown("""
    - **Architecture**: EfficientNet-B2
    - **Input Pipeline**: 
      1. Otsu Ink Extraction & Binarization
      2. Stage 2: Median Filter (3x3)
      3. Stage 3: High-Boost Filtering (k=1.5)
      4. Stage 4: Sobel Gradient Magnitude
    """)

# Load Model
@st.cache_resource
def get_model(ckpt_path):
    return pipeline.load_classifier(ckpt_path)

try:
    model, calibrated_thr = get_model(model_path)
    active_thr = calibrated_thr if use_calibrated else custom_threshold
    st.sidebar.success(f"Model loaded successfully!\nCalibrated EER Threshold: {calibrated_thr:.4f}")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model = None
    active_thr = custom_threshold

# Main layout: Upload & Inference
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("1. Upload Signature Image")
    uploaded_file = st.file_uploader(
        "Choose a signature image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg", "bmp", "tiff"]
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded Original Signature", use_container_width=True)

with col_result:
    st.subheader("2. Analysis & Verification")
    if uploaded_file is not None and model is not None:
        with st.spinner("Processing image through DIP stages & running EfficientNet-B2..."):
            try:
                canon_img, tensor = pipeline.preprocess_image(file_bytes)
                prob_forged = pipeline.predict_signature(model, tensor)
                prob_genuine = 1.0 - prob_forged
                is_forged = prob_forged >= active_thr

                # Classification Result Banner
                if is_forged:
                    st.error(f"⚠️ **Result: FORGED SIGNATURE** (Confidence: {prob_forged*100:.2f}%)")
                else:
                    st.success(f"✅ **Result: GENUINE SIGNATURE** (Confidence: {prob_genuine*100:.2f}%)")

                # Probabilities gauge
                st.write(f"**Forged Probability:** `{prob_forged*100:.2f}%` (Decision Threshold: `{active_thr*100:.2f}%`)")
                st.progress(float(np.clip(prob_forged, 0.0, 1.0)))

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("Genuine Probability", f"{prob_genuine*100:.2f}%")
                with m_col2:
                    st.metric("Forged Probability", f"{prob_forged*100:.2f}%")

                # Display DIP Preprocessed Signature
                with st.expander("🔍 View Digital Image Processing (DIP) Pipeline Output", expanded=False):
                    s1 = pipeline.contrast_stretching(canon_img)
                    s2 = pipeline.median_filter(s1)
                    s3 = pipeline.high_boost_filter(s2, k=1.5)
                    s4 = pipeline.sobel_edge_detection(s3)

                    st.markdown("**Canonical Binarized & Letterboxed Signature**")
                    st.image(canon_img, width=384, clamp=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Stage 2: Median Filter")
                        st.image(s2, clamp=True)
                        st.caption("Stage 3: High-Boost Filter")
                        st.image(s3, clamp=True)
                    with c2:
                        st.caption("Stage 1: Contrast Stretching")
                        st.image(s1, clamp=True)
                        st.caption("Stage 4: Sobel Gradient Magnitude")
                        st.image(s4, clamp=True)

            except Exception as ex:
                st.error(f"Inference error: {ex}")
    elif uploaded_file is None:
        st.info("👆 Please upload a signature image on the left panel to begin verification.")
