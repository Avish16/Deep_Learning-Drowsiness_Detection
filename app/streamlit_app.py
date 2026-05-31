import streamlit as st
import numpy as np
import cv2
import time
from pathlib import Path
from PIL import Image as PILImage
import tensorflow as tf

st.set_page_config(page_title="DrowsAI", page_icon="👁️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2.5rem; }
.app-title {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg, #00D4FF, #00FF88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-sub { color: #8899AA; font-size: 0.95rem; margin-bottom: 0.8rem; }
.pill {
    display: inline-block; border: 1px solid #00D4FF; color: #00D4FF;
    border-radius: 20px; padding: 3px 14px; font-size: 0.78rem;
    font-weight: 600; margin-right: 8px; margin-bottom: 1rem;
}
.step-bar {
    display: flex; align-items: center; gap: 6px;
    margin: 0.8rem 0 1.2rem 0;
}
.step {
    background: #0D1B2A; border: 1px solid #1E2D3E;
    border-radius: 20px; padding: 5px 14px;
    color: #8899AA; font-size: 0.8rem; font-weight: 600;
}
.step-active {
    background: rgba(0,212,255,0.15); border: 1px solid #00D4FF;
    border-radius: 20px; padding: 5px 14px;
    color: #00D4FF; font-size: 0.8rem; font-weight: 600;
}
.arrow { color: #556677; font-size: 0.9rem; }
.col-label {
    font-size: 0.8rem; font-weight: 700; color: #8899AA;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.badge-awake {
    background: linear-gradient(135deg, #003d1f, #006b35);
    border: 2px solid #00FF88; border-radius: 10px;
    color: #00FF88; font-size: 1.6rem; font-weight: 800;
    padding: 16px; text-align: center; margin-bottom: 0.8rem;
}
.badge-drowsy {
    background: linear-gradient(135deg, #3d0000, #6b0000);
    border: 2px solid #FF4444; border-radius: 10px;
    color: #FF4444; font-size: 1.6rem; font-weight: 800;
    padding: 16px; text-align: center; margin-bottom: 0.8rem;
}
.conf-track {
    background: #1E2D3E; border-radius: 6px;
    height: 10px; width: 100%; margin: 4px 0 12px 0;
}
.conf-fill-awake {
    background: linear-gradient(90deg, #00D4FF, #00FF88);
    height: 10px; border-radius: 6px;
}
.conf-fill-drowsy {
    background: linear-gradient(90deg, #FF4444, #FF8800);
    height: 10px; border-radius: 6px;
}
.meta-line { color: #556677; font-size: 0.78rem; margin-top: 0.6rem; }
.detect-box {
    font-size: 0.8rem; border-radius: 6px; padding: 6px 10px; margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="app-title" style="margin-top: 1rem;">👁️ DrowsAI</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Driver Drowsiness Detection · CNN on Infrared Eye Images</div>', unsafe_allow_html=True)
st.markdown("""
<span class="pill">DATASET: 84,898 images</span>
<span class="pill">ACCURACY: 98.6%</span>
<span class="pill">MODEL: CNN</span>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox("Select Model", [
        "Deep Augmented CNN (Best)", "Augmented CNN", "Baseline CNN"
    ])
    threshold = st.slider("Confidence Threshold", 0.50, 0.95, 0.70, 0.01)
    st.info("ℹ️ Raw model output = P(drowsy). If above threshold → DROWSY alert.")
    st.markdown("---")
    st.warning("⚠️ Research prototype only. Not for use in real vehicles.")

# Weights
BASE = Path(__file__).parent.parent
WEIGHTS = {
    "Deep Augmented CNN (Best)": BASE / "weights" / "cnn_mrl_driver_aug_deep.keras",
    "Augmented CNN":             BASE / "weights" / "cnn_mrl_driver_aug.keras",
    "Baseline CNN":              BASE / "weights" / "baseline_cnn_mrl.keras",
}

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(str(path))

mp = WEIGHTS[model_choice]
if not mp.exists():
    st.error(f"Weight file not found: {mp}")
    st.stop()
model = load_model(mp)

def detect_face_and_eyes(uploaded_file):
    img = PILImage.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape
    annotated = img_np.copy()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = None
    for scale in [1.1, 1.05, 1.15, 1.2]:
        detected = face_cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=5, minSize=(60, 60)
        )
        if len(detected) > 0:
            faces = detected
            break

    if faces is not None and len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        fx, fy, fw, fh = faces[0]

        # Draw blue face box
        cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh), (100, 149, 237), 2)
        cv2.putText(annotated, "Face", (fx, fy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 149, 237), 1)

        # Eye strip: 22%-48% of face height
        ey1 = max(0, fy + int(fh * 0.22))
        ey2 = min(h_img, fy + int(fh * 0.48))
        ex1 = max(0, fx + int(fw * 0.05))
        ex2 = min(w_img, fx + int(fw * 0.95))

        # Draw cyan eye region box
        cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), (0, 212, 255), 2)
        cv2.putText(annotated, "Eye region", (ex1, ey1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 212, 255), 1)

        eye_crop = gray[ey1:ey2, ex1:ex2]
        return eye_crop, "face_detected", annotated
    else:
        # No face — use full image as grayscale
        cv2.putText(annotated, "No face detected", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 1)
        return gray, "no_face", annotated

def preprocess_image(eye_array):
    eye = cv2.resize(eye_array, (64, 64))
    eye = eye.astype("float32") / 255.0
    eye = np.expand_dims(eye, axis=-1)
    eye = np.expand_dims(eye, axis=0)
    return eye

def run_predict(img_array):
    t0 = time.time()
    prob = float(model.predict(img_array, verbose=0)[0][0])
    elapsed = (time.time() - t0) * 1000
    label = "DROWSY" if prob > threshold else "AWAKE"
    confidence = prob if prob > threshold else 1.0 - prob
    return label, confidence, prob, elapsed

# Tabs
tab1, tab2 = st.tabs(["🔍 Live Detection", "📊 Model Performance"])

with tab1:
    # Pipeline step bar
    st.markdown("""
    <div class="step-bar">
        <span class="step-active">1 · Upload Image</span>
        <span class="arrow">→</span>
        <span class="step">2 · Detect Face</span>
        <span class="arrow">→</span>
        <span class="step">3 · Extract Eyes</span>
        <span class="arrow">→</span>
        <span class="step">4 · CNN Inference</span>
        <span class="arrow">→</span>
        <span class="step">5 · Alert</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Upload an image")
    st.caption("Upload a face photo or infrared eye crop. The app detects the face, extracts the eye region, and runs the CNN.")

    uploaded = st.file_uploader("", type=["jpg","jpeg","png","bmp"],
                                 label_visibility="collapsed")

    if uploaded:
        eye_crop, method, annotated = detect_face_and_eyes(uploaded)
        arr = preprocess_image(eye_crop)
        label, confidence, prob, elapsed = run_predict(arr)

        # Active pipeline steps
        step2 = "step-active" if method == "face_detected" else "step"
        step3 = "step-active" if method == "face_detected" else "step"
        step4 = "step-active"
        step5 = "step-active"
        st.markdown(f"""
        <div class="step-bar">
            <span class="step-active">1 · Upload Image</span>
            <span class="arrow">→</span>
            <span class="{step2}">2 · Detect Face</span>
            <span class="arrow">→</span>
            <span class="{step3}">3 · Extract Eyes</span>
            <span class="arrow">→</span>
            <span class="{step4}">4 · CNN Inference</span>
            <span class="arrow">→</span>
            <span class="{step5}">5 · Alert</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1.2, 1, 1.3])

        with col1:
            st.markdown('<div class="col-label">Original Image</div>',
                        unsafe_allow_html=True)
            st.image(annotated, use_container_width=True,
                     caption="Blue = face · Cyan = eye region")

        with col2:
            st.markdown('<div class="col-label">Eye Crop (Model Input)</div>',
                        unsafe_allow_html=True)
            st.image(eye_crop, use_container_width=True,
                     caption="Detected eye region")
            thumb = cv2.resize(eye_crop, (64, 64))
            st.image(thumb, use_container_width=True,
                     caption="64×64 model input")

        with col3:
            st.markdown('<div class="col-label">Result</div>',
                        unsafe_allow_html=True)
            icon = "✅" if label == "AWAKE" else "⚠️"
            badge = "badge-awake" if label == "AWAKE" else "badge-drowsy"
            st.markdown(f'<div class="{badge}">{icon} {label}</div>',
                        unsafe_allow_html=True)
            pct = int(confidence * 100)
            fill = "conf-fill-awake" if label == "AWAKE" else "conf-fill-drowsy"
            st.markdown(f"""
            <p style="margin:0 0 3px 0;font-weight:600;font-size:0.9rem;">
                Confidence: {pct}%</p>
            <div class="conf-track">
                <div class="{fill}" style="width:{pct}%"></div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Awake", f"{round((1-prob)*100,1)}%")
            c2.metric("Drowsy", f"{round(prob*100,1)}%")
            if label == "DROWSY":
                st.error("⚠️ Drowsiness detected. Pull over safely.")
            det_color = "#00FF88" if method == "face_detected" else "#FFCC00"
            det_msg = "✅ Face detected — eye region extracted" if method == "face_detected" else "⚠️ No face detected — using full image"
            st.markdown(
                f'<div class="detect-box" style="color:{det_color};'
                f'background:rgba(0,0,0,0.2);border-left:3px solid {det_color};">'
                f'{det_msg}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="meta-line">Model: <b>{model_choice}</b> · '
                f'Inference: <b>{elapsed:.1f}ms</b> · '
                f'Threshold: <b>{threshold}</b></p>',
                unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Model Performance")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Baseline CNN", "98.59%", "Test Accuracy")
        st.caption("~500K params · F1: ~0.99")
    with c2:
        st.metric("Augmented CNN", "98.15%", "Test Accuracy")
        st.caption("~500K params · F1: ~0.98")
    with c3:
        st.metric("Deep Aug CNN ★", "98.39%", "Test Accuracy")
        st.caption("~800K params · F1: ~0.98")

    st.markdown("### 🏗️ Architecture")
    st.code("""
Input (64×64×1 grayscale)
    │
    ├─ Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
    ├─ Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
    ├─ Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
    │  [Deep model adds Conv2D(256) block]
    │
    ├─ Flatten → Dense(128, ReLU) → Dropout
    └─ Dense(1, Sigmoid) → P(drowsy)
    """, language="text")

    st.markdown("### ⚙️ Training Config")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
- **Optimizer:** Adam (lr=1e-3)
- **Loss:** Binary Cross-Entropy
- **Batch size:** 64
- **Input:** 64×64×1 grayscale
        """)
    with t2:
        st.markdown("""
- **Augmentation:** RandomFlip + RandomRotation(±5°)
- **Dataset:** MRL Eye (84,898 IR images)
- **Split:** ~60% train / 20% val / 20% test
- **Classes:** awake (0) · sleepy (1)
        """)

    st.markdown("### 📦 Dataset")
    st.markdown("""
**MRL Eye Dataset** — 84,898 PNG grayscale infrared eye crops, binary labels.
[View on Kaggle →](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset)
    """)
