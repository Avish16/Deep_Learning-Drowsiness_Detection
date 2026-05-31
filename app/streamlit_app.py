import time
import pathlib
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DrowsAI — Driver Drowsiness Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .drows-title {
            font-size: 2.8rem; font-weight: 800;
            background: linear-gradient(90deg, #00D4FF, #00FF88);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.15rem;
        }
        .drows-subtitle { color: #8899AA; font-size: 1.05rem; margin-bottom: 1rem; }

        .pill-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
        .pill {
            background: rgba(0,212,255,0.12); border: 1px solid rgba(0,212,255,0.35);
            border-radius: 999px; padding: 0.35rem 1rem;
            font-size: 0.82rem; font-weight: 600; color: #00D4FF; letter-spacing: 0.04em;
        }

        .glass-card {
            background: rgba(13,27,42,0.75);
            border: 1px solid rgba(0,212,255,0.18);
            border-radius: 1rem; padding: 1.4rem 1.6rem;
            backdrop-filter: blur(12px); margin-bottom: 1rem;
        }

        /* Sample image cards */
        .sample-card {
            border: 2px solid rgba(0,212,255,0.18);
            border-radius: 0.75rem; padding: 0.5rem;
            margin-bottom: 0.4rem; transition: border-color 0.2s;
        }
        .sample-card-selected {
            border: 2px solid #00D4FF;
            border-radius: 0.75rem; padding: 0.5rem;
            margin-bottom: 0.4rem;
            box-shadow: 0 0 12px rgba(0,212,255,0.3);
        }

        /* Metric cards for Tab 2 */
        .metric-card {
            background: rgba(13,27,42,0.75);
            border: 1px solid rgba(0,212,255,0.18);
            border-radius: 1rem; padding: 1.2rem 1.4rem;
            text-align: center;
        }
        .metric-card-best {
            background: rgba(13,27,42,0.75);
            border: 2px solid #00D4FF;
            border-radius: 1rem; padding: 1.2rem 1.4rem;
            text-align: center;
            box-shadow: 0 0 18px rgba(0,212,255,0.2);
        }
        .metric-title { font-size: 0.9rem; font-weight: 700; color: #00D4FF; margin-bottom: 0.6rem; }
        .metric-acc { font-size: 1.9rem; font-weight: 800; color: #00FF88; }
        .metric-sub { font-size: 0.8rem; color: #8899AA; margin-top: 0.3rem; }

        /* Result badges */
        .badge-awake {
            display: inline-block;
            background: rgba(0,255,136,0.15); border: 2px solid #00FF88;
            border-radius: 0.75rem; padding: 0.7rem 1.6rem;
            font-size: 2rem; font-weight: 800; color: #00FF88;
            text-align: center; width: 100%;
        }
        .badge-drowsy {
            display: inline-block;
            background: rgba(255,68,68,0.15); border: 2px solid #FF4444;
            border-radius: 0.75rem; padding: 0.7rem 1.6rem;
            font-size: 2rem; font-weight: 800; color: #FF4444;
            text-align: center; width: 100%;
        }

        /* Confidence bar */
        .conf-track {
            background: rgba(255,255,255,0.08); border-radius: 999px;
            height: 14px; width: 100%; margin: 0.5rem 0 1rem 0; overflow: hidden;
        }
        .conf-fill-awake  { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#00FF88,#00D4FF); }
        .conf-fill-drowsy { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#FF8800,#FF4444); }

        /* Warning / info boxes */
        .warn-box {
            background: rgba(255,68,68,0.12); border-left: 4px solid #FF4444;
            border-radius: 0.5rem; padding: 0.8rem 1rem;
            color: #FF7777; font-weight: 600; margin-top: 0.75rem;
        }
        .disclaimer {
            background: rgba(255,200,0,0.08); border-left: 4px solid #FFCC00;
            border-radius: 0.5rem; padding: 0.6rem 1rem;
            color: #BBAA44; font-size: 0.82rem; margin-top: 1rem;
        }
        .info-card {
            background: rgba(0,212,255,0.06); border: 1px solid rgba(0,212,255,0.2);
            border-radius: 0.6rem; padding: 0.75rem 1rem;
            color: #8899AA; font-size: 0.83rem; line-height: 1.5;
        }

        /* Meta / secondary text */
        .meta-line { color: #5577AA; font-size: 0.83rem; margin-top: 0.5rem; }

        /* Section headers */
        .section-label {
            font-size: 0.78rem; font-weight: 700; letter-spacing: 0.1em;
            color: #5577AA; text-transform: uppercase; margin-bottom: 0.5rem;
        }

        table { width: 100%; }
        th { color: #00D4FF !important; }
        #MainMenu { visibility: hidden; }
        footer    { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
WEIGHTS = {
    "Deep Augmented CNN (Best)": ROOT / "weights" / "cnn_mrl_driver_aug_deep.keras",
    "Augmented CNN":             ROOT / "weights" / "cnn_mrl_driver_aug.keras",
    "Baseline CNN":              ROOT / "weights" / "baseline_cnn_mrl.keras",
}
SAMPLE_DIR = ROOT / "app" / "sample_images"

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

# ── Inference helpers ─────────────────────────────────────────────────────────
def prepare_image(file_obj):
    """Load any image as grayscale uint8 numpy array."""
    return np.array(Image.open(file_obj).convert("L"))

def preprocess_image(gray_array):
    """Resize to 64×64, normalise to [0,1], return (1,64,64,1) float32."""
    import cv2
    eye = cv2.resize(gray_array, (64, 64))
    eye = eye.astype("float32") / 255.0
    eye = np.expand_dims(eye, axis=-1)
    eye = np.expand_dims(eye, axis=0)
    return eye

def predict(model, img_array, threshold: float = 0.70):
    prob = float(model.predict(img_array, verbose=0)[0][0])
    label = "DROWSY" if prob > threshold else "AWAKE"
    confidence = prob if prob > threshold else 1.0 - prob
    return label, confidence, prob

def run_inference(file_obj, model, threshold):
    t0 = time.perf_counter()
    gray = prepare_image(file_obj)
    arr  = preprocess_image(gray)
    label, confidence, raw_prob = predict(model, arr, threshold)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return gray, label, confidence, raw_prob, elapsed_ms

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="drows-title">👁️ DrowsAI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="drows-subtitle">Driver Drowsiness Detection &nbsp;·&nbsp; '
    'Deep Learning on Infrared Eye Images</p>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="pill-row">
        <span class="pill">DATASET: 84,898 images</span>
        <span class="pill">ACCURACY: 98.6%</span>
        <span class="pill">MODEL: CNN</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model_choice = st.selectbox("Select Model", list(WEIGHTS.keys()), index=0)
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.50, max_value=0.95, value=0.70, step=0.01,
        help="Sigmoid output > threshold → DROWSY. Lower = more sensitive.",
    )
    st.markdown("---")
    st.markdown(
        '<div class="info-card">'
        "Model trained on the <strong>MRL Eye Dataset</strong> — "
        "84,898 infrared grayscale eye crops. "
        "Binary classification: <strong>Awake</strong> vs <strong>Drowsy</strong>."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer" style="margin-top:1rem;">⚠️ Research prototype only. '
        "Not for use in real vehicles without safety validation.</div>",
        unsafe_allow_html=True,
    )

# ── Load model ────────────────────────────────────────────────────────────────
weight_path = WEIGHTS[model_choice]
if not weight_path.exists():
    st.error(
        f"**Model weights not found:** `{weight_path}`\n\n"
        "Make sure the `weights/` directory contains the `.keras` files."
    )
    st.stop()

try:
    model = load_model(str(weight_path))
except Exception as exc:
    st.error(f"**Failed to load model:** {exc}")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_detect, tab_perf = st.tabs(["🔍 Live Detection", "📊 Model Performance"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Detection
# ═════════════════════════════════════════════════════════════════════════════
with tab_detect:

    awake_samples  = sorted((SAMPLE_DIR / "awake").glob("*.png"))  if (SAMPLE_DIR / "awake").exists()  else []
    sleepy_samples = sorted((SAMPLE_DIR / "sleepy").glob("*.png")) if (SAMPLE_DIR / "sleepy").exists() else []

    # ── Section A: Sample gallery ─────────────────────────────────────────────
    if awake_samples or sleepy_samples:
        st.markdown("### Try the model — select a sample image")
        st.caption("Synthetic infrared eye images matching the MRL training distribution.")

        def _sample_row(samples, ground_truth, row_label):
            st.markdown(f'<p class="section-label">{row_label}</p>', unsafe_allow_html=True)
            cols = st.columns(len(samples))
            for col, sf in zip(cols, samples):
                is_sel = st.session_state.get("sample_path") == str(sf)
                with col:
                    card_cls = "sample-card-selected" if is_sel else "sample-card"
                    st.markdown(f'<div class="{card_cls}">', unsafe_allow_html=True)
                    st.image(str(sf), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    btn_label = "✓ Selected" if is_sel else "Select"
                    if st.button(btn_label, key=f"btn_{sf.stem}", use_container_width=True):
                        st.session_state["sample_path"]          = str(sf)
                        st.session_state["sample_ground_truth"]  = ground_truth
                        st.session_state.pop("uploaded_gray", None)
                        st.rerun()

        _sample_row(awake_samples,  "Awake",  "👁️  Awake samples — open eye, bright pupil")
        _sample_row(sleepy_samples, "Drowsy", "😴  Drowsy samples — closed eye, eyelid covering pupil")

    # ── Section B: Upload ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📤 Upload your own IR eye image")
    st.caption(
        "Upload a grayscale infrared eye crop from the MRL dataset or similar NIR camera. "
        "JPG · PNG · BMP accepted."
    )
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        st.session_state.pop("sample_path", None)
        st.session_state.pop("sample_ground_truth", None)

    # ── Determine active input ────────────────────────────────────────────────
    active_file  = None
    source_label = None

    if uploaded is not None:
        active_file  = uploaded
        source_label = "Uploaded image"
    elif "sample_path" in st.session_state:
        _p = pathlib.Path(st.session_state["sample_path"])
        active_file  = open(_p, "rb")
        source_label = f"Sample — {st.session_state.get('sample_ground_truth', _p.stem)}"

    # ── Results panel ─────────────────────────────────────────────────────────
    if active_file is not None:
        gray, label, confidence, raw_prob, elapsed_ms = run_inference(
            active_file, model, threshold
        )
        st.markdown("---")
        st.markdown("### Result")

        col_img, col_res = st.columns([1, 1.6], gap="large")

        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="meta-line">{source_label}</p>', unsafe_allow_html=True)
            st.image(Image.fromarray(gray), caption="Input (grayscale)", use_container_width=True)
            import cv2 as _cv2
            img64 = Image.fromarray(_cv2.resize(gray, (64, 64)))
            st.image(img64, caption="64×64 model input", use_container_width=True, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_res:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            if label == "AWAKE":
                st.markdown('<div class="badge-awake">✅ AWAKE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-drowsy">⚠️ DROWSY</div>', unsafe_allow_html=True)

            pct        = int(confidence * 100)
            fill_class = "conf-fill-awake" if label == "AWAKE" else "conf-fill-drowsy"
            st.markdown(
                f"""
                <p style="margin:0.9rem 0 0.25rem 0; font-weight:600;">
                    Confidence: <span style="color:#E0E0E0">{pct}%</span>
                </p>
                <div class="conf-track">
                    <div class="{fill_class}" style="width:{pct}%"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            awake_pct  = round((1 - raw_prob) * 100, 1)
            drowsy_pct = round(raw_prob * 100, 1)
            c1, c2 = st.columns(2)
            c1.metric("Awake",  f"{awake_pct}%")
            c2.metric("Drowsy", f"{drowsy_pct}%")

            if label == "DROWSY":
                st.markdown(
                    '<div class="warn-box">⚠️ WARNING: Drowsiness detected. '
                    "Pull over safely and rest before continuing.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<p class="meta-line" style="margin-top:1rem;">'
                f"Model: <strong>{model_choice}</strong> &nbsp;|&nbsp; "
                f"Inference: <strong>{elapsed_ms:.1f} ms</strong> &nbsp;|&nbsp; "
                f"Threshold: <strong>{threshold:.2f}</strong></p>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center;padding:2.5rem 1rem;color:#445566;">
                <div style="font-size:2.5rem">👆</div>
                <div style="font-size:1.05rem;margin-top:0.5rem;">
                    Select a sample above or upload an IR eye image to run detection
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ═════════════════════════════════════════════════════════════════════════════
with tab_perf:

    # ── Metric cards ─────────────────────────────────────────────────────────
    st.markdown("### Model Comparison")
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Baseline CNN</div>
                <div class="metric-acc">98.59%</div>
                <div class="metric-sub">Test Accuracy</div>
                <div class="metric-sub" style="margin-top:0.6rem;">
                    F1: 0.99 &nbsp;·&nbsp; Precision: 0.99<br>
                    Recall: 0.98 &nbsp;·&nbsp; Params: ~500 K
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Augmented CNN</div>
                <div class="metric-acc">98.15%</div>
                <div class="metric-sub">Test Accuracy</div>
                <div class="metric-sub" style="margin-top:0.6rem;">
                    F1: 0.98 &nbsp;·&nbsp; Precision: 0.99<br>
                    Recall: 0.98 &nbsp;·&nbsp; Params: ~500 K
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card-best">
                <div class="metric-title">Deep Augmented CNN ★</div>
                <div class="metric-acc">98.39%</div>
                <div class="metric-sub">Test Accuracy</div>
                <div class="metric-sub" style="margin-top:0.6rem;">
                    F1: 0.98 &nbsp;·&nbsp; Precision: 0.99<br>
                    Recall: 0.99 &nbsp;·&nbsp; Params: ~800 K
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Architecture ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    st.code(
        """\
Input (64×64×1 grayscale)
    │
    ├─ Conv2D(32,  3×3, ReLU) → BatchNorm → MaxPool(2×2)
    ├─ Conv2D(64,  3×3, ReLU) → BatchNorm → MaxPool(2×2)
    ├─ Conv2D(128, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
    │  [Deep model adds: Conv2D(256, 3×3, ReLU) → BatchNorm → MaxPool(2×2)]
    │
    ├─ Flatten → Dense(128, ReLU) → Dropout(0.5)
    └─ Dense(1, Sigmoid) → P(drowsy) ∈ [0, 1]""",
        language="text",
    )

    # ── Training config ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Training Configuration")
    cfg_l, cfg_r = st.columns(2, gap="large")

    with cfg_l:
        st.markdown(
            """
            <div class="glass-card">
                <b style="color:#00D4FF;">Optimisation</b><br><br>
                Optimizer: <strong>Adam</strong> (lr = 1e-3)<br>
                Loss: <strong>Binary Cross-Entropy</strong><br>
                Batch size: <strong>64</strong><br>
                Early stopping on <code>val_loss</code> (patience 3)<br>
                Input: <strong>64×64×1</strong> grayscale
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cfg_r:
        st.markdown(
            """
            <div class="glass-card">
                <b style="color:#00D4FF;">Data Augmentation</b><br><br>
                RandomFlip (horizontal)<br>
                RandomRotation (±5°)<br><br>
                <b style="color:#00D4FF;">Dataset</b><br><br>
                MRL Eye Dataset — 84,898 IR images<br>
                Split: 60 / 20 / 20 (train / val / test)<br>
                Labels: <strong>awake = 0</strong> &nbsp;·&nbsp; <strong>sleepy = 1</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📦 Dataset")
    ds_l, ds_r = st.columns(2, gap="large")

    with ds_l:
        st.markdown(
            """
            <div class="glass-card">
                <b style="color:#00D4FF;">MRL Eye Dataset</b><br><br>
                84,898 PNG grayscale infrared eye crops<br>
                Binary labels: <strong>awake / sleepy</strong><br>
                ~50 / 50 class balance<br>
                Source: Kaggle
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ds_r:
        st.markdown(
            """
            <div class="glass-card">
                <b style="color:#00D4FF;">Access</b><br><br>
                Kaggle dataset:<br>
                <code>akashshingha850/mrl-eye-dataset</code><br><br>
                MRL Eye Dataset — Machine Vision Research Laboratory,
                Brno University of Technology
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="disclaimer">⚠️ Research prototype only. '
        "Not for use in real vehicles without safety validation.</div>",
        unsafe_allow_html=True,
    )
