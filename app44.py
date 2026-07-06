"""
AI-Powered Crop Health Monitoring System
=========================================
A Streamlit application for detecting sugarcane leaf diseases using a
MobileNetV3Small deep learning model.

Final Year Project — Annesha Panda, Suman Das, Mritunjoy Paul
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

from disease_info import DISEASE_INFO
from leaf_validation import validate_leaf
from utlis import preprocess_image

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("crop_health_app")

# =====================================================
# CONFIG
# =====================================================
@dataclass(frozen=True)
class Config:
    PAGE_TITLE: str = "AI-Powered Crop Health Monitoring"
    PAGE_ICON: str = "🌿"
    MODEL_PATH: str = "best_model.keras"
    MODEL_NAME: str = "MobileNetV3Small"
    MODEL_ACCURACY: str = "88.24%"
    CONFIDENCE_THRESHOLD: float = 85.0
    MAX_UPLOAD_FILES: int = 20
    ALLOWED_TYPES: Tuple[str, ...] = ("jpg", "jpeg", "png")
    GRADCAM_ALPHA: float = 0.45
    GRADCAM_ACTIVE_THRESHOLD: float = 0.5
    SEVERITY_MILD_MAX: float = 15.0
    SEVERITY_MODERATE_MAX: float = 40.0
    SANITY_MIN_CONFIDENCE: float = 35.0
    CLASS_NAMES: Tuple[str, ...] = (
        "Banded Chlorosis",
        "Brown Spot",
        "BrownRust",
        "Grassy shoot",
        "Pokkah Boeng",
        "Sett Rot",
        "Viral Disease",
        "Yellow Leaf",
        "smut",
    )


CFG = Config()

st.set_page_config(
    page_title=CFG.PAGE_TITLE,
    page_icon=CFG.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# THEME / CUSTOM CSS
# =====================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet">

<style>
:root{
    --cane-green:#1B3B2F;
    --fresh-blade:#2F6B32;
    --jaggery-gold:#A8710F;
    --parchment:#F2F4EC;
    --parchment-2:#E8EBDD;
    --rust:#8F2E22;
    --ink:#0D140F;
    --ink-soft:#2B342E;
}

html, body, [class*="css"]  { font-family: 'Manrope', sans-serif; color: var(--ink); }
.stApp{ background-color: var(--parchment); color: var(--ink); }
.stApp p, .stApp li, .stApp span, .stApp label { color: var(--ink); }
.stMarkdown, .stCaption, [data-testid="stCaptionContainer"] { color: var(--ink-soft) !important; }

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

/* ---------- Remove Streamlit's default rainbow/blue top decoration bar ---------- */
[data-testid="stDecoration"]{ display:none !important; }
header[data-testid="stHeader"]{
    background: transparent !important;
    box-shadow: none !important;
}

/* ---------- Recolor default blue accent widgets to match theme ---------- */
/* Sliders */
div[data-testid="stSlider"] [role="slider"]{
    background-color: var(--cane-green) !important;
    border-color: var(--cane-green) !important;
}
div[data-testid="stSlider"] .rc-slider-track,
div[data-baseweb="slider"] div[style*="background: rgb(255"]{
    background: var(--fresh-blade) !important;
}
/* Toggles / switches */
div[data-testid="stCheckbox"] label div[data-checked="true"],
div[role="switch"][aria-checked="true"]{
    background-color: var(--cane-green) !important;
}
/* Checkboxes */
span[data-baseweb="checkbox"] div[aria-checked="true"]{
    background-color: var(--cane-green) !important;
    border-color: var(--cane-green) !important;
}
/* Radio buttons (selected dot) */
div[role="radiogroup"] label div:first-child > div{
    border-color: var(--cane-green) !important;
}
div[role="radiogroup"] label div[aria-checked="true"] > div:first-child{
    background-color: var(--cane-green) !important;
}
/* Tabs underline / active tab */
button[data-baseweb="tab"][aria-selected="true"]{
    color: var(--cane-green) !important;
}
div[data-baseweb="tab-highlight"]{
    background-color: var(--jaggery-gold) !important;
}
/* Buttons */
button[kind="primary"], button[data-testid="baseButton-primary"]{
    background-color: var(--cane-green) !important;
    border-color: var(--cane-green) !important;
    color: #fff !important;
}
/* Focus outlines */
*:focus{ outline-color: var(--fresh-blade) !important; }

.hero-wrap{ text-align:center; padding: 8px 0 4px; }
.hero-eyebrow{
    font-family:'IBM Plex Mono', monospace;
    font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase;
    color: var(--fresh-blade); font-weight:600;
}
.main-title{
    font-family:'Fraunces', serif; font-size:2.6rem; font-weight:700;
    color: var(--cane-green); margin: 6px 0 4px;
}
.sub-title{ color: var(--ink-soft); font-size:1.05rem; max-width: 560px; margin: 0 auto; }

.metric-card{
    background:#fff; border-radius:16px; padding:20px 18px;
    box-shadow: 0 1px 2px rgba(27,59,47,0.06), 0 10px 24px rgba(27,59,47,0.07);
    text-align:center;
}
.metric-icon{ font-size:1.6rem; margin-bottom:6px; }
.metric-value{ font-family:'Fraunces', serif; font-size:1.7rem; font-weight:700; color: var(--cane-green); }
.metric-label{
    font-family:'IBM Plex Mono', monospace; font-size:0.72rem;
    letter-spacing:0.05em; text-transform:uppercase; color: var(--ink-soft);
}

section[data-testid="stFileUploader"]{
    background:#fff; border: 2px dashed rgba(27,59,47,0.25); border-radius:16px; padding: 14px;
}

.result-card{
    background:#fff; border-radius:18px; padding:22px;
    box-shadow: 0 1px 2px rgba(27,59,47,0.06), 0 10px 24px rgba(27,59,47,0.07);
    margin-bottom:6px;
}
.result-disease{ font-family:'Fraunces', serif; font-size:1.5rem; color: var(--cane-green); font-weight:700; margin:0 0 2px; }
.result-label{
    font-family:'IBM Plex Mono', monospace; font-size:0.7rem;
    text-transform:uppercase; letter-spacing:0.06em; color: var(--ink-soft);
}
.confidence-pill{
    display:inline-block;
    background: radial-gradient(circle at 30% 30%, #E3A94E, var(--jaggery-gold));
    color:#fff; font-family:'IBM Plex Mono', monospace; font-weight:600;
    padding:6px 16px; border-radius:999px; font-size:0.95rem;
}
.low-confidence-pill{
    display:inline-block;
    background: radial-gradient(circle at 30% 30%, #C97A6E, var(--rust));
    color:#fff; font-family:'IBM Plex Mono', monospace; font-weight:600;
    padding:6px 16px; border-radius:999px; font-size:0.95rem;
}
.warning-banner{
    background:#FBF3E4; border:1px solid rgba(201,138,44,0.35); border-radius:10px;
    padding:8px 12px; margin-top:10px; font-size:0.85rem; color: var(--ink-soft);
}
.severity-badge{
    display:inline-block; font-family:'IBM Plex Mono', monospace; font-weight:600;
    font-size:0.78rem; letter-spacing:0.04em; text-transform:uppercase;
    padding:4px 12px; border-radius:999px; margin-left:8px; color:#fff;
}
.severity-mild{ background:#4C8C4A; }
.severity-moderate{ background:#C98A2C; }
.severity-severe{ background:#B34B3C; }
.compare-card{
    background:#fff; border-radius:14px; padding:12px;
    box-shadow: 0 1px 2px rgba(27,59,47,0.06), 0 8px 18px rgba(27,59,47,0.07);
    text-align:center;
}
.compare-card .name{ font-weight:600; color: var(--cane-green); font-size:0.85rem; margin-top:6px; }
.compare-card .meta{ color: var(--ink-soft); font-size:0.78rem; }

.invalid-card{
    background:#FBEEEC; border:1px solid rgba(179,75,60,0.25); border-radius:16px;
    padding:20px; text-align:center;
}
.invalid-card h4{ color: var(--rust); margin:6px 0 4px; }
.invalid-card p{ color: var(--ink-soft); margin:0; }

.error-card{
    background:#FBEEEC; border:1px solid rgba(179,75,60,0.35); border-radius:16px;
    padding:18px; text-align:left;
}
.error-card h4{ color: var(--rust); margin:0 0 6px; }

.info-card{ background: var(--parchment); border-radius:12px; padding:14px 16px; margin-top:10px; }
.info-card h5{
    font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;
    color: var(--fresh-blade); margin:0 0 6px;
}
.info-card p{ margin:0; color: var(--ink); line-height:1.5; font-size:0.92rem; }

section[data-testid="stSidebar"]{ background: var(--cane-green); }
section[data-testid="stSidebar"] * { color: #F2F4EC !important; }
section[data-testid="stSidebar"] hr{ border-color: rgba(255,255,255,0.15); }

hr{ border-color: var(--parchment-2); }
</style>
""", unsafe_allow_html=True)

DARK_MODE_CSS = """
<style>
.stApp{ background-color:#12160F !important; color:#EAEDE3 !important; }
.stApp p, .stApp li, .stApp span, .stApp label,
.stMarkdown, .stCaption, [data-testid="stCaptionContainer"] { color:#D3D8C8 !important; }
.main-title, .metric-value, .result-disease { color:#8FD9A8 !important; }
.hero-eyebrow, .info-card h5 { color:#6FBF73 !important; }
.sub-title, .metric-label, .result-label { color:#AEB6A0 !important; }
.metric-card, .result-card { background:#1B2117 !important; box-shadow:0 1px 2px rgba(0,0,0,0.45), 0 10px 24px rgba(0,0,0,0.4) !important; }
.info-card{ background:#171C14 !important; }
.info-card p{ color:#EAEDE3 !important; }
.warning-banner{ background:#2A2314 !important; color:#E3D3AE !important; border-color: rgba(227,169,78,0.35) !important; }
section[data-testid="stFileUploader"]{ background:#1B2117 !important; border-color: rgba(143,217,168,0.3) !important; }
.invalid-card{ background:#2A1712 !important; border-color: rgba(224,122,99,0.4) !important; }
.invalid-card p{ color:#D3D8C8 !important; }
.invalid-card h4{ color:#E3937E !important; }
.error-card{ background:#2A1712 !important; border-color: rgba(224,122,99,0.5) !important; }
.error-card h4{ color:#E3937E !important; }
.severity-badge{ filter:brightness(1.05); }
hr{ border-color:#232A1E !important; }
</style>
"""


def apply_dark_mode(enabled: bool) -> None:
    """Inject the dark-mode CSS override on top of the base theme."""
    if enabled:
        st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)


# =====================================================
# MODEL LOADING
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Optional[tf.keras.Model]:
    """Load and cache the trained Keras model. Returns None on failure."""
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully from %s", model_path)
        return model
    except (OSError, IOError, ValueError) as exc:
        logger.exception("Failed to load model from %s", model_path)
        st.session_state["model_load_error"] = str(exc)
        return None


def run_inference(model: tf.keras.Model, rgb_input: np.ndarray) -> np.ndarray:
    """Run a forward pass and return the class-probability vector."""
    prediction = model.predict(rgb_input, verbose=0)
    return prediction[0]


@st.cache_resource(show_spinner=False)
def get_last_conv_layer_name(_model: tf.keras.Model) -> Optional[str]:
    """Find the name of the last 4D-output (convolutional) layer for Grad-CAM."""
    for layer in reversed(_model.layers):
        try:
            shape = layer.output_shape
        except AttributeError:
            continue
        if isinstance(shape, list):
            shape = shape[0]
        if shape is not None and len(shape) == 4:
            return layer.name
    return None


def compute_gradcam(
    model: tf.keras.Model, rgb_input: np.ndarray, class_index: int, layer_name: str
) -> Optional[np.ndarray]:
    """Compute a Grad-CAM heatmap (values in [0, 1]) for the given class."""
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(rgb_input)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val <= 0:
            return np.zeros(heatmap.shape, dtype=np.float32)
        heatmap = heatmap / max_val
        return heatmap.numpy()
    except Exception:
        logger.exception("Grad-CAM computation failed for layer %s", layer_name)
        return None


def to_pil_image(image) -> Image.Image:
    """Normalize whatever preprocess_image returns into a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.array(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a Grad-CAM heatmap onto the original leaf image."""
    base = np.array(image).astype(np.float32)
    h, w = base.shape[:2]
    heat_img = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR)
    heat = np.array(heat_img).astype(np.float32) / 255.0

    color = np.zeros_like(base)
    color[..., 0] = 200
    color[..., 1] = 40 + (1 - heat) * 110
    color[..., 2] = 20

    heat3 = np.stack([heat] * 3, axis=-1)
    blended = base * (1 - alpha * heat3) + color * (alpha * heat3)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def compute_severity(heatmap: np.ndarray) -> Tuple[str, float]:
    """Estimate disease severity from the proportion of strongly-activated pixels."""
    affected_pct = float(np.mean(heatmap > CFG.GRADCAM_ACTIVE_THRESHOLD)) * 100
    if affected_pct < CFG.SEVERITY_MILD_MAX:
        return "Mild", affected_pct
    if affected_pct < CFG.SEVERITY_MODERATE_MAX:
        return "Moderate", affected_pct
    return "Severe", affected_pct


# =====================================================
# SIDEBAR
# =====================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🌿 Project Information")
        st.markdown(
            "**AI-Powered Crop Health Monitoring**\n\n"
            "This project detects sugarcane diseases using "
            "Deep Learning and Computer Vision."
        )
        st.markdown("---")
        st.markdown(f"**Model**  \n{CFG.MODEL_NAME}")
        st.markdown(f"**Classes**  \n{len(CFG.CLASS_NAMES)} Disease Classes")
        st.markdown(f"**Accuracy**  \n{CFG.MODEL_ACCURACY}")
        st.markdown("""
**Technologies**

- TensorFlow
- Streamlit
- OpenCV
- MobileNetV3
""")
        st.markdown("---")
        with st.expander("⚙️ Settings", expanded=True):
            threshold = st.slider(
                "Low-confidence warning threshold (%)",
                min_value=50, max_value=99,
                value=int(CFG.CONFIDENCE_THRESHOLD), step=1,
                help="Predictions below this confidence are flagged for manual review.",
            )
            st.session_state["confidence_threshold"] = threshold

            dark_mode = st.toggle(
                "🌙 Dark mode",
                value=st.session_state.get("dark_mode", False),
            )
            st.session_state["dark_mode"] = dark_mode

            enable_gradcam = st.toggle(
                "🔥 Grad-CAM heatmap + severity",
                value=st.session_state.get("enable_gradcam", True),
                help="Highlights the leaf regions that most influenced the prediction and estimates severity from the affected area.",
            )
            st.session_state["enable_gradcam"] = enable_gradcam
        st.markdown("---")
        st.markdown("""
**Final Year Project**

Team Members
- Annesha Panda
- Suman Das
- Mritunjoy Paul
""")


# =====================================================
# HERO / HEADER
# =====================================================
def render_header() -> None:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">Sugarcane Disease Detection</div>
        <div class="main-title">AI-Powered Crop Health Monitoring System</div>
        <div class="sub-title">Early detection of sugarcane diseases using deep learning — upload a leaf photo to get an instant diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)


def render_dashboard() -> None:
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, "🎯", CFG.MODEL_ACCURACY, "Accuracy"),
        (col2, "🦠", str(len(CFG.CLASS_NAMES)), "Diseases"),
        (col3, "🤖", CFG.MODEL_NAME, "Model"),
        (col4, "📤", "Multiple", "Upload"),
    ]
    for col, icon, value, label in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_confidence_explainer() -> None:
    """A plain-language guide to interpreting confidence, top-3, and Grad-CAM/severity output."""
    with st.expander("❓ How to read these results"):
        st.markdown("""
**Confidence score** reflects how strongly the model favors its top prediction
relative to the other classes — it is a statistical measure, not a direct
measure of how visually severe the disease is on the leaf.

- **Above the threshold** (set in the sidebar): the model is reasonably certain.
  Still worth a quick visual sanity check, especially for treatment decisions.
- **Below the threshold**: the leaf may show mixed or unfamiliar symptoms, poor
  lighting, or overlap between disease classes. Treat the result as a starting
  point, not a diagnosis.

**Top 3 predictions** show the next most likely classes. A close second place
is a signal the case may be borderline and worth a second photo or expert look.

**Grad-CAM heatmap & severity** show *where* the model focused and estimate
*how much* of the leaf is strongly activated — they describe the model's
attention, not a clinical severity grading. Always confirm treatment decisions
with an agronomist or extension officer.

**Feedback** — use the ✅ / ❌ buttons under each result to tell the model
whether it got it right. This is logged for this session only and can be
exported as a feedback report to help track real-world accuracy over time.
""")


# =====================================================
# CHART HELPER
# =====================================================
def plot_probabilities(class_names: Tuple[str, ...], probs: np.ndarray, top_idx: int) -> go.Figure:
    """Build a horizontal bar chart of class probabilities, highlighting the top prediction."""
    order = np.argsort(probs)
    names = [class_names[i] for i in order]
    values = [float(probs[i]) * 100 for i in order]

    bar_colors = []
    for i in order:
        if i == top_idx:
            bar_colors.append("#C98A2C")
        else:
            intensity = 0.30 + 0.70 * float(probs[i])
            bar_colors.append(f"rgba(27,59,47,{intensity:.2f})")

    ticktext = [f"<b>{n}</b>" if i == top_idx else n for i, n in zip(order, names)]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="#1B3B2F", width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#0D140F", size=13, family="IBM Plex Mono, monospace"),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=50, t=10, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.35,
        showlegend=False,
        xaxis=dict(
            title=dict(text="Probability (%)", font=dict(color="#0D140F", size=13, family="Manrope, sans-serif")),
            range=[0, 112],
            gridcolor="#E3E7D8",
            zeroline=False,
            tickfont=dict(color="#0D140F", size=12),
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=names,
            ticktext=ticktext,
            tickfont=dict(color="#0D140F", size=13),
        ),
        font=dict(family="Manrope, sans-serif", color="#0D140F"),
    )
    return fig


# =====================================================
# VALIDATION HELPERS
# =====================================================
def validate_upload_batch(files: List) -> Tuple[List, List[str]]:
    """Filter uploaded files by size / count limits before processing."""
    warnings = []
    valid_files = files

    if len(files) > CFG.MAX_UPLOAD_FILES:
        warnings.append(
            f"Only the first {CFG.MAX_UPLOAD_FILES} of {len(files)} images will be "
            f"processed in this batch."
        )
        valid_files = files[: CFG.MAX_UPLOAD_FILES]

    return valid_files, warnings


def safe_preprocess(uploaded_file) -> Tuple[Optional[object], Optional[np.ndarray], Optional[str]]:
    """Wrap preprocess_image with error handling for corrupt/unreadable files."""
    try:
        image, rgb_input, extra = preprocess_image(uploaded_file)
        return image, rgb_input, None
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        logger.exception("Preprocessing failed for %s", uploaded_file.name)
        return None, None, str(exc)


# =====================================================
# SINGLE-IMAGE RESULT RENDERING
# =====================================================
def render_feedback_widget(image_name: str, predicted_disease: str, confidence: float, row_index: int) -> Optional[dict]:
    """Capture user agreement/disagreement with a prediction for trust tracking."""
    st.markdown("**📝 Was this prediction correct?**")
    verdict = st.radio(
        "Feedback",
        options=["Not reviewed", "✅ Correct", "❌ Incorrect"],
        index=0,
        horizontal=True,
        key=f"feedback_choice_{row_index}",
        label_visibility="collapsed",
    )

    if verdict == "Not reviewed":
        return None

    record = {
        "Image": image_name,
        "Predicted": predicted_disease,
        "Confidence (%)": round(confidence, 2),
        "User Verdict": "Correct" if verdict == "✅ Correct" else "Incorrect",
        "Corrected Disease": "",
        "Note": "",
    }

    if verdict == "❌ Incorrect":
        corrected = st.selectbox(
            "What is the correct condition?",
            options=list(CFG.CLASS_NAMES),
            key=f"feedback_correction_{row_index}",
        )
        note = st.text_input(
            "Optional note (symptoms, growth stage, field conditions, etc.)",
            key=f"feedback_note_{row_index}",
        )
        record["Corrected Disease"] = corrected
        record["Note"] = note

    st.caption("Logged for this session — included in the exportable feedback report below.")
    return record


def render_result(
    uploaded_file,
    image,
    rgb_input: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    model: tf.keras.Model,
    conv_layer_name: Optional[str],
    enable_gradcam: bool,
    row_index: int,
) -> Tuple[dict, Optional[Image.Image], Optional[dict]]:
    """Render the result card, heatmap, top-3 predictions, and probability chart for one image."""
    pred_index = int(np.argmax(probs))
    disease = CFG.CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index]) * 100
    top3_indices = np.argsort(probs)[-3:][::-1]

    pil_image = to_pil_image(image)
    heatmap = None
    severity_label, affected_pct = None, None
    display_image = pil_image

    if enable_gradcam and conv_layer_name is not None:
        heatmap = compute_gradcam(model, rgb_input, pred_index, conv_layer_name)
        if heatmap is not None:
            severity_label, affected_pct = compute_severity(heatmap)
            display_image = overlay_heatmap(pil_image, heatmap, alpha=CFG.GRADCAM_ALPHA)

    col1, col2 = st.columns([1, 1])

    with col1:
        if heatmap is not None:
            show_heatmap = st.toggle(
                "Show Grad-CAM overlay", value=True, key=f"heatmap_toggle_{row_index}"
            )
            st.image(
                display_image if show_heatmap else pil_image,
                caption=uploaded_file.name,
                use_container_width=True,
            )
        else:
            st.image(pil_image, caption=uploaded_file.name, use_container_width=True)

        with st.expander("🔍 View full size"):
            st.image(pil_image, use_container_width=False)

        add_to_compare = st.checkbox("➕ Add to comparison tray", key=f"compare_{row_index}")

    with col2:
        pill_class = "confidence-pill" if confidence >= threshold else "low-confidence-pill"
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Detected Condition</div>
            <div class="result-disease">{disease}</div>
            <span class="{pill_class}">{confidence:.2f}% confidence</span>
            {f'<span class="severity-badge severity-{severity_label.lower()}">{severity_label} severity</span>' if severity_label else ''}
        </div>
        """, unsafe_allow_html=True)

        if affected_pct is not None:
            st.caption(f"🔥 Grad-CAM estimates ~{affected_pct:.1f}% of the leaf region is strongly activated.")

        if confidence < threshold:
            st.markdown(
                '<div class="warning-banner">⚠️ Confidence is below the review '
                'threshold — consider re-taking the photo in better lighting or '
                'having an agronomist verify this result.</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("**Top 3 Predictions**")
        for idx in top3_indices:
            st.write(f"{CFG.CLASS_NAMES[idx]} — {probs[idx] * 100:.2f}%")

        if disease in DISEASE_INFO:
            info = DISEASE_INFO[disease]
            st.markdown(f"""
            <div class="info-card">
                <h5>Description</h5>
                <p>{info.get("description", "No description available.")}</p>
            </div>
            <div class="info-card">
                <h5>Recommended Treatment</h5>
                <p>{info.get("treatment", "No treatment information available.")}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card">
                <h5>Description</h5>
                <p>Detailed information for this class is not yet available.</p>
            </div>
            """, unsafe_allow_html=True)

    st.subheader(f"Prediction Probabilities — {uploaded_file.name}")
    st.plotly_chart(
        plot_probabilities(CFG.CLASS_NAMES, probs, pred_index),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    feedback_record = render_feedback_widget(uploaded_file.name, disease, confidence, row_index)

    result = {
        "Image": uploaded_file.name,
        "Disease": disease,
        "Confidence (%)": round(confidence, 2),
        "Severity": severity_label or "N/A",
        "Needs Review": confidence < threshold,
    }

    compare_snapshot = None
    if add_to_compare:
        compare_snapshot = {
            "name": uploaded_file.name,
            "image": pil_image,
            "disease": disease,
            "confidence": confidence,
        }

    return result, compare_snapshot, feedback_record


# =====================================================
# BATCH SUMMARY
# =====================================================
def render_batch_summary(results: List[dict]) -> None:
    st.divider()
    st.header("📊 Batch Prediction Summary")

    if not results:
        st.info("No valid leaf images were found in this batch.")
        return

    df = pd.DataFrame(results)

    search_col, disease_col, conf_col = st.columns([1.2, 1.2, 1.4])
    with search_col:
        search_term = st.text_input("🔎 Search by image name", value="", placeholder="e.g. leaf_03.jpg")
    with disease_col:
        disease_options = sorted(df["Disease"].unique().tolist())
        selected_diseases = st.multiselect("Filter by disease", options=disease_options, default=[])
    with conf_col:
        conf_range = st.slider("Confidence range (%)", min_value=0, max_value=100, value=(0, 100))

    filtered_df = df.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df["Image"].str.contains(search_term, case=False, na=False)]
    if selected_diseases:
        filtered_df = filtered_df[filtered_df["Disease"].isin(selected_diseases)]
    filtered_df = filtered_df[
        (filtered_df["Confidence (%)"] >= conf_range[0]) & (filtered_df["Confidence (%)"] <= conf_range[1])
    ]

    st.dataframe(filtered_df, use_container_width=True)
    st.caption(f"Showing {len(filtered_df)} of {len(df)} result(s).")

    flagged = df[df["Needs Review"]] if "Needs Review" in df.columns else pd.DataFrame()
    if not flagged.empty:
        st.warning(f"⚠️ {len(flagged)} image(s) fell below the confidence threshold and may need manual review.")

    st.subheader("Disease Distribution")
    disease_counts = filtered_df["Disease"].value_counts()
    if disease_counts.empty:
        st.info("No results match the current filters.")
    else:
        st.bar_chart(disease_counts, color="#4C8C4A")

    csv = filtered_df.to_csv(index=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name=f"prediction_results_{timestamp}.csv",
        mime="text/csv",
    )


def render_comparison_tray(snapshots: List[dict]) -> None:
    """Render a side-by-side gallery of images the user flagged for comparison."""
    if not snapshots:
        return

    st.divider()
    st.header("🖼️ Comparison Tray")
    st.caption(f"{len(snapshots)} image(s) selected for side-by-side comparison.")

    cols = st.columns(min(len(snapshots), 4))
    for i, snap in enumerate(snapshots):
        with cols[i % len(cols)]:
            st.image(snap["image"], use_container_width=True)
            st.markdown(f"""
            <div class="compare-card">
                <div class="name">{snap['name']}</div>
                <div class="meta">{snap['disease']} — {snap['confidence']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)


def render_feedback_summary(feedback_records: List[dict]) -> None:
    """Summarize user-submitted correctness feedback and offer it as a downloadable report."""
    st.divider()
    st.header("🗳️ Feedback & Model Trust")

    if not feedback_records:
        st.info(
            "No feedback submitted yet. Use the ✅ / ❌ buttons under each result above "
            "to help track real-world accuracy."
        )
        return

    fb_df = pd.DataFrame(feedback_records)
    total = len(fb_df)
    correct_count = int((fb_df["User Verdict"] == "Correct").sum())
    accuracy = (correct_count / total) * 100 if total else 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("User-Reported Accuracy", f"{accuracy:.1f}%")
    with col2:
        st.metric("Reviewed Predictions", str(total))
    with col3:
        st.metric("Corrections Logged", str(total - correct_count))

    st.dataframe(fb_df, use_container_width=True)

    csv = fb_df.to_csv(index=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="📥 Download Feedback Report",
        data=csv,
        file_name=f"feedback_report_{timestamp}.csv",
        mime="text/csv",
    )
    st.caption(
        "Feedback is kept for this browser session only — export the CSV to preserve it "
        "or feed it back into future model retraining."
    )


# =====================================================
# MAIN APP
# =====================================================
def main() -> None:
    render_sidebar()
    apply_dark_mode(st.session_state.get("dark_mode", False))

    render_header()
    st.write("")
    render_dashboard()
    st.write("")
    st.divider()

    model = load_model(CFG.MODEL_PATH)
    if model is None:
        st.markdown(f"""
        <div class="error-card">
            <h4>⚠️ Model failed to load</h4>
            <p>The disease classification model could not be loaded from
            <code>{CFG.MODEL_PATH}</code>. Please verify the model file is present
            and compatible, then refresh the page.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    enable_gradcam = st.session_state.get("enable_gradcam", True)
    conv_layer_name = get_last_conv_layer_name(model) if enable_gradcam else None
    if enable_gradcam and conv_layer_name is None:
        st.info("ℹ️ Grad-CAM is enabled but no convolutional layer could be found in this model, so heatmaps will be skipped.")

    render_confidence_explainer()

    st.markdown("### 📤 Upload or Capture Leaf Images")
    tab_upload, tab_camera = st.tabs(["📁 Upload Files", "📷 Use Camera"])

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Drag and drop sugarcane leaf images, or click to browse",
            type=list(CFG.ALLOWED_TYPES),
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    with tab_camera:
        camera_file = st.camera_input("Take a photo of a sugarcane leaf", label_visibility="collapsed")

    files_to_process: List = list(uploaded_files) if uploaded_files else []
    if camera_file is not None:
        files_to_process.append(camera_file)

    if not files_to_process:
        st.markdown("""
        <div style="text-align:center; padding: 40px 0; color:#4B564D;">
            📤 Upload one or more sugarcane leaf images above, or take a photo, to begin prediction.
        </div>
        """, unsafe_allow_html=True)
        return

    threshold = st.session_state.get("confidence_threshold", CFG.CONFIDENCE_THRESHOLD)
    files_to_process, batch_warnings = validate_upload_batch(files_to_process)
    for warning in batch_warnings:
        st.warning(warning)

    results = []
    compare_snapshots = []
    feedback_records = []
    rejected_count = 0
    progress = st.progress(0, text="Starting batch analysis...")

    for i, uploaded_file in enumerate(files_to_process, start=1):
        st.divider()
        progress.progress(i / len(files_to_process), text=f"Analyzing {uploaded_file.name} ({i}/{len(files_to_process)})")

        image, rgb_input, error = safe_preprocess(uploaded_file)
        if error is not None:
            st.markdown(f"""
            <div class="invalid-card">
                <div style="font-size:1.8rem;">⚠️</div>
                <h4>Could Not Process — {uploaded_file.name}</h4>
                <p>This file appears to be corrupted or in an unsupported format.</p>
            </div>
            """, unsafe_allow_html=True)
            rejected_count += 1
            continue

        try:
            is_leaf, message = validate_leaf(image)
        except Exception:
            logger.exception("Leaf validation failed for %s", uploaded_file.name)
            is_leaf, message = False, "Unable to validate this image."

        if not is_leaf:
            st.markdown(f"""
            <div class="invalid-card">
                <div style="font-size:1.8rem;">⚠️</div>
                <h4>Invalid Image — {uploaded_file.name}</h4>
                <p>{message}</p>
                <p>Please upload a clear sugarcane leaf image.</p>
            </div>
            """, unsafe_allow_html=True)
            rejected_count += 1
            continue

        try:
            with st.spinner(f"Running model inference on {uploaded_file.name}..."):
                probs = run_inference(model, rgb_input)
        except Exception:
            logger.exception("Inference failed for %s", uploaded_file.name)
            st.markdown(f"""
            <div class="invalid-card">
                <div style="font-size:1.8rem;">⚠️</div>
                <h4>Prediction Failed — {uploaded_file.name}</h4>
                <p>An unexpected error occurred while analyzing this image. Please try again.</p>
            </div>
            """, unsafe_allow_html=True)
            rejected_count += 1
            continue

        top_confidence = float(np.max(probs)) * 100
        if top_confidence < CFG.SANITY_MIN_CONFIDENCE:
            st.markdown(f"""
            <div class="invalid-card">
                <div style="font-size:1.8rem;">⚠️</div>
                <h4>Invalid Image — {uploaded_file.name}</h4>
                <p>This does not appear to be a valid sugarcane leaf photo. The model's best
                guess is only {top_confidence:.1f}% confident — close to random guessing across
                {len(CFG.CLASS_NAMES)} disease classes.</p>
                <p>Please upload a clear, well-lit photo of a single sugarcane leaf.</p>
            </div>
            """, unsafe_allow_html=True)
            rejected_count += 1
            continue

        result, compare_snapshot, feedback_record = render_result(
            uploaded_file, image, rgb_input, probs, threshold,
            model, conv_layer_name, enable_gradcam, row_index=i,
        )
        results.append(result)
        if compare_snapshot is not None:
            compare_snapshots.append(compare_snapshot)
        if feedback_record is not None:
            feedback_records.append(feedback_record)

    progress.empty()

    if rejected_count:
        st.info(f"ℹ️ {rejected_count} of {len(files_to_process)} uploaded image(s) were rejected as invalid and excluded from the results below.")

    render_batch_summary(results)
    render_comparison_tray(compare_snapshots)
    render_feedback_summary(feedback_records)

    st.divider()
    st.caption("🌱 AI-Powered Crop Health Monitoring System | Final Year Project")


if __name__ == "__main__":
    main()