import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion-MNIST Classifier",
    #page_icon="👗",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Class labels ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot",
]

CLASS_EMOJIS = ["👕", "👖", "🧥", "👗", "🥼", "👡", "👔", "👟", "👜", "🥾"]

# ── Load model (cached so it only loads once) ──────────────────────────────────
@st.cache_resource
def load_model():
    """
    Load the saved Keras model.
    Looks in models/ folder first (structured layout),
    then falls back to the same directory as app.py.
    """
    for path in ["notebook/models/best_lenet5_fashion.keras",
                 "models/best_lenet5_fashion.keras",
                 "../models/best_lenet5_fashion.keras",
                 "best_lenet5_fashion.keras"]:
        if os.path.exists(path):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(path)
                return model, path
            except Exception as e:
                st.error(f"Error loading model from {path}: {e}")
                return None, None
    return None, None

# ── Image preprocessing ────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a 28×28 grayscale array
    normalised to [0, 1] with the correct shape for the model.

    Fashion-MNIST convention: white item on black background.
    Most real photos have dark item on white background, so we invert.
    """
    img = img.convert("L")                          # convert to grayscale
    img = ImageOps.invert(img)                      # invert: dark bg → bright item
    img = img.resize((28, 28), Image.LANCZOS)       # resize to model input size
    arr = np.array(img, dtype="float32") / 255.0    # normalise to [0, 1]
    arr = arr.reshape(1, 28, 28, 1)                 # (batch, H, W, channels)
    return arr

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
  }
  .hero-banner h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
  }
  .hero-banner p { font-size: 0.95rem; opacity: 0.7; margin: 0; }

  .result-card {
    background: #f0f7ff;
    border: 1.5px solid #c2dbfc;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
  }
  .result-card .emoji  { font-size: 3rem; }
  .result-card .label  { font-size: 1.5rem; font-weight: 600; color: #1a3a6b; margin: 0.3rem 0; }
  .result-card .conf   { font-size: 0.95rem; color: #4a6fa5; }

  .bar-row {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 7px; font-size: 0.82rem;
  }
  .bar-label { width: 120px; text-align: right; color: #444; flex-shrink: 0; }
  .bar-bg    { flex: 1; background: #e8edf5; border-radius: 6px; height: 10px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 6px; transition: width 0.5s ease; }
  .bar-pct   { width: 42px; color: #555; font-variant-numeric: tabular-nums; }

  [data-testid="stFileUploader"] > div { border-radius: 12px !important; }
  [data-testid="stSidebar"] { background:#123 ; color:white }
</style>
""", unsafe_allow_html=True)

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>👗 Fashion-MNIST Classifier</h1>
  <p>LeNet-5 CNN · 10 clothing categories · Upload an image to classify</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Fashion Items")
    for emoji, name in zip(CLASS_EMOJIS, CLASS_NAMES):
        st.markdown(f"{emoji} &nbsp; {name}", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Tips for best results")
    st.markdown(
        "- Use a **clear, close-up** image of a single clothing item\n"
        "- Works best with a **plain or white background**\n"
        "- The model is trained on **28×28 grayscale** — resizing is automatic\n"
        "- Try photographing items laid flat for best accuracy"
    )
    st.markdown("---")
    st.caption("CCS 3572 / CSE 3582 · Deep Learning Mini Project · USJ Faculty of Computing")

# ── Load model ─────────────────────────────────────────────────────────────────
model, model_path = load_model()

if model is None:
    st.warning(
        "**Model file not found.**  \n"
        "Make sure `best_lenet5_fashion.keras` is in the `models/` folder.  \n"
        "Run the Jupyter notebook first to train and save the model."
    )
    st.info(
        "**Demo mode** — The app UI is fully functional. "
        "Add the model file to `models/` to enable real predictions."
    )
else:
    st.success(f"Model loaded from `{model_path}`", icon="✅")

# ── Upload & predict ───────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### Upload Image")
    uploaded = st.file_uploader(
        "Choose a clothing image",
        type=["jpg", "jpeg", "png", "webp"],
        help="PNG, JPG, JPEG, or WebP · any size (auto-resized to 28×28)",
    )

    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

        # Show preprocessed version so user understands the pipeline
        with st.expander("See preprocessed input (28×28)"):
            arr_display = preprocess_image(pil_img)
            st.image(
                arr_display.reshape(28, 28),
                caption="What the model actually sees",
                width=140,
                clamp=True
            )

with col2:
    st.markdown("#### Prediction")

    if uploaded is None:
        st.markdown(
            "<div style='padding:3rem 1rem; text-align:center; "
            "color:#aaa; border:1.5px dashed #ddd; border-radius:12px;'>"
            "Upload an image to get started</div>",
            unsafe_allow_html=True,
        )
    elif model is None:
        st.info("Train and save the model first to see predictions.")
    else:
        with st.spinner("Classifying…"):
            arr   = preprocess_image(pil_img)
            probs = model.predict(arr, verbose=0)[0]   # shape (10,)

        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # ── Top result card ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card">
          <div class="emoji">{CLASS_EMOJIS[pred_idx]}</div>
          <div class="label">{CLASS_NAMES[pred_idx]}</div>
          <div class="conf">Confidence: <strong>{confidence:.1%}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bar chart — sorted highest first ────────────────────────
        st.markdown("<br>**All class probabilities**", unsafe_allow_html=True)

        sorted_idx = np.argsort(probs)[::-1]
        colors = [
            "#1d6feb", "#4a90d9", "#7ab3e8", "#a8cdf0",
            "#c8dff5", "#dde9f7", "#e8f0fa", "#f0f5fc", "#f5f8fe", "#fafcff"
        ]

        bars_html = ""
        for rank, i in enumerate(sorted_idx):
            pct   = float(probs[i])
            bar_w = max(1, round(pct * 100))
            bold  = "font-weight:600;" if i == pred_idx else ""
            bars_html += f"""
            <div class="bar-row">
              <div class="bar-label" style="{bold}">{CLASS_NAMES[i]}</div>
              <div class="bar-bg">
                <div class="bar-fill" style="width:{bar_w}%;background:{colors[rank]};"></div>
              </div>
              <div class="bar-pct">{pct:.1%}</div>
            </div>"""

        st.markdown(bars_html, unsafe_allow_html=True)

# ── How the model works ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### How the model works")

info_cols = st.columns(4)
steps = [
    ("🖼️", "Input",       "28×28 grayscale image normalised to [0, 1]"),
    ("🔬", "Conv layers", "6 then 16 filters extract edges, textures, shapes"),
    ("📦", "Pooling",     "Average pooling reduces size, adds shift-tolerance"),
    ("🎯", "Output",      "Softmax over 10 classes → predicted category"),
]
for col, (icon, title, desc) in zip(info_cols, steps):
    with col:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:1.8rem'>{icon}</div>"
            f"<div style='font-weight:600;font-size:0.9rem;margin:6px 0 4px'>{title}</div>"
            f"<div style='font-size:0.78rem;color:#777;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: LeNet-5 &nbsp;|&nbsp; "
    "Dataset: Fashion-MNIST (70,000 images, 10 classes) &nbsp;|&nbsp; "
    "Framework: TensorFlow / Keras &nbsp;|&nbsp; "
    "CCS 3572 / CSE 3582 · USJ"
)
