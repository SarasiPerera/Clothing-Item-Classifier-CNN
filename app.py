import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clothing Item Classifier",
    page_icon="👗",
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
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;500;600;700&display=swap');

  :root {
    --ink:        #0d0d0d;
    --parchment:  #faf7f2;
    --accent:     #c8a97e;
    --accent-dim: #e8ddd0;
    --muted:      #8a8070;
    --surface:    #ffffff;
    --border:     #e2dbd0;
    --deep:       #1c1610;
    --highlight:  #f5efe6;
  }

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--parchment) !important;
    color: var(--ink);
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--deep) !important;
    border-right: 1px solid #2e2620;
  }
  [data-testid="stSidebar"] * {
    color: #d4c9b8 !important;
  }
  [data-testid="stSidebar"] .sidebar-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--accent) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #3a3028;
  }
  [data-testid="stSidebar"] hr {
    border-color: #2e2620 !important;
  }
  [data-testid="stSidebar"] .stCaption {
    color: #6b5f50 !important;
    font-size: 0.72rem !important;
    line-height: 1.5 !important;
  }

  /* ── Hero ── */
  .hero-wrap {
    position: relative;
    overflow: hidden;
    background: var(--deep);
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2.5rem;
    text-align: center;
  }
  .hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(200,169,126,0.18) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(200,169,126,0.10) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
  }
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #f5efe6;
    line-height: 1.15;
    margin: 0 0 0.7rem;
  }
  .hero-title em {
    font-style: italic;
    color: var(--accent);
  }
  .hero-sub {
    font-size: 0.85rem;
    color: #8a7d6b;
    letter-spacing: 0.05em;
    margin: 0;
  }
  .hero-divider {
    width: 48px;
    height: 1px;
    background: var(--accent);
    margin: 1.2rem auto 1rem;
    opacity: 0.6;
  }

  /* ── Section headers ── */
  .section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] > div {
    border-radius: 14px !important;
    border: 1.5px dashed var(--accent-dim) !important;
    background: var(--highlight) !important;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploader"] > div:hover {
    border-color: var(--accent) !important;
  }

  /* ── Empty prediction state ── */
  .empty-pred {
    border: 1.5px dashed var(--border);
    border-radius: 14px;
    padding: 3rem 1rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
    background: var(--highlight);
    line-height: 1.7;
  }
  .empty-pred .empty-icon {
    font-size: 2.4rem;
    margin-bottom: 0.8rem;
    display: block;
    opacity: 0.5;
  }

  /* ── Result card ── */
  .result-card {
    background: var(--deep);
    border-radius: 18px;
    padding: 2rem 1.5rem 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
  }
  .result-card::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(200,169,126,0.15) 0%, transparent 70%);
    border-radius: 50%;
  }
  .result-card .rc-emoji {
    font-size: 3.2rem;
    line-height: 1;
    margin-bottom: 0.6rem;
  }
  .result-card .rc-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #f5efe6;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
  }
  .result-card .rc-badge {
    display: inline-block;
    background: rgba(200,169,126,0.18);
    border: 1px solid rgba(200,169,126,0.35);
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--accent);
    margin-top: 0.2rem;
  }

  /* ── Probability bars ── */
  .prob-section-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.9rem;
  }
  .bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 0.8rem;
  }
  .bar-label {
    width: 112px;
    text-align: right;
    color: var(--ink);
    flex-shrink: 0;
    line-height: 1.2;
  }
  .bar-label.top {
    font-weight: 700;
    color: var(--deep);
  }
  .bar-bg {
    flex: 1;
    background: var(--accent-dim);
    border-radius: 999px;
    height: 7px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s cubic-bezier(0.22,1,0.36,1);
  }
  .bar-pct {
    width: 40px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    font-size: 0.75rem;
  }
  .bar-pct.top {
    color: var(--ink);
    font-weight: 600;
  }

  /* ── How it works cards ── */
  .hw-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 1rem;
  }
  .hw-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 0.9rem;
    text-align: center;
  }
  .hw-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
  .hw-title {
    font-weight: 700;
    font-size: 0.8rem;
    color: var(--deep);
    margin-bottom: 0.35rem;
    letter-spacing: 0.03em;
  }
  .hw-desc {
    font-size: 0.72rem;
    color: var(--muted);
    line-height: 1.5;
  }

  /* ── Alerts ── */
  [data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 3px !important;
  }

  /* ── Footer ── */
  .footer-bar {
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    font-size: 0.72rem;
    color: var(--muted);
    text-align: center;
    letter-spacing: 0.04em;
  }
  .footer-bar span {
    color: var(--accent-dim);
    margin: 0 0.5rem;
  }

  /* ── Preprocessed expander ── */
  [data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--highlight) !important;
  }

  /* ── Success/warning boxes ── */
  div[data-testid="stAlert"] {
    font-size: 0.83rem !important;
  }

  /* ── Image display ── */
  [data-testid="stImage"] img {
    border-radius: 12px;
  }
</style>
""", unsafe_allow_html=True)

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">Deep Learning · LeNet-5 CNN</div>
  <h1 class="hero-title">Fashion<em>MNIST</em><br>Classifier</h1>
  <div class="hero-divider"></div>
  <p class="hero-sub">10 clothing categories &nbsp;·&nbsp; Upload an image to classify</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Categories</div>', unsafe_allow_html=True)
    for emoji, name in zip(CLASS_EMOJIS, CLASS_NAMES):
        st.markdown(
            f"<div style='padding:5px 0; display:flex; gap:10px; align-items:center;'>"
            f"<span style='font-size:1.1rem'>{emoji}</span>"
            f"<span style='font-size:0.84rem'>{name}</span></div>",
            unsafe_allow_html=True
        )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; font-weight:700; letter-spacing:0.15em; "
        "text-transform:uppercase; color:#8a7d6b; margin-bottom:0.6rem;'>Tips</div>",
        unsafe_allow_html=True
    )
    for tip in [
        "Clear, close-up shot of a **single item**",
        "Plain or **white background** works best",
        "Auto-resized to **28×28 grayscale**",
        "Lay items **flat** for best accuracy",
    ]:
        st.markdown(
            f"<div style='font-size:0.78rem; color:#9a8d7b; padding:3px 0 3px 8px; "
            f"border-left:2px solid #3a3028; margin-bottom:6px;'>{tip}</div>",
            unsafe_allow_html=True
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
    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a clothing image",
        type=["jpg", "jpeg", "png", "webp"],
        help="PNG, JPG, JPEG, or WebP · any size (auto-resized to 28×28)",
        label_visibility="collapsed",
    )

    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Uploaded image", use_container_width=True)
        with st.expander("See preprocessed input (28×28)"):
            arr_display = preprocess_image(pil_img)
            st.image(
                arr_display.reshape(28, 28),
                caption="What the model actually sees",
                width=140,
                clamp=True
            )

with col2:
    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.markdown(
            '<div class="empty-pred">'
            '<span class="empty-icon">🔍</span>'
            'Upload a clothing image<br>to get a prediction'
            '</div>',
            unsafe_allow_html=True,
        )
    elif model is None:
        st.info("Train and save the model first to see predictions.")
    else:
        with st.spinner("Classifying…"):
            arr   = preprocess_image(pil_img)
            probs = model.predict(arr, verbose=0)[0]

        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # ── Top result card ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card">
          <div class="rc-emoji">{CLASS_EMOJIS[pred_idx]}</div>
          <div class="rc-label">{CLASS_NAMES[pred_idx]}</div>
          <div class="rc-badge">{confidence:.1%} confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bars ────────────────────────────────────────────────────
        st.markdown('<div class="prob-section-title">All class probabilities</div>', unsafe_allow_html=True)

        sorted_idx = np.argsort(probs)[::-1]

        # Warm gold → cool tan gradient stops
        fill_colors = [
            "#c8a97e", "#c9ac86", "#cab08f", "#cbb49a",
            "#ccb8a4", "#cebbaa", "#d0bfb0", "#d2c3b7",
            "#d5c8be", "#d8cdc6",
        ]

        bars_html = ""
        for rank, i in enumerate(sorted_idx):
            pct   = float(probs[i])
            bar_w = max(1, round(pct * 100))
            is_top = (i == pred_idx)
            label_cls = "bar-label top" if is_top else "bar-label"
            pct_cls   = "bar-pct top"   if is_top else "bar-pct"
            fill_col  = fill_colors[rank]
            bars_html += f"""
            <div class="bar-row">
              <div class="{label_cls}">{CLASS_NAMES[i]}</div>
              <div class="bar-bg">
                <div class="bar-fill" style="width:{bar_w}%; background:{fill_col};"></div>
              </div>
              <div class="{pct_cls}">{pct:.1%}</div>
            </div>"""

        st.markdown(bars_html, unsafe_allow_html=True)

# ── How the model works ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label" style="margin-top:1rem;">How It Works</div>', unsafe_allow_html=True)

steps = [
    ("🖼️", "Input",        "28×28 grayscale image normalised to [0, 1]"),
    ("🔬", "Conv Layers",  "6 then 16 filters extract edges, textures, shapes"),
    ("📦", "Pooling",      "Average pooling reduces size, adds shift-tolerance"),
    ("🎯", "Output",       "Softmax over 10 classes → predicted category"),
]

hw_html = '<div class="hw-grid">'
for icon, title, desc in steps:
    hw_html += f"""
    <div class="hw-card">
      <div class="hw-icon">{icon}</div>
      <div class="hw-title">{title}</div>
      <div class="hw-desc">{desc}</div>
    </div>"""
hw_html += '</div>'
st.markdown(hw_html, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-bar">'
    'Model: LeNet-5 <span>·</span> '
    'Dataset: Fashion-MNIST (70,000 images, 10 classes) <span>·</span> '
    'Framework: TensorFlow / Keras <span>·</span> '
    'CCS 3572 / CSE 3582 · USJ'
    '</div>',
    unsafe_allow_html=True,
)