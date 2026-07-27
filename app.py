import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Banana Ripeness Classifier",
    page_icon="🍌",
    layout="centered",
)

# Minimal Clean Styling
st.markdown("""
<style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-title">Banana Ripeness Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a banana image to check if it is Ripe or Unripe.</div>', unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenetv3_transfer.keras")

model = load_model()

# Prediction Function
def predict(model, pil_image):
    img = pil_image.convert("RGB").resize((128, 128))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    preds = model.predict(arr, verbose=0)[0]
    
    # Class mapping based on alphabetical folder order (ripe, unripe)
    prob_ripe = float(preds[0])
    prob_unripe = float(preds[1])
    
    label = "Ripe" if prob_ripe >= prob_unripe else "Unripe"
    confidence = max(prob_ripe, prob_unripe) * 100
    return label, confidence

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # Display Image cleanly
    st.image(img, caption="Uploaded Image", width=300)
    
    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing..."):
            label, confidence = predict(model, img)
            
        st.success(f"Result: **{label}** ({confidence:.2f}% confidence)")
