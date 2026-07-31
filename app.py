import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Banana Ripeness Classifier",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg",
    layout="centered",
)

st.markdown("""
<style>
    .main-title {
        font-size: 1.6rem;
        font-weight: 500;
        margin-bottom: 0.25rem;
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
    
    # Return both probabilities explicitly instead of just the max confidence
    return label, prob_ripe, prob_unripe

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # Display Image cleanly
    st.image(img, caption="Uploaded Image", width=300)
    
    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing..."):
            label, prob_ripe, prob_unripe = predict(model, img)
            
        st.success(f"Result: **{label}**")
        
        # Display the breakdown using columns and progress bars
        st.write("### Confidence Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Ripe", value=f"{prob_ripe * 100:.2f}%")
            st.progress(prob_ripe)
            
        with col2:
            st.metric(label="Unripe", value=f"{prob_unripe * 100:.2f}%")
            st.progress(prob_unripe)
