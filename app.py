# -*- coding: utf-8 -*-

from fastai.vision.all import *
from PIL import Image
import streamlit as st
import os
import requests

# Load the model
@st.cache_resource
def load_model():
    model_path = "eagle_crow.pkl"
    
    
    # Download the model if it doesn't already exist
    if not os.path.exists(model_path):
        url = "https://github.com/maranurpraveen/DAV-6150/blob/main/eagle_crow.pkl"  # Raw URL to your file
        with open(model_path, "wb") as f:
            response = requests.get(url)
            f.write(response.content)
    
    return load_learner(model_path)

model = load_model()

# Streamlit UI
st.title("Eagle or Crow Classifier")
st.write("Upload an image to classify if it is an Eagle or Crow.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = PILImage.create(uploaded_file)
    st.image(image.to_thumb(256, 256), caption="Uploaded Image", use_column_width=True)

    # Classify image
    pred_class, pred_idx, probs = model.predict(image)
    st.write(f"Prediction: {pred_class}")
    st.write(f"Confidence: {probs[pred_idx] * 100:.2f}%")
