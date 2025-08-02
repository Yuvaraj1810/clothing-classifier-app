
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('models/best_model.h5')
class_names = ['tshirt', 'shirt', 'jeans', 'dress', 'shoes']  # Update as needed

st.set_page_config(page_title="Clothing Classifier", page_icon="🧥")

st.title("👚 E-Commerce Clothing Image Classifier")
st.write("Upload an image of clothing to classify it (e.g., shirt, jeans, dress).")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.success(f"🧠 Predicted Class: **{predicted_class}**")
