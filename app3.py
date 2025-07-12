import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# ✅ THIS MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Semiconductor Wafer Detector", layout="centered")

# Path to your model file
model_path ="saved_model/wafer_cnn_model.h5"

# Check if model exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()

try:
    st.write("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

class_names = ['Pass', 'Fail']

st.title("🔍 Semiconductor Wafer Pass/Fail Classifier")
st.write("Upload an image or take a photo to classify the wafer as Pass or Fail.")

input_method = st.radio("Choose Image Input Method:", ("Upload Image", "Use Camera"))
image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            st.image(image, caption="Uploaded Wafer Image", use_column_width=True)
        except Exception as e:
            st.error("⚠ Could not read the uploaded image.")

elif input_method == "Use Camera":
    camera_image = st.camera_input("Take a wafer photo...")
    if camera_image:
        try:
            image = Image.open(camera_image)
            st.image(image, caption="Captured Wafer Image", use_column_width=True)
        except Exception as e:
            st.error("⚠ Could not access the image from camera.")

if image:
    with st.spinner("Classifying..."):
        # Convert to RGB because your model expects 3 channels
        image = image.convert("RGB")
        image = image.resize((64, 64))

        img_array = np.asarray(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]

        if prediction.shape[0] == 2:
            predicted_index = int(np.argmax(prediction))
            predicted_class = class_names[predicted_index]
            confidence = float(prediction[predicted_index])
        else:
            prob = float(prediction[0])
            predicted_class = "Pass" if prob < 0.5 else "Fail"
            confidence = 1 - prob if predicted_class == "Pass" else prob

        st.markdown(f"### 🧠 Prediction: *{predicted_class}*")
        st.markdown(f"### 📊 Confidence: *{confidence * 100:.2f}%*")

st.write("Note: This model was trained on the WM-811K dataset. 'Pass' indicates no defect pattern, while 'Fail' indicates a defect.")
