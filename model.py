import streamlit as st  # type: ignore
import tensorflow as tf
import numpy as np      # type: ignore
from PIL import Image   # type: ignore
import io
import os

# -------------------------------
# Dynamically build model path
# -------------------------------

# The directory where this script lives
script_dir = os.path.dirname(__file__)

# Your model filename
MODEL_FILENAME = "/my_model.keras"

# Absolute path to the model file
model_path = os.path.join(script_dir, MODEL_FILENAME)

# -------------------------------
# Load the model safely
# -------------------------------
if not os.path.exists("/my_model.keras"):
    st.error(f"🚫 Model file not found at:\n\n{"/my_model.keras}")
    st.stop()

try:
    model = tf.keras.models.load_model("/my_model.keras")
    st.success("✅ Model loaded successfully!")
    print(model.summary())
except Exception as e:
    st.error(f"⚠️ Failed to load the model:\n\n{str(e)}")
    st.stop()

# -------------------------------
# Class labels (adjust if needed)
# -------------------------------
class_names = ['Normal', 'Defective']

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Semiconducter Wafer Detector", layout="centered")
st.title("🔍 Prediction of Semiconductor Wafer")
st.write("Upload an image or take a photo to detect anomalies in the product.")

# Choose input method
input_method = st.radio("Choose Image Input Method:", ("Upload Image", "Use Camera"))

image = None

# Upload image block
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(io.BytesIO(uploaded_file.read()))
        except Exception as e:
            st.error("⚠️ Could not read the image. Please upload a valid image file.")

elif input_method == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        try:
            image = Image.open(camera_image)
        except Exception as e:
            st.error("⚠️ Could not access the image from camera.")

# -------------------------------
# Make prediction
# -------------------------------
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.asarray(image).astype(np.float32) / 255.0

    # If grayscale → expand to 3 channels
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    # If RGBA → drop alpha
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    try:
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.markdown(f"### 🧠 Prediction: *{predicted_class}*")
        st.markdown(f"### 📊 Confidence: *{confidence * 100:.2f}%*")
    except Exception as e:
        st.error(f"⚠️ Error during prediction:\n\n{str(e)}")
