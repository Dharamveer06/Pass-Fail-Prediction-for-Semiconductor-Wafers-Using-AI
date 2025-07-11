import streamlit as st  # type: ignore
import tensorflow as tf
import numpy as np      # type: ignore
from PIL import Image   # type: ignore
import io
import os

# Check files in current directory
st.write("Files in current directory:", os.listdir("."))

# Path to model file
model_path = "my_model.h5"

# Check if model file exists
if not os.path.exists("my_model.h5"):
    st.error(f"🚫 Model file `{my_model.h5}` not found in your project directory.")
    st.stop()

# Try loading the model safely
try:
    model = tf.keras.models.load_model("my_model.h5")
    st.success("✅ Model loaded successfully!")
    # Optional: print model summary in console (not on Streamlit)
    print(model.summary())
except Exception as e:
    st.error(f"⚠️ Failed to load the model. Details:\n\n{str(e)}")
    st.stop()

# Define class labels based on your training
class_names = ['Normal', 'Defective']  # Change if your classes differ

# Page setup
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

# Camera input block
elif input_method == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        try:
            image = Image.open(camera_image)
        except Exception as e:
            st.error("⚠️ Could not access the image from camera.")

# If image is loaded, make prediction
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    img_array = np.asarray(image).astype(np.float32) / 255.0

    # Handle grayscale images
    if img_array.ndim == 2:
        # expand grayscale to 3 channels
        img_array = np.stack([img_array] * 3, axis=-1)

    if img_array.shape[-1] == 4:
        # drop alpha channel if present
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    try:
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        # Display results
        st.markdown(f"### 🧠 Prediction: *{predicted_class}*")
        st.markdown(f"### 📊 Confidence: *{confidence * 100:.2f}%*")
    except Exception as e:
        st.error(f"⚠️ Error during prediction:\n\n{str(e)}")
