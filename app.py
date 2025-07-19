import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Page config
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🧠", layout="centered")

# Title and description
st.markdown(
    """
    <style>
    .header {
        background-color: #c00384;  /* dark pink */
        color: white;
        padding: 15px 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        border-radius: 8px;
        margin-bottom: 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    <div class="header">
        Handwritten Digit Recognition (MNIST)
    </div>
    """,
    unsafe_allow_html=True
)


# File uploader
uploaded_file = st.file_uploader("📤 Upload an image (PNG/JPG).", type=["png", "jpg", "jpeg"])
st.write("Please upload a grayscale image of a handwritten digit, preferably 28x28 pixels for best results.")

if uploaded_file is not None:
    st.subheader("🖼️ Uploaded Image")
    image = Image.open(uploaded_file).convert('L')  # Grayscale
    st.image(image, caption='Your Image', use_column_width=False, width=200)

    # Preprocessing
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255

    # Predict
    predictions = model.predict(img_array)
    prediction = np.argmax(predictions)
    confidence = predictions[0][prediction]

    st.write(f"### 📝 Predicted Digit: {prediction}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")


st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ffe6f0;  /* light pink background */
        color: #c00384;  /* dark pink text */
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        border-top: 1px solid #c00384;
    }
    </style>
    <div class="footer">
       Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

    

