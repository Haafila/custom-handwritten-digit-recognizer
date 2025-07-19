import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

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
    <p>This app predicts digits (0–9) from either a digit you draw or an image you upload. It uses a model trained on the MNIST dataset and gives results based       on your real input.</p>
    """,
    unsafe_allow_html=True
)



st.subheader("📤Upload an Image(PNG/JPG).")

# File uploader
uploaded_file = st.file_uploader("Please upload a grayscale image of a handwritten digit, preferably 28x28 pixels for best results.", type=["png", "jpg", "jpeg"])

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

    st.success(f"📝 Predicted Digit: {pred_digit}")
    st.info(f"📊 Confidence: {confidence * 100:.2f}%")


st.subheader("✍️ Draw a Digit.")
st.write("Draw your digit in the center of the box for best results.")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=100,
    width=100,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict Drawn Digit"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, 0]  # Get grayscale
        img = Image.fromarray((img).astype("uint8"))
        img = img.resize((28, 28))
        img = ImageOps.invert(img)  # Make white background black and vice versa
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

        prediction = model.predict(img_array)
        pred_digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"📝 Predicted Digit: {pred_digit}")
        st.info(f"📊 Confidence: {confidence * 100:.2f}%")

    

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

    

