import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 2. Set Parameters
img_size = (224, 224)

# 3. Streamlit Interface
st.title("Plant Disease Classification")
st.write("This app predicts the disease in a plant leaf image. Upload an image of a leaf to check for diseases.")

# 4. File Upload
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Display the predicted class
    class_names = ['class1', 'class2', 'class3']  # Replace with your actual class names
    st.write(f"Predicted Class: {class_names[predicted_class]}")

    # Optionally, display the confidence of prediction
    confidence = np.max(predictions[0])
    st.write(f"Prediction Confidence: {confidence * 100:.2f}%")

# Add a footer to the app
st.write("---")
st.write("Built with Streamlit and TensorFlow")
st.write("Model trained for plant disease classification.")
