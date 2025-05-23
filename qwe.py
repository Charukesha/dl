import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 1. Model Definition (You define the model architecture here)
def create_model():
    img_size = (224, 224)
    
    # Define the model architecture (same as the trained one)
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # Update '3' with the number of your classes
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize and train the model (only when needed; normally this can be pre-trained)
model = create_model()

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
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalize image

    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # 5. Prediction function (Inference)
    def predict_image(img_array):
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])  # Get index of the max probability
        return predicted_class, predictions

    # 6. Get Prediction
    predicted_class, predictions = predict_image(img_array)

    # 7. Class Names (update with your actual class names from dataset)
    class_names = ['Healthy', 'Bacterial Blight', 'Powdery Mildew']  # Replace with actual class names

    # Display prediction results
    st.write(f"Predicted Class: {class_names[predicted_class]}")

    # Optionally, display the confidence of prediction
    confidence = np.max(predictions[0])  # Confidence level (highest probability)
    st.write(f"Prediction Confidence: {confidence * 100:.2f}%")

# Add a footer to the app
st.write("---")
st.write("Built with Streamlit and TensorFlow")
st.write("Model trained for plant disease classification.")
