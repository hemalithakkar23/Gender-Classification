import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------
# Load model (cached)
# --------------------------
@st.cache_resource
def load_model():
    # Load your model safely with compile=False
    model = tf.keras.models.load_model("best_gender_animal_model.h5", compile=False)
    return model

# --------------------------
# Preprocess image (PIL Image)
# --------------------------
@st.cache_data
def preprocess_image(_img: Image.Image, target_size=(224, 224)):
    img = _img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------------
# Streamlit UI
# --------------------------
st.title("Animal Gender/Species Classifier")

# Option 1: File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Option 2: Camera input
camera_image = st.camera_input("Or take a picture with your camera")

# Use whichever is available
img = None
if uploaded_file is not None:
    img = Image.open(uploaded_file)
elif camera_image is not None:
    img = Image.open(camera_image)

if img is not None:
    st.image(img, caption="Selected Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(img)
    model = load_model()
    prediction = model.predict(img_array)

    # Replace these with your model's actual class names
    class_names = ['Cat Male', 'Cat Female', 'Dog Male', 'Dog Female']  
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"Prediction: **{predicted_class}**")
    st.write("Confidence Scores:")
    for name, score in zip(class_names, prediction[0]):
        st.write(f"{name}: {score:.2f}")
