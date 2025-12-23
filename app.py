import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


@st.cache(allow_output_mutation=True)
def load_model_fn():
    return tf.keras.models.load_model("best_gender_animal_model.h5", compile=False)


model = load_model_fn()
IMG_SIZE = 224
CLASSES = ["Male", "Female", "Animal"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.set_page_config(page_title="Face + Animal Classifier", page_icon="🐾")
st.title("🐾 Gender + Animal Classification App")

mode = st.sidebar.radio("Choose Mode", ["Upload Image", "Live Camera"])

def classify(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:  # human face
        x, y, w, h = faces[0]
        crop = image[y:y+h, x:x+w]
        processed = preprocess(crop)
    else:  # no face → animal
        processed = preprocess(image)

    preds = model.predict(processed)[0]
    idx = np.argmax(preds)
    return CLASSES[idx], preds[idx]

# Upload image mode
if mode == "Upload Image":
    file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if file:
        img = np.array(Image.open(file))
        st.image(img)

        if st.button("Predict"):
            label, conf = classify(img)
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {conf:.2f}")

# Live camera mode
else:
    camera = st.camera_input("Take a Picture")
    if camera:
        img = np.array(Image.open(camera))
        label, conf = classify(img)
        st.image(img, caption=f"{label} ({conf:.2f})")









# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image

# # Fix: use compile=False to avoid batch_shape deserialization issues
# @st.cache_resource
# def load_model_fn():
#     return tf.keras.models.load_model("best_gender_animal_model.h5", compile=False)

# model = load_model_fn()
# IMG_SIZE = 224
# CLASSES = ["Male", "Female", "Animal"]

# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# def preprocess(img):
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = img.astype("float32") / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# st.set_page_config(page_title="Face + Animal Classifier", page_icon="🐾")
# st.title("🐾 Gender + Animal Classification App")

# mode = st.sidebar.radio("Choose Mode", ["Upload Image", "Live Camera"])

# def classify(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 5)

#     if len(faces) > 0:  # human face
#         x, y, w, h = faces[0]
#         crop = image[y:y+h, x:x+w]
#         processed = preprocess(crop)
#     else:  # no face → animal
#         processed = preprocess(image)

#     preds = model.predict(processed)[0]
#     idx = np.argmax(preds)
#     return CLASSES[idx], preds[idx]

# # Upload image mode
# if mode == "Upload Image":
#     file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
#     if file:
#         img = np.array(Image.open(file))
#         st.image(img)

#         if st.button("Predict"):
#             label, conf = classify(img)
#             st.success(f"Prediction: {label}")
#             st.info(f"Confidence: {conf:.2f}")

# # Live camera mode
# else:
#     camera = st.camera_input("Take a Picture")
#     if camera:
#         img = np.array(Image.open(camera))
#         label, conf = classify(img)
#         st.image(img, caption=f"{label} ({conf:.2f})")
