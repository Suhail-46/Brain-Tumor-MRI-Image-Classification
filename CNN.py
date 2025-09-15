import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image

# ==============================
# Load trained model
# ==============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("/content/best_model_cnn.h5")  # path to your trained model
    return model

model = load_model()

# Class labels (order must match training)
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# ==============================
# App UI
# ==============================
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image, and the model will classify the tumor type.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # resize as per your model
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds[0])
    confidence = np.max(preds[0])

    # Show prediction result
    st.markdown(f"### 🏷️ Prediction: **{CLASS_NAMES[predicted_class]}**")
    st.markdown(f"🔮 Confidence: **{confidence:.2f}**")

    # Show confidence scores for all classes
    st.write("### 📊 Confidence Scores")
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, preds[0], color=["#2E86AB", "#F6A01A", "#28B463", "#9B59B6"])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)

else:
    st.info("👆 Please upload an MRI image to get started.")
