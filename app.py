import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ---------------------------
# CACHE THE MODEL
# ---------------------------
@st.cache_data
def load_cnn_model():
    return load_model('Person_cnn_classifier.h5')

model = load_cnn_model()

# ---------------------------
# CLASS NAMES
# ---------------------------
class_names = ['Deepu', 'Vamsi', 'VenkataLakshmi', 'VenkataRao']

# ---------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="My Family Image Classification", layout='centered')
st.sidebar.title("Upload Your Image")
st.markdown("""
# Family Image Classifier
Upload an image, and this app will predict which family member it belongs to using a Vanilla CNN model.
""")

# ---------------------------
# FILE UPLOADER
# ---------------------------
upload_file = st.sidebar.file_uploader("Upload your Image", type=['jpg', 'jpeg', 'png'])

if upload_file is not None:
    # Load image
    img = Image.open(upload_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized)/255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_batch)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    
    # Display prediction
    st.success(f"Predicted Class: **{predicted_class}**")
    
    # Display probabilities
    st.subheader("Prediction Probabilities:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")
