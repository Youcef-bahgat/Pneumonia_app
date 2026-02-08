import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from PIL import Image

# عنوان الموقع
st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a Chest X-ray image and the model will predict if it's NORMAL or PNEUMONIA.")

# رفع صورة
uploaded_file = st.file_uploader("Choose a Chest X-ray image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # حفظ مؤقت للمعالجة
    img.save("temp.jpg")
    
    # تجهيز الصورة
    img_size = (299, 299)
    img = load_img("temp.jpg", target_size=img_size, color_mode='rgb')
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    
    # تحميل الموديل
    model = load_model('X-Ray_Pneumonia_classification.h5')
    
    # Prediction
    pred = model.predict(img_arr)[0][0]
    label = 'PNEUMONIA' if pred > 0.5 else 'NORMAL'
    
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{pred:.2f}**")
