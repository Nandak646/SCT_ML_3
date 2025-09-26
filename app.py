# clean_app.py
import streamlit as st
import cv2
import numpy as np
import joblib
from pdf2image import convert_from_bytes

# Load trained model
MODEL_PATH = r"C:\Users\nanda\OneDrive\Documents\Nanda's ML Tasks\SVM\PetImages\cat_dog_svm_model.pkl"
clf = joblib.load(MODEL_PATH)

# Helper functions
def classify_image(img):
    img_resized = cv2.resize(img, (64, 64))
    img_flat = img_resized.flatten().reshape(1, -1)
    pred = clf.predict(img_flat)[0]
    return "Cat 🐱" if pred == 0 else "Dog 🐶"

def process_pdf(pdf_file):
    pages = convert_from_bytes(pdf_file.read())
    images = [cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY) for page in pages]
    return images

# Page config
st.set_page_config(page_title="🐱 Cat vs Dog Classifier", layout="wide", page_icon="🐾")

# Header
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🐱🐶 Cat vs Dog Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload images or PDFs to see if it's a Cat or Dog!</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload section
st.subheader("📤 Upload Images or PDFs")
uploaded_files = st.file_uploader("Select files", type=["jpg", "png", "pdf"], accept_multiple_files=True)

results = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "application/pdf":
                st.info(f"📄 {uploaded_file.name} (PDF detected)")
                pdf_images = process_pdf(uploaded_file)
                for idx, img in enumerate(pdf_images):
                    label = classify_image(img)
                    color = "#4CAF50" if label.startswith("Cat") else "#2196F3"
                    with st.container():
                        st.image(img, width=250)
                        st.markdown(
                            f"<div style='border:3px solid {color}; padding:10px; border-radius:10px; text-align:center;'>"
                            f"<h3>{label}</h3>"
                            f"</div>", unsafe_allow_html=True
                        )
                        results.append({"File": uploaded_file.name, "Label": label})

            else:
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                label = classify_image(img)
                color = "#4CAF50" if label.startswith("Cat") else "#2196F3"
                with st.container():
                    st.image(img, width=250)
                    st.markdown(
                        f"<div style='border:3px solid {color}; padding:10px; border-radius:10px; text-align:center;'>"
                        f"<h3>{label}</h3>"
                        f"</div>", unsafe_allow_html=True
                    )
                    results.append({"File": uploaded_file.name, "Label": label})
        except Exception as e:
            st.error(f"❌ Could not process {uploaded_file.name}: {e}")

# Summary Table
if results:
    st.markdown("---")
    st.subheader("📊 Predictions Summary")
    import pandas as pd
    df = pd.DataFrame(results)
    st.table(df)

st.markdown("---")
st.markdown("<p style='text-align:center;'></p>", unsafe_allow_html=True)
