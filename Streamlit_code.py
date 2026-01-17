import os
import base64
import numpy as np
import streamlit as st
import joblib

# ---------------- Load Joblib Files ----------------
model = joblib.load("Model.pkl")
norm = joblib.load("Norm.pkl")

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Heart Disease Detector",
    layout="centered"
)

# ---------------- Image ----------------
img_path = "Heart_image1.jpeg"
if os.path.exists(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(f"""
    <div style="display:flex; justify-content:center; margin-bottom:20px;">
        <img src="data:image/png;base64,{b64}"
             style="width:1200px;
             height:200px;
             border:4px solid #5E8A8E;
             border-radius:20px;
             box-shadow:0 0 12px rgba(94,138,142,0.5);
             object-fit:fill;">
    </div>
    """, unsafe_allow_html=True)

# ---------------- Inputs ----------------
st.markdown("## 🩺 Patient Medical Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

with col2:
    trestbps = st.number_input("Resting Blood Pressure(restbps)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col3:
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])
thalt = st.selectbox("Thalassemia", [0, 1, 2, 3])

# ---------------- Prediction ----------------
if st.button("Predict"):

    sex = 1 if sex == "Male" else 0

    input_data = np.array([[ 
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thalt
    ]])

    input_data = norm.transform(input_data)

    prediction_prob = model.predict(input_data)[0]
    prediction = 1 if prediction_prob > 0.5 else 0

    if prediction == 1:
        st.error("Heart Disease Detected! Please consult a doctor.")
    else:
        st.success("No Heart Disease Detected. Stay healthy!")

    st.write(f" Prediction Confidence: **{prediction_prob:.2f}**")
