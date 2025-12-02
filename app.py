import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")  # Your trained model file
    scaler = joblib.load("scaler.pkl")  # Scaler if used
    return model, scaler

model, scaler = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔍 Diabetes Prediction App")
st.write("Upload your dataset or enter patient values manually.")

# -----------------------------
# File Upload Section
# -----------------------------
uploaded_file = st.file_uploader("Upload Excel/CSV file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File Uploaded Successfully!")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error reading file: {e}")

# -----------------------------
# Manual Input Section
# -----------------------------
st.subheader("Enter Patient Details")

preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 85)
bmi = st.number_input("BMI", 0.0, 70.0, 22.5)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

if st.button("Predict"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

