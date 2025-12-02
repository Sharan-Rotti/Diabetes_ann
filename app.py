# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.preprocessing import StandardScaler

# # -----------------------------
# # Load Model and Scaler
# # -----------------------------
# @st.cache_resource
# def load_model():
#     model = joblib.load("model.pkl")  # Your trained model file
#     scaler = joblib.load("scaler.pkl")  # Scaler if used
#     return model, scaler

# model, scaler = load_model('trail_model_dibates.keras')

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.title("🔍 Diabetes Prediction App")
# st.write("Upload your dataset or enter patient values manually.")

# # -----------------------------
# # File Upload Section
# # -----------------------------
# uploaded_file = st.file_uploader("Upload Excel/CSV file", type=["csv", "xlsx"])

# if uploaded_file is not None:
#     try:
#         if uploaded_file.name.endswith(".csv"):
#             df = pd.read_csv(uploaded_file)
#         else:
#             df = pd.read_excel(uploaded_file)

#         st.success("File Uploaded Successfully!")
#         st.dataframe(df.head())

#     except Exception as e:
#         st.error(f"Error reading file: {e}")

# # -----------------------------
# # Manual Input Section
# # -----------------------------
# st.subheader("Enter Patient Details")

# preg = st.number_input("Pregnancies", 0, 20, 1)
# glucose = st.number_input("Glucose Level", 0, 300, 120)
# bp = st.number_input("Blood Pressure", 0, 200, 70)
# skin = st.number_input("Skin Thickness", 0, 100, 20)
# insulin = st.number_input("Insulin Level", 0, 900, 85)
# bmi = st.number_input("BMI", 0.0, 70.0, 22.5)
# dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
# age = st.number_input("Age", 1, 120, 30)

# input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

# if st.button("Predict"):
#     scaled_input = scaler.transform(input_data)
#     prediction = model.predict(scaled_input)

#     if prediction[0] == 1:
#         st.error("⚠️ High Risk of Diabetes")
#     else:
#         st.success("✅ Low Risk of Diabetes")



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page configuration
st.set_page_config(
    page_title="Diabetes Health Indicators App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
FILE_NAME = "health_diabetes.csv"

@st.cache_data
def load_data():
    """Load the dataset and perform basic type conversions."""
    try:
        data = pd.read_csv(FILE_NAME)
        # Convert the target variable to a category for clear plotting
        data['Diabetes_012'] = data['Diabetes_012'].astype('category')
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{FILE_NAME}' was not found. "
                 "Please ensure it is in the same directory as app.py.")
        return pd.DataFrame()

# Load the data
df = load_data()

st.title("🩺 Diabetes Health Indicators Data Explorer")
st.markdown("---")

if not df.empty:
    # --- Sidebar for Navigation/Filtering (Optional but good practice) ---
    st.sidebar.header("Settings")
    
    # Simple data viewing option
    show_raw = st.sidebar.checkbox("Show Raw Data Sample", False)
    
    # --- Main Content ---
    
    st.header("1. Dataset Overview")
    st.write(f"Data loaded successfully with {len(df)} rows and {len(df.columns)} columns.")

    if show_raw:
        st.subheader("Raw Data Sample")
        st.dataframe(df.head())

    st.subheader("Summary Statistics (Numerical Columns)")
    st.write(df.describe())

    st.markdown("---")

    st.header("2. Key Distribution: Diabetes Status")
    st.markdown("The target variable 'Diabetes_012' is encoded as: **0 (No Diabetes), 1 (Prediabetes), 2 (Diabetes)**.")

    # Calculate counts and create the plot
    diabetes_counts = df['Diabetes_012'].value_counts().sort_index()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=diabetes_counts.index, y=diabetes_counts.values, ax=ax, palette="viridis")
    
    # Add labels and title
    ax.set_title('Counts of Diabetes Status', fontsize=16)
    ax.set_xlabel('Diabetes Status (0: No, 1: Pre, 2: Yes)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    st.markdown("---")

    st.header("3. BMI Distribution")
    
    # Create the figure for BMI
    fig_bmi, ax_bmi = plt.subplots(figsize=(8, 5))
    sns.histplot(df['BMI'], bins=30, kde=True, ax=ax_bmi, color='skyblue')
    
    # Add labels and title
    ax_bmi.set_title('Distribution of Body Mass Index (BMI)', fontsize=16)
    ax_bmi.set_xlabel('BMI', fontsize=12)
    ax_bmi.set_ylabel('Frequency', fontsize=12)
    
    # Display the plot in Streamlit
    st.pyplot(fig_bmi)
