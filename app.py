



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set the page configuration
st.set_page_config(
    page_title="Diabetes Health & Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
DATA_FILE = "health_diabetes.csv"
MODEL_FILE = "trail_model_diabetes.keras"

@st.cache_data
def load_data():
    """Load the dataset."""
    try:
        data = pd.read_csv(DATA_FILE)
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{DATA_FILE}' was not found.")
        return pd.DataFrame()

@st.cache_resource
def load_prediction_model():
    """Load the Keras model."""
    try:
        model = load_model(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading model '{MODEL_FILE}': {e}")
        return None

# Load resources
df = load_data()
model = load_prediction_model()

# --- App Layout ---
st.title("🩺 Diabetes Health & Prediction App")

# Create tabs for Data Exploration and Prediction
tab1, tab2 = st.tabs(["📊 Data Explorer", "🔮 Predict Diabetes Status"])

# ==========================================
# TAB 1: DATA EXPLORER
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("health_diabetes.csv")

# Calculate correlation with Diabetes_012
# We use numeric conversion just in case, though data seems float64 already
correlations = df.corr()['Diabetes_012'].drop('Diabetes_012')

# Sort by absolute values to find the "strongest" relationships (positive or negative)
top_10_corr = correlations.abs().sort_values(ascending=False).head(10)

# Get the actual correlation values (with sign) for these top 10
top_10_actual = correlations[top_10_corr.index]

# Sort them for plotting logic (e.g., highest positive to lowest or vice versa)
# Let's sort by the actual value for the visual
top_10_actual_sorted = top_10_actual.sort_values(ascending=True)

print("Top 10 Correlated Features:")
print(top_10_actual_sorted)

# Create the plot using pure Matplotlib
plt.figure(figsize=(10, 6))
# Create a horizontal bar plot
bars = plt.barh(top_10_actual_sorted.index, top_10_actual_sorted.values, color='skyblue')

# Add values to the end of the bars
for bar in bars:
    width = bar.get_width()
    label_x_pos = width if width > 0 else width - 0.05
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
             va='center', ha='left' if width > 0 else 'right')

plt.title('Top 10 Features Correlated with Diabetes', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('top10_correlation.png')

# ==========================================
# TAB 2: PREDICTION
# ==========================================
with tab2:
    st.header("Predict Your Health Status")
    st.markdown("Adjust the values below to match your health profile, then click **Predict**.")

    if model is None:
        st.warning("Model not found. Please ensure 'trail_model_diabetes.keras' is in the directory.")
    else:
        # Create a form for inputs
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            # Define mapping for binary/ordinal inputs to make them user-friendly
            binary_map = {0: 'No', 1: 'Yes'}
            
            with col1:
                st.subheader("General Health")
                high_bp = st.selectbox("High Blood Pressure?", options=[0, 1], format_func=lambda x: binary_map[x])
                high_chol = st.selectbox("High Cholesterol?", options=[0, 1], format_func=lambda x: binary_map[x])
                chol_check = st.selectbox("Cholesterol Check in 5 yrs?", options=[0, 1], format_func=lambda x: binary_map[x])
                bmi = st.slider("BMI (Body Mass Index)", 12, 98, 28)
                gen_hlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

            with col2:
                st.subheader("Lifestyle & Habits")
                smoker = st.selectbox("Smoked >100 cigs in life?", options=[0, 1], format_func=lambda x: binary_map[x])
                phys_activity = st.selectbox("Physical Activity (past 30 days)?", options=[0, 1], format_func=lambda x: binary_map[x])
                fruits = st.selectbox("Consume Fruit 1+ times/day?", options=[0, 1], format_func=lambda x: binary_map[x])
                veggies = st.selectbox("Consume Veggies 1+ times/day?", options=[0, 1], format_func=lambda x: binary_map[x])
                hvy_alcohol = st.selectbox("Heavy Alcohol Consumption?", options=[0, 1], format_func=lambda x: binary_map[x])

            with col3:
                st.subheader("Medical History & Demographics")
                stroke = st.selectbox("Ever had a Stroke?", options=[0, 1], format_func=lambda x: binary_map[x])
                heart_disease = st.selectbox("Heart Disease or Attack?", options=[0, 1], format_func=lambda x: binary_map[x])
                diff_walk = st.selectbox("Difficulty Walking?", options=[0, 1], format_func=lambda x: binary_map[x])
                sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
                age = st.slider("Age Category (1=18-24, 13=80+)", 1, 13, 8)
            
            st.markdown("---")
            col4, col5 = st.columns(2)
            with col4:
                ment_hlth = st.slider("Days of Poor Mental Health (past 30 days)", 0, 30, 0)
                phys_hlth = st.slider("Days of Poor Physical Health (past 30 days)", 0, 30, 0)
            with col5:
                education = st.slider("Education Level (1-6)", 1, 6, 4)
                income = st.slider("Income Scale (1-8)", 1, 8, 5)
                any_healthcare = st.selectbox("Have any Healthcare coverage?", options=[0, 1], format_func=lambda x: binary_map[x])
                no_doc_cost = st.selectbox("Skipped Doctor due to cost?", options=[0, 1], format_func=lambda x: binary_map[x])

            # Submit button
            submitted = st.form_submit_button("Predict Diabetes Status")

        if submitted:
            # Prepare input array in the EXACT order of training columns
            # Order: HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, 
            # PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, 
            # GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
            
            input_data = np.array([[
                high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease,
                phys_activity, fruits, veggies, hvy_alcohol, any_healthcare, no_doc_cost,
                gen_hlth, ment_hlth, phys_hlth, diff_walk, sex, age, education, income
            ]])

            # Make prediction
            try:
                prediction_probs = model.predict(input_data)
                prediction_class = np.argmax(prediction_probs, axis=1)[0]
                
                st.markdown("### Prediction Result:")
                
                if prediction_class == 0:
                    st.success(f"**Result: No Diabetes** (Class 0)")
                    st.write("You are likely healthy based on these indicators.")
                elif prediction_class == 1:
                    st.warning(f"**Result: Prediabetes** (Class 1)")
                    st.write("You may be at risk. Consult a healthcare professional.")
                else:
                    st.error(f"**Result: Diabetes** (Class 2)")
                    st.write("The model predicts a high likelihood of diabetes. Please consult a doctor immediately.")

                # Show probabilities if available
                st.write("Prediction Probabilities:", prediction_probs)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
