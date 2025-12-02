import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model

# ------------------------------
# Streamlit & Seaborn settings
# ------------------------------
st.set_page_config(
    page_title="Diabetes Health & Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)
sns.set_theme(style="whitegrid")

# ------------------------------
# Config
# ------------------------------
DATA_FILE = "health_diabetes.csv"
MODEL_FILE = "trail_model_diabetes.keras"

# ------------------------------
# Data Loading & Preprocessing
# ------------------------------
@st.cache_data
def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    """Load the dataset and create human-readable labels for EDA."""
    try:
        data = pd.read_csv(path)

        # Target label
        data["Diabetes_Status"] = data["Diabetes_012"].map(
            {0.0: "No Diabetes", 1.0: "Pre-diabetes", 2.0: "Diabetes"}
        )

        # Age mapping
        age_labels = {
            1.0: "18-24", 2.0: "25-29", 3.0: "30-34", 4.0: "35-39",
            5.0: "40-44", 6.0: "45-49", 7.0: "50-54", 8.0: "55-59",
            9.0: "60-64", 10.0: "65-69", 11.0: "70-74", 12.0: "75-79",
            13.0: "80+"
        }
        data["Age_Group"] = data["Age"].map(age_labels)

        # General Health mapping
        data["GenHlth_Label"] = data["GenHlth"].map(
            {1.0: "Excellent", 2.0: "Very Good", 3.0: "Good", 4.0: "Fair", 5.0: "Poor"}
        )

        # Sex mapping
        data["Sex_Label"] = data["Sex"].map({0.0: "Female", 1.0: "Male"})

        return data

    except FileNotFoundError:
        st.error(
            f"Error: The file '{path}' was not found. "
            f"Please ensure it is in the same directory as app.py."
        )
        return pd.DataFrame()


@st.cache_resource
def load_prediction_model(model_path: str = MODEL_FILE):
    """Load the trained Keras model for prediction."""
    try:
        model = keras_load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model '{model_path}': {e}")
        return None


# Load resources once
df = load_data()
model = load_prediction_model()


# ------------------------------
# Plot Functions
# ------------------------------
def plot_1_target_distribution(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(x="Diabetes_Status", data=df, palette="viridis")
    plt.title("Plot 1: Distribution of Diabetes Status", fontsize=16)
    plt.xlabel("Diabetes Status", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    return plt.gcf()


def plot_2_top_10_correlation(df):
    # numeric_only for robustness with non-numeric columns
    corr_matrix = df.corr(numeric_only=True)
    if "Diabetes_012" not in corr_matrix.columns:
        return None

    diabetes_corr = corr_matrix["Diabetes_012"].drop("Diabetes_012")
    top_10 = diabetes_corr.abs().sort_values(ascending=False).head(10)
    top_10_features = diabetes_corr[top_10.index].sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_10_features.index, top_10_features.values, color="#4c72b0")

    ax.set_title("Plot 2: Top 10 Features Correlated with Diabetes", fontsize=16)
    ax.set_xlabel("Correlation Coefficient (Pearson)", fontsize=12)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.04
        ax.text(
            label_x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    return fig


def plot_3_bmi_distribution(df):
    plt.figure(figsize=(9, 5))
    sns.histplot(df["BMI"], bins=50, kde=True, color="darkcyan")
    plt.title("Plot 3: Distribution of Body Mass Index (BMI)", fontsize=16)
    plt.xlabel("BMI", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    return plt.gcf()


def plot_4_bmi_vs_diabetes(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Diabetes_Status", y="BMI", data=df, palette="Spectral")
    plt.title("Plot 4: BMI Distribution by Diabetes Status", fontsize=16)
    plt.xlabel("Diabetes Status", fontsize=12)
    plt.ylabel("Body Mass Index (BMI)", fontsize=12)
    plt.tight_layout()
    return plt.gcf()


def plot_5_age_vs_diabetes(df):
    age_diabetes_counts = df.groupby(["Age_Group", "Diabetes_Status"]).size().unstack(fill_value=0)
    age_diabetes_proportions = age_diabetes_counts.apply(lambda x: x / x.sum(), axis=1) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    age_diabetes_proportions.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)

    plt.title(
        "Plot 5: Distribution of Diabetes Status Across Age Groups (Proportions)",
        fontsize=16,
    )
    plt.xlabel("Age Group", fontsize=12)
    plt.ylabel("Percentage within Age Group", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Diabetes Status", loc="upper left")
    plt.tight_layout()
    return plt.gcf()


def plot_6_genhlth_vs_diabetes(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(
        x="GenHlth_Label",
        hue="Diabetes_Status",
        data=df,
        order=["Excellent", "Very Good", "Good", "Fair", "Poor"],
        palette="magma",
    )
    plt.title("Plot 6: Diabetes Status Counts by Self-Reported General Health", fontsize=16)
    plt.xlabel("Self-Reported General Health", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(title="Diabetes Status")
    plt.tight_layout()
    return plt.gcf()


def plot_7_highbp_vs_diabetes(df):
    plt.figure(figsize=(8, 5))
    ctab = pd.crosstab(df["HighBP"], df["Diabetes_Status"], normalize="index")
    sns.heatmap(
        ctab,
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        cbar=False,
        yticklabels=["No HighBP", "HighBP"],
    )
    plt.title("Plot 7: Proportion of Diabetes Status by High Blood Pressure", fontsize=16)
    plt.xlabel("Diabetes Status", fontsize=12)
    plt.ylabel("High Blood Pressure", fontsize=12)
    plt.tight_layout()
    return plt.gcf()


def plot_8_physhlth_vs_diabetes(df):
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Diabetes_Status",
        y="PhysHlth",
        data=df,
        palette="cubehelix",
        inner="quartile",
    )
    plt.title(
        "Plot 8: Physical Health (Days of Poor Health) Distribution by Diabetes Status",
        fontsize=16,
    )
    plt.xlabel("Diabetes Status", fontsize=12)
    plt.ylabel("Days of Poor Physical Health (Past 30 Days)", fontsize=12)
    plt.ylim(0, 31)
    plt.tight_layout()
    return plt.gcf()


def plot_9_sex_vs_diabetes(df):
    plt.figure(figsize=(8, 5))
    ctab = pd.crosstab(df["Sex_Label"], df["Diabetes_Status"], normalize="index")
    sns.heatmap(ctab, annot=True, fmt=".1%", cmap="Blues", cbar=False)
    plt.title("Plot 9: Proportion of Diabetes Status by Sex", fontsize=16)
    plt.xlabel("Diabetes Status", fontsize=12)
    plt.ylabel("Sex", fontsize=12)
    plt.tight_layout()
    return plt.gcf()


def plot_10_income_vs_diabetes(df):
    plt.figure(figsize=(10, 6))
    income_prevalence = (
        df[df["Diabetes_012"] == 2.0]
        .groupby("Income")
        .size()
        .div(df.groupby("Income").size())
        * 100
    )
    income_prevalence = income_prevalence.fillna(0).sort_index()

    sns.barplot(x=income_prevalence.index, y=income_prevalence.values, palette="RdYlGn_r")
    plt.title(
        "Plot 10: Diabetes Prevalence by Income Category (1=Low, 8=High)", fontsize=16
    )
    plt.xlabel("Income Category", fontsize=12)
    plt.ylabel("Percentage with Diabetes", fontsize=12)
    plt.tight_layout()
    return plt.gcf()


# ------------------------------
# Main Streamlit App
# ------------------------------
st.title("🩺 Diabetes Health Indicators App: EDA and Prediction")

if not df.empty and model is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🔮 Predict Status", "📈 EDA Plots"])

    # ==========================
    # TAB 1: DATA EXPLORER
    # ==========================
    with tab1:
        st.header("1. Dataset Overview")
        st.write(f"Data loaded successfully: {len(df)} rows and {len(df.columns)} columns.")

        if st.checkbox("Show Raw Data Sample"):
            st.dataframe(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe())

    # ==========================
    # TAB 2: PREDICTION
    # ==========================
    with tab2:
        st.header("2. Predict Your Health Status")
        st.markdown(
            "Adjust the values below to match your health profile, then click **Predict**."
        )

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            binary_map = {0: "No", 1: "Yes"}

            with col1:
                st.subheader("General Health")
                high_bp = st.selectbox(
                    "High Blood Pressure?", options=[0, 1], format_func=lambda x: binary_map[x]
                )
                high_chol = st.selectbox(
                    "High Cholesterol?", options=[0, 1], format_func=lambda x: binary_map[x]
                )
                chol_check = st.selectbox(
                    "Cholesterol Check in 5 yrs?", options=[0, 1],
                    format_func=lambda x: binary_map[x],
                )
                bmi = st.slider("BMI (Body Mass Index)", 12, 98, 28)
                gen_hlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

            with col2:
                st.subheader("Lifestyle & Habits")
                smoker = st.selectbox(
                    "Smoked >100 cigs in life?", options=[0, 1], format_func=lambda x: binary_map[x]
                )
                phys_activity = st.selectbox(
                    "Physical Activity (past 30 days)?",
                    options=[0, 1],
                    format_func=lambda x: binary_map[x],
                )
                fruits = st.selectbox(
                    "Consume Fruit 1+ times/day?",
                    options=[0, 1],
                    format_func=lambda x: binary_map[x],
                )
                veggies = st.selectbox(
                    "Consume Veggies 1+ times/day?",
                    options=[0, 1],
                    format_func=lambda x: binary_map[x],
                )
                hvy_alcohol = st.selectbox(
                    "Heavy Alcohol Consumption?",
                    options=[0, 1],
                    format_func=lambda x: binary_map[x],
                )

            with col3:
                st.subheader("Medical History & Demographics")
                stroke = st.selectbox(
                    "Ever had a Stroke?", options=[0, 1], format_func=lambda x: binary_map[x]
                )
                heart_disease = st.selectbox(
                    "Heart Disease or Attack?",
                    options=[0, 1],
                    format_func=lambda x: binary_map[x],
                )
                diff_walk = st.selectbox(
                    "Difficulty Walking?", options=[0, 1], format_func=lambda x: binary_map[x]
                )
                sex = st.selectbox(
                    "Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male"
                )
                age = st.slider("Age Category (1=18-24, 13=80+)", 1, 13, 8)

            st.markdown("---")
            col4, col5 = st.columns(2)
            with col4:
                ment_hlth = st.slider("Days of Poor Mental Health (past 30 days)", 0, 30, 0)
                phys_hlth = st.slider("Days of Poor Physical Health (past 30 days)", 0, 30, 0)

                submitted = st.form_submit_button("Predict Diabetes Status")

        if submitted:
            # Order must exactly match training feature order
            input_data = np.array(
                [[
                    high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease,
                    phys_activity, fruits, veggies, hvy_alcohol, any_healthcare, no_doc_cost,
                    gen_hlth, ment_hlth, phys_hlth, diff_walk, sex, age, education, income
                ]]
            )

            try:
                prediction_probs = model.predict(input_data)
                prediction_class = int(np.argmax(prediction_probs, axis=1)[0])

                st.markdown("### Prediction Result:")

                if prediction_class == 0:
                    st.success("**Result: No Diabetes** (Class 0)")
                elif prediction_class == 1:
                    st.warning("**Result: Prediabetes** (Class 1)")
                else:
                    st.error("**Result: Diabetes** (Class 2)")

                st.write(
                    "Prediction Probabilities (0: No, 1: Pre, 2: Yes):",
                    prediction_probs,
                )

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    # ==========================
    # TAB 3: EDA PLOTS
    # ==========================
    with tab3:
        st.header("3. Comprehensive Exploratory Data Analysis (EDA) Plots")
        st.markdown(
            "These 10 charts highlight key distributions and relationships with diabetes status."
        )

        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(plot_3_bmi_distribution(df))
        with col4:
            st.pyplot(plot_4_bmi_vs_diabetes(df))

        st.markdown("---")

        col5, col6 = st.columns(2)
        with col5:
            st.pyplot(plot_5_age_vs_diabetes(df))
        with col6:
            st.pyplot(plot_6_genhlth_vs_diabetes(df))

        st.markdown("---")

        col7, col8 = st.columns(2)
        with col7:
            st.pyplot(plot_7_highbp_vs_diabetes(df))
        with col8:
            st.pyplot(plot_8_physhlth_vs_diabetes(df))

        st.markdown("---")

        col9, col10 = st.columns(2)
        with col9:
            st.pyplot(plot_9_sex_vs_diabetes(df))
        with col10:
            st.pyplot(plot_10_income_vs_diabetes(df))

else:
    st.warning(
        "Please ensure both `health_diabetes.csv` and "
        "`trail_model_diabetes.keras` are correctly placed in the app's directory."
    )
