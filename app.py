import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("rf_model.pkl")

# Page config
st.set_page_config(page_title="Breast Cancer Detector", layout="wide")

# Title
st.markdown(
    """
    <h2 style='text-align:center;'>🔬 Breast Cancer Detection (30 Features)</h2>
    <p style='text-align:center; font-size:17px;'>
        Provide the tumor feature measurements below to predict whether the tumor is <b>Benign</b> or <b>Malignant</b>.
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# 30 features
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Create 3 columns for cleaner UI
col1, col2, col3 = st.columns(3)

inputs = []

st.sidebar.header("📌 Enter 30 Features")

for i, feature in enumerate(feature_names):
    if i % 3 == 0:
        val = col1.number_input(feature, format="%.4f", value=0.0)
    elif i % 3 == 1:
        val = col2.number_input(feature, format="%.4f", value=0.0)
    else:
        val = col3.number_input(feature, format="%.4f", value=0.0)
    inputs.append(val)

inputs = np.array(inputs).reshape(1, -1)

# Centered Predict Button
center_btn = st.columns([4, 2, 4])
with center_btn[1]:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

# Show result
if predict_btn:
    prediction = model.predict(inputs)[0]
    proba = model.predict_proba(inputs)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    # Result Card
    if prediction == 1:
        color = "#ff4c4c"
        label = "Malignant"
        icon = "⚠️"
    else:
        color = "#4CAF50"
        label = "Benign"
        icon = "✅"

    st.markdown(
        f"""
        <div style='background-color:{color};
                    padding:25px;
                    border-radius:15px;
                    text-align:center;
                    color:white;
                    font-size:26px;
                    font-weight:bold;'>
            {icon} The tumor is <b>{label}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability Table
    st.subheader("📊 Prediction Probabilities")
    st.write(f"**Benign Probability:** {proba[0]:.4f}")
    st.write(f"**Malignant Probability:** {proba[1]:.4f}")
