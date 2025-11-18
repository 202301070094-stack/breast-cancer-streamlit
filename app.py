import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("rf_model.pkl")

st.title("Breast Cancer Detection Using Random Forest")
st.write("Enter the tumor features below to predict whether the tumor is benign or malignant.")

# 30 features
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

inputs = []

st.sidebar.header("Enter Input Features")

for feature in feature_names:
    val = st.sidebar.number_input(f"{feature}", value=0.0)
    inputs.append(val)

inputs = np.array(inputs).reshape(1, -1)

if st.sidebar.button("Predict"):
    prediction = model.predict(inputs)[0]
    proba = model.predict_proba(inputs)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ The tumor is **Malignant**.")
    else:
        st.success("✅ The tumor is **Benign**.")

    st.write(f"**Probability (Benign):** {proba[0]:.4f}")
    st.write(f"**Probability (Malignant):** {proba[1]:.4f}")
