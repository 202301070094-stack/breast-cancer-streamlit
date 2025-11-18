import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("rf_model.pkl")

# --------------------------- #
#        PAGE LAYOUT
# --------------------------- #
st.set_page_config(page_title="Breast Cancer Detector", layout="wide")

st.markdown(
    """
    <h2 style='text-align: center;'>🔬 Breast Cancer Detection (10 Features)</h2>
    <p style='text-align: center; font-size:16px;'>
        Enter the tumor measurements below to check whether the cancer is likely <b>Benign</b> or <b>Malignant</b>.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# --------------------------- #
#     TOP 10 IMPORTANT FEATURES
# --------------------------- #
feature_names = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "radius_worst"
]

inputs = []

st.sidebar.header("📌 Enter Input Values")
st.sidebar.write("Provide values for the 10 most important tumor features.")

# Two-column layout for cleaner UI
col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        val = col1.number_input(feature, value=0.0, format="%.4f")
    else:
        val = col2.number_input(feature, value=0.0, format="%.4f")
    inputs.append(val)

inputs = np.array(inputs).reshape(1, -1)

# --------------------------- #
#        PREDICT BUTTON
# --------------------------- #
center = st.columns([3, 1, 3])
with center[1]:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

# --------------------------- #
#        SHOW RESULTS
# --------------------------- #
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
        <div style="
            background-color:{color};
            padding:20px;
            border-radius:15px;
            text-align:center;
            color:white;
            font-size:24px;
            font-weight:bold;">
            {icon} The tumor is <b>{label}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability Table
    st.subheader("📊 Prediction Probabilities")
    st.write(f"**Benign:** {proba[0]:.4f}")
    st.write(f"**Malignant:** {proba[1]:.4f}")
