import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ML Model Prediction App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_path = Path(__file__).parent / 'model.pkl'
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    """Load the scaler (if exists)"""
    try:
        scaler_path = Path(__file__).parent / 'scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            return scaler
        else:
            st.info("No scaler file found. Proceeding without scaling.")
            return None
    except Exception as e:
        st.warning(f"Could not load scaler: {str(e)}")
        return None

# Load model and scaler
model = load_model()
scaler = load_scaler()

# Title and description
st.title("🤖 Machine Learning Prediction App")
st.markdown("---")
st.subheader("Enter Feature Values for Prediction")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This app uses a trained machine learning model to make predictions.
    
    **How to use:**
    1. Enter feature values in the input fields
    2. Click 'Predict' button
    3. View the prediction results
    """)
    
    st.header("📊 Model Info")
    if model:
        st.success("✅ Model loaded successfully")
        try:
            st.write(f"**Model Type:** {type(model).__name__}")
        except:
            pass
    else:
        st.error("❌ Model not loaded")
    
    if scaler:
        st.success("✅ Scaler loaded successfully")
    else:
        st.warning("⚠️ No scaler loaded")

# Main content area
if model is None:
    st.error("⚠️ Model could not be loaded. Please ensure 'model.pkl' exists in the app directory.")
    st.stop()

# Input section
st.markdown("### 📝 Input Features")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

# Example feature inputs (customize these based on your model)
# You can add/remove features as needed
with col1:
    feature1 = st.number_input(
        "Feature 1", 
        min_value=0.0, 
        max_value=100.0, 
        value=0.0,
        step=0.1,
        help="Enter value for Feature 1"
    )
    feature2 = st.number_input(
        "Feature 2", 
        min_value=0.0, 
        max_value=100.0, 
        value=0.0,
        step=0.1,
        help="Enter value for Feature 2"
    )
    feature3 = st.number_input(
        "Feature 3", 
        min_value=0.0, 
        max_value=100.0, 
        value=0.0,
        step=0.1,
        help="Enter value for Feature 3"
    )

with col2:
    feature4 = st.number_input(
        "Feature 4", 
        min_value=0.0, 
        max_value=100.0, 
        value=0.0,
        step=0.1,
        help="Enter value for Feature 4"
    )
    feature5 = st.number_input(
        "Feature 5", 
        min_value=0.0, 
        max_value=100.0, 
        value=0.0,
        step=0.1,
        help="Enter value for Feature 5"
    )
    # Example selectbox for categorical feature
    feature6 = st.selectbox(
        "Feature 6 (Category)",
        options=["Option A", "Option B", "Option C"],
        help="Select category for Feature 6"
    )

with col3:
    feature7 = st.number_input(
        "Feature 7", 
        min_value=0.0, 
        max_value=100.0, 
        value=0.0,
        step=0.1,
        help="Enter value for Feature 7"
    )
    feature8 = st.number_input(
        "Feature 8", 
        min_value=0.0, 
        max_value=100.0, 
        value=0.0,
        step=0.1,
        help="Enter value for Feature 8"
    )
    feature9 = st.number_input(
        "Feature 9", 
        min_value=0.0, 
        max_value=100.0, 
        value=0.0,
        step=0.1,
        help="Enter value for Feature 9"
    )

st.markdown("---")

# Prediction button
predict_button = st.button("🎯 Predict", use_container_width=True)

if predict_button:
    try:
        # Create feature array
        # Adjust this based on your model's features
        # Handle categorical features if needed
        feature6_encoded = {"Option A": 0, "Option B": 1, "Option C": 2}.get(feature6, 0)
        
        features = np.array([[
            feature1, feature2, feature3, feature4, feature5, 
            feature6_encoded, feature7, feature8, feature9
        ]])
        
        # Display input data
        with st.expander("📋 View Input Data"):
            input_df = pd.DataFrame(
                features,
                columns=[f"Feature {i+1}" for i in range(features.shape[1])]
            )
            st.dataframe(input_df, use_container_width=True)
        
        # Apply preprocessing (scaling)
        if scaler is not None:
            features_scaled = scaler.transform(features)
            st.info("✓ Preprocessing applied (scaling)")
        else:
            features_scaled = features
            st.info("✓ No preprocessing applied")
        
        # Make prediction
        with st.spinner("🔮 Making prediction..."):
            prediction = model.predict(features_scaled)
            
            # Try to get prediction probability if available
            try:
                prediction_proba = model.predict_proba(features_scaled)
                has_proba = True
            except:
                has_proba = False
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        result_col1, result_col2 = st.columns([1, 2])
        
        with result_col1:
            st.success("✅ Prediction Complete!")
            st.metric(
                label="Predicted Value",
                value=f"{prediction[0]}"
            )
        
        with result_col2:
            if has_proba:
                st.markdown("**Prediction Probabilities:**")
                proba_df = pd.DataFrame(
                    prediction_proba,
                    columns=[f"Class {i}" for i in range(prediction_proba.shape[1])]
                )
                st.dataframe(proba_df, use_container_width=True)
                
                # Visualize probabilities if classification
                if prediction_proba.shape[1] > 1:
                    st.bar_chart(proba_df.T)
        
        # Additional information
        st.info(f"""
        **Prediction Details:**
        - Model Type: {type(model).__name__}
        - Number of Features: {features.shape[1]}
        - Preprocessing: {'Applied' if scaler else 'Not Applied'}
        - Prediction: {prediction[0]}
        """)
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.error("Please check your input values and model compatibility.")
        with st.expander("🔍 Error Details"):
            st.code(str(e))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit 🎈 | Powered by Machine Learning 🤖</p>
</div>
""", unsafe_allow_html=True)
