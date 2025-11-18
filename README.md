# Breast Cancer Detection using Random Forest & Streamlit

This project builds and deploys a machine learning model for breast cancer detection using a dataset of more than 9000 samples. The workflow includes preprocessing, model training, and deployment using Streamlit Cloud.

## 📌 Project Features
- 9000+ rows breast cancer dataset  
- 30 numerical tumor features  
- Random Forest Classifier  
- Streamlit web app with user-friendly UI  
- Cloud deployment using Streamlit Cloud  
- Model saved as `rf_model.pkl`

---

## 🔧 Steps Followed

### 1. Data Preprocessing (Colab)
- Loaded dataset  
- Removed irrelevant columns  
- Encoded target variable  
- Used 30 numeric tumor features  
- Train/test split  

### 2. Model Training
- Trained a Random Forest Classifier  
- Evaluated performance  
- Saved model with joblib:

joblib.dump(model, "rf_model.pkl")

markdown
Copy code

### 3. Streamlit App Development
- Sidebar inputs for 30 features  
- Predict button  
- Shows:
  - Benign / Malignant  
  - Prediction probabilities  

### 4. Deployment (Streamlit Cloud)
- Uploaded all files to GitHub:
  - `app.py`
  - `rf_model.pkl`
  - `requirements.txt`
- Streamlit Cloud automatically deployed the app  
- Public URL generated  

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py