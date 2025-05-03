import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from PIL import Image

# --- Set Page Config ---
st.set_page_config(page_title="Loan Default Predictor", page_icon="📉", layout="centered")

# --- Optional Logo ---
try:
    logo = Image.open("logo.png")
    st.image(logo, width=120)
except:
    st.markdown("## 📉 Loan Default Prediction App")

st.markdown("Enter applicant info to predict loan default risk.")

# --- Load Model ---
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- Check if Model Supports `predict_proba()` ---
if hasattr(model, 'predict_proba'):
    st.success("✅ The model supports `predict_proba()` for predicting probabilities.")
else:
    st.error("❌ Model does not support `predict_proba()`. Ensure you're using a classifier like XGBoost or RandomForest.")

# --- Input Fields ---
loan_amount = st.number_input("💷 Loan Amount (£)", min_value=0.0, format="%.2f")
term = st.selectbox("Loan Term", ["36 months", "60 months"])
income = st.number_input("💷 Annual Income (£)", min_value=0.0, format="%.2f")
credit_score = st.slider("Credit Score", 300, 850)
employment_length = st.slider("Employment Length (years)", 0, 40)
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])
purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvement", "Credit Card", "Other"])

# --- Credit Score Indicator ---
score_color = "🔴 Poor"
if credit_score > 750:
    score_color = "🟢 Excellent"
elif credit_score > 650:
    score_color = "🟡 Fair"
elif credit_score > 550:
    score_color = "🟠 Low"
st.markdown(f"**Credit Score Status:** {score_color}")

# --- Preprocess Inputs ---
def preprocess():
    term_encoded = 0 if term == "36 months" else 1
    home = {"Rent": 0, "Own": 1, "Mortgage": 2}[home_ownership]
    purp = {"Debt Consolidation": 0, "Home Improvement": 1, "Credit Card": 2, "Other": 3}[purpose]
    return pd.DataFrame([[
        loan_amount, term_encoded, income, credit_score,
        employment_length, home, purp
    ]], columns=[
        "loan_amount", "term", "annual_income", "credit_score",
        "employment_length", "home_ownership", "purpose"
    ])

# --- Predict Button ---
if st.button("Predict"):
    data = preprocess()
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    result = "❌ Not Good (Likely to Default)" if prediction == 1 else "✅ Good (Low Default Risk)"
    st.subheader(f"Prediction: {result}")
    st.metric(label="Default Risk Probability", value=f"{proba:.2%}")

    if credit_score < 600:
        st.info("💡 Tip: A credit score below 600 may significantly increase default risk. Try to improve your credit behavior.")

    # --- SHAP Feature Importance ---
   # --- SHAP Feature Importance ---
st.subheader("📊 SHAP Feature Importance")
try:
    # Generate SHAP explainer object
    explainer = shap.Explainer(model)
    
    # Compute SHAP values
    shap_values = explainer(data)
    
    # Plot the SHAP values using SHAP's built-in plotting method
    shap.summary_plot(shap_values, data)
    
    # SHAP automatically displays the plot, no need for st.pyplot()
except Exception as e:
    st.warning(f"⚠️ SHAP plot could not be generated. Reason: {e}")
