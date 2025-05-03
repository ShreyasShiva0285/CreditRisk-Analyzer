import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# Load model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📉 Loan Default Prediction App")
st.markdown("Enter applicant info to predict default risk.")

# Inputs
loan_amount = st.number_input("Loan Amount", min_value=0)
term = st.selectbox("Term", ["36 months", "60 months"])
income = st.number_input("Annual Income", min_value=0)
credit_score = st.slider("Credit Score", 300, 850)
employment_length = st.slider("Employment Length (years)", 0, 40)
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])
purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvement", "Credit Card", "Other"])

# Preprocess input
def preprocess():
    term_encoded = 0 if term == "36 months" else 1
    home = {"Rent": 0, "Own": 1, "Mortgage": 2}[home_ownership]
    purp = {
        "Debt Consolidation": 0,
        "Home Improvement": 1,
        "Credit Card": 2,
        "Other": 3
    }[purpose]
    return pd.DataFrame([[
        loan_amount, term_encoded, income, credit_score,
        employment_length, home, purp
    ]], columns=[
        "loan_amount", "term", "annual_income", "credit_score",
        "employment_length", "home_ownership", "purpose"
    ])

if st.button("Predict"):
    data = preprocess()
    pred = model.predict(data)[0]
    result = "❌ Likely to Default" if pred == 1 else "✅ Not Likely to Default"
    st.subheader(f"Prediction: {result}")
