import streamlit as st
import pandas as pd
import pickle
import base64
import numpy as np

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="📉",
    layout="centered"
)

# --- Load Model ---
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- App Header ---
st.title("📉 Loan Default Prediction App")
st.markdown("Enter applicant details to predict the likelihood of defaulting on a loan.")

# --- Input Fields ---
loan_amount = st.number_input("💷 Loan Amount (£)", min_value=0.0, format="%.2f")
term = st.selectbox("Loan Term", ["36 months", "60 months"])
income = st.number_input("💷 Annual Income (£)", min_value=0.0, format="%.2f")
credit_score = st.slider("Credit Score", 300, 850)

# --- Show Credit Score Status ---
if credit_score > 750:
    st.markdown("**Credit Score Status:** 🟢 Excellent")
elif credit_score > 650:
    st.markdown("**Credit Score Status:** 🟡 Fair")
elif credit_score > 550:
    st.markdown("**Credit Score Status:** 🟠 Low")
else:
    st.markdown("**Credit Score Status:** 🔴 Poor")

employment_length = st.slider("Employment Length (years)", 0, 40)
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])

purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvement", "Credit Card", "Other"])
interest_rate = 0.1  # Default interest
if purpose == "Debt Consolidation":
    interest_rate = st.slider("Interest Rate for Debt Consolidation", 5.0, 25.0, step=0.5) / 100
elif purpose == "Home Improvement":
    interest_rate = st.slider("Interest Rate for Home Improvement", 5.0, 20.0, step=0.5) / 100
else:
    interest_rate = st.slider("Interest Rate", 5.0, 30.0, step=0.5) / 100

# --- Preprocess Input ---
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

# --- Predict ---
if st.button("Predict"):
    data = preprocess()
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.subheader("Prediction: ❌ Not Good (Likely to Default)")
        st.markdown(f"**Probability of Default:** {proba:.2%}")
        st.warning("💡 Suggestion: Improve credit score, reduce loan request, or increase income.")
    else:
        st.subheader("Prediction: ✅ Good (Low Default Risk)")
        st.markdown(f"**Probability of Default:** {proba:.2%}")
        st.success("💡 Suggestion: You have strong loan approval chances!")

    # --- Risk Summary ---
    st.subheader("📊 Risk Factors Summary")
    months = 36 if term == "36 months" else 60
    monthly_rate = interest_rate / 12
    monthly_payment = (loan_amount * monthly_rate) / (1 - (1 + monthly_rate) ** -months)
    total_repayment = monthly_payment * months
    total_interest = total_repayment - loan_amount

    st.markdown(f"**Monthly Payment:** £{monthly_payment:.2f}")
    st.markdown(f"**Total Repayment Over {months//12} Years:** £{total_repayment:.2f}")
    st.markdown(f"**Total Interest Paid:** £{total_interest:.2f}")
