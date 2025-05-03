import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt

# --- MUST BE FIRST: Set Streamlit Page Config ---
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="📉",
    layout="centered"
)

# --- Load Model ---
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- Inputs ---
customer_name = st.text_input("Customer's Name", help="Enter the name of the applicant.")
customer_age = st.number_input("Customer's Age", min_value=18, max_value=120, help="Enter the age of the applicant.")

loan_amount = st.number_input("💷 Loan Amount (£)", min_value=0.0, format="%.2f", help="Total loan amount requested by the applicant.")
term = st.selectbox("Loan Term", ["36 months", "60 months"], help="Loan repayment duration.")
income = st.number_input("💷 Annual Income (£)", min_value=0.0, format="%.2f", help="Applicant's annual income before tax.")
credit_score = st.slider("Credit Score", 300, 850, help="Higher credit score reduces default risk.")
employment_length = st.slider("Employment Length (years)", 0, 40, help="Years employed at current job.")
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"], help="Applicant's housing status.")
purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvement", "Credit Card", "Other"], help="Purpose of the loan.")
interest_rate = st.selectbox("Interest Rate", [0.05, 0.06, 0.07, 0.08, 0.09], help="Select an interest rate for the loan.")

# --- Credit Score Status ---
score_color = "🔴 Poor"
if credit_score > 750:
    score_color = "🟢 Excellent"
elif credit_score > 650:
    score_color = "🟡 Fair"
elif credit_score > 550:
    score_color = "🟠 Low"
st.markdown(f"**Credit Score Status:** {score_color}")

# --- Preprocessing Function ---
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

# --- Prediction Button ---
if st.button("Predict"):
    data = preprocess()
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    # --- Risk Factors Summary ---
    months = 36 if term == "36 months" else 60
    monthly_rate = interest_rate / 12
    monthly_payment = (loan_amount * monthly_rate) / (1 - (1 + monthly_rate) ** -months)
    total_repayment = monthly_payment * months
    total_interest = total_repayment - loan_amount

    # Display Customer Name and Age
    st.subheader(f"🔑 Customer Information")
    st.markdown(f"**Customer's Name:** {customer_name}")
    st.markdown(f"**Age:** {customer_age} years")

# --- Risk Factors Summary ---
if st.button("Predict"):
    data = preprocess()
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    # --- Risk Factors Summary ---
    months = 36 if term == "36 months" else 60
    monthly_rate = interest_rate / 12
    monthly_payment = (loan_amount * monthly_rate) / (1 - (1 + monthly_rate) ** -months)
    total_repayment = monthly_payment * months
    total_interest = total_repayment - loan_amount

    # Display Customer Name and Age
    st.subheader(f"🔑 Customer Information")
    st.markdown(f"**Customer's Name:** {customer_name}")
    st.markdown(f"**Age:** {customer_age} years")

    # Risk Summary
    st.subheader("📊 Risk Factors Summary")
    
    # Debt-to-Income Ratio (DTI)
    monthly_income = income / 12  # Monthly income
    monthly_debt_payments = 500  # Example debt payments
    dti = monthly_debt_payments / monthly_income
    st.markdown(f"**Debt-to-Income Ratio (DTI):** {dti:.2f} ({'Good' if dti < 0.36 else 'High Risk'})")

    # Loan-to-Value Ratio (LTV) (for secured loans)
    property_value = 300000  # Example for home loans
    ltv = loan_amount / property_value
    st.markdown(f"**Loan-to-Value Ratio (LTV):** {ltv:.2f} ({'Good' if ltv < 0.8 else 'High Risk'})")

    # Loan cost details
    st.markdown(f"**Monthly Payment:** £{monthly_payment:.2f}")
    st.markdown(f"**Total Repayment Over {months // 12} Years:** £{total_repayment:.2f}")
    st.markdown(f"**Total Interest Paid:** £{total_interest:.2f}")

    # Loan Decision Result
    result = "❌ Not Good (Likely to Default)" if prediction == 1 else "✅ Good (Low Default Risk)"
    st.subheader(f"Prediction: {result}")
    st.metric(label="Default Risk Probability", value=f"{proba:.2%}")

    if prediction == 1:
        st.info("💡 The applicant is at higher risk of default. We recommend considering lower loan amounts or securing the loan with an asset.")
    else:
        st.success("✅ The applicant is at a low risk of default. The loan seems manageable given the financial status.")
