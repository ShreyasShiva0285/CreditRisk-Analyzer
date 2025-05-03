import streamlit as st
import pandas as pd
import pickle
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
customer_name = st.text_input("Customer's Name", help="Enter the name of the applicant.")
customer_age = st.number_input("Customer's Age", min_value=18, max_value=120, help="Enter the age of the applicant.")
loan_amount = st.number_input("💷 Loan Amount (£)", min_value=0.0, format="%.2f")
asset_value = st.number_input("💎 Asset Value (£)", min_value=0.0, format="%.2f", help="Enter the value of the asset (e.g., house value) securing the loan.")
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

# --- Check for valid term before processing ---
if 'term' not in st.session_state:
    st.session_state.term = "36 months"  # Default value

term = st.selectbox("Loan Term", ["36 months", "60 months"], index=["36 months", "60 months"].index(st.session_state.term))

# --- Preprocess Input ---
def preprocess():
    st.write(f"Term selected: {term}")  # Debugging: Check the term value
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

# --- Calculate Additional Ratios ---
def calculate_ratios():
    # Debt-to-Income Ratio (DTI)
    total_debt_payments = loan_amount * 0.05  # Assume 5% of loan amount is monthly debt repayment (this could be adjusted based on more realistic values)
    dti_ratio = total_debt_payments / income
    
    # Loan-to-Value Ratio (LTV)
    ltv_ratio = loan_amount / asset_value if asset_value != 0 else 0
    
    # Credit Utilization
    credit_utilization = 0.3  # Placeholder value for now. You can replace it with the actual data if available.
    
    return dti_ratio, ltv_ratio, credit_utilization

# --- Predict ---
if st.button("Predict"):
    # Check if all input fields are filled
    if not customer_name or customer_age < 18 or loan_amount <= 0 or income <= 0 or credit_score <= 0:
        st.error("Please fill in all the fields correctly.")
    else:
        data = preprocess()
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0][1]

        # Display Customer Info
        st.subheader(f"🔑 Customer Information")
        st.markdown(f"**Customer's Name:** {customer_name}")
        st.markdown(f"**Age:** {customer_age} years")

        # Loan Default Prediction
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

        # Additional Financial Metrics
        dti_ratio, ltv_ratio, credit_utilization = calculate_ratios()

        st.markdown(f"**Monthly Payment:** £{monthly_payment:.2f}")
        st.markdown(f"**Total Repayment Over {months//12} Years:** £{total_repayment:.2f}")
        st.markdown(f"**Total Interest Paid:** £{total_interest:.2f}")

        # Display Financial Metrics
        st.subheader("📊 Additional Financial Metrics")
        st.markdown(f"**Debt-to-Income Ratio (DTI):** {dti_ratio:.2%}")
        st.markdown(f"**Loan-to-Value Ratio (LTV):** {ltv_ratio:.2%}")
        st.markdown(f"**Credit Utilization:** {credit_utilization:.2%}")
