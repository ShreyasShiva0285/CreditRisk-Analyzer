import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
from PIL import Image
import base64

# --- Page Config ---
st.set_page_config(page_title="Loan Default Predictor", page_icon="📉", layout="centered")

# --- Set Background (optional) ---
def set_background(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

set_background("download.jpeg")

# --- Logo ---
try:
    logo = Image.open("logo.png")
    st.image(logo, width=120)
except:
    st.markdown("## 📉 Loan Default Prediction App")

st.markdown("Enter applicant info to predict loan default risk.")

# --- Load Model ---
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- Inputs ---
loan_amount = st.number_input("💷 Loan Amount (£)", min_value=0.0, format="%.2f")
term = st.selectbox("Loan Term", ["36 months", "60 months"])
income = st.number_input("💷 Annual Income (£)", min_value=0.0, format="%.2f")
credit_score = st.slider("Credit Score", 300, 850)
employment_length = st.slider("Employment Length (years)", 0, 40)
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])
purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvement", "Credit Card", "Other"])

# --- Credit Score Status ---
score_color = "🔴 Poor"
if credit_score > 750:
    score_color = "🟢 Excellent"
elif credit_score > 650:
    score_color = "🟡 Fair"
elif credit_score > 550:
    score_color = "🟠 Low"
st.markdown(f"**Credit Score Status:** {score_color}")

# --- Preprocessing ---
def preprocess():
    term_encoded = 0 if term == "36 months" else 1
    home = {"Rent": 0, "Own": 1, "Mortgage": 2}[home_ownership]
    purp = {"Debt Consolidation": 0, "Home Improvement": 1, "Credit Card": 2, "Other": 3}[purpose]
    return pd.DataFrame([[loan_amount, term_encoded, income, credit_score, employment_length, home, purp]],
                        columns=["loan_amount", "term", "annual_income", "credit_score",
                                 "employment_length", "home_ownership", "purpose"])

# --- Prediction Button ---
if st.button("Predict"):
    data = preprocess()
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    result = "❌ Not Good (Likely to Default)" if prediction == 1 else "✅ Good (Low Default Risk)"
    st.subheader(f"Prediction: {result}")
    st.metric(label="Default Risk Probability", value=f"{proba:.2%}")
    st.progress(int(proba * 100))

    # --- Risk Summary Table ---
   # --- Risk Summary Table ---
st.subheader("🔍 Risk Factors Summary")
risk_table = pd.DataFrame(columns=["Feature", "Value", "Risk Indicator"])

# Loan calculations
months = 36 if term == "36 months" else 60
annual_interest_rate = 0.12  # Assume 12% annual interest
monthly_interest_rate = annual_interest_rate / 12
monthly_payment = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** months) / ((1 + monthly_interest_rate) ** months - 1) if loan_amount > 0 else 0
total_repayment = monthly_payment * months
total_interest = total_repayment - loan_amount

def add_risk(feature, value, condition, message):
    indicator = "⚠️ " + message if condition else "✅ OK"
    return {"Feature": feature, "Value": value, "Risk Indicator": indicator}

risk_rows = [
    add_risk("Credit Score", credit_score, credit_score < 600, "Low Credit Score"),
    add_risk("Annual Income", f"£{income:,.2f}", income < 30000, "Low Income"),
    add_risk("Loan Amount", f"£{loan_amount:,.2f}", loan_amount > income * 0.5, "High Relative Loan"),
    add_risk("Employment Length", f"{employment_length} years", employment_length < 2, "Unstable Employment"),
    add_risk("Loan Term", term, term == "60 months", "Long Term Increases Risk"),
    {"Feature": "Monthly Repayment", "Value": f"£{monthly_payment:,.2f}", "Risk Indicator": "ℹ️ Calculated"},
    {"Feature": "Total Repayment", "Value": f"£{total_repayment:,.2f}", "Risk Indicator": "ℹ️ Includes Interest"},
    {"Feature": "Estimated Interest", "Value": f"£{total_interest:,.2f}", "Risk Indicator": "ℹ️ Over Loan Term"},
]

risk_table = pd.DataFrame(risk_rows)
st.dataframe(risk_table, use_container_width=True)
