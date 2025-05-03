import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import io

# Load model
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📉 Loan Default Prediction App")
st.markdown("Enter applicant info to predict default risk.")

# Inputs with tooltips
loan_amount = st.number_input("💷 Loan Amount (£)", min_value=0, format="%.2f",
                              help="Total loan amount requested by the applicant.")
term = st.selectbox("Loan Term", ["36 months", "60 months"], help="Loan repayment duration.")
income = st.number_input("💷 Annual Income (£)", min_value=0, format="%.2f",
                         help="Applicant's annual income before tax.")

# Credit Score input with color indicator
credit_score = st.slider("Credit Score", 300, 850, help="Higher credit score reduces default risk.")
score_color = "🔴 Poor"
if credit_score > 750:
    score_color = "🟢 Excellent"
elif credit_score > 650:
    score_color = "🟡 Fair"
elif credit_score > 550:
    score_color = "🟠 Low"
st.markdown(f"Credit Score Status: **{score_color}**")

employment_length = st.slider("Employment Length (years)", 0, 40, help="Years employed at current job.")
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"],
                              help="Current housing status of the applicant.")
purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvement", "Credit Card", "Other"],
                       help="Purpose of the loan.")

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

# Predict and display
if st.button("Predict"):
    data = preprocess()
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]  # Probability of default

    result = "❌ Not Good (Likely to Default)" if prediction == 1 else "✅ Good (Low Default Risk)"
    st.subheader(f"Prediction: {result}")
    st.metric(label="Default Risk Probability", value=f"{proba:.2%}")

    # Feature importance
    st.subheader("📈 Feature Importance")
    importance = model.feature_importances_
    features = data.columns
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by="Importance")
    st.bar_chart(imp_df.set_index("Feature"))

    # Tip based on credit score
    if credit_score < 600:
        st.info("💡 Tip: A credit score below 600 may significantly increase default risk. Try to improve your credit behavior.")

    # Download prediction result
    data["Prediction"] = "Not Good" if prediction == 1 else "Good"
    data["Default_Risk_Probability"] = f"{proba:.2%}"
    csv = data.to_csv(index=False)
    st.download_button("📥 Download Prediction Result", csv, file_name="loan_prediction.csv", mime="text/csv")

# UI Tip
st.markdown("🌗 **Tip:** Use the gear icon in the top-right corner to toggle Light/Dark mode in Streamlit.")
