import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from PIL import Image
import base64

# --- MUST BE FIRST: Set Streamlit Page Config ---
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="📉",
    layout="centered"
)

# --- Set Background Image ---
def set_background(image_path):
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

set_background("download.jpeg")  # Ensure this file is in the same directory

# --- Optional Logo (top) ---
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
loan_amount = st.number_input("💷 Loan Amount (£)", min_value=0.0, format="%.2f",
                              help="Total loan amount requested by the applicant.")
term = st.selectbox("Loan Term", ["36 months", "60 months"], help="Loan repayment duration.")
income = st.number_input("💷 Annual Income (£)", min_value=0.0, format="%.2f",
                         help="Applicant's annual income before tax.")
credit_score = st.slider("Credit Score", 300, 850, help="Higher credit score reduces default risk.")
employment_length = st.slider("Employment Length (years)", 0, 40, help="Years employed at current job.")
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"], help="Applicant's housing status.")
purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvement", "Credit Card", "Other"],
                       help="Purpose of the loan.")

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

# --- Credit Score Status ---
def credit_score_status(credit_score):
    score_color = "🔴 Poor"
    status_message = "You have a poor credit score, which may significantly impact your ability to get a loan."
    if credit_score > 750:
        score_color = "🟢 Excellent"
        status_message = "Excellent credit score! You're in a great position for getting a loan."
    elif credit_score > 650:
        score_color = "🟡 Fair"
        status_message = "Fair credit score. You're eligible for loans but might face higher interest rates."
    elif credit_score > 550:
        score_color = "🟠 Low"
        status_message = "Low credit score. You might face difficulties getting approved for a loan."

    return score_color, status_message

# --- Prediction Button ---
if st.button("Predict"):
    data = preprocess()
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    result = "❌ Not Good (Likely to Default)" if prediction == 1 else "✅ Good (Low Default Risk)"
    st.subheader(f"Prediction: {result}")
    st.metric(label="Default Risk Probability", value=f"{proba:.2%}")

    # --- Credit Score Status Below Credit Score ---
    score_color, status_message = credit_score_status(credit_score)
    st.markdown(f"**Credit Score Status:** {score_color}")
    st.markdown(f"**Status Message:** {status_message}")

    # --- Additional Feedback and Tips ---
    if prediction == 1:
        st.warning("**Tip:** Improve your credit score and reduce existing debts to lower the default risk. Consider exploring other financial assistance options, such as debt counseling.")
    else:
        st.success("**Tip:** Great! Maintain a healthy credit score and ensure timely repayments to keep your loan eligibility high.")

    if credit_score < 600:
        st.info("💡 Tip: A credit score below 600 may significantly increase default risk. Try to improve your credit behavior.")

    # --- Feature Importance ---
    st.subheader("📈 Feature Importance")
    importance = model.feature_importances_
    features = data.columns
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by="Importance")
    st.bar_chart(imp_df.set_index("Feature"))

    # --- Download Results ---
    data["Prediction"] = "Not Good" if prediction == 1 else "Good"
    data["Default_Risk_Probability"] = f"{proba:.2%}"
    csv = data.to_csv(index=False)
    st.download_button("📥 Download Prediction Result", csv, file_name="loan_prediction.csv", mime="text/csv")

# --- File Upload for Test Data (Optional) ---
uploaded_file = st.file_uploader("Upload CSV with 'target' column", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "target" in df.columns:
        X_test = df.drop(columns=["target"])
        y_test = df["target"]
        st.success("Test data uploaded. Click 'Predict' to view the results.")

# --- UI Tip ---
st.markdown("🌗 Tip: Use the gear icon (⚙️) to toggle between Light and Dark mode in Streamlit.")
