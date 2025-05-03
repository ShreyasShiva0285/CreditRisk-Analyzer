import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from PIL import Image
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="📉",
    layout="centered"
)

# --- Set Background (Optional) ---
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
    except:
        pass

set_background("download.jpeg")  # Optional background

# --- Logo (Optional) ---
try:
    logo = Image.open("logo.png")
    st.image(logo, width=120)
except:
    st.title("📉 Loan Default Prediction App")

st.markdown("Enter applicant information below:")

# --- Load Model ---
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- Input Form ---
loan_amount = st.number_input("Loan Amount (£)", min_value=0.0, format="%.2f")
term = st.selectbox("Loan Term", ["36 months", "60 months"])
income = st.number_input("Annual Income (£)", min_value=0.0, format="%.2f")
credit_score = st.slider("Credit Score", 300, 850)
employment_length = st.slider("Employment Length (years)", 0, 40)
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])
purpose = st.selectbox("Purpose", ["Debt Consolidation", "Home Improvement", "Credit Card", "Other"])

# --- Preprocess Input ---
def preprocess():
    term_encoded = 0 if term == "36 months" else 1
    home = {"Rent": 0, "Own": 1, "Mortgage": 2}[home_ownership]
    purp = {"Debt Consolidation": 0, "Home Improvement": 1, "Credit Card": 2, "Other": 3}[purpose]
    return pd.DataFrame([[loan_amount, term_encoded, income, credit_score,
                          employment_length, home, purp]],
                        columns=["loan_amount", "term", "annual_income", "credit_score",
                                 "employment_length", "home_ownership", "purpose"])

# --- Predict Button ---
if st.button("Predict"):
    data = preprocess()
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    result = "Not Good (Likely to Default)" if prediction == 1 else "Good (Low Default Risk)"
    st.subheader(f"Prediction: {result}")
    st.metric(label="Default Risk Probability", value=f"{proba:.2%}")

    # --- Feature Importance ---
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    features = data.columns
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by="Importance")
    st.bar_chart(imp_df.set_index("Feature"))

    # --- Download Prediction Result ---
    data["Prediction"] = result
    data["Default_Risk_Probability"] = f"{proba:.2%}"
    csv = data.to_csv(index=False)
    st.download_button("Download Prediction Result", csv, file_name="loan_prediction.csv", mime="text/csv")
