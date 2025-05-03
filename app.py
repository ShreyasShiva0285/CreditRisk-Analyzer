import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc
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

set_background("Cal.jpeg")  # Ensure this file is in the same directory

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

# --- Load Sample Test Data (Optional ROC Support) ---
X_test, y_test = None, None
try:
    test_data = pd.read_csv("sample_test_data.csv")  # Optional
    if "target" in test_data.columns:
        X_test = test_data.drop(columns=["target"])
        y_test = test_data["target"]
except:
    pass

# --- Prediction Button ---
if st.button("Predict"):
    data = preprocess()
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    result = "❌ Not Good (Likely to Default)" if prediction == 1 else "✅ Good (Low Default Risk)"
    st.subheader(f"Prediction: {result}")
    st.metric(label="Default Risk Probability", value=f"{proba:.2%}")

    if credit_score < 600:
        st.info("💡 Tip: A credit score below 600 may significantly increase default risk. Try to improve your credit behavior.")

    # --- Feature Importance ---
    st.subheader("📈 Feature Importance")
    importance = model.feature_importances_
    features = data.columns
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by="Importance")
    st.bar_chart(imp_df.set_index("Feature"))

    # --- ROC Curve (if test data available) ---
        if X_test is not None and y_test is not None:
        st.subheader("📊 ROC Curve")

        # Preprocess X_test to match the model's expected format
        try:
            X_test_cleaned = X_test.copy()

            # If there are any object columns, encode them as categorical codes
            for col in X_test_cleaned.columns:
                if X_test_cleaned[col].dtype == 'object':
                    X_test_cleaned[col] = X_test_cleaned[col].astype('category').cat.codes

            # Ensure only numeric columns are used
            X_test_cleaned = X_test_cleaned.select_dtypes(include='number')

            y_scores = model.predict_proba(X_test_cleaned)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")
            ax.legend(loc="lower right")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"🚨 Error while plotting ROC curve: {e}")
