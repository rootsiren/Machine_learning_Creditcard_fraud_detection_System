

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Load trained model

st.sidebar.header("⚡ Model Info")
model = joblib.load("fraud_rf_model.pkl")   # retrained RF model without PCA
st.sidebar.success("Model loaded ✅")
st.sidebar.write("Trained directly on Kaggle features (V1–V28, Time, Amount)")


# Required columns

required_cols = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12",
    "V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24",
    "V25","V26","V27","V28","Amount"
]

# App UI

st.title("💳 Credit Card Fraud Detection Dashboard")

st.subheader("1️⃣ Upload Transactions CSV")
uploaded_file = st.file_uploader("Upload a Kaggle-style creditcard.csv file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check columns
    if all(col in df.columns for col in required_cols):
        st.success("✅ File accepted")
        st.write("Preview:")
        st.dataframe(df.head())

        # Predictions
        X = df[required_cols]
        preds = model.predict(X)

        df["Prediction"] = preds
        st.subheader("2️⃣ Predictions")
        st.dataframe(df[["Time","Amount","Prediction"]].head(20))

        # Fraud stats
        fraud_count = (df["Prediction"] == 1).sum()
        legit_count = (df["Prediction"] == 0).sum()
        st.write(f"🟢 Legit transactions: {legit_count}")
        st.write(f"🔴 Fraud transactions: {fraud_count}")

        # Downloadable results
        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions CSV", csv_out, "predictions.csv", "text/csv")

    else:
        st.error("❌ Uploaded CSV does not have the required Kaggle columns.")

