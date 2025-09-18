# fraud_detection_app_raw_only.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection Dashboard")
st.sidebar.header("Model Settings")

# -----------------------------
# Load trained model & preprocessing objects
# -----------------------------
model = joblib.load("fraud_rf_model.pkl")          # Your trained RandomForest model
scaler = joblib.load("scaler.pkl")                # Scaler for 'Amount' used in training
pca = joblib.load("pca_transform.pkl")           # PCA object from training
best_thresh = 0.5
st.sidebar.write(f"Prediction threshold: {best_thresh}")

# -----------------------------
# CSV Upload
# -----------------------------
st.subheader("Upload CSV with raw transactions")
uploaded_file = st.file_uploader("CSV should contain only 'Time' and 'Amount'", type="csv")

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    # Check columns
    if not set(['Time','Amount']).issubset(raw_df.columns):
        st.error("CSV must contain columns: 'Time' and 'Amount'")
    else:
        st.write("Preview of uploaded transactions:")
        st.dataframe(raw_df.head())

        # Scale 'Amount'
        raw_df['Amount'] = scaler.transform(raw_df[['Amount']])

        # Apply PCA to generate features for the model
        X_features = pca.transform(raw_df)

        # Predict
        probs = model.predict_proba(X_features)[:, 1]
        preds = (probs >= best_thresh).astype(int)

        # Add results to dataframe
        raw_df['Fraud_Probability'] = probs
        raw_df['Predicted_Class'] = ['Fraud' if x == 1 else 'Legit' for x in preds]

        st.subheader("Prediction Results")
        st.dataframe(raw_df)

        # Allow download of predictions
        csv = raw_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

else:
    st.info("Upload a CSV file containing raw transactions to predict fraud.")
