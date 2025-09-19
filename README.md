# Machine_learning_Creditcard_fraud_detection_System
# 💳 Credit Card Fraud Detection

## 📌 Overview
This project implements a **machine learning model** to detect fraudulent credit card transactions.  
The model is trained on the **Kaggle Credit Card Fraud Detection dataset** and handles the **class imbalance problem** effectively.  

✅ Demonstrates skills in:
- Data preprocessing & handling imbalanced datasets  
- Model training & evaluation (Random Forest)  
- Saving/loading models with `joblib`  
- Building interactive dashboards with **Streamlit**  

---

## 📊 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Rows:** 284,807 transactions  
- **Features:**  
  - `Time` – seconds elapsed between transactions  
  - `V1–V28` – PCA-transformed features (anonymized)  
  - `Amount` – transaction amount  
  - `Class` – target variable (0 = legit, 1 = fraud)  

⚠️ Dataset is highly **imbalanced** → frauds ≈ 0.17% of all transactions.

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib, Imbalanced-learn  
- **Optional Frontend:** Streamlit  

---

## 🚀 Model Training
1. Loaded and preprocessed dataset.  
2. Handled imbalance using **SMOTE** (Synthetic Minority Oversampling).  
3. Trained **Random Forest Classifier** with tuned hyperparameters.  
4. Saved model as `fraud_rf_model.pkl`.  

### 📈 Evaluation
- **Precision (fraud):** 0.82  
- **Recall (fraud):** 0.83  
- **F1-score (fraud):** 0.82  
- **Accuracy:** 99.9%  

**Confusion Matrix:**
[[56846, 18], # True Legit, False Fraud
[ 17, 81]] # False Legit, True Fraud


**📂 Files **

fraud_rf_model.pkl → Trained Random Forest model

demo_transactions.csv → 20 legit + 10 fraud unseen samples for testing

main.py → Streamlit dashboard


## ⚠️ Drawbacks & Limitations

- **Only PCA Features:** The dataset provides anonymized PCA-transformed features (`V1–V28`). No raw transactional details (merchant, location, device info, IP, etc.) are available.  
- **Limited Real-World Interpretability:** Since the features are anonymized, it is not possible to explain *why* a transaction is marked fraudulent in business terms.  
- **Not Deployable as-is:** In real banking systems, additional signals (user behavior, geolocation, merchant risk scores, etc.) are needed. This dataset alone cannot build a production-ready fraud detection system.  
- **Generalization Risk:** Fraud patterns change over time; the 2013 dataset may not reflect modern fraud techniques.  
