# 📊 Customer Churn Prediction — Internship Project

**Data Science Intern @ Persevex (Nov 2025 – Jan 2026)**

An end-to-end data science and machine learning project analyzing customer behavior to predict account churn for an e-commerce platform.

## 🔗 Project Components
- Jupyter Notebook: `customer_churn_analysis.ipynb`
- Production Pipeline: `optimized_model.pkl`

## 📋 Project Phases

### Phase 1 – Problem Statement & Business Impact
- **The Challenge**: Rising competition makes retaining active e-commerce accounts difficult. Since one account can represent multiple users, losing a single account results in a compounding loss of active customers.
- **Objective**: Build a high-precision churn forecasting pipeline to trigger targeted, segmented offers for at-risk accounts.
- **Why it matters**: Retaining an existing account is significantly cheaper than acquiring a new one, making early churn prediction valuable for marketing efficiency.

### Phase 2 – Exploratory Data Analysis (EDA)
- Overall churn rate observed: **16.84%** *(verify against notebook)*
- Churn is concentrated in early account lifecycle — highest drop-off in the first 6 months *(verify exact ratio against notebook)*
- Accounts with 3–4 users showed the highest churn volume
- Positive correlation found between customer care complaints (`Complain_ly`) and churn
- Tier 3 cities and single (unmarried) account holders showed relatively higher churn *(verify exact % against notebook)*

### Phase 3 – Data Cleaning & Preprocessing
- Fixed input errors (special characters, corrupted strings) in columns like `Tenure`, `Cashback`, `Login_device`
- Identified and treated outliers in `Tenure`, `CC_Contacted_LY`, `rev_per_month`, `cashback` using log transformation
- Missing value imputation: mode for categorical columns, **KNN Imputer** for numerical columns
- Feature scaling applied post-imputation

### Phase 4 – Model Building & Hyperparameter Tuning
- Benchmarked Logistic Regression, SVM, Bagging, Random Forest, and XGBoost
- Trained on both original and **SMOTE-oversampled** data to address class imbalance
- Used Grid Search / Randomized Search cross-validation for tuning

### Phase 5 – Model Validation & Explainability (SHAP)
- Ran statistical significance tests on categorical features *(confirm test type and p-values against notebook)*
- Applied **SHAP (KernelExplainer & TreeExplainer)** for model interpretability
- **Top churn drivers identified**: contract type, tenure, and account complaint volume

## 📁 Dataset Profile
- **Data volume**: 11,260 entries | 19 features
- **Key variables**: `AccountID`, `Tenure`, `City_Tier`, `CC_Contacted_LY`, `Payment`, `Gender`, `Service_Score`, `Account_user_count`, `account_segment`, `CC_Agent_Score`, `Marital_Status`, `rev_per_month`, `Complain_ly`, `rev_growth_yoy`, `coupon_used_for_payment`, `Day_Since_CC_connect`, `cashback`, `Login_device`, `Churn`

## Results
- **Best model:** Tuned SVM
- **Accuracy:** 97.09%
- **Recall:** 93.7%
- **AUC-ROC:** 0.957

## Tools & Libraries
Python, Pandas, NumPy, Scikit-learn, SHAP, Imbalanced-learn (SMOTE)
