# 📊 Customer_Churn_prediction

An interactive data science and machine learning repository designed to analyze customer behavior, identify institutional drop-off channels, and predict multi-user account churn for an E-Commerce platform.

🔗 Project Components
* Jupyter Notebook: `customer_churn_analysis.ipynb`
* Production Pipeline: `optimized_model.pkl`

📋 Project Phases

### Phase 1 – Problem Statement & Business Impact
* **The Challenge**: Rising competition makes retaining active e-commerce accounts difficult. Because one account aggregates multiple individual users, losing a single account results in a compounding loss of active customers.
* **Objective**: Build a high-precision churn forecasting pipeline to trigger targeted, segmented offers for at-risk accounts.
* **Financial Imperative**: Account retention is **5 to 7 times cheaper** than new customer acquisition, maximizing the efficiency of defensive marketing budgets.

### Phase 2 – Exploratory Data Analysis (EDA)
* **Target Layout**: The customer base exhibits a baseline **16.84% churn rate** (approximately 1 out of every 5 accounts leaves).
* **The Onboarding Leak**: Churn spikes heavily during early lifecycle stages. Nearly **1 in 3 accounts churn within their first 6 months**. Drop-off drops significantly after 12 months.
* **Friction Points**: Accounts with **3 to 4 users** exhibit the highest absolute churn volume. A direct positive correlation exists between customer care complaints (`Complain_ly`) and account drop-off.
* **Demographic Triggers**: Tier 3 cities show weak user stickiness with a **30% churn rate** relative to population size. Single account holders churn at a noticeably higher frequency than married account holders.

### Phase 3 – Data Cleaning & Preprocessing
* **Feature Fixes & Cleaning**: Fixed systemic input errors, such as removing special characters like `#`, `$`, `@`, `+`, and corrupted strings (`&&&&`) across columns like `Tenure`, `Cashback`, and `Login_device`.
* **Outlier Strategy**: Identified extreme variances within `Tenure`, `CC_Contacted_LY`, `rev_per_month`, and `cashback`. Outliers were isolated, converted into missing values, and compressed safely using **log transformation** to prevent coefficient skew.
* **Missing Value Imputation**: 
  * Categorical nulls filled using **Mode Imputation** (leveraging the dominant `Mobile` login trend).
  * Numerical nulls resolved using a **KNN Imputer** to preserve hidden multivariate patterns.
* **Feature Scaling**: Scaled all continuous variables post-imputation using standard normalization tools.

### Phase 4 – Model Building & Hyperparameter Tuning
* **Algorithmic Suite**: Benchmarked structural parametric models (**Logistic Regression**, **SVM**) against robust ensemble frameworks (**Bagging**, **Random Forest**, **XGBoost**).
* **Resampling Layout**: Trained configurations over both the native dataset and **oversampled variations** to safeguard model classification metrics against minority class data imbalances.
* **Optimization**: Applied Grid Search and Randomized Search cross-validation routines across the ensemble models to secure the most viable business pipeline.

### Phase 5 – Model Validation & Explainability (SHAP)
* **Statistical Guardrails**: Executed **Chi-Square Tests** across categorical descriptors, proving all object-type properties were statistically significant inputs ($p < 0.05$) relative to account churn.
* **Global Interpretability**: Utilized **SHAP (SHapley Additive exPlanations)** values alongside traditional feature importance metrics to explain underlying classification layers.
* **Key Risk Vectors**: Low relationship tenure, elevated volumes of unresolved support complaints, and minimal transactional cashback rewards were mathematically pinpointed as the strongest operational triggers of account churn.

---

📁 Dataset Profile
* **Data Volume**: 11,260 entries | 19 operational dimensions.
* **Core Variables**: `AccountID`, `Tenure`, `City_Tier`, `CC_Contacted_LY`, `Payment`, `Gender`, `Service_Score`, `Account_user_count`, `account_segment`, `CC_Agent_Score`, `Marital_Status`, `rev_per_month`, `Complain_ly`, `rev_growth_yoy`, `coupon_used_for_payment`, `Day_Since_CC_connect`, `cashback`, `Login_device`, and `Churn`.

📌 Key Project Insights
* **Primary Payment Method**: **Debit Cards** dominate e-commerce checkouts (~42.6%).
* **Core Account Segment**: **Regular Plus** accounts represent the largest user block but sustain the highest baseline churn.
* **Critical Mitigation Window**: Early proactive support validation within **0 to 3 days** of a customer service interaction significantly lowers active account churn probability.

* **Highest Risk Segment**: Employees with 3–5 years of tenure.
* **Model Target Metric**: Achieved balanced predictive recall across both active and leaving classes using SMOTE.
