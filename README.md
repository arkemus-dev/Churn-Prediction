# 📞 Telco Customer Churn Prediction

A Machine Learning project to predict customer attrition using **CatBoost** on the IBM Telco Dataset (from Kaggle).
This project focuses on handling **Imbalanced Data** and detecting **Data Leakage** in real-world scenarios.

> **🤖 AI Collaboration Note:**
> This project was developed using an iterative **AI-assisted workflow**.
> Generative AI (Gemini) was utilized for:
> * **Code Optimization:** Refactoring Pandas operations and CatBoost configuration.
> * **Debugging:** Identifying and claryfying error messages
> * **Theoretical Reinforcement:** Validating strategies for handling imbalanced datasets (Class Weights vs SMOTE).

---

## 🎯 Project Goal
The objective is to identify customers at risk of leaving (Churn) to enable proactive retention strategies.
The dataset presented significant challenges, including a **73/27 class imbalance** and several "hidden" leakage columns that required careful cleaning.

## 🏆 Key Results
* **Model:** CatBoost Classifier (Gradient Boosting)
* **Performance (Recall on Churn):** **0.83** (Detects 83% of customers who are about to leave).
* **ROC-AUC:** **~0.92** (Excellent discrimination capability).
* **Strategy:** Optimized for **Recall** (minimizing False Negatives) using `auto_class_weights='Balanced'`.

## 📊 Business Insights (Top Predictors)
The model identified these behavioral drivers as the most critical:
1.  **Number of Referrals:** High impact. Customers who refer friends are socially invested and significantly less likely to churn.
2.  **Contract Type:** "Month-to-month" contracts are the highest risk factor compared to 1-2 year contracts.
3.  **Dependents:** Customers with family dependents show higher stability and lower churn rates.

## 🛠️ Methodology & Tech Stack

### 1. Data Cleaning & Leakage Removal 🕵️‍♂️
The raw dataset contained features that "spoiled" the answer. The following were identified and removed to ensure a realistic model:
* `Satisfaction Score` (Post-churn survey data).
* `Customer Status` (Directly correlates with Churn label, detected after the first prediction, the model was learning "too well").
* `Churn Score` & `Churn Reason`.
* **Handling Missing Values:** Detected and fixed hidden NaNs in `Total Charges` (imputed with 0 for new customers).

### 2. Modeling Strategy 🚀
* **Algorithm:** CatBoost Classifier.
* **Categorical Handling:** Leveraged CatBoost's native support for string features (no One-Hot Encoding needed), preserving information.
* **Imbalance Handling:** Used Class Weights to penalize errors on the minority class (`Churn = Yes`) more heavily than on the majority class.

### 3. Environment
* **IDE:** VS Code (Local)
* **Python Version:** 3.12+
* **Libraries:** `pandas`, `catboost`, `scikit-learn`, `seaborn`, `joblib`.

## 📂 Project Structure
```text
├── data/
│   ├── raw/            # Original IBM Dataset (telco.csv)
│   └── processed/      # Cleaned data (No leakage, encoded target)
├── notebooks/
│   ├── 01_initial_analysis.ipynb  # EDA & Leakage Detection
│   ├── 02_data_cleaning.ipynb     # NaN Imputation & Cleaning
│   └── 03_modeling.ipynb          # Training & Evaluation (CatBoost)
├── models/             # Saved .cbm model
└── README.md