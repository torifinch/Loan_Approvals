# Loan Approval Prediction with Machine Learning
Predicting Loan Status with Logistic Regression, Random Forest, and XGBoost

## 🔍 Overview

This project applies machine learning to predict loan approval outcomes based on applicant details. Using a publicly available dataset, we explore data cleaning, feature engineering, class balancing with SMOTE, and model evaluation to identify the most effective approach for accurate predictions.

---

## 📁 Dataset Summary

The dataset includes variables like:
- **Applicant and Coapplicant Income**
- **Loan Amount & Term**
- **Credit History**
- **Dependents, Marital Status, Education**
- **Target**: `loan_status` (1 = approved, 0 = denied)

---

## ⚙️ Preprocessing Workflow

| Step               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| Missing Values   | Imputed using mean/mode or flagged with missingness indicators              |
| Feature Engineering | Created `total_income`, `loan_to_income`, and `is_family`                     |
| Encoding         | Applied one-hot encoding for categorical variables                          |
| Scaling          | StandardScaler used prior to PCA and logistic regression                    |
| Class Balancing  | SMOTE applied to address class imbalance in the training dataset            |

---

## Model Performance

###  Logistic Regression (Baseline)
- **Accuracy**: 0.83  
- Weak at identifying rejected loans (Recall = 0.33 for class 0)

###  Random Forest (Tuned with GridSearchCV)
- **Accuracy**: 0.84  
- Strong, balanced performance  
- **Best performing model overall**

###  XGBoost (Tuned with GridSearchCV)
- **Accuracy**: 0.82  
- Competitive, but slightly lower than Random Forest

---

## 📊 Feature Importances (Random Forest)

| Rank | Feature              | Importance |
|------|----------------------|------------|
| 1    | credit_history       | 0.1848     |
| 2    | loan_to_income       | 0.1339     |
| 3    | applicant_income     | 0.1326     |
| 4    | total_income         | 0.1135     |
| 5    | loan_amount          | 0.1060     |
| 6    | coapplicant_income   | 0.0805     |
| 7    | dependents_1         | 0.0381     |
| 8    | is_family            | 0.0359     |
| 9    | loan_amount_term     | 0.0320     |
| 10   | property_area_Urban  | 0.0253     |

---

##  Key Takeaways

- SMOTE significantly improved performance on the minority class.
- Random Forest offered the best balance of precision and recall.
- Custom engineered features like `loan_to_income` were highly predictive.

---

##  Future Directions

- Investigate fairness and bias across demographic features like gender and employment type.
- Experiment with other ensemble algorithms (e.g., LightGBM, CatBoost).
- Use SHAP values for explainable AI and model transparency.

---

##  Tech Stack

- **Languages & Tools**: Python, Jupyter Notebooks
- **Libraries**: pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, seaborn, matplotlib



