# Loan-Prediction-Using-Machine-Learning

## 📌 Overview
This project is about building a machine learning system that predicts whether a loan application is likely to be **approved or rejected** based on an applicant’s personal, financial, and credit information.

The idea is simple:  
banks deal with thousands of loan applications, and manual screening can be slow and inconsistent. This model helps automate the **initial screening process** and highlights the key factors that influence loan approval decisions.

---

## 📂 Dataset
The project uses the dataset **`LoanApprovalPrediction.csv`**, which contains information such as:
- Applicant demographics
- Income details
- Loan amount and tenure
- Credit history
- Property location
- Loan approval status (target variable)

---

## 🔄 Project Workflow

### 1️⃣ Data Loading & First Look
- Loaded the dataset using **pandas**
- Explored the data using:
  - `head()` to preview records
  - `info()` to understand data types
  - `describe()` for statistical summary
- Checked missing values to understand data quality issues

---

### 2️⃣ Data Preprocessing
To prepare the data for modeling, several cleaning and transformation steps were performed:

- **Missing Values**
  - Rows with missing values in critical columns (`Dependents`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`) were removed to maintain data reliability.

- **Categorical Encoding**
  - Binary variables were mapped to numerical values:
    - Gender, Married, Education, Self_Employed
  - One-Hot Encoding was applied to `Property_Area`.

- **Feature Scaling**
  - Numerical features were standardized using `StandardScaler` to ensure fair model learning.

- **Class Imbalance Handling**
  - Since approved and rejected loans were not evenly distributed, **SMOTE** was applied to balance the target variable.

- **Feature Selection**
  - `Loan_ID` was removed as it does not contribute to prediction.

---

### 3️⃣ Exploratory Data Analysis (EDA)
EDA helped understand the data and uncover meaningful patterns:

- Distribution of income, loan amount, and tenure
- Comparison of approval rates across education, marital status, and property area
- Correlation heatmap to identify strong feature relationships

📌 **Key Insight**  
`Credit_History` turned out to be the **most influential feature** — applicants with a good credit history had a much higher chance of loan approval.

---

### 4️⃣ Model Training
The dataset was split into **80% training** and **20% testing** data using stratified sampling.

Three classification models were trained:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

### 5️⃣ Model Evaluation & Comparison
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

#### 📊 Performance Comparison

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.7228   | 0.8116    | 0.7887 | 0.8000   |
| Decision Tree       | 0.7129   | 0.8000    | 0.7887 | 0.7943   |
| **Random Forest**   | **0.7624** | **0.8219** | **0.8451** | **0.8333** |

Confusion matrices and comparison plots were also generated for better visualization.

---

### 6️⃣ Best Model
🏆 **Random Forest Classifier** performed the best overall, achieving an accuracy of **76.24%**, along with strong precision and recall scores.

---

### 7️⃣ Model Saving & Prediction
- The trained Random Forest model was saved as `best_loan_model.pkl`
- The fitted scaler was saved as `scaler.pkl`
- A reusable function `predict_loan_approval()` was created to:
  - Apply the same preprocessing steps
  - Predict approval status
  - Return approval probability and confidence level

---

## 💡 Business Insights & Recommendations
- **Credit History matters most** – it heavily influences approval decisions
- The model can be used for **automated first-level screening**
- Applications with **borderline probabilities (50–65%)** should be reviewed manually
- Periodic retraining with fresh data will keep the model accurate and relevant

---

## 📁 Files Generated
- `best_loan_model.pkl` – Trained Random Forest model
- `scaler.pkl` – StandardScaler used for preprocessing
- `model_comparison.png` – Model performance comparison
- `project_report.txt` – Detailed project documentation

---

## 🚀 How to Use the Prediction Function

```python
import pickle
import pandas as pd

# Load model and scaler
with open('best_loan_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# Example input
sample_input = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 1500,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1.0,
    'Property_Area': 'Urban'
}

result = predict_loan_approval(sample_input)

print(f"Loan Approved: {result['approved']}")
print(f"Approval Probability: {result['approval_probability']:.2%}")
print(f"Confidence: {result['confidence']}")
