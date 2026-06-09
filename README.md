# 💳 Credit Risk Modeling System

An end-to-end machine learning project that predicts whether a loan applicant is a **good** or **bad** credit risk, built using Python, Scikit-learn, and deployed as an interactive web app with Streamlit.

---

## 🎯 Project Overview

Credit risk assessment is one of the most critical functions in financial services. This project simulates a real-world credit risk modeling pipeline — from raw data exploration to a live prediction dashboard — using the German Credit Dataset.

---

## 📊 Dataset

- **Source:** German Credit Dataset
- **Size:** 1,000 applicants, 11 features
- **Target:** `Risk` — Good (1) or Bad (0)
- **Class distribution:** 55% Good, 45% Bad (mild imbalance handled via `class_weight="balanced"`)

---

## 🔍 Key Insights from EDA

- Applicants with **loan duration ≥ 60 months** were 100% classified as bad risk
- **Bad risk applicants** borrow 38% more (avg ₹3,881) and have 40% longer loan durations (25.4 months) vs good risk applicants
- **Checking account balance** is a strong risk signal — rich checking account = overwhelmingly good risk
- **Credit amount and Duration** have 0.61 correlation — larger loans tend to have longer repayment periods
- **Highly skilled workers** (Job=3) borrow on average 3x more than unskilled non-residents

---

## 🤖 Models Trained

| Model | Accuracy |
|---|---|
| Decision Tree | 58.1% |
| Random Forest | 61.9% |
| Extra Trees | 64.7% ✅ Best |
| XGBoost | 63.8% |

- Hyperparameter tuning done using **GridSearchCV** with 5-fold cross validation
- Final model: **Extra Trees Classifier**

---

## 🛠️ Tech Stack

- **Python** — Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning** — Scikit-learn (ExtraTreesClassifier, GridSearchCV, LabelEncoder)
- **Boosting** — XGBoost
- **Deployment** — Streamlit
- **Model Persistence** — Joblib

---

## 📁 Project Structure

```
credit-risk-model/
├── analysis_model.ipynb          # EDA + model training notebook
├── app.py                        # Streamlit web app
├── german_credit_data.csv        # Dataset
├── extra_trees_credit_model.pkl  # Saved best model
├── Sex_encode.pkl                # Label encoders
├── Housing_encode.pkl
├── Saving accounts_encode.pkl
├── Checking account_encode.pkl
└── target_encode.pkl
```

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/aanam-shaikh/Credit-Risk-Model.git
cd Credit-Risk-Model
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit joblib
```

3. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🌐 App Features

- Input applicant details via interactive dropdowns and sliders
- Real-time credit risk prediction
- Color-coded result — 🟢 **GOOD** risk or 🔴 **BAD** risk

---

## 👩‍💻 Author

**Aanam Shaikh**  
MSc Data Science | KES Shroff College, Mumbai  
[LinkedIn](https://www.linkedin.com/in/aanam-shaikh-941a01256/) | [GitHub](https://github.com/aanam-shaikh)
