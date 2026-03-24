# 🏥 Hospital Readmission Predictor — Diabetic Patients

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Model](https://img.shields.io/badge/Model-XGBoost-EC4D37)](https://xgboost.readthedocs.io)
[![Optimization](https://img.shields.io/badge/HPO-Optuna-5A67D8)](https://optuna.org)
[![Interpretability](https://img.shields.io/badge/XAI-SHAP-orange)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Predicting which diabetic patients are at high risk of readmission within 30 days of discharge** — with a clinical focus on minimizing missed cases over false alarms.

---

## 🎯 Business Problem

Early hospital readmission is one of the most expensive and preventable problems in healthcare management. In the US alone, diabetic patient readmissions cost over **$2.5 billion annually**.

A readmission within 30 days often signals a failure in the initial treatment or post-discharge follow-up. This model gives clinical staff a tool to **identify high-risk patients before discharge** — enabling proactive intervention rather than reactive treatment.

**The core decision:** In a context where a false negative (missed high-risk patient) has direct human consequences, which metric do you optimize — and why?

This project makes that decision explicitly and defends it.

---

## 📊 Dataset

| Attribute | Value |
|---|---|
| Source | UCI / Kaggle Diabetes Dataset |
| Records | 101,766 real hospitalizations |
| Time span | 1999–2008 (10 years) |
| Features | 50 clinical variables |
| Target | Binary: readmitted < 30 days |

---

## ⚙️ Technical Methodology

### 1. Clinically-Guided Data Cleaning

Preprocessing guided by medical logic, not just statistical criteria:

- **Cohort filtering:** Excluded patients discharged to hospice or who died — readmission is impossible in these cases. Including them would introduce irreducible noise.
- **Missing data:** Dropped variables with >50% nullity (`weight`, `payer_code`). Strategic imputation for the rest.
- **Duplicate encounters:** One patient, one model input — removed repeated hospitalizations to avoid data leakage.

### 2. Feature Engineering

- **ICD-9 grouping:** Simplified hundreds of diagnostic codes into 9 clinically meaningful categories (circulatory, respiratory, digestive, etc.)
- **HL7 discharge encoding:** Standardized discharge disposition codes following HL7 healthcare data standards
- **Patient history features:** Weighted `number_inpatient` (prior visits) and `time_in_hospital` as strong readmission signals
- **Medication interaction:** Analyzed `change` (medication adjustment) and insulin usage as clinical indicators

### 3. Class Imbalance Handling

Readmission-positive cases were a minority class. Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for the minority class — preventing model bias toward predicting "no readmission" and improving clinical sensitivity.

### 4. Modeling & Optimization

- **Algorithm:** XGBoost Classifier
- **Hyperparameter optimization:** Bayesian search with **Optuna** (100 trials)
- **Primary metric:** **Recall** — in medicine, missing a high-risk patient (False Negative) is more costly than flagging a low-risk one (False Positive)

### 5. Threshold Tuning — The Key Clinical Decision

| Threshold | High-Risk Cases Detected | False Positives |
|---|---|---|
| 0.50 (default) | Baseline | Lower |
| **0.20 (chosen)** | **+1,650 additional cases** | Higher |

**Chosen threshold: 0.20** — accepting more false positives to ensure high-risk patients aren't sent home without intervention.

### 6. Interpretability with SHAP

SHAP values reveal *why* the model makes each prediction, enabling clinical staff to understand and trust the output:

- `number_inpatient` → strongest predictor (prior hospitalizations)
- `num_lab_procedures` → proxy for patient complexity
- `discharge_disposition_id` → where the patient goes after discharge significantly affects risk

---

## 📈 Results

- **+1,650 high-risk cases detected** that a standard 0.5 threshold would have missed
- Clinically defensible recall-precision tradeoff with documented justification
- Model serialized and ready for API deployment

---

## 🗂️ Project Structure

```
hospital-readmission-prediction-diabetes/
├── data/
│   ├── raw/                    # Original clinical dataset
│   └── processed/              # Cleaned and feature-engineered data
├── notebooks/
│   └── eda_modeling.ipynb      # Full EDA, SMOTE, modeling pipeline
├── src/
│   └── features/               # Feature engineering scripts
├── models/
│   └── xgb_readmission.joblib  # Serialized production model
├── reports/
│   └── figures/                # SHAP plots, confusion matrices
├── static/                     # App assets
├── app.py                      # Streamlit inference app
├── train_model_production.py   # Production training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/AlexSc97/hospital-readmission-prediction-diabetes.git
cd hospital-readmission-prediction-diabetes

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Or retrain the model
python train_model_production.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Data Processing** | Pandas · NumPy |
| **Machine Learning** | Scikit-Learn · XGBoost |
| **Hyperparameter Optimization** | Optuna |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP |
| **Visualization** | Seaborn · Matplotlib |
| **App** | Streamlit |
| **Serialization** | joblib |

---

## 💡 Key Takeaways

This project demonstrates that good ML in healthcare isn't about achieving the highest accuracy — it's about making **informed decisions on which errors are acceptable**.

The choice to lower the classification threshold from 0.5 to 0.2 wasn't arbitrary: it was a deliberate tradeoff between clinical risk (missing a high-risk patient) and operational cost (additional reviews for flagged low-risk patients). That kind of decision requires domain knowledge, not just technical skill.

---

## License

[MIT](LICENSE) — Free to use and adapt with attribution.
