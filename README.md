# 🔬 Breast Cancer Classification

A Machine Learning system that classifies breast tumors as **malignant** or **benign** using Logistic Regression and Random Forest. Achieves **98.2% accuracy** with a complete ML Pipeline.

---

## 🎯 What It Does

```
Input:  30 tumor measurements (radius, texture, perimeter, area...)
Output: Malignant 🔴 or Benign 🟢 + confidence score

Patient 1: Malignant (confidence: 99.1%)
Patient 2: Benign    (confidence: 98.7%)
Patient 3: Malignant (confidence: 88.0%)
```

---

## 📊 Results

| Model | Accuracy | AUC | Precision | Recall |
|-------|----------|-----|-----------|--------|
| Logistic Regression | **0.982** | **0.995** | 0.98 | 0.98 |
| Decision Tree | 0.939 | — | 0.94 | 0.93 |
| Random Forest | 0.956 | — | 0.96 | 0.95 |

**Winner: Logistic Regression** — highest accuracy, highest AUC, most stable (CV ±0.009)

---

## 🏗️ Pipeline

```
569 Patient Records (30 features each)
      ↓
Train/Test Split (80/20, stratified)
      ↓
StandardScaler Normalization
      ↓
Model Training (LR, Decision Tree, Random Forest)
      ↓
Cross Validation (5-fold, CV=0.981 ±0.007)
      ↓
Evaluation (Accuracy, Precision, Recall, AUC, ROC Curve)
      ↓
Production Pipeline (saved with joblib)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Models | scikit-learn (LR, DecisionTree, RandomForest) |
| Pipeline | sklearn.pipeline.Pipeline |
| Evaluation | ROC Curve, AUC, Confusion Matrix |
| Data | Wisconsin Breast Cancer Dataset (569 samples) |
| Language | Python 3.11 |

---

## 📦 Installation

```bash
git clone https://github.com/RashaAlorabi/cancer-classification.git
cd cancer-classification

python3 -m venv ai-env
source ai-env/bin/activate

pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
jupyter notebook cancer_classification.ipynb
```

Load saved model and predict:

```python
import joblib
import numpy as np

# Load production-ready pipeline
pipeline = joblib.load('models/model_v1.pkl')

# Predict on new patient
new_patient = np.array([[...30 measurements...]])
diagnosis = pipeline.predict(new_patient)
confidence = pipeline.predict_proba(new_patient).max()

print(f"Diagnosis: {'Benign' if diagnosis[0]==1 else 'Malignant'}")
print(f"Confidence: {confidence*100:.1f}%")
```

---

## 🔑 Key Learnings

**Why Logistic Regression beat Random Forest here:**
- Data has near-linear relationships between features and diagnosis
- Simpler model = more stable (lower variance)
- Occam's Razor: simplest model with best results wins

**Critical Medical Tradeoff:**
```
False Negative (missed cancer) = danger to patient's life ← WORST
False Positive (healthy flagged) = unnecessary worry

Therefore: Maximize RECALL over Precision in medical applications
```

**ROC Curve & AUC:**
```
AUC = 0.995 → Near-perfect discrimination
AUC = 0.5   → Random guessing (baseline)
```

---

## 🧠 Concepts Demonstrated

```
Data Leakage Prevention:  fit_transform on train, transform on test
Cross Validation:         5-fold CV proves model reliability
Bias-Variance Tradeoff:   Decision Tree depth tuning
Ensemble Learning:        Random Forest = 100 trees voting
Production Pipeline:      sklearn Pipeline + joblib saving
Model Versioning:         Saved with metadata JSON
```

---

## 📁 Project Structure

```
cancer-classification/
├── cancer_classification.ipynb   # Main notebook
├── models/
│   ├── model_v1.pkl             # Saved pipeline
│   └── model_v1_info.json       # Model metadata
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📋 Requirements

```txt
scikit-learn
numpy
pandas
matplotlib
seaborn
joblib
jupyter
```

---

## 🚧 Future Improvements

- [ ] Add SHAP values for model explainability
- [ ] Build Streamlit web interface
- [ ] Add more datasets for validation
- [ ] Deploy as REST API with FastAPI

---

## ⚠️ Disclaimer

This project is for educational purposes only. Not intended for actual medical diagnosis.

---

## 👤 Author

**Rasha** — Senior Software Engineer → AI Solutions Engineer

