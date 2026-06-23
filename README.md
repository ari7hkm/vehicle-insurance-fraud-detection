# Vehicle Insurance Fraud Detection

Automated detection of fraudulent vehicle insurance claims using Random Forest and SMOTE on a highly imbalanced real-world dataset.

---

## Problem Statement

Insurance fraud costs companies millions annually. Traditional manual review is slow, expensive, and reactive. This project builds a machine learning system to **automatically score insurance claims in real-time** — flagging fraudulent ones before payout occurs.

---

## Dataset

- **Source:** [Vehicle Claim Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
- **Size:** 15,420 records, 33 features
- **Features:** Demographics, vehicle info, accident details, policy type
- **Target:** `FraudFound_P` (0 = Legitimate, 1 = Fraud)
- **Class Imbalance:** Only ~6% fraud cases (923 out of 15,420)

---

## Methodology

### Preprocessing Pipeline
1. **Label Encoding** — binary variables encoded to 0/1
2. **Ordinal Encoding** — ordered categoricals mapped to meaningful numeric values (VehiclePrice, AgeOfVehicle, BasePolicy)
3. **Feature Removal** — dropped irrelevant columns (PolicyNumber, DayOfWeek, etc.)
4. **One-Hot Encoding** — remaining categorical features (Make, PolicyType, MaritalStatus, etc.)
5. **Low-Variance Filter** — removed near-constant one-hot encoded features
6. **Outlier Handling** — IQR method on Age column + binning into categories
7. **SMOTE** — oversampled minority fraud class to balance training data

### Model
- **Random Forest Classifier** (700 estimators, max depth 12)
- **Tuning:** RandomizedSearchCV for hyperparameter optimization
- **Evaluation:** 80/20 train-test split

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 89% |
| Precision (Fraud) | 84% |
| **Recall (Fraud)** | **96%** |
| F1-Score (Fraud) | 90% |
| AUC-ROC | 0.98 |
| AUC-PR | 0.98 |

**96% Recall** means the model catches 96 out of every 100 real fraud cases — minimizing costly missed detections.

---

## Key Findings

Top predictive features identified by Random Forest:
- **Fault** (Policy Holder vs. Third Party) — most important
- **PolicyType**
- **BasePolicy**
- **VehicleCategory**

These features can directly inform business investigation rules.

---

## Business Impact

- **Cost Reduction:** Catching 96% of fraud prevents millions in false payouts
- **Operational Efficiency:** Investigators focus only on flagged claims
- **Real-time Scoring:** Model can be deployed via API to score claims at submission

---

## Tech Stack

```
Python | Pandas | NumPy | Scikit-Learn | Imbalanced-Learn | Matplotlib | Seaborn
```

---

## Files

```
Fraud_Detection_problem6.ipynb   ← Main notebook
README.md
requirements.txt
```

---

## Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
```
