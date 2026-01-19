# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid")

# =========================================================
# 2. LOAD CSV FILE
# =========================================================
# File link : https://drive.google.com/file/d/1x3wvEkpRUsaJpOtiihVV7UZkJYOlAsjv/view?usp=sharing
df = pd.read_csv("credit_card_fraud.csv")  

print("Dataset Shape:", df.shape)
print(df.head())

# =========================================================
# 3. BASIC CHECKS
# =========================================================

print(df.info())
print("Missing Values:\n", df.isnull().sum())

# =========================================================
# 4. FRAUD DISTRIBUTION
# =========================================================

print("\nFraud Distribution:")
print(df['isFraud'].value_counts(normalize=True))

plt.figure(figsize=(6,4))
sns.countplot(x='isFraud', data=df)
plt.title("Fraud vs Non-Fraud Distribution")
plt.show()

# =========================================================
# 5. RULE-BASED FRAUD CHECK
# =========================================================

rule_confusion = pd.crosstab(
    df['isFraud'],
    df['isFlaggedFraud'],
    rownames=['Actual Fraud'],
    colnames=['Rule Flagged']
)

print("\nRule Engine Confusion Matrix:")
print(rule_confusion)

plt.figure(figsize=(5,4))
sns.heatmap(rule_confusion, annot=True, fmt='d', cmap='Reds')
plt.title("Rule Engine vs Actual Fraud")
plt.show()

# =========================================================
# 6. TRANSACTION TYPE VS FRAUD
# =========================================================

type_fraud_rate = df.groupby('type')['isFraud'].mean().sort_values(ascending=False)
print(type_fraud_rate)

plt.figure(figsize=(7,5))
type_fraud_rate.plot(kind='bar')
plt.title("Fraud Rate by Transaction Type")
plt.ylabel("Fraud Rate")
plt.show()

# =========================================================
# 7. LEDGER CONSISTENCY FEATURES
# =========================================================

df['orig_balance_error'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
df['dest_balance_error'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

# =========================================================
# 8. BEHAVIORAL FEATURES
# =========================================================

df['drain_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
df['zero_balance_flag'] = (df['newbalanceOrig'] == 0).astype(int)
df['is_cashout_transfer'] = df['type'].isin(['CASH_OUT','TRANSFER']).astype(int)
df['amount_log'] = np.log1p(df['amount'])

plt.figure(figsize=(7,5))
sns.boxplot(x='isFraud', y='drain_ratio', data=df)
plt.title("Drain Ratio vs Fraud")
plt.show()

# =========================================================
# 9. VELOCITY & FREQUENCY FEATURES
# =========================================================

df['txn_count_orig'] = df.groupby('nameOrig')['step'].transform('count')
df['amount_sum_orig'] = df.groupby('nameOrig')['amount'].transform('sum')
df['avg_amount_orig'] = df.groupby('nameOrig')['amount'].transform('mean')
df['amount_deviation'] = df['amount'] / (df['avg_amount_orig'] + 1)

# =========================================================
# 10. GRAPH / NETWORK FEATURES
# =========================================================

df['dest_in_degree'] = df.groupby('nameDest')['nameOrig'].transform('nunique')
df['is_self_loop'] = (df['nameOrig'] == df['nameDest']).astype(int)

# =========================================================
# 11. TIME-BASED FRAUD PATTERN
# =========================================================

fraud_by_step = df.groupby('step')['isFraud'].mean()

plt.figure(figsize=(8,5))
fraud_by_step.plot()
plt.title("Fraud Rate Over Time (step)")
plt.xlabel("Step")
plt.ylabel("Fraud Rate")
plt.show()

# =========================================================
# 12. ENCODE CATEGORICAL DATA
# =========================================================

le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])

# =========================================================
# 13. FEATURE SELECTION
# =========================================================

features = [
    'amount_log',
    'oldbalanceOrg',
    'newbalanceOrig',
    'orig_balance_error',
    'dest_balance_error',
    'drain_ratio',
    'zero_balance_flag',
    'is_cashout_transfer',
    'txn_count_orig',
    'amount_sum_orig',
    'amount_deviation',
    'dest_in_degree',
    'is_self_loop',
    'type_encoded'
]

X = df[features]
y = df['isFraud']

# =========================================================
# 14. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 15. LOGISTIC REGRESSION
# =========================================================

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)

print("\nLOGISTIC REGRESSION")
print(classification_report(y_test, lr.predict(X_test)))

# =========================================================
# 16. RANDOM FOREST
# =========================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:,1]

print("\nRANDOM FOREST")
print(classification_report(y_test, rf.predict(X_test)))

# =========================================================
# 17. XGBOOST (FINAL MODEL)
# =========================================================

scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    random_state=42
)

xgb.fit(X_train, y_train)
xgb_probs = xgb.predict_proba(X_test)[:,1]

print("\nXGBOOST")
print(classification_report(y_test, xgb.predict(X_test)))

# =========================================================
# 18. ROC & PR CURVES
# =========================================================

roc = roc_auc_score(y_test, xgb_probs)
precision, recall, _ = precision_recall_curve(y_test, xgb_probs)
pr_auc = auc(recall, precision)

fpr, tpr, _ = roc_curve(y_test, xgb_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve – XGBoost")
plt.show()

plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
plt.legend()
plt.title("Precision–Recall Curve – XGBoost")
plt.show()

# =========================================================
# 19. THRESHOLD TUNING
# =========================================================

threshold = 0.25
custom_preds = (xgb_probs >= threshold).astype(int)

print("\nCUSTOM THRESHOLD RESULTS (0.25)")
print(classification_report(y_test, custom_preds))

# =========================================================
# 20. UNSUPERVISED FRAUD (ISOLATION FOREST)
# =========================================================

iso = IsolationForest(contamination=0.002, random_state=42)
df['anomaly_score'] = iso.fit_predict(X)

plt.figure(figsize=(6,4))
sns.countplot(x='anomaly_score', data=df)
plt.title("Isolation Forest Anomaly Detection")
plt.show()

# =========================================================
# 21. FEATURE IMPORTANCE
# =========================================================

importance = pd.Series(xgb.feature_importances_, index=features).sort_values()

plt.figure(figsize=(8,6))
importance.tail(10).plot(kind='barh')
plt.title("Top 10 Fraud Features (XGBoost)")
plt.show()

# =========================================================
# 22. REAL-TIME FRAUD SIMULATION
# =========================================================

print("\nREAL-TIME FRAUD SIMULATION")
for _, txn in df.sample(10, random_state=42).iterrows():
    prob = xgb.predict_proba(txn[features].values.reshape(1,-1))[0][1]
    if prob > threshold:
        print("🚨 FRAUD ALERT:", txn['nameOrig'], "Prob:", round(prob,3))
    else:
        print("✅ SAFE:", txn['nameOrig'], "Prob:", round(prob,3))

print("\n✔ ALL-IN-ONE FRAUD DETECTION PIPELINE COMPLETED SUCCESSFULLY")
