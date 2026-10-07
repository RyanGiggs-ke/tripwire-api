"""Train a fraud model.
Usage: python train.py [paysim.csv]
PaySim is a public simulated mobile-money dataset (search "PaySim" on Kaggle).
With no file, a small synthetic dataset is generated so the pipeline still runs.
"""
import sys, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
FEATURES = ["amount", "hour", "orig_drain", "dest_zero_before", "type_code"]

def featurize(df):
    out = pd.DataFrame()
    out["amount"] = df["amount"]
    out["hour"] = df["step"] % 24
    # share of the sender's balance emptied: classic account-drain signal
    out["orig_drain"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    out["dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)
    out["type_code"] = df["type"].map({t: i for i, t in enumerate(TYPES)}).fillna(0)
    return out[FEATURES]

def synthetic(n=60000, seed=1):
    r = np.random.default_rng(seed)
    fraud = r.random(n) < 0.02
    amount = np.where(fraud, r.uniform(20000, 300000, n), np.exp(r.normal(7, 1.2, n)))
    bal = amount * np.where(fraud, r.uniform(1.0, 1.1, n), r.uniform(1.5, 40, n))
    return pd.DataFrame({
        "step": np.where(fraud, r.choice([1, 2, 3, 4, 23], n), r.integers(0, 24, n)),
        "type": np.where(fraud, r.choice(["TRANSFER", "CASH_OUT"], n), r.choice(TYPES, n)),
        "amount": amount, "oldbalanceOrg": bal,
        "oldbalanceDest": np.where(fraud, 0, r.uniform(0, 5e5, n)),
        "isFraud": fraud.astype(int)})

df = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else synthetic()
X, y = featurize(df), df["isFraud"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
model = HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]
pred = p >= 0.5
print("precision", round(precision_score(yte, pred), 3),
      "recall", round(recall_score(yte, pred), 3),
      "PR-AUC", round(average_precision_score(yte, p), 3))
joblib.dump(model, "model.joblib")
print("saved model.joblib")
