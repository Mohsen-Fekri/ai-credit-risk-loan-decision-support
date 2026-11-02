#!/usr/bin/env python3
"""Train the XGBoost credit-risk pipeline on the prepared Lending Club data and write artifacts/."""

from credit_risk.data import load_lending_club
from credit_risk.train import train

if __name__ == "__main__":
    # 0.20 is where the policy layer starts treating a file as elevated risk (base rate is about 15%).
    metrics = train(data=load_lending_club(), threshold=0.2)
    print("Trained default-risk model")
    for key in ("roc_auc", "pr_auc", "ks", "brier", "default_rate", "precision", "recall", "f1", "n_test"):
        print(f"  {key}: {metrics[key]}")
