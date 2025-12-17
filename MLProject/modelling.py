import os
import argparse
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data(data_dir: str):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze().values
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze().values
    return X_train, X_test, y_train, y_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    mlflow.set_experiment("Workflow-CI-Training")
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="ci_retrain_logreg"):
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        joblib.dump(model, os.path.join(args.out_dir, "model.joblib"))

        with open(os.path.join(args.out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        print("✅ CI training done. Outputs saved to:", args.out_dir)

if __name__ == "__main__":
    main()
