import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)


def load_data(data_dir: str):
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"data_dir harus folder. Diterima: {data_dir}")

    x_train_path = os.path.join(data_dir, "X_train.npy")
    x_test_path = os.path.join(data_dir, "X_test.npy")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")

    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File tidak ditemukan: {p}")

    X_train = np.load(x_train_path, allow_pickle=False)
    X_test = np.load(x_test_path, allow_pickle=False)
    y_train = pd.read_csv(y_train_path).squeeze().to_numpy()
    y_test = pd.read_csv(y_test_path).squeeze().to_numpy()
    return X_train, X_test, y_train, y_test


def save_confusion_matrix(y_true, y_pred, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def ensure_active_run():
    """
    Saat dijalankan via `mlflow run`, MLflow Projects biasanya sudah membuat run_id
    dan menyimpannya di env var MLFLOW_RUN_ID, tapi belum menjadi active run di proses python.
    Jadi kita attach run itu biar logging tidak mismatch.
    """
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if run_id:
        # attach ke run yang sudah dibuat oleh MLflow Projects
        mlflow.start_run(run_id=run_id)
        return True
    else:
        # fallback kalau dijalankan manual (python modelling.py)
        mlflow.start_run()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # JANGAN set_experiment saat pakai MLflow Projects (lebih aman via CLI),
    # tapi kalau dijalankan manual, ini tetap oke.
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment("Workflow-CI-Training")

    attached = ensure_active_run()

    try:
        X_train, X_test, y_train, y_test = load_data(args.data_dir)

        model = LogisticRegression(max_iter=2000, solver="lbfgs")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        mlflow.log_params({
            "model": "LogisticRegression",
            "max_iter": 2000,
            "solver": "lbfgs",
        })
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Save outputs (local)
        model_path = os.path.join(args.out_dir, "model.pkl")
        joblib.dump(model, model_path)

        metrics_path = os.path.join(args.out_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
        save_confusion_matrix(y_test, y_pred, cm_path)

        # Log artifacts
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifacts(args.out_dir, artifact_path="outputs")

        print("✅ Training selesai.")
        print("Metrics:", metrics)
        print("Outputs saved to:", args.out_dir)

    finally:
        # kalau kita yang start_run (attached maupun fallback), tetap end_run biar clean
        mlflow.end_run()


if __name__ == "__main__":
    main()
