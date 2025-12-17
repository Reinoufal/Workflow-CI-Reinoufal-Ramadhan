import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, classification_report
)


def load_data(data_dir: str):
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"data_dir harus folder. Diterima: {data_dir}")

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze().values
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze().values
    return X_train, X_test, y_train, y_test


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder berisi dataset preprocessing")
    parser.add_argument("--out-dir", default="outputs", help="Folder output artifacts (untuk CI upload)")
    args = parser.parse_args()

    # ========= IMPORTANT FIX =========
    # Jika script dijalankan via `mlflow run`, maka run_id sudah disediakan oleh MLflow.
    env_run_id = os.environ.get("MLFLOW_RUN_ID")
    if env_run_id:
        mlflow.start_run(run_id=env_run_id)
    else:
        mlflow.start_run(run_name="local_run")

    try:
        # (Opsional) Jangan set_experiment di sini saat pakai MLflow Project.
        # Experiment sudah di-handle oleh `mlflow run` (bisa diatur dari CLI kalau perlu).

        X_train, X_test, y_train, y_test = load_data(args.data_dir)
        out_dir = ensure_dir(args.out_dir)

        # ====== Model (simple baseline) ======
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # ====== Evaluate ======
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        # ====== Manual logging (aman untuk CI) ======
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "max_iter": 1000,
        })
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # ====== Log model ======
        mlflow.sklearn.log_model(model, artifact_path="model")

        # ====== Create artifacts locally (out_dir) then log to MLflow ======
        # 1) Confusion matrix
        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(values_format="d")
        plt.title("Confusion Matrix (Test)")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()
        mlflow.log_artifact(cm_path)

        # 2) ROC curve
        roc_path = os.path.join(out_dir, "roc_curve.png")
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("ROC Curve (Test)")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150)
        plt.close()
        mlflow.log_artifact(roc_path)

        # 3) PR curve
        pr_path = os.path.join(out_dir, "pr_curve.png")
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
        plt.title("Precision-Recall Curve (Test)")
        plt.tight_layout()
        plt.savefig(pr_path, dpi=150)
        plt.close()
        mlflow.log_artifact(pr_path)

        # 4) classification report
        report_path = os.path.join(out_dir, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact(report_path)

        # 5) metrics json (biar gampang dilihat di artifacts)
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        print("✅ CI Training done. Metrics:", metrics)
        print(f"✅ Outputs saved to: {out_dir}")

    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
