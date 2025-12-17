import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)

def load_data(data_dir: str):
    # data_dir harus folder yang berisi X_train.npy dst
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(
            f"data_dir harus folder. Diterima: {data_dir}"
        )

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze().values
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze().values
    return X_train, X_test, y_train, y_test

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    ensure_dir(out_dir)

    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    # ===== TRAIN =====
    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=42
    )

    # PENTING:
    # - JANGAN mlflow.start_run() di sini jika dijalankan via `mlflow run`
    # - `mlflow run` sudah mengaktifkan run (MLFLOW_RUN_ID)
    model.fit(X_train, y_train)

    # ===== EVAL =====
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    # ===== LOG PARAMS/METRICS =====
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "solver": "liblinear",
        "max_iter": 2000,
        "random_state": 42,
    })
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # ===== ARTIFACTS (gambar) =====
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    disp.plot(values_format="d")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    mlflow.log_artifact(cm_path)

    roc_path = os.path.join(out_dir, "roc_curve.png")
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve (Test)")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()
    mlflow.log_artifact(roc_path)

    pr_path = os.path.join(out_dir, "pr_curve.png")
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision-Recall Curve (Test)")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()
    mlflow.log_artifact(pr_path)

    # ===== SAVE MODEL ARTIFACT =====
    # Ini penting untuk Docker build (nanti model_uri = runs:/RUN_ID/model)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        pip_requirements=[
            "mlflow==2.19.0",
            "scikit-learn==1.5.2",
            "numpy",
            "pandas",
            "matplotlib",
            "joblib",
        ],
    )

    # ===== SAVE OUTPUTS LOKAL (buat upload artifact CI) =====
    save_json(metrics, os.path.join(out_dir, "metrics.json"))

    # Simpan RUN_ID biar workflow gampang ambil untuk build docker
    run_id = os.environ.get("MLFLOW_RUN_ID", "")
    with open(os.path.join(out_dir, "run_id.txt"), "w", encoding="utf-8") as f:
        f.write(run_id)

    print("✅ Training done. Metrics:", metrics)
    print("✅ RUN_ID:", run_id)

if __name__ == "__main__":
    main()
