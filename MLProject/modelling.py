import os
import argparse
import json
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def load_data(data_dir: str):
    """
    Expected files in data_dir:
      - X_train.npy
      - X_test.npy
      - y_train.csv
      - y_test.csv
    """
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(
            f"data_dir harus folder. Diterima: {data_dir} "
            f"(cek path pada workflow dan struktur repo)"
        )

    x_train_path = os.path.join(data_dir, "X_train.npy")
    x_test_path = os.path.join(data_dir, "X_test.npy")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")

    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File tidak ditemukan: {p}")

    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze().values
    y_test = pd.read_csv(y_test_path).squeeze().values

    return X_train, X_test, y_train, y_test


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder dataset preprocessing (berisi X_train.npy, dst)")
    parser.add_argument("--out-dir", required=True, help="Folder output hasil training")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # (CI) Jangan set tracking_uri ke localhost. Biarkan default local file store.
    mlflow.set_experiment("Workflow-CI-Training")

    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    # Model baseline yang ringan untuk CI
    model = LogisticRegression(max_iter=2000, solver="lbfgs")

    # PENTING:
    # Saat mlflow run MLProject, MLflow sudah membuat "active run".
    # Jadi jangan start_run lagi untuk menghindari error "active run ID does not match..."
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # untuk roc_auc perlu probabilitas
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["test_roc_auc"] = float(roc_auc_score(y_test, y_proba))

    # Log ke MLflow (aman di MLflow Project, karena run sudah aktif)
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "max_iter": 2000,
        "solver": "lbfgs",
    })
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Simpan metrics ke file output juga (biar bisa dijadikan artifact CI)
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    mlflow.log_artifact(metrics_path)

    # ====== INI YANG WAJIB UNTUK DOCKER (ADVANCE) ======
    # Buat MLflow Model directory yang berisi MLmodel, conda.yaml, python_env.yaml, dst.
    model_dir = os.path.join(args.out_dir, "model")
    mlflow.sklearn.save_model(model, path=model_dir)

    # Log model folder sebagai artifact MLflow juga (opsional, tapi bagus)
    # Ini akan muncul sebagai artifact "model" di MLflow run
    mlflow.log_artifacts(model_dir, artifact_path="model")

    # Simpan juga versi joblib (opsional)
    # (Tidak wajib untuk build-docker, tapi kadang berguna)
    try:
        import joblib
        joblib_path = os.path.join(args.out_dir, "model.joblib")
        joblib.dump(model, joblib_path)
        mlflow.log_artifact(joblib_path)
    except Exception:
        pass

    print("✅ Training selesai.")
    print("✅ Output saved to:", args.out_dir)
    print("✅ MLflow model dir:", model_dir)
    print("✅ Metrics:", metrics)


if __name__ == "__main__":
    main()
