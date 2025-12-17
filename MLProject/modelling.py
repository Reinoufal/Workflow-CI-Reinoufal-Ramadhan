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
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze().values
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze().values
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ⚠️ Jangan set_experiment + jangan start_run saat dipanggil via `mlflow run`
    # MLflow Projects sudah mengatur run aktifnya sendiri.
    # Kalau mau tetap rapih, boleh set tag:
    mlflow.set_tag("stage", "ci_retraining")

    # Autolog boleh, tapi pastikan versi sklearn kompatibel (kita rapikan nanti)
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)

    # simpan model sebagai artefak workflow (untuk Skilled)
    model_path = os.path.join(args.out_dir, "model.joblib")
    joblib.dump(model, model_path)

    # simpan metrics file
    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"accuracy={acc}\n")
        f.write(f"precision={prec}\n")
        f.write(f"recall={rec}\n")
        f.write(f"f1={f1}\n")
        f.write(f"roc_auc={roc}\n")

    # log artifact tambahan (opsional tapi bagus)
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(metrics_path)

    print("✅ CI training done.")
    print("Saved:", model_path, metrics_path)


if __name__ == "__main__":
    main()
