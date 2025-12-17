import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_data(data_dir: str):
    # Validasi: harus folder
    if not os.path.isdir(data_dir):
        raise NotADirectoryError(f"data_dir harus folder. Diterima: {data_dir}")

    xtr = os.path.join(data_dir, "X_train.npy")
    xte = os.path.join(data_dir, "X_test.npy")
    ytr = os.path.join(data_dir, "y_train.csv")
    yte = os.path.join(data_dir, "y_test.csv")

    for p in [xtr, xte, ytr, yte]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"File tidak ditemukan: {p}")

    X_train = np.load(xtr)
    X_test  = np.load(xte)
    y_train = pd.read_csv(ytr).squeeze().values
    y_test  = pd.read_csv(yte).squeeze().values
    return X_train, X_test, y_train, y_test

def save_confusion_matrix(y_true, y_pred, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_roc_curve(model, X_test, y_test, out_path: str):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve (Test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_pr_curve(model, X_test, y_test, out_path: str):
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title("Precision-Recall Curve (Test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder preprocessing (X_train.npy, X_test.npy, y_train.csv, y_test.csv)")
    parser.add_argument("--out-dir", required=True, help="Folder output artifacts (akan di-upload workflow)")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    ensure_dir(args.out_dir)

    # Autolog untuk Basic/Skilled CI (boleh)
    mlflow.sklearn.autolog(log_models=True)

    # Model baseline (CI retrain)
    model = LogisticRegression(max_iter=1000)

    # TRAIN
    model.fit(X_train, y_train)

    # EVAL
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    # Karena autolog aktif, metrics/params/model akan tercatat otomatis.
    # Tapi tetap kita simpan file tambahan untuk artefak CI.
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    roc_path = os.path.join(args.out_dir, "roc_curve.png")
    pr_path = os.path.join(args.out_dir, "pr_curve.png")

    save_confusion_matrix(y_test, y_pred, cm_path)
    save_roc_curve(model, X_test, y_test, roc_path)
    save_pr_curve(model, X_test, y_test, pr_path)

    print("✅ Training selesai.")
    print("✅ Metrics:", metrics)
    print("✅ Outputs saved to:", args.out_dir)

if __name__ == "__main__":
    main()
