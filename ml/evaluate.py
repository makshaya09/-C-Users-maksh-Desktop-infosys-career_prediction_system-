"""
Model Evaluation Module for Career Prediction System.
Calculates Accuracy, Precision, Recall, F1-Score, Coverage, Confusion Matrix, and Classification Report.
"""

import json
import os
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def calculate_coverage(y_probs: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculates prediction coverage:
    Coverage = number of valid predictions (confidence >= threshold) / total prediction attempts.
    With default threshold 0.0, any model outputting a valid probability distribution has 100% (1.0) coverage.
    """
    if len(y_probs) == 0:
        return 0.0
    max_probs = np.max(y_probs, axis=1)
    valid_predictions = np.sum(max_probs >= threshold)
    total_attempts = len(y_probs)
    return float(valid_predictions / total_attempts)


def evaluate_model(
    model,
    X_test_vec,
    y_test_encoded,
    target_names: List[str],
    reports_dir: str = None
) -> Dict[str, Any]:
    """
    Evaluates the model on test data, generates metrics, confusion matrix, and classification report.
    """
    if reports_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Model predictions
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)

    # Core Metrics
    accuracy = float(accuracy_score(y_test_encoded, y_pred))
    precision = float(precision_score(y_test_encoded, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test_encoded, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test_encoded, y_pred, average="weighted", zero_division=0))
    coverage = calculate_coverage(y_proba, threshold=0.20)  # Coverage with a 20% minimum confidence threshold

    # Classification Report
    clf_report_str = classification_report(
        y_test_encoded,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    clf_report_dict = classification_report(
        y_test_encoded,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)

    # Plot and save Confusion Matrix
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True
    )
    plt.title("Career Prediction Baseline Confusion Matrix (Logistic Regression)", fontsize=13, pad=15)
    plt.xlabel("Predicted Career", fontsize=11)
    plt.ylabel("Actual Career", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Save Classification Report as text file
    clf_path = os.path.join(reports_dir, "classification_report.txt")
    with open(clf_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("CAREER PREDICTION MODEL - CLASSIFICATION REPORT (MILESTONE 1)\n")
        f.write("=" * 65 + "\n\n")
        f.write(clf_report_str)
        f.write("\n\n" + "=" * 65 + "\n")
        f.write(f"Summary Metrics:\n")
        f.write(f"Accuracy : {accuracy * 100:.2f}%\n")
        f.write(f"Precision: {precision * 100:.2f}%\n")
        f.write(f"Recall   : {recall * 100:.2f}%\n")
        f.write(f"F1-Score : {f1 * 100:.2f}%\n")
        f.write(f"Coverage : {coverage * 100:.2f}%\n")
        f.write("=" * 65 + "\n")

    # Save Metrics as JSON
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "coverage": round(coverage, 4),
        "total_test_samples": int(len(y_test_encoded)),
        "classes": target_names,
        "classification_report": clf_report_dict
    }
    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    print("Run ml/train_model.py to train and evaluate the baseline model.")
