"""
Continuous Integration (CI) Model Accuracy Gate.
Evaluates actual trained model accuracy against a configurable threshold.
Exits with code 0 if accuracy >= threshold, or exits with code 1 to fail the CI pipeline.
Milestone 3: Skill Gap Analysis, API & CI Integration.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml.config import DATA_DIR
from ml.model_comparison import get_best_model_artifacts
from ml.random_forest import load_dataset


def run_accuracy_gate(threshold: float = 0.80, random_state: int = 42) -> dict:
    """
    Loads the winning model and test dataset, computes actual accuracy,
    and validates against the required accuracy threshold.
    """
    dataset_path = os.path.join(DATA_DIR, "career_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Career dataset not found at {dataset_path}")

    # Load dataset
    df = load_dataset(dataset_path)
    X = df["feature_text"]
    y = df["career"]

    # Load winning model artifacts
    model, vectorizer, label_encoder, model_name = get_best_model_artifacts()

    # Encode labels using the model's fitted label encoder
    y_encoded = label_encoder.transform(y)
    target_names = list(label_encoder.classes_)

    # Create stratified test split (20%)
    _, X_test, _, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=random_state,
        stratify=y_encoded
    )

    # Vectorize test features
    X_test_vec = vectorizer.transform(X_test)

    # Compute actual predictions
    y_pred = model.predict(X_test_vec)

    # Calculate actual metrics
    actual_accuracy = float(accuracy_score(y_test, y_pred))
    actual_precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    actual_recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    actual_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    passed = actual_accuracy >= threshold

    print("\n" + "=" * 65)
    print("AUTOMATED CI MODEL ACCURACY GATE EVALUATION")
    print("=" * 65)
    print(f"Evaluated Model      : {model_name}")
    print(f"Test Samples         : {len(y_test)}")
    print(f"Configured Threshold : {threshold * 100:.2f}%")
    print(f"Actual Accuracy      : {actual_accuracy * 100:.2f}%")
    print(f"Actual Precision     : {actual_precision * 100:.2f}%")
    print(f"Actual Recall        : {actual_recall * 100:.2f}%")
    print(f"Actual F1-Score      : {actual_f1 * 100:.2f}%")
    print("-" * 65)

    if passed:
        print(f"[PASSED] Accuracy ({actual_accuracy*100:.2f}%) meets threshold ({threshold*100:.2f}%). Gate Passed!")
    else:
        print(f"[FAILED] Accuracy ({actual_accuracy*100:.2f}%) below threshold ({threshold*100:.2f}%). Gate Failed!")
    print("=" * 65 + "\n")

    return {
        "model_name": model_name,
        "threshold": threshold,
        "actual_accuracy": round(actual_accuracy, 4),
        "actual_precision": round(actual_precision, 4),
        "actual_recall": round(actual_recall, 4),
        "actual_f1": round(actual_f1, 4),
        "passed": passed,
        "test_samples": int(len(y_test))
    }


def main():
    parser = argparse.ArgumentParser(description="Automated CI Model Accuracy Gate")
    # Read threshold from env var ACCURACY_THRESHOLD if set, default to 0.80
    default_threshold = float(os.environ.get("ACCURACY_THRESHOLD", 0.80))
    parser.add_argument(
        "--threshold",
        type=float,
        default=default_threshold,
        help="Minimum required accuracy threshold (e.g. 0.80 for 80%%)"
    )
    args = parser.parse_args()

    result = run_accuracy_gate(threshold=args.threshold)
    if not result["passed"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
