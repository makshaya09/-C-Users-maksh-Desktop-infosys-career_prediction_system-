"""
Model Comparison and Dynamic Model Selection Module.
Milestone 2: Advanced ML & Recommendation Engine.
"""

import os
import sys
import json
import joblib
import pandas as pd
from typing import Dict, Any, Tuple

# Set path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml.config import MODELS_DIR, RF_MODEL_DIR, XGB_MODEL_DIR, RESULTS_DIR, REPORTS_DIR
from ml.random_forest import train_random_forest
from ml.xgboost_model import train_xgboost
from ml.train_model import train_baseline_model


def run_model_comparison(force_retrain: bool = False) -> Dict[str, Any]:
    """
    Runs or loads evaluation metrics for Logistic Regression, Random Forest, and XGBoost.
    Compares models on Accuracy, Precision, Recall, and F1-Score.
    Selects the winning model automatically based on highest F1-Score.
    Saves results to results/model_comparison.json.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    models_comparison = []

    # 1. Logistic Regression (Baseline)
    lr_metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    if force_retrain or not os.path.exists(lr_metrics_path):
        print("[Comparison] Training Logistic Regression baseline...")
        train_baseline_model()

    with open(lr_metrics_path, "r", encoding="utf-8") as f:
        lr_data = json.load(f)

    models_comparison.append({
        "model": "Logistic Regression (Baseline)",
        "model_key": "logistic_regression",
        "accuracy": lr_data.get("accuracy", 0.0),
        "precision": lr_data.get("precision", 0.0),
        "recall": lr_data.get("recall", 0.0),
        "f1_score": lr_data.get("f1_score", 0.0),
        "model_dir": MODELS_DIR
    })

    # 2. Random Forest
    rf_metrics_path = os.path.join(RF_MODEL_DIR, "rf_metrics.json")
    if force_retrain or not os.path.exists(rf_metrics_path):
        print("[Comparison] Training Random Forest with GridSearchCV...")
        train_random_forest()

    with open(rf_metrics_path, "r", encoding="utf-8") as f:
        rf_data = json.load(f)

    models_comparison.append({
        "model": "Random Forest",
        "model_key": "random_forest",
        "accuracy": rf_data.get("accuracy", 0.0),
        "precision": rf_data.get("precision", 0.0),
        "recall": rf_data.get("recall", 0.0),
        "f1_score": rf_data.get("f1_score", 0.0),
        "best_params": rf_data.get("best_params", {}),
        "model_dir": RF_MODEL_DIR
    })

    # 3. XGBoost
    xgb_metrics_path = os.path.join(XGB_MODEL_DIR, "xgb_metrics.json")
    if force_retrain or not os.path.exists(xgb_metrics_path):
        print("[Comparison] Training XGBoost with GridSearchCV...")
        train_xgboost()

    with open(xgb_metrics_path, "r", encoding="utf-8") as f:
        xgb_data = json.load(f)

    models_comparison.append({
        "model": "XGBoost",
        "model_key": "xgboost",
        "accuracy": xgb_data.get("accuracy", 0.0),
        "precision": xgb_data.get("precision", 0.0),
        "recall": xgb_data.get("recall", 0.0),
        "f1_score": xgb_data.get("f1_score", 0.0),
        "best_params": xgb_data.get("best_params", {}),
        "model_dir": XGB_MODEL_DIR
    })

    # Dynamic selection: Sort primarily by F1-Score, then Accuracy, then Model Complexity (Ensemble over baseline)
    sorted_models = sorted(
        models_comparison,
        key=lambda m: (
            m["f1_score"],
            m["accuracy"],
            2 if m["model_key"] == "random_forest" else (1 if m["model_key"] == "xgboost" else 0)
        ),
        reverse=True
    )
    best_model_info = sorted_models[0]

    comparison_results = {
        "models": models_comparison,
        "best_model": best_model_info["model"],
        "best_model_key": best_model_info["model_key"],
        "best_f1_score": best_model_info["f1_score"],
        "best_accuracy": best_model_info["accuracy"],
        "selection_criteria": "Highest F1-Score (Weighted) with Accuracy tie-breaker"
    }

    comparison_file = os.path.join(RESULTS_DIR, "model_comparison.json")
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=4)

    print("\n" + "=" * 65)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Model':<30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    for m in models_comparison:
        print(f"{m['model']:<30} | {m['accuracy']*100:>8.2f}% | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}% | {m['f1_score']*100:>8.2f}%")
    print("=" * 65)
    print(f"[*] Selected Best Model: {best_model_info['model']} (F1: {best_model_info['f1_score']*100:.2f}%)")
    print("=" * 65 + "\n")

    return comparison_results


def get_best_model_artifacts() -> Tuple[Any, Any, Any, str]:
    """
    Loads and returns (model, vectorizer, label_encoder, model_name) for the winning model.
    Falls back to Random Forest or Logistic Regression if artifacts are missing.
    """
    comparison_file = os.path.join(RESULTS_DIR, "model_comparison.json")
    if not os.path.exists(comparison_file):
        run_model_comparison()

    with open(comparison_file, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    best_key = comp_data.get("best_model_key", "random_forest")

    if best_key == "xgboost":
        m_path = os.path.join(XGB_MODEL_DIR, "xgboost_model.pkl")
        v_path = os.path.join(XGB_MODEL_DIR, "tfidf_vectorizer.pkl")
        l_path = os.path.join(XGB_MODEL_DIR, "label_encoder.pkl")
        name = "XGBoost"
    elif best_key == "random_forest":
        m_path = os.path.join(RF_MODEL_DIR, "random_forest_model.pkl")
        v_path = os.path.join(RF_MODEL_DIR, "tfidf_vectorizer.pkl")
        l_path = os.path.join(RF_MODEL_DIR, "label_encoder.pkl")
        name = "Random Forest"
    else:
        m_path = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
        v_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        l_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
        name = "Logistic Regression"

    # Verify files exist, fallback if needed
    if not (os.path.exists(m_path) and os.path.exists(v_path) and os.path.exists(l_path)):
        # Fallback to base logistic regression
        m_path = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
        v_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        l_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
        name = "Logistic Regression (Fallback)"

    model = joblib.load(m_path)
    vectorizer = joblib.load(v_path)
    label_encoder = joblib.load(l_path)
    return model, vectorizer, label_encoder, name


if __name__ == "__main__":
    run_model_comparison(force_retrain=False)
