"""
XGBoost Classifier Pipeline with Cross-Validated Hyperparameter Tuning.
Milestone 2: Advanced ML & Recommendation Engine.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Set path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from nlp.preprocessing import combine_profile_features
from ml.config import XGB_MODEL_DIR, XGB_PARAM_GRID, DATA_DIR, RESULTS_DIR


def load_dataset(dataset_path: str = None) -> pd.DataFrame:
    """Loads and cleans dataset for XGBoost training."""
    if dataset_path is None:
        dataset_path = os.path.join(DATA_DIR, "career_dataset.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Career dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["career", "skills"]).drop_duplicates()

    # Fill optional columns
    for col in ["education", "certifications", "projects"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    if "experience" in df.columns:
        df["experience"] = df["experience"].fillna(0)

    # Format unified feature text
    df["feature_text"] = [
        combine_profile_features({
            "education": row.get("education", ""),
            "skills": row.get("skills", ""),
            "experience": row.get("experience", 0),
            "certifications": row.get("certifications", ""),
            "projects": row.get("projects", "")
        })
        for _, row in df.iterrows()
    ]
    return df


def train_xgboost(
    dataset_path: str = None,
    model_dir: str = None,
    param_grid: dict = None,
    cv_folds: int = 5,
    random_state: int = 42
):
    """
    Trains an XGBoost classifier with GridSearchCV hyperparameter tuning.
    Saves best model, vectorizer, and label encoder.
    """
    if model_dir is None:
        model_dir = XGB_MODEL_DIR
    if param_grid is None:
        param_grid = XGB_PARAM_GRID

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[XGBoost] 1/5 Loading dataset...")
    df = load_dataset(dataset_path)
    X = df["feature_text"]
    y = df["career"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    target_names = list(label_encoder.classes_)
    num_classes = len(target_names)

    print("[XGBoost] 2/5 Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=random_state,
        stratify=y_encoded
    )

    print("[XGBoost] 3/5 Vectorizing text with TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)

    print(f"[XGBoost] 4/5 Running GridSearchCV ({cv_folds}-fold Stratified CV)...")
    base_xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=random_state,
        use_label_encoder=False
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_vec, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"[XGBoost] Best Hyperparameters: {best_params}")

    # Evaluate on test set
    print("[XGBoost] 5/5 Evaluating best model on test set...")
    y_pred = best_model.predict(X_test_vec)
    y_proba = best_model.predict_proba(X_test_vec)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    clf_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save artifacts
    model_path = os.path.join(model_dir, "xgboost_model.pkl")
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    meta_path = os.path.join(model_dir, "xgb_metrics.json")

    joblib.dump(best_model, model_path)
    joblib.dump(tfidf_vectorizer, vec_path)
    joblib.dump(label_encoder, le_path)

    metrics = {
        "model_name": "XGBoost",
        "best_params": best_params,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "cv_best_score": round(float(grid_search.best_score_), 4),
        "classes": target_names,
        "classification_report": clf_report,
        "confusion_matrix": cm
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"[XGBoost] Test Accuracy : {accuracy * 100:.2f}% | F1-Score: {f1 * 100:.2f}%")
    print(f"[XGBoost] Model and artifacts saved to: {model_dir}")
    return best_model, tfidf_vectorizer, label_encoder, metrics


if __name__ == "__main__":
    train_xgboost()
