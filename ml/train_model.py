"""
Model Training Module for Career Prediction System (Milestone 1).
Trains a Baseline Logistic Regression Classifier on the Career Dataset.
"""

import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Add parent directory to path to enable relative imports when run as script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from nlp.preprocessing import combine_profile_features
from ml.evaluate import evaluate_model


def load_and_preprocess_data(dataset_path: str):
    """
    Loads dataset, handles missing values, removes duplicates, and prepares feature text.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Clean missing values
    df = df.dropna(subset=["career", "skills"])
    df = df.drop_duplicates()

    # Fill optional missing columns
    for col in ["education", "certifications", "projects"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    if "experience" in df.columns:
        df["experience"] = df["experience"].fillna(0)

    # Construct unified feature text representation
    feature_texts = []
    for _, row in df.iterrows():
        profile = {
            "education": row.get("education", ""),
            "skills": row.get("skills", ""),
            "experience": row.get("experience", 0),
            "certifications": row.get("certifications", ""),
            "projects": row.get("projects", "")
        }
        feature_texts.append(combine_profile_features(profile))

    df["feature_text"] = feature_texts
    return df


def train_baseline_model(
    dataset_path: str = None,
    models_dir: str = None,
    reports_dir: str = None,
    random_state: int = 42
):
    """
    Trains the baseline Logistic Regression model and saves artifacts.
    """
    base_dir = parent_dir

    if dataset_path is None:
        dataset_path = os.path.join(base_dir, "data", "career_dataset.csv")
    if models_dir is None:
        models_dir = os.path.join(base_dir, "models")
    if reports_dir is None:
        reports_dir = os.path.join(base_dir, "reports")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("[1/5] Loading and preprocessing dataset...")
    df = load_and_preprocess_data(dataset_path)
    print(f"      Total processed samples: {len(df)}")
    print(f"      Target classes: {list(df['career'].unique())}")

    X = df["feature_text"]
    y = df["career"]

    # Encode target career labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    target_names = list(label_encoder.classes_)

    # Stratified Train/Test Split (80% train, 20% test)
    print("[2/5] Splitting data into train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=random_state,
        stratify=y_encoded
    )

    # TF-IDF Feature Extraction
    print("[3/5] Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)

    # Train Logistic Regression Baseline Classifier
    print("[4/5] Training Logistic Regression model (random_state=42)...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        C=1.0
    )
    model.fit(X_train_vec, y_train)

    # Save Model Artifacts
    print("[5/5] Saving model artifacts to models/ directory...")
    model_path = os.path.join(models_dir, "logistic_regression_model.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    le_path = os.path.join(models_dir, "label_encoder.pkl")

    joblib.dump(model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    joblib.dump(label_encoder, le_path)

    print(f"      Saved: {model_path}")
    print(f"      Saved: {vectorizer_path}")
    print(f"      Saved: {le_path}")

    # Evaluate and Generate Reports
    print("\n--- Evaluating Model & Generating Milestone 1 Reports ---")
    metrics = evaluate_model(
        model=model,
        X_test_vec=X_test_vec,
        y_test_encoded=y_test,
        target_names=target_names,
        reports_dir=reports_dir
    )

    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score : {metrics['f1_score'] * 100:.2f}%")
    print(f"Coverage : {metrics['coverage'] * 100:.2f}%")
    print("\nReports successfully generated in reports/ directory.")

    return model, tfidf_vectorizer, label_encoder, metrics


if __name__ == "__main__":
    train_baseline_model()
