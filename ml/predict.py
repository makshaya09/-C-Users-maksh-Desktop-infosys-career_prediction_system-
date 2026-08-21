"""
Career Prediction Module.
Loads trained Logistic Regression model and TF-IDF vectorizer to generate
career predictions and top-3 ranked recommendations with probabilities.
"""

import os
import sys
from typing import Dict, Any, List
import joblib
import numpy as np

# Add parent directory to path to enable relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from nlp.preprocessing import combine_profile_features

# Cache loaded models in memory for fast inference
_model = None
_vectorizer = None
_label_encoder = None


def load_model_artifacts(models_dir: str = None):
    """
    Loads saved model, vectorizer, and label encoder from models/ directory.
    """
    global _model, _vectorizer, _label_encoder

    if _model is not None and _vectorizer is not None and _label_encoder is not None:
        return _model, _vectorizer, _label_encoder

    if models_dir is None:
        models_dir = os.path.join(parent_dir, "models")

    model_path = os.path.join(models_dir, "logistic_regression_model.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    le_path = os.path.join(models_dir, "label_encoder.pkl")

    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(le_path)):
        raise FileNotFoundError(
            "Trained model artifacts not found. Please run 'python ml/train_model.py' first."
        )

    _model = joblib.load(model_path)
    _vectorizer = joblib.load(vectorizer_path)
    _label_encoder = joblib.load(le_path)

    return _model, _vectorizer, _label_encoder


def predict_career(profile: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    """
    Predicts career category and top-K recommendations for a structured user profile.

    Input profile schema:
    {
        "education": "B.Tech",
        "skills": ["Python", "Machine Learning", "SQL"],  # or comma-separated string
        "experience": 2.0,
        "certifications": ["AWS Certified"],
        "projects": ["Recommendation System"]
    }

    Returns:
    {
        "prediction": "Data Scientist",
        "confidence": 0.84,
        "recommendations": [
            {"career": "Data Scientist", "probability": 0.84},
            {"career": "Data Analyst", "probability": 0.10},
            {"career": "Machine Learning Engineer", "probability": 0.06}
        ],
        "feature_text": "Education: B.Tech | Skills: Python, ..."
    }
    """
    model, vectorizer, label_encoder = load_model_artifacts()

    # Convert profile into unified string matching training pipeline
    feature_text = combine_profile_features(profile)

    # Vectorize
    feature_vec = vectorizer.transform([feature_text])

    # Predict probabilities
    probabilities = model.predict_proba(feature_vec)[0]
    classes = label_encoder.classes_

    # Sort indices by probability descending
    top_indices = np.argsort(probabilities)[::-1]

    recommendations: List[Dict[str, Any]] = []
    for idx in top_indices[:top_k]:
        career_name = str(classes[idx])
        prob = float(probabilities[idx])
        recommendations.append({
            "career": career_name,
            "probability": round(prob, 4),
            "percentage": round(prob * 100, 1)
        })

    top_prediction = recommendations[0]["career"]
    confidence = recommendations[0]["probability"]

    return {
        "prediction": top_prediction,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 1),
        "recommendations": recommendations,
        "feature_text": feature_text
    }


if __name__ == "__main__":
    sample_profile = {
        "education": "B.Tech",
        "skills": ["Python", "Machine Learning", "Pandas", "Scikit-learn", "SQL"],
        "experience": 2,
        "certifications": "IBM Data Science",
        "projects": "Customer Churn Prediction"
    }
    try:
        res = predict_career(sample_profile)
        print("Sample Prediction Result:")
        print(f"Predicted Career: {res['prediction']} ({res['confidence_percentage']}%)")
        print("Top Recommendations:")
        for r in res["recommendations"]:
            print(f"  - {r['career']}: {r['percentage']}%")
    except Exception as e:
        print(f"Prediction test note: {e}")
