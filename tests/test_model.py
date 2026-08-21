"""
Unit tests for Model Training, Inference, and Career Prediction.
"""

import os
import sys
import pytest

# Ensure parent directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml.predict import predict_career, load_model_artifacts
from ml.evaluate import calculate_coverage
import numpy as np


def test_model_artifacts_load():
    model, vectorizer, label_encoder = load_model_artifacts()
    assert model is not None
    assert vectorizer is not None
    assert label_encoder is not None
    assert hasattr(model, "predict_proba")
    assert len(label_encoder.classes_) == 7


def test_predict_career_structure():
    profile = {
        "education": "B.Tech",
        "skills": ["Python", "Machine Learning", "Pandas", "Scikit-learn", "SQL", "Statistics"],
        "experience": 2,
        "certifications": "IBM Data Science",
        "projects": "Customer Churn Prediction"
    }
    result = predict_career(profile, top_k=3)

    assert "prediction" in result
    assert "confidence" in result
    assert "confidence_percentage" in result
    assert "recommendations" in result

    # Check top 3 recommendations
    recs = result["recommendations"]
    assert len(recs) == 3

    # Check sorting order: prob[0] >= prob[1] >= prob[2]
    assert recs[0]["probability"] >= recs[1]["probability"]
    assert recs[1]["probability"] >= recs[2]["probability"]

    # Check top prediction matches the first recommendation
    assert result["prediction"] == recs[0]["career"]
    assert result["confidence"] == recs[0]["probability"]


def test_data_scientist_prediction():
    profile = {
        "education": "M.Tech",
        "skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "Pandas", "Scikit-learn"],
        "experience": 3,
        "certifications": "TensorFlow Developer",
        "projects": "Neural Network Classifier"
    }
    result = predict_career(profile)
    assert result["prediction"] in ["Data Scientist", "Machine Learning Engineer"]


def test_web_developer_prediction():
    profile = {
        "education": "BCA",
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Express", "Bootstrap"],
        "experience": 2,
        "certifications": "Full Stack Web Development",
        "projects": "E-Commerce Web Portal"
    }
    result = predict_career(profile)
    assert result["prediction"] == "Web Developer"


def test_cybersecurity_prediction():
    profile = {
        "education": "B.Tech",
        "skills": ["Network Security", "Wireshark", "Ethical Hacking", "Nmap", "Metasploit", "Firewall"],
        "experience": 2,
        "certifications": "CompTIA Security+",
        "projects": "Vulnerability Assessment Sandbox"
    }
    result = predict_career(profile)
    assert result["prediction"] == "Cybersecurity Analyst"


def test_calculate_coverage():
    # Simulated probabilities: 5 samples, all valid
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.6, 0.3, 0.1],
        [0.9, 0.05, 0.05],
        [0.7, 0.2, 0.1],
        [0.5, 0.3, 0.2]
    ])
    cov = calculate_coverage(probs, threshold=0.20)
    assert cov == 1.0
