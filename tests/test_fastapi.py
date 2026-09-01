"""
Unit and Integration Tests for FastAPI REST Service (Milestone 3).
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure parent directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_fastapi_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Career Prediction & Skill Gap REST API"
    assert "active_model" in data
    assert len(data["endpoints"]) >= 4


def test_fastapi_predict_endpoint_success(client):
    payload = {
        "name": "Jane Data",
        "email": "jane@example.com",
        "education": "B.Tech",
        "skills": ["Python", "Machine Learning", "Pandas", "Scikit-learn", "SQL"],
        "experience": 2.0,
        "top_k": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["prediction"] in ["Data Scientist", "Machine Learning Engineer"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["confidence_percentage"] <= 100.0
    assert len(data["recommendations"]) == 3
    for rec in data["recommendations"]:
        assert "career" in rec
        assert "probability" in rec
        assert "percentage" in rec


def test_fastapi_predict_invalid_skills_fails(client):
    payload = {
        "name": "Jane Data",
        "email": "jane@example.com",
        "education": "B.Tech",
        "skills": [],  # Empty list should fail validation
        "experience": 2.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["error_type"] == "ValidationError"


def test_fastapi_recommend_endpoint_success(client):
    payload = {
        "name": "Alex Web",
        "email": "alex@example.com",
        "education": "BCA",
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Express", "REST API"],
        "experience": 2.0,
        "top_k": 5
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["prediction"] == "Web Developer"
    assert len(data["recommendations"]) == 5
    top_rec = data["recommendations"][0]
    assert top_rec["rank"] == 1
    assert top_rec["career"] == "Web Developer"
    assert "overall_score" in top_rec
    assert "confidence_score" in top_rec
    assert "skill_alignment" in top_rec
    assert "semantic_similarity" in top_rec
    assert "matched_skills" in top_rec
    assert "missing_skills" in top_rec
    assert "React" in top_rec["matched_skills"]


def test_fastapi_recommend_invalid_experience_fails(client):
    payload = {
        "name": "Alex",
        "email": "alex@example.com",
        "education": "B.Tech",
        "skills": ["Python", "SQL"],
        "experience": -5.0  # Negative experience should fail
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422


def test_fastapi_skill_gap_report_endpoint(client):
    payload = {
        "skills": ["Python", "SQL", "Pandas"],
        "target_career": "Data Scientist"
    }
    response = client.post("/skill-gap-report", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["target_career"] == "Data Scientist"
    assert "Python" in data["matched_skills"]
    assert "SQL" in data["matched_skills"]
    assert len(data["missing_skills"]) > 0
    assert 0.0 <= data["skill_gap_percentage"] <= 100.0
    assert 0.0 <= data["competency_match_score"] <= 100.0
    assert len(data["improvement_suggestions"]) > 0

    first_sug = data["improvement_suggestions"][0]
    assert "skill" in first_sug
    assert "priority" in first_sug
    assert "recommended_action" in first_sug
    assert len(first_sug["learning_resources"]) > 0


def test_fastapi_openapi_json(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "paths" in data
    assert "/health" in data["paths"]
    assert "/predict" in data["paths"]
    assert "/recommend" in data["paths"]
    assert "/skill-gap-report" in data["paths"]
