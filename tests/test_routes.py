"""
Integration tests for Flask application routes and endpoints.
"""

import io
import os
import sys
import pytest

# Ensure parent directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI-Based Career Prediction" in response.data


def test_upload_page_route(client):
    response = client.get("/upload")
    assert response.status_code == 200
    assert b"Upload Resume" in response.data


def test_profile_page_route(client):
    response = client.get("/profile")
    assert response.status_code == 200
    assert b"Structured User Profile Ingestion Form" in response.data


def test_report_page_route(client):
    response = client.get("/report")
    assert response.status_code == 200
    assert b"Model Comparison" in response.data or b"Model Performance" in response.data
    assert b"Confusion Matrix" in response.data


def test_parse_route_with_txt_file(client):
    resume_content = """
    Jane Developer
    Email: jane.dev@example.com
    Education: B.Tech in Information Technology
    Skills: HTML, CSS, JavaScript, React, Node.js, Express, MongoDB
    Experience: 2 years of experience
    Projects: Social Media Web App
    """
    data = {
        "resume": (io.BytesIO(resume_content.encode("utf-8")), "resume.txt")
    }
    response = client.post("/parse", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Resume Parsed Successfully" in response.data
    assert b"React" in response.data or b"JavaScript" in response.data


def test_predict_route_success(client):
    form_data = {
        "name": "Jane Developer",
        "email": "jane.dev@example.com",
        "education": "B.Tech",
        "degree_major": "Information Technology",
        "skills": "HTML, CSS, JavaScript, React, Node.js, Express, MongoDB",
        "experience": "2",
        "certifications": "Meta Front-End Developer",
        "projects": "Social Media Web App",
        "preferred_career": ""
    }
    response = client.post("/predict", data=form_data)
    assert response.status_code == 200
    assert b"Top Career Recommendation" in response.data or b"Top Recommendation" in response.data
    assert b"Web Developer" in response.data


def test_predict_route_validation_failure(client):
    # Missing required name and skills
    invalid_data = {
        "name": "",
        "email": "invalid-email",
        "education": "",
        "skills": "",
        "experience": "-5"
    }
    response = client.post("/predict", data=invalid_data)
    assert response.status_code == 200
    assert b"Validation Error" in response.data or b"cannot be empty" in response.data


def test_api_predict_json(client):
    payload = {
        "name": "Alice Cyber",
        "email": "alice@security.org",
        "education": "B.Tech",
        "skills": ["Network Security", "Wireshark", "Ethical Hacking", "Nmap", "Metasploit"],
        "experience": 3,
        "certifications": ["CompTIA Security+"],
        "projects": ["Intrusion Detection System"]
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["data"]["prediction"] == "Cybersecurity Analyst"
    assert len(json_data["data"]["recommendations"]) == 3
