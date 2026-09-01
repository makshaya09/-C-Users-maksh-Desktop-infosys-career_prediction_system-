"""
Complete End-to-End Flow & Regression Tests.
Verifies input -> parsing -> validation -> prediction -> recommendation -> skill gap -> report pipeline.
"""

import os
import tempfile
import pytest
from fastapi.testclient import TestClient

import careercast
from api.main import app as fastapi_app

client = TestClient(fastapi_app)


def test_complete_end_to_end_pipeline():
    # 1. Raw resume input text
    raw_resume = """
    Rohan Verma
    Email: rohan.verma@cloudtech.io
    Education: B.Tech in Information Technology
    Experience: 3 years of experience as DevOps & Cloud Engineer
    Technical Skills: AWS, Docker, Kubernetes, Linux, Terraform, CI/CD, Jenkins, Python, Git
    Projects: Multi-region Kubernetes infrastructure automation with Terraform
    Certifications: AWS Certified Solutions Architect Associate
    """

    # 2. NLP Resume Parsing
    parsed = careercast.parse_resume(raw_resume)
    assert "AWS" in parsed["skills"]
    assert "Docker" in parsed["skills"]
    assert "Kubernetes" in parsed["skills"]
    assert parsed["primary_education"] == "B.Tech"
    assert parsed["experience"] == 3.0

    # 3. Profile Assembly & Validation
    profile = {
        "name": "Rohan Verma",
        "email": "rohan.verma@cloudtech.io",
        "education": parsed["primary_education"],
        "skills": parsed["skills"],
        "experience": parsed["experience"],
        "certifications": ", ".join(parsed["certifications"]),
        "projects": "; ".join(parsed["projects"])
    }
    is_valid, errors = careercast.validate_profile(profile)
    assert is_valid is True
    assert len(errors) == 0

    # 4. Feature Combination
    combined_features = careercast.combine_profile_features(profile)
    assert "Education: B.Tech" in combined_features
    assert "Skills:" in combined_features

    # 5. Career Prediction & Probability Distribution
    pred_res = careercast.predict_career(profile, top_k=3)
    assert pred_res["prediction"] == "Cloud Engineer"
    assert pred_res["confidence"] > 0.0
    assert len(pred_res["recommendations"]) == 3

    # 6. Multi-Modal Top-K Recommendation
    rec_res = careercast.recommend_careers(profile, top_k=5)
    assert rec_res["prediction"] == "Cloud Engineer"
    assert rec_res["overall_score"] > 30.0
    assert "user_skills" in rec_res

    # 7. Skill Gap Analysis & Roadmap
    gap_res = careercast.analyze_skill_gap(
        candidate_skills=profile["skills"],
        target_career="Cloud Engineer",
        profile=profile
    )
    assert gap_res["target_career"] == "Cloud Engineer"
    assert "AWS" in gap_res["matched_skills"]
    assert "Docker" in gap_res["matched_skills"]
    assert gap_res["competency_match_score"] > 40.0
    assert len(gap_res["improvement_suggestions"]) > 0

    # 8. Report Generation (Markdown, JSON, PDF)
    md_report = careercast.generate_markdown_report(profile, rec_res, gap_res)
    assert "# AI Career Assessment & Skill Gap Analysis Report" in md_report
    assert "Rohan Verma" in md_report

    json_report = careercast.generate_json_report(profile, rec_res, gap_res)
    assert '"report_metadata"' in json_report

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = os.path.join(tmp_dir, "e2e_report.pdf")
        pdf_bytes = careercast.generate_pdf_report(profile, rec_res, gap_res, output_path=pdf_path)
        assert os.path.exists(pdf_path)
        assert len(pdf_bytes) > 2000
        assert pdf_bytes.startswith(b"%PDF")


def test_api_endpoints_e2e_integration():
    # 1. Health endpoint
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True

    # 2. Predict endpoint
    predict_payload = {
        "education": "B.Tech",
        "skills": "Python, Machine Learning, Deep Learning, SQL",
        "experience": 2.0,
        "top_k": 3
    }
    resp = client.post("/predict", json=predict_payload)
    assert resp.status_code == 200
    p_data = resp.json()
    assert p_data["status"] == "success"
    assert "prediction" in p_data
    assert len(p_data["recommendations"]) == 3

    # 3. Recommend endpoint
    rec_payload = {
        "education": "B.Tech",
        "skills": "HTML, CSS, JavaScript, React, Node.js",
        "experience": 2.0,
        "top_k": 5
    }
    resp = client.post("/recommend", json=rec_payload)
    assert resp.status_code == 200
    r_data = resp.json()
    assert r_data["status"] == "success"
    assert r_data["prediction"] == "Web Developer"

    # 4. Skill Gap Report endpoint
    gap_payload = {
        "skills": "AWS, Docker, Linux",
        "target_career": "Cloud Engineer",
        "education": "B.Tech",
        "experience": 2.0
    }
    resp = client.post("/skill-gap-report", json=gap_payload)
    assert resp.status_code == 200
    g_data = resp.json()
    assert g_data["target_career"] == "Cloud Engineer"
    assert "AWS" in g_data["matched_skills"]
    assert len(g_data["improvement_suggestions"]) > 0


def test_api_invalid_requests_handled_gracefully():
    # Blank skills in predict
    resp = client.post("/predict", json={"skills": "", "education": "B.Tech"})
    assert resp.status_code == 422

    # Negative experience in recommend
    resp = client.post("/recommend", json={"skills": "Python", "experience": -5.0})
    assert resp.status_code == 422
