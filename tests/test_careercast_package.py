"""
Unit and Integration Tests for CareerCast Top-Level Python API Package.
"""

import pytest
import careercast


def test_careercast_version_and_metadata():
    assert hasattr(careercast, "__version__")
    assert careercast.__version__ == "1.0.0"


def test_careercast_resume_parsing_api():
    sample_text = """
    John Doe
    Email: john.doe@example.com
    Education: B.Tech in Computer Science
    Experience: 3 years of experience as Software Developer
    Technical Skills: Python, Java, SQL, React, Git, Docker
    Projects: E-Commerce Microservices Platform
    """
    parsed = careercast.parse_resume(sample_text)
    assert isinstance(parsed, dict)
    assert "skills" in parsed
    assert "Python" in parsed["skills"]
    assert "Java" in parsed["skills"]
    assert "SQL" in parsed["skills"]
    assert parsed["primary_education"] == "B.Tech"
    assert parsed["experience"] == 3.0


def test_careercast_profile_validation_api():
    valid_profile = {
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "education": "B.Tech",
        "skills": "Python, Machine Learning",
        "experience": 2.0
    }
    is_valid, errors = careercast.validate_profile(valid_profile)
    assert is_valid is True
    assert len(errors) == 0

    invalid_profile = {
        "name": "",
        "email": "invalid-email",
        "education": "",
        "skills": "",
        "experience": -5
    }
    is_valid, errors = careercast.validate_profile(invalid_profile)
    assert is_valid is False
    assert len(errors) >= 3


def test_careercast_predict_career_api():
    profile = {
        "education": "B.Tech",
        "skills": ["Python", "Machine Learning", "Pandas", "Scikit-learn", "SQL"],
        "experience": 2.0
    }
    pred_res = careercast.predict_career(profile, top_k=3)
    assert "prediction" in pred_res
    assert "confidence" in pred_res
    assert "confidence_percentage" in pred_res
    assert len(pred_res["recommendations"]) == 3
    assert pred_res["confidence"] > 0.0


def test_careercast_recommend_careers_api():
    profile = {
        "education": "B.Tech",
        "skills": "AWS, Docker, Kubernetes, Linux, Terraform, CI/CD",
        "experience": 2.5
    }
    rec_res = careercast.recommend_careers(profile, top_k=5)
    assert "prediction" in rec_res
    assert "overall_score" in rec_res
    assert len(rec_res["recommendations"]) <= 5
    assert rec_res["prediction"] == "Cloud Engineer"


def test_careercast_analyze_skill_gap_api():
    candidate_skills = ["Python", "SQL", "Pandas"]
    gap_res = careercast.analyze_skill_gap(candidate_skills, target_career="Data Scientist")
    assert gap_res["target_career"] == "Data Scientist"
    assert "Python" in gap_res["matched_skills"]
    assert "SQL" in gap_res["matched_skills"]
    assert "Pandas" in gap_res["matched_skills"]
    assert "Machine Learning" in gap_res["missing_skills"]
    assert gap_res["matched_count"] == 3
    assert gap_res["skill_gap_percentage"] > 0
    assert len(gap_res["improvement_suggestions"]) > 0


def test_careercast_report_generation_api():
    profile = {
        "name": "Alex Smith",
        "email": "alex@example.com",
        "education": "M.Tech",
        "skills": "Python, SQL, Machine Learning",
        "experience": 3.0
    }
    rec_res = careercast.recommend_careers(profile, top_k=3)
    gap_res = careercast.analyze_skill_gap(profile["skills"], target_career="Data Scientist")

    # Markdown report
    md_str = careercast.generate_markdown_report(profile, rec_res, gap_res)
    assert "# AI Career Assessment & Skill Gap Analysis Report" in md_str
    assert "Alex Smith" in md_str

    # JSON report
    json_str = careercast.generate_json_report(profile, rec_res, gap_res)
    assert "report_metadata" in json_str
    assert "Alex Smith" in json_str

    # PDF report
    pdf_bytes = careercast.generate_pdf_report(profile, rec_res, gap_res)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
