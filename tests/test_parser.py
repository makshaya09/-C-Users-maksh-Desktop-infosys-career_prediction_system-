"""
Unit tests for Resume Parser and SpaCy Entity Extraction.
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

from nlp.preprocessing import clean_text, normalize_text, combine_profile_features
from nlp.resume_parser import (
    extract_text,
    extract_entities,
    extract_skills,
    extract_education,
    extract_roles,
    parse_resume
)


def test_clean_text():
    raw = "John Doe \n\n Email: john@example.com \t Website: https://johndoe.dev • Python • SQL"
    cleaned = clean_text(raw)
    assert "https://" not in cleaned
    assert "john@example.com" not in cleaned
    assert "Python" in cleaned
    assert "SQL" in cleaned


def test_combine_profile_features():
    profile = {
        "education": "B.Tech",
        "skills": ["Python", "Machine Learning", "SQL"],
        "experience": 3,
        "certifications": ["AWS Certified"],
        "projects": ["Churn Prediction"]
    }
    feature_str = combine_profile_features(profile)
    assert "Education: B.Tech" in feature_str
    assert "Python, Machine Learning, SQL" in feature_str
    assert "Experience: 3 years" in feature_str
    assert "AWS Certified" in feature_str
    assert "Churn Prediction" in feature_str


def test_extract_skills_from_text():
    sample_text = """
    Experienced Data Professional with expertise in Python, SQL, Machine Learning,
    Pandas, NumPy, Scikit-learn, and Tableau. Built predictive models with TensorFlow.
    """
    skills = extract_skills(sample_text)
    skills_lower = [s.lower() for s in skills]

    assert "python" in skills_lower
    assert "sql" in skills_lower
    assert "machine learning" in skills_lower
    assert "pandas" in skills_lower
    assert "scikit-learn" in skills_lower or "sklearn" in skills_lower
    assert "tableau" in skills_lower


def test_extract_education_from_text():
    sample_text = "Education: Bachelor of Technology (B.Tech) in Computer Science, followed by M.Tech."
    education = extract_education(sample_text)
    edu_lower = [e.lower() for e in education]

    assert "b.tech" in edu_lower or "bachelor of technology" in edu_lower
    assert "m.tech" in edu_lower


def test_extract_roles_from_text():
    sample_text = "Previously worked as a Software Developer and later as a Data Scientist at Tech Corp."
    roles = extract_roles(sample_text)
    roles_lower = [r.lower() for r in roles]

    assert "software developer" in roles_lower
    assert "data scientist" in roles_lower


def test_parse_resume_from_txt_stream():
    sample_resume = """
    Alex Smith
    Email: alex.smith@example.com
    Degree: B.Tech in Computer Science
    Experience: 4 years of experience as Cloud Engineer
    Skills: AWS, Docker, Kubernetes, Linux, Terraform, CI/CD
    Certifications: AWS Solutions Architect Associate
    Projects: High Availability Microservices Infrastructure
    """
    stream = io.BytesIO(sample_resume.encode("utf-8"))
    result = parse_resume(stream, filename="resume.txt")

    assert isinstance(result, dict)
    assert "skills" in result
    assert "education" in result
    assert "roles" in result

    skills_lower = [s.lower() for s in result["skills"]]
    assert "aws" in skills_lower
    assert "docker" in skills_lower
    assert "kubernetes" in skills_lower
    assert result["experience"] == 4.0


def test_parse_resume_empty_raises_error():
    empty_stream = io.BytesIO(b"")
    with pytest.raises(ValueError):
        parse_resume(empty_stream, filename="empty.txt")
