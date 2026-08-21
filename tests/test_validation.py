"""
Unit tests for User Profile Form Validation.
"""

import os
import sys
import pytest

# Ensure parent directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app import validate_profile_data


def test_valid_profile():
    valid_data = {
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "education": "B.Tech",
        "degree_major": "Computer Science",
        "skills": "Python, SQL, Machine Learning",
        "experience": "2",
        "certifications": "AWS Certified",
        "projects": "Recommendation System"
    }
    is_valid, errors = validate_profile_data(valid_data)
    assert is_valid is True
    assert len(errors) == 0


def test_empty_name():
    data = {
        "name": "   ",
        "email": "jane@example.com",
        "education": "B.Tech",
        "skills": "Python",
        "experience": "1"
    }
    is_valid, errors = validate_profile_data(data)
    assert is_valid is False
    assert any("Name" in err for err in errors)


def test_invalid_email():
    data = {
        "name": "Jane",
        "email": "invalid-email-format",
        "education": "B.Tech",
        "skills": "Python",
        "experience": "1"
    }
    is_valid, errors = validate_profile_data(data)
    assert is_valid is False
    assert any("email" in err.lower() for err in errors)


def test_empty_education():
    data = {
        "name": "Jane",
        "email": "jane@example.com",
        "education": "",
        "skills": "Python",
        "experience": "1"
    }
    is_valid, errors = validate_profile_data(data)
    assert is_valid is False
    assert any("Education" in err for err in errors)


def test_empty_skills():
    data = {
        "name": "Jane",
        "email": "jane@example.com",
        "education": "B.Tech",
        "skills": "",
        "experience": "1"
    }
    is_valid, errors = validate_profile_data(data)
    assert is_valid is False
    assert any("skill" in err.lower() for err in errors)


def test_negative_experience():
    data = {
        "name": "Jane",
        "email": "jane@example.com",
        "education": "B.Tech",
        "skills": "Python",
        "experience": "-3"
    }
    is_valid, errors = validate_profile_data(data)
    assert is_valid is False
    assert any("negative" in err.lower() for err in errors)


def test_non_numeric_experience():
    data = {
        "name": "Jane",
        "email": "jane@example.com",
        "education": "B.Tech",
        "skills": "Python",
        "experience": "five years"
    }
    is_valid, errors = validate_profile_data(data)
    assert is_valid is False
    assert any("numeric" in err.lower() for err in errors)
