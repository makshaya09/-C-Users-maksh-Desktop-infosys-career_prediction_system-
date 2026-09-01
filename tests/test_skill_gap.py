"""
Unit and Integration Tests for Skill Gap Analysis Module (Milestone 3).
"""

import os
import sys
import pytest

# Ensure parent directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from skill_gap.analyzer import analyze_skill_gap, get_required_skills_for_career
from skill_gap.suggestions import (
    generate_improvement_suggestions,
    get_suggestion_for_skill,
    SKILL_SUGGESTIONS_DB
)
from ml.recommendation_engine import CAREER_REQUIRED_SKILLS


def test_get_required_skills_for_all_careers():
    for career in CAREER_REQUIRED_SKILLS.keys():
        skills = get_required_skills_for_career(career)
        assert isinstance(skills, list)
        assert len(skills) >= 5
        assert all(isinstance(s, str) for s in skills)


def test_get_required_skills_case_insensitive():
    skills_upper = get_required_skills_for_career("DATA SCIENTIST")
    skills_lower = get_required_skills_for_career("data scientist")
    assert skills_upper == skills_lower
    assert "Python" in skills_upper
    assert "SQL" in skills_upper


def test_analyze_skill_gap_data_scientist():
    candidate_skills = ["Python", "SQL", "Pandas", "Scikit-learn"]
    res = analyze_skill_gap(candidate_skills, target_career="Data Scientist")

    assert res["status"] == "success"
    assert res["target_career"] == "Data Scientist"
    assert "Python" in res["matched_skills"]
    assert "SQL" in res["matched_skills"]
    assert "Pandas" in res["matched_skills"]
    assert "Scikit-learn" in res["matched_skills"]
    assert res["matched_count"] >= 4

    # Missing skills should be present
    assert len(res["missing_skills"]) > 0
    assert "Deep Learning" in res["missing_skills"] or "TensorFlow" in res["missing_skills"]

    # Gap and Match percentage validation
    assert 0.0 <= res["skill_gap_percentage"] <= 100.0
    assert 0.0 <= res["competency_match_score"] <= 100.0
    assert round(res["skill_gap_percentage"] + res["competency_match_score"], 1) == 100.0

    # Suggestions validation
    assert len(res["improvement_suggestions"]) == res["missing_count"]
    first_sug = res["improvement_suggestions"][0]
    assert "skill" in first_sug
    assert "priority" in first_sug
    assert "recommended_action" in first_sug
    assert "learning_resources" in first_sug
    assert "practical_project" in first_sug
    assert "estimated_time" in first_sug


def test_analyze_skill_gap_all_skills_matched():
    req_skills = list(CAREER_REQUIRED_SKILLS["Web Developer"])
    res = analyze_skill_gap(req_skills, target_career="Web Developer")

    assert res["matched_count"] == len(req_skills)
    assert res["missing_count"] == 0
    assert res["skill_gap_percentage"] == 0.0
    assert res["competency_match_score"] == 100.0
    assert len(res["improvement_suggestions"]) == 0


def test_analyze_skill_gap_no_skills():
    res = analyze_skill_gap([], target_career="Cloud Engineer")

    assert res["matched_count"] == 0
    assert res["missing_count"] == len(CAREER_REQUIRED_SKILLS["Cloud Engineer"])
    assert res["skill_gap_percentage"] == 100.0
    assert res["competency_match_score"] == 0.0
    assert len(res["improvement_suggestions"]) == res["missing_count"]


def test_analyze_skill_gap_dynamic_target_career():
    # If target_career is omitted, it should be predicted from candidate skills
    cyber_skills = ["Network Security", "Wireshark", "Ethical Hacking", "Nmap", "Firewall"]
    res = analyze_skill_gap(cyber_skills, target_career=None)

    assert res["status"] == "success"
    assert res["target_career"] == "Cybersecurity Analyst"
    assert "Network Security" in res["matched_skills"]


def test_suggestions_database_integrity():
    for skill, data in SKILL_SUGGESTIONS_DB.items():
        assert "priority" in data
        assert "category" in data
        assert "recommended_action" in data
        assert isinstance(data["learning_resources"], list)
        assert len(data["learning_resources"]) > 0
        assert "practical_project" in data
        assert "estimated_time" in data


def test_fallback_suggestion_for_unknown_skill():
    sug = get_suggestion_for_skill("Quantum Computing Algorithms", target_career="Research Scientist")
    assert sug["skill"] == "Quantum Computing Algorithms"
    assert sug["priority"] in ["High", "Medium", "Normal"]
    assert "Quantum Computing Algorithms" in sug["recommended_action"]
    assert len(sug["learning_resources"]) > 0
    assert len(sug["practical_project"]) > 0
