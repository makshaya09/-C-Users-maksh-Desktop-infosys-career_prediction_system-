"""
Unit and Integration Tests for Milestone 2: Advanced ML & Recommendation Engine.
"""

import os
import sys
import pytest
import numpy as np

# Ensure parent directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml.skill_embeddings import (
    encode_skills,
    encode_job_description,
    calculate_skill_similarity,
    get_career_semantic_similarity,
    get_all_career_similarities
)
from ml.recommendation_engine import (
    calculate_skill_alignment,
    recommend_careers,
    parse_user_skills
)
from ml.evaluation import (
    calculate_mrr,
    calculate_ndcg_at_k,
    evaluate_linkedin_transitions,
    evaluate_semeval_benchmark
)
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# 1. Skill Embeddings & Semantic Similarity Tests
def test_encode_skills_and_similarity():
    user_skills = ["Python", "Machine Learning", "SQL", "Pandas"]
    emb = encode_skills(user_skills)
    assert emb is not None
    assert emb.shape[0] == 1
    assert emb.shape[1] > 0

    desc_emb = encode_job_description("Machine learning data analysis in Python")
    sim = calculate_skill_similarity(emb, desc_emb)
    assert 0.0 <= sim <= 1.0
    assert sim > 0.3  # Should have strong semantic overlap


def test_encode_empty_skills_returns_zero_vector():
    emb = encode_skills("")
    assert emb is not None
    assert np.all(emb == 0)


def test_get_all_career_similarities():
    skills = "AWS, Docker, Kubernetes, Terraform, Linux"
    sims = get_all_career_similarities(skills)
    assert isinstance(sims, dict)
    assert "Cloud Engineer" in sims
    assert "Data Scientist" in sims
    # Cloud Engineer similarity should be higher than unrelated careers
    assert sims["Cloud Engineer"] >= sims.get("Web Developer", 0.0)


# 2. Skill Alignment Tests
def test_calculate_skill_alignment_exact_and_missing():
    user_skills = ["Python", "SQL", "Machine Learning"]
    align = calculate_skill_alignment(user_skills, "Data Scientist")

    assert align["career"] == "Data Scientist"
    assert "Python" in align["matched_skills"]
    assert "SQL" in align["matched_skills"]
    assert "Machine Learning" in align["matched_skills"]
    assert align["matched_count"] >= 3
    assert len(align["missing_skills"]) > 0
    assert align["exact_match_percentage"] > 0
    assert 0 <= align["alignment_score"] <= 100


def test_parse_user_skills():
    raw_str = "Python, SQL, , Machine Learning, python, AWS "
    parsed = parse_user_skills(raw_str)
    assert len(parsed) == 4  # Deduplicated case-insensitively
    assert "Python" in parsed
    assert "SQL" in parsed
    assert "AWS" in parsed


# 3. Top-K Recommendation Engine Tests
def test_recommend_careers_top_5():
    profile = {
        "education": "B.Tech",
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Express"],
        "experience": 2,
        "certifications": "Meta Front-End Developer",
        "projects": "Social Media Web App"
    }
    result = recommend_careers(profile, top_k=5)

    assert "prediction" in result
    assert "overall_score" in result
    assert "active_model" in result
    assert len(result["recommendations"]) == 5

    # Check that recommendations are properly sorted by overall_score descending
    scores = [r["overall_score"] for r in result["recommendations"]]
    assert scores == sorted(scores, reverse=True)

    # Top recommendation should be Web Developer
    top_rec = result["recommendations"][0]
    assert top_rec["career"] == "Web Developer"
    assert top_rec["rank"] == 1
    assert "matched_skills" in top_rec
    assert "missing_skills" in top_rec
    assert "React" in top_rec["matched_skills"]


def test_recommend_careers_cybersecurity():
    profile = {
        "education": "B.Tech",
        "skills": ["Network Security", "Wireshark", "Ethical Hacking", "Nmap", "Firewall"],
        "experience": 3,
        "certifications": "CompTIA Security+",
        "projects": "Vulnerability Assessment"
    }
    result = recommend_careers(profile, top_k=5)
    assert result["prediction"] == "Cybersecurity Analyst"
    assert result["recommendations"][0]["rank"] == 1


# 4. Recommendation API Tests
def test_api_recommend_success(client):
    payload = {
        "name": "Alex Mercer",
        "email": "alex@example.com",
        "education": "B.Tech",
        "skills": ["Python", "Machine Learning", "Pandas", "Scikit-learn", "SQL"],
        "experience": 2,
        "top_k": 5
    }
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "success"
    assert data["prediction"] in ["Data Scientist", "Machine Learning Engineer"]
    assert len(data["recommendations"]) == 5
    first = data["recommendations"][0]
    assert "rank" in first
    assert "confidence_score" in first
    assert "skill_alignment" in first
    assert "semantic_similarity" in first
    assert "matched_skills" in first
    assert "missing_skills" in first


def test_api_recommend_empty_skills_fails(client):
    payload = {
        "name": "Alex",
        "email": "alex@example.com",
        "education": "B.Tech",
        "skills": "",
        "experience": 2
    }
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 422
    data = response.get_json()
    assert "error" in data


# 5. Ranking Metrics Calculation Tests (MRR & NDCG@K)
def test_calculate_mrr():
    # Sample 1: rank 1, Sample 2: rank 2, Sample 3: rank 4
    rankings = [1, 2, 4]
    # Expected MRR: (1/1 + 1/2 + 1/4) / 3 = (1 + 0.5 + 0.25) / 3 = 1.75 / 3 = 0.5833
    mrr = calculate_mrr(rankings)
    assert round(mrr, 4) == 0.5833


def test_calculate_ndcg_at_k():
    rankings = [1, 2, 0]
    ndcg = calculate_ndcg_at_k(rankings, k=5)
    assert 0.0 <= ndcg <= 1.0


# 6. Benchmark Evaluation Tests
def test_evaluate_linkedin_transitions():
    res = evaluate_linkedin_transitions()
    assert res["status"] == "evaluated"
    assert res["total_samples"] > 0
    assert res["top_1_accuracy"] >= 70.0
    assert res["top_3_accuracy"] >= 90.0
    assert res["mrr"] > 0.7


def test_evaluate_semeval_benchmark():
    res = evaluate_semeval_benchmark()
    assert res["status"] == "evaluated"
    assert res["total_samples"] > 0
    assert res["top_1_accuracy"] >= 70.0
    assert res["top_3_accuracy"] >= 90.0
    assert res["mrr"] > 0.7
