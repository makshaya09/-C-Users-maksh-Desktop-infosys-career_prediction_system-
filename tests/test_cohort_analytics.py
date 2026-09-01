"""
Unit and Integration Tests for Cohort Analytics & Multi-Career Comparison Module.
"""

import pytest
from careercast.analytics.cohort import (
    compute_cohort_analytics,
    compute_career_comparison,
    get_sample_cohort
)


def test_get_sample_cohort():
    sample = get_sample_cohort()
    assert isinstance(sample, list)
    assert len(sample) >= 4
    for s in sample:
        assert "name" in s
        assert "skills" in s
        assert "prediction" in s


def test_compute_cohort_analytics_non_empty():
    sample = get_sample_cohort()
    analytics = compute_cohort_analytics(sample)

    assert analytics["total_candidates"] == len(sample)
    assert analytics["average_experience"] > 0
    assert analytics["average_confidence"] > 0
    assert analytics["average_competency_score"] > 0
    assert len(analytics["career_distribution"]) > 0
    assert len(analytics["top_candidate_skills"]) > 0
    assert len(analytics["confidence_bins"]) == 4
    assert len(analytics["records_summary"]) == len(sample)


def test_compute_cohort_analytics_empty_dataset():
    analytics = compute_cohort_analytics([])

    assert analytics["total_candidates"] == 0
    assert analytics["average_experience"] == 0.0
    assert analytics["average_confidence"] == 0.0
    assert analytics["average_competency_score"] == 0.0
    assert analytics["career_distribution"] == {}
    assert analytics["top_candidate_skills"] == {}
    assert analytics["top_missing_skills"] == {}
    assert analytics["records_summary"] == []


def test_compute_cohort_analytics_single_record():
    single = [{
        "name": "Solo Dev",
        "education": "B.Tech",
        "experience": 4.0,
        "skills": "Java, C++, Git",
        "prediction": "Software Developer",
        "confidence_percentage": 88.0,
        "competency_match_score": 50.0,
        "missing_skills": ["Linux", "Docker"]
    }]
    analytics = compute_cohort_analytics(single)
    assert analytics["total_candidates"] == 1
    assert analytics["average_experience"] == 4.0
    assert analytics["average_confidence"] == 88.0
    assert analytics["career_distribution"]["Software Developer"] == 1


def test_compute_career_comparison_multiple_careers():
    skills = "Python, SQL, Machine Learning, Deep Learning, Pandas"
    careers = ["Data Scientist", "Data Analyst", "Web Developer", "Cloud Engineer"]
    comparisons = compute_career_comparison(skills, target_careers=careers)

    assert len(comparisons) == 4
    # Data Scientist should have higher overall score than Web Developer for ML profile
    ds_comp = next(c for c in comparisons if c["career"] == "Data Scientist")
    web_comp = next(c for c in comparisons if c["career"] == "Web Developer")

    assert ds_comp["overall_score"] > web_comp["overall_score"]
    assert "Python" in ds_comp["matched_skills"]
    assert "Machine Learning" in ds_comp["matched_skills"]
    assert ds_comp["rank"] == 1


def test_compute_career_comparison_all_careers_default():
    skills = "AWS, Docker, Kubernetes, Linux, Terraform, CI/CD"
    comparisons = compute_career_comparison(skills)

    assert len(comparisons) == 7
    top_c = comparisons[0]
    assert top_c["career"] == "Cloud Engineer"
    assert top_c["rank"] == 1
