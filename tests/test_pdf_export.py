"""
Unit and Integration Tests for PDF Report Export Module.
"""

import os
import tempfile
import pytest
from careercast.reports.generator import (
    generate_pdf_report,
    generate_markdown_report,
    generate_json_report,
    REPORTLAB_AVAILABLE
)
from careercast.analytics.cohort import compute_career_comparison
from ml.recommendation_engine import recommend_careers
from skill_gap.analyzer import analyze_skill_gap


@pytest.fixture
def sample_assessment_data():
    profile = {
        "name": "Sarah Connor",
        "email": "sarah.connor@example.com",
        "education": "B.Tech in CS",
        "skills": "Network Security, Wireshark, Linux, Ethical Hacking, Firewall",
        "experience": 3.0,
        "certifications": "CompTIA Security+",
        "projects": "SOC SIEM Intrusion Detection"
    }
    rec_res = recommend_careers(profile, top_k=3)
    gap_res = analyze_skill_gap(profile["skills"], target_career="Cybersecurity Analyst", profile=profile)
    comp_res = compute_career_comparison(profile["skills"], profile=profile)
    return profile, rec_res, gap_res, comp_res


def test_pdf_report_generation_returns_bytes(sample_assessment_data):
    if not REPORTLAB_AVAILABLE:
        pytest.skip("ReportLab not installed")

    profile, rec_res, gap_res, comp_res = sample_assessment_data
    pdf_bytes = generate_pdf_report(
        profile=profile,
        rec_result=rec_res,
        gap_result=gap_res,
        comparison_results=comp_res
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF")


def test_pdf_report_saves_to_file(sample_assessment_data):
    if not REPORTLAB_AVAILABLE:
        pytest.skip("ReportLab not installed")

    profile, rec_res, gap_res, comp_res = sample_assessment_data
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = os.path.join(tmp_dir, "assessment_report.pdf")
        generate_pdf_report(
            profile=profile,
            rec_result=rec_res,
            gap_result=gap_res,
            output_path=pdf_path,
            comparison_results=comp_res
        )
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 2000


def test_pdf_report_with_no_comparison(sample_assessment_data):
    if not REPORTLAB_AVAILABLE:
        pytest.skip("ReportLab not installed")

    profile, rec_res, gap_res, _ = sample_assessment_data
    pdf_bytes = generate_pdf_report(
        profile=profile,
        rec_result=rec_res,
        gap_result=gap_res,
        comparison_results=None
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_pdf_report_with_minimal_profile():
    if not REPORTLAB_AVAILABLE:
        pytest.skip("ReportLab not installed")

    profile = {"skills": "Python, SQL"}
    rec_res = recommend_careers(profile, top_k=2)
    gap_res = analyze_skill_gap(profile["skills"], target_career="Data Scientist", profile=profile)

    pdf_bytes = generate_pdf_report(
        profile=profile,
        rec_result=rec_res,
        gap_result=gap_res
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
