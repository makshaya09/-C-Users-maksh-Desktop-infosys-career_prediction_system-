"""
Unit and Integration Tests for CareerCast Command Line Interface (CLI).
"""

import os
import sys
import tempfile
import pytest
from careercast.cli.main import main, create_parser


def test_cli_parser_creation():
    parser = create_parser()
    assert parser is not None
    assert parser.prog == "careercast"


def test_cli_no_arguments_returns_zero(capsys):
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage: careercast" in captured.out


def test_cli_parse_command(capsys):
    sample_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "sample_resumes",
        "sample_data_scientist_resume.txt"
    )
    exit_code = main(["parse", "--file", sample_file])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "RESUME PARSING RESULT" in captured.out
    assert "Python" in captured.out


def test_cli_parse_json_format(capsys):
    sample_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "sample_resumes",
        "sample_cloud_engineer_resume.txt"
    )
    exit_code = main(["parse", "--file", sample_file, "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"skills"' in captured.out
    assert '"AWS"' in captured.out or '"aws"' in captured.out.lower()


def test_cli_parse_nonexistent_file(capsys):
    exit_code = main(["parse", "--file", "non_existent_file.pdf"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err.lower()


def test_cli_predict_command(capsys):
    exit_code = main([
        "predict",
        "--skills", "Python, Machine Learning, SQL, Pandas",
        "--education", "B.Tech",
        "--experience", "2.0",
        "--top-k", "3"
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "CAREER PREDICTION RESULT" in captured.out
    assert "Data Scientist" in captured.out or "Machine Learning" in captured.out


def test_cli_predict_json_format(capsys):
    exit_code = main([
        "predict",
        "--skills", "Java, C++, Data Structures, Algorithms",
        "--json"
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"prediction"' in captured.out
    assert '"recommendations"' in captured.out


def test_cli_predict_missing_skills_fails(capsys):
    exit_code = main(["predict", "--skills", ""])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "validation error" in captured.err.lower()


def test_cli_recommend_command(capsys):
    exit_code = main([
        "recommend",
        "--skills", "HTML, CSS, JavaScript, React, Node.js",
        "--top-k", "3"
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "MULTI-MODAL CAREER RECOMMENDATIONS" in captured.out
    assert "Web Developer" in captured.out


def test_cli_gap_command(capsys):
    exit_code = main([
        "gap",
        "--skills", "Python, SQL",
        "--career", "Data Scientist"
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "SKILL GAP ANALYSIS" in captured.out
    assert "Competency Match Score" in captured.out
    assert "Missing Skills" in captured.out


def test_cli_gap_json_format(capsys):
    exit_code = main([
        "gap",
        "--skills", "AWS, Docker",
        "--career", "Cloud Engineer",
        "--json"
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"matched_skills"' in captured.out
    assert '"missing_skills"' in captured.out


def test_cli_report_markdown(capsys):
    exit_code = main([
        "report",
        "--skills", "Python, SQL, Machine Learning",
        "--career", "Data Scientist",
        "--format", "md"
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "# AI Career Assessment & Skill Gap Analysis Report" in captured.out


def test_cli_report_json(capsys):
    exit_code = main([
        "report",
        "--skills", "Python, SQL, Machine Learning",
        "--career", "Data Scientist",
        "--format", "json"
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"report_metadata"' in captured.out


def test_cli_report_pdf_file_output(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_out = os.path.join(temp_dir, "test_cli_report.pdf")
        exit_code = main([
            "report",
            "--skills", "Python, SQL, Machine Learning",
            "--career", "Data Scientist",
            "--format", "pdf",
            "--output", pdf_out
        ])
        assert exit_code == 0
        assert os.path.exists(pdf_out)
        assert os.path.getsize(pdf_out) > 1000
