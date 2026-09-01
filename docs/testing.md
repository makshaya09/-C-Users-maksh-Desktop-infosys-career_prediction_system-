# Testing & Quality Assurance Guide — CareerCast

CareerCast incorporates a comprehensive, multi-layer testing strategy spanning unit tests, integration tests, API schema validations, CLI end-to-end tests, regression pipelines, and CI accuracy gates.

---

## 1. Test Suite Overview

| Test Module | Suite Description | Coverage Focus |
|---|---|---|
| `tests/test_careercast_package.py` | Top-level Python API & packaging | Import integrity, public API contracts, validation, predictions |
| `tests/test_cli.py` | Command-Line Interface | CLI subcommands (`parse`, `predict`, `recommend`, `gap`, `report`), `--json`, error handling |
| `tests/test_pdf_export.py` | PDF generation & export | Vector PDF structure, multi-career table rendering, byte stream integrity |
| `tests/test_cohort_analytics.py` | Cohort calculations & comparisons | Aggregate metrics, distributions, empty dataset handling, career comparison |
| `tests/test_e2e_regression.py` | Full end-to-end flow | Ingestion → Parsing → Prediction → Recommendation → Skill Gap → PDF Export |
| `tests/test_fastapi.py` | FastAPI REST endpoints | `/health`, `/predict`, `/recommend`, `/skill-gap-report`, Pydantic 422 validations |
| `tests/test_skill_gap.py` | Skill gap analysis & KB | Matched/missing skill computation, suggestions database lookup |
| `tests/test_milestone2.py` | ML models & recommendation engine | Sentence-BERT embeddings, cosine similarity, Top-K ranking, SemEval/LinkedIn benchmarks |
| `tests/test_mlflow.py` | MLOps & Experiment Tracking | MLflow SQLite tracking, run logging, model registry |
| `tests/test_ci_gate.py` | CI Accuracy Gate | Accuracy threshold verification script |
| `tests/test_model.py` | Baseline Logistic Regression model | Model artifact loading, probability calculation |
| `tests/test_parser.py` | SpaCy resume parser | PDF/TXT extraction, skill extraction, regex experience |
| `tests/test_routes.py` | Flask web routes | Upload resume, profile form submission, report download |
| `tests/test_validation.py` | Form validation rules | Email regex, required fields, numeric experience |

---

## 2. Running Tests

### Run Full Test Suite
```bash
pytest -q
```

### Run with Coverage Reporting
```bash
pytest --cov=careercast --cov=ml --cov=nlp --cov=skill_gap --cov=api --cov-report=term-missing
```

### Run Specific Test Suites
```bash
# Test CLI only
pytest tests/test_cli.py -v

# Test PDF report export
pytest tests/test_pdf_export.py -v

# Test FastAPI endpoints
pytest tests/test_fastapi.py -v

# Test Top-level CareerCast package
pytest tests/test_careercast_package.py -v
```

---

## 3. Automated CI Accuracy Gate

In addition to pytest unit tests, CareerCast includes an automated CI accuracy gate script (`ml/ci_accuracy_gate.py`) triggered during CI runs:

```bash
python ml/ci_accuracy_gate.py
```

- Verifies that the winning model achieves an accuracy exceeding the defined `ACCURACY_THRESHOLD` (e.g., 0.75).
- Exits with returncode `0` if passed, or `1` if accuracy regresses.
