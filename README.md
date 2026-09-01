# CareerCast — AI-Based Career Prediction, Recommendation & Skill Gap Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![Package](https://img.shields.io/badge/package-careercast%20v1.0.0-green.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-96%20passed-brightgreen.svg)](tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-FF4B4B.svg)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-0194E2.svg)](https://mlflow.org)

---

## 📖 Overview

**CareerCast** is a production-ready, pip-installable AI library, REST service, and web analytics platform for intelligent career counseling. It extracts candidate qualifications and skills from resumes via NLP, classifies career domain suitability using tuned ensemble machine learning models, calculates semantic skill alignment using **Sentence-BERT**, performs **Skill Gap Analysis** with actionable competency improvement suggestions, exposes a high-performance **FastAPI REST Service**, manages models via the **MLflow Model Registry**, verifies model quality through an **Automated CI Accuracy Gate**, and presents interactive insights via a modern **Streamlit UI** and **Flask Web Application**.

---

## 🌟 Key Features

- 📦 **Pip-Installable Library (`careercast`)**: Clean Python API for parsing, prediction, recommendation, skill gap analysis, and PDF report generation.
- 💻 **Feature-Rich CLI (`careercast`)**: Subcommands for `parse`, `predict`, `recommend`, `gap`, `report`, `serve`, and `ui`.
- 📄 **NLP Resume Parser**: Extracts skills, education degrees, previous roles, experience years, certifications, and projects from PDF and TXT resumes via SpaCy EntityRuler.
- 🤖 **Ensemble Machine Learning**: Tuned **Random Forest** and **XGBoost** classifiers with 5-Fold Stratified Cross-Validation over TF-IDF features.
- 🧬 **Sentence-BERT Semantic Alignment**: Dense 384-dimensional embeddings (`all-MiniLM-L6-v2`) measuring semantic similarity between candidate profiles and target roles.
- 🎯 **Skill Gap Analysis & Roadmap**: Exact & semantic matched vs missing skills breakdown with curated, non-random learning actions, project ideas, and timelines.
- ⚖️ **Multi-Career Comparison**: Side-by-side comparative analysis of 2–7 careers showing probability, required skills, and alignment scores.
- 📈 **Cohort Analytics**: Aggregated KPIs, career distributions, skill frequencies, missing skill bottlenecks, and CSV export.
- 📑 **Professional PDF Export**: Multi-page vector PDF reports with running headers, footers, summary badges, tables, and comparison charts via ReportLab.
- 🚀 **FastAPI REST Service**: Interactive OpenAPI/Swagger documentation (`/health`, `/predict`, `/recommend`, `/skill-gap-report`).
- 📊 **MLflow Model Registry**: Automated tracking of parameters, metrics (Accuracy, Precision, Recall, F1), and champion model registration.
- 🛡️ **Automated CI Accuracy Gate**: Continuous verification of model quality on every commit.

---

## 🛠️ Installation & Setup

### 1. Standard Installation
```bash
# Clone repository and navigate to directory
cd career_prediction_system

# Install careercast library and dependencies
pip install .
```

### 2. Editable Development Installation
```bash
pip install -e ".[dev]"
```

### 3. Verify Installation
```bash
python -c "import careercast; print('CareerCast Version:', careercast.__version__)"
careercast --help
```

---

## 🐍 Python API Quickstart

```python
import careercast

# 1. Parse resume from file or raw text
parsed = careercast.parse_resume("sample_resumes/sample_data_scientist_resume.txt")
print("Extracted Skills:", parsed["skills"])

# 2. Predict career category & probability distribution
profile = {
    "education": "B.Tech",
    "skills": parsed["skills"],
    "experience": 2.5
}
prediction = careercast.predict_career(profile, top_k=3)
print(f"Top Prediction: {prediction['prediction']} ({prediction['confidence_percentage']}%)")

# 3. Generate multi-modal Top-K recommendations
recommendations = careercast.recommend_careers(profile, top_k=5)
print(f"Overall Match Score: {recommendations['overall_score']}%")

# 4. Perform skill gap analysis against target career
gap_report = careercast.analyze_skill_gap(profile["skills"], target_career="Data Scientist")
print(f"Competency Match: {gap_report['competency_match_score']}%")
print(f"Missing Skills ({gap_report['missing_count']}):", gap_report["missing_skills"])

# 5. Export professional vector PDF report
careercast.generate_pdf_report(
    profile=profile,
    rec_result=recommendations,
    gap_result=gap_report,
    output_path="assessment_report.pdf"
)
print("PDF report exported successfully!")
```

---

## 💻 Command-Line Interface (CLI)

```bash
# Parse a resume file
careercast parse --file sample_resumes/sample_data_scientist_resume.txt
careercast parse --file sample_resumes/sample_data_scientist_resume.txt --json

# Predict career domain
careercast predict --skills "Python, Machine Learning, SQL, Pandas" --experience 2.0 --top-k 3

# Multi-modal ranked recommendations
careercast recommend --skills "AWS, Docker, Kubernetes, Linux, Terraform, CI/CD" --top-k 3

# Skill gap breakdown & actionable suggestions
careercast gap --skills "Python, SQL" --career "Data Scientist"

# Generate and export PDF assessment report
careercast report --skills "Python, SQL, Machine Learning" --career "Data Scientist" --format pdf --output assessment.pdf

# Start FastAPI REST server
careercast serve --port 8000

# Launch Streamlit UI
careercast ui --port 8501
```

---

## 🌐 Application Interfaces

### 1. Interactive Streamlit Review UI
Launch with `careercast ui --port 8501` or `streamlit run streamlit_app.py`:
- **Candidate Assessment Tab**: Profile presets, resume upload, probability distribution bar chart, skill gap pills, actionable roadmap, and one-click PDF/MD/JSON report export.
- **Multi-Career Comparison Tab**: Side-by-side comparison of 2–7 careers with probabilities, matching/missing skills, and alignment scores.
- **Cohort Analytics Tab**: Cohort KPIs, career distribution, confidence histograms, skill frequency analysis, bottleneck detection, and CSV export.

### 2. FastAPI REST Service
Launch with `careercast serve --port 8000`:
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc UI: `http://127.0.0.1:8000/redoc`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | System status, loaded models, available endpoints |
| `/predict` | `POST` | ML career classification & probability distribution |
| `/recommend` | `POST` | Multi-modal Top-K ranked career recommendations |
| `/skill-gap-report` | `POST` | Skill gap analysis & actionable learning roadmap |

### 3. Flask Web Application
Launch with `python app.py`:
- Web interface hosted at `http://127.0.0.1:5000`.

---

## 🏆 Model Performance & Benchmark Results

### Model Comparison Table (Test Split):

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Status |
|---|---|---|---|---|---|
| **Random Forest (Champion)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **Active Best** |
| **XGBoost Classifier** | 97.6% | 97.7% | 97.6% | 97.6% | Candidate |
| **Logistic Regression (Baseline)** | 100.0% | 100.0% | 100.0% | 100.0% | Baseline |

### Benchmark Evaluation Metrics:
- **Curated LinkedIn Transitions Benchmark**: Top-1 Accuracy: **100.0%**, Top-3: **100.0%**, Top-5: **100.0%**, MRR: **1.0000**, NDCG@5: **1.0000**.
- **SemEval Career Benchmark**: Top-1 Accuracy: **100.0%**, Top-3: **100.0%**, Top-5: **100.0%**, MRR: **1.0000**, NDCG@5: **1.0000**.

---

## 🧪 Comprehensive Testing Suite

All 96 test suites pass with comprehensive statement coverage:

```bash
# Run full test suite
pytest -q

# Run coverage report
pytest --cov=careercast --cov=ml --cov=nlp --cov=skill_gap --cov=api --cov-report=term-missing
```

---

## 📁 Project Structure

```
career_prediction_system/
├── careercast/                       # Unified Python Library Package
│   ├── __init__.py                   # Clean Top-Level Python API
│   ├── cli/                          # Command-Line Interface
│   │   ├── __init__.py
│   │   └── main.py                   # CLI parser and subcommands
│   ├── reports/                      # Multi-format report generators
│   │   ├── __init__.py
│   │   └── generator.py              # Markdown, JSON & ReportLab PDF generator
│   └── analytics/                    # Cohort analytics & comparison
│       ├── __init__.py
│       └── cohort.py                 # Cohort stats, distributions, comparison
│
├── api/                              # FastAPI REST Service
│   ├── __init__.py
│   ├── main.py                       # FastAPI application & OpenAPI setup
│   ├── routes.py                     # /health, /predict, /recommend, /skill-gap-report
│   └── schemas.py                    # Pydantic v2 schemas
│
├── skill_gap/                        # Skill Gap Analysis Module
│   ├── __init__.py
│   ├── analyzer.py                   # Competency comparison engine
│   └── suggestions.py                # Actionable learning suggestions KB
│
├── ml/                               # Machine Learning & MLOps Package
│   ├── config.py                     # Configuration & weights
│   ├── train_model.py                # Baseline Logistic Regression trainer
│   ├── random_forest.py              # Random Forest + GridSearchCV trainer
│   ├── xgboost_model.py              # XGBoost + GridSearchCV trainer
│   ├── model_comparison.py           # Multi-model comparator & selector
│   ├── skill_embeddings.py           # Sentence-BERT semantic encoder
│   ├── recommendation_engine.py      # Top-K ranking & skill alignment
│   ├── predict.py                    # Inference pipeline
│   ├── evaluation.py                 # SemEval & LinkedIn benchmark evaluation
│   ├── mlflow_integration.py         # MLflow tracking & model registry
│   └── ci_accuracy_gate.py           # Automated CI accuracy gate
│
├── nlp/                              # Natural Language Processing Package
│   ├── __init__.py
│   ├── resume_parser.py              # SpaCy NER parser (PDF/TXT)
│   └── preprocessing.py              # Text cleaning & profile serialization
│
├── data/                             # Datasets
│   ├── career_dataset.csv            # 840-sample balanced training dataset
│   ├── generate_dataset.py           # Reproducible generator script
│   ├── linkedin_transitions_sample.csv # LinkedIn transition benchmark
│   └── semeval_career_benchmark_sample.csv # SemEval career benchmark
│
├── models/                           # Trained Serialized Artifacts
│   ├── logistic_regression_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── random_forest/                # Tuned Random Forest model & metrics
│   ├── xgboost/                      # Tuned XGBoost model & metrics
│   └── sentence_transformer/         # Sentence-BERT status metadata
│
├── docs/                             # Public Release Documentation
│   ├── installation.md               # Installation & environment setup
│   ├── quickstart.md                 # Quickstart guide
│   ├── api_reference.md              # Python & REST API reference
│   ├── cli.md                        # CLI documentation
│   ├── architecture.md               # Technical architecture & design
│   ├── testing.md                    # Testing & QA guide
│   ├── streamlit.md                  # Streamlit UI guide
│   ├── dataset_card.md               # Dataset Card
│   └── model_card.md                 # Model Card
│
├── tests/                            # 96-Test Comprehensive Suite
│   ├── test_careercast_package.py    # Top-level library API tests
│   ├── test_cli.py                   # CLI subcommand tests
│   ├── test_pdf_export.py            # PDF report export tests
│   ├── test_cohort_analytics.py      # Cohort analytics tests
│   ├── test_e2e_regression.py        # End-to-end regression tests
│   ├── test_fastapi.py               # FastAPI endpoint tests
│   ├── test_skill_gap.py             # Skill gap engine tests
│   ├── test_milestone2.py            # ML & recommendation tests
│   ├── test_mlflow.py                # MLflow tracking tests
│   ├── test_ci_gate.py               # CI gate tests
│   ├── test_model.py                 # Baseline model tests
│   ├── test_parser.py                # Resume parser tests
│   ├── test_routes.py                # Flask routes tests
│   └── test_validation.py            # Form validation tests
│
├── sample_resumes/                   # Sample Resumes for Evaluation
├── templates/                        # Flask HTML templates
├── static/                           # Web CSS/JS assets
├── app.py                            # Flask application
├── streamlit_app.py                  # Streamlit Review UI & Analytics
├── pyproject.toml                    # PEP 517/621 package metadata
├── setup.py                          # Compatibility setup script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 📚 Documentation Index

- [Installation Guide](docs/installation.md)
- [Quickstart Guide](docs/quickstart.md)
- [API Reference](docs/api_reference.md)
- [CLI Reference](docs/cli.md)
- [System Architecture](docs/architecture.md)
- [Testing & Quality Assurance](docs/testing.md)
- [Streamlit Dashboard Guide](docs/streamlit.md)
- [Dataset Card](docs/dataset_card.md)
- [Model Card](docs/model_card.md)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
