# Installation & Setup Guide — CareerCast

`careercast` is a production-ready Python library and command-line utility for AI-driven career prediction, multi-modal recommendations, and skill gap analysis.

---

## 1. System Requirements

- **Operating System**: Windows 10/11, macOS (12+), or Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+)
- **Python Version**: Python 3.10, 3.11, 3.12, 3.13, or 3.14
- **Hardware**: Minimum 4 GB RAM (8 GB recommended for S-BERT embedding caching)
- **Disk Space**: ~500 MB for virtual environment and model artifacts

---

## 2. Standard Installation

### From Source Repository

Clone the project repository and install in your active Python environment:

```bash
# Navigate to the project directory
cd career_prediction_system

# Install the library and dependencies
pip install .
```

### Editable Development Installation

For active development, testing, and customization:

```bash
pip install -e .
```

### Installing Development & Testing Dependencies

To install optional testing dependencies (`pytest`, `pytest-cov`):

```bash
pip install -e ".[dev]"
```

---

## 3. Verifying Installation

Verify that the Python package and CLI are properly installed:

```bash
# Verify Python library import
python -c "import careercast; print('CareerCast Version:', careercast.__version__)"

# Verify CLI tool
careercast --help
careercast --version
```

---

## 4. NLP Model & SpaCy Setup

CareerCast uses SpaCy for resume entity extraction. The package automatically initializes the entity ruler and tokenization pipelines. If you wish to install the optional pre-trained SpaCy English statistical model:

```bash
python -m spacy download en_core_web_sm
```

*(Note: CareerCast includes a robust built-in rule-based EntityRuler fallback so it functions seamlessly even if `en_core_web_sm` is not downloaded).*

---

## 5. Running the Application Servers

### Running the FastAPI REST Service

```bash
# Via CLI command
careercast serve --port 8000

# Or via Uvicorn directly
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI documentation is accessible at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Running the Streamlit Review UI

```bash
# Via CLI command
careercast ui --port 8501

# Or via Streamlit directly
streamlit run streamlit_app.py
```

Access the dashboard at `http://localhost:8501`.

### Running the Flask Web Application

```bash
python app.py
```

Access the Flask web interface at `http://localhost:5000`.
