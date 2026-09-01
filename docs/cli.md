# Command-Line Interface (CLI) Guide — CareerCast

The `careercast` CLI provides commands for resume parsing, career prediction, recommendations, skill gap analysis, server management, and multi-format report exports.

---

## 1. Overview & General Syntax

```bash
careercast [-h] [-v] <command> [options]
```

### Global Flags
- `-h`, `--help`: Displays help message and command descriptions.
- `-v`, `--version`: Displays the installed library version (`careercast 1.0.0`).

---

## 2. CLI Subcommands

### 1. `careercast parse`
Extracts text and structured entities (skills, education, roles, experience, certifications, projects) from a PDF or TXT resume file.

```bash
careercast parse -f <file_path> [--json]
```

#### Options:
- `-f`, `--file` (*required*): Path to PDF or TXT resume file.
- `--json` (*optional*): Outputs machine-readable JSON format.

#### Example:
```bash
careercast parse --file sample_resumes/sample_data_scientist_resume.txt
careercast parse --file sample_resumes/sample_data_scientist_resume.txt --json
```

---

### 2. `careercast predict`
Classifies candidate profile suitability and predicts career domain probabilities using trained ML models.

```bash
careercast predict [options]
```

#### Options:
- `-s`, `--skills`: Comma-separated candidate skills.
- `-e`, `--education`: Education degree (default: `B.Tech`).
- `-x`, `--experience`: Years of experience (default: `1.0`).
- `-f`, `--file`: Path to resume file to auto-extract skills & experience.
- `-k`, `--top-k`: Number of top predictions to display (default: `3`).
- `--json`: Format output as JSON.

#### Example:
```bash
careercast predict --skills "Python, Machine Learning, SQL, Pandas" --experience 2.0 --top-k 3
careercast predict --file sample_resumes/sample_web_developer_resume.txt --json
```

---

### 3. `careercast recommend`
Generates multi-modal ranked career recommendations combining ML confidence, Sentence-BERT semantic similarity, and skill alignment.

```bash
careercast recommend [options]
```

#### Options:
- `-s`, `--skills`: Comma-separated candidate skills.
- `-f`, `--file`: Path to resume file.
- `-e`, `--education`: Education degree.
- `-x`, `--experience`: Years of experience.
- `-k`, `--top-k`: Number of recommendations to rank (default: `5`).
- `--json`: Format output as JSON.

#### Example:
```bash
careercast recommend --skills "AWS, Docker, Kubernetes, Linux, Terraform, CI/CD" --top-k 3
```

---

### 4. `careercast gap`
Compares candidate skills against canonical career requirements, identifies matched and missing skills, calculates gap %, and provides actionable learning suggestions.

```bash
careercast gap [options]
```

#### Options:
- `-s`, `--skills`: Comma-separated candidate skills.
- `-c`, `--career`: Target career domain (e.g., `Data Scientist`, `Cloud Engineer`). If omitted, auto-predicted.
- `-f`, `--file`: Path to resume file.
- `--json`: Format output as JSON.

#### Example:
```bash
careercast gap --skills "Python, SQL, Pandas" --career "Data Scientist"
careercast gap --skills "Java, C++, Git" --career "Software Developer" --json
```

---

### 5. `careercast report`
Generates a complete career assessment report in Markdown, JSON, or professional vector PDF format.

```bash
careercast report [options]
```

#### Options:
- `-s`, `--skills`: Comma-separated candidate skills.
- `-f`, `--file`: Path to resume file.
- `-c`, `--career`: Target career domain.
- `--name`: Candidate name.
- `--email`: Candidate email.
- `--format`: Report format: `md` (default), `json`, or `pdf`.
- `-o`, `--output`: Output file destination.

#### Examples:
```bash
# Print formatted Markdown to terminal
careercast report --skills "Python, SQL, Machine Learning" --career "Data Scientist"

# Save JSON report to disk
careercast report --skills "Python, SQL, Machine Learning" --format json -o assessment.json

# Generate professional vector PDF report
careercast report --file sample_resumes/sample_cloud_engineer_resume.txt --format pdf -o cloud_assessment.pdf
```

---

### 6. `careercast serve`
Starts the FastAPI high-performance REST API service with Swagger documentation.

```bash
careercast serve [--host 127.0.0.1] [--port 8000]
```

---

### 7. `careercast ui`
Launches the interactive Streamlit Review UI, Multi-Career Comparison view, and Cohort Analytics dashboard.

```bash
careercast ui [--port 8501]
```
