# Python & REST API Reference — CareerCast

Complete reference documentation for the CareerCast Python API, data models, schemas, and REST endpoints.

---

## 1. Top-Level Python API (`careercast`)

### `careercast.parse_resume(file_input, filename="") -> dict`
Extracts structured candidate data from a resume file path, raw string, bytes, or file stream.
- **Parameters**:
  - `file_input` (*str | bytes | BytesIO*): File path, raw resume string, or file stream.
  - `filename` (*str, optional*): Name of file with extension (`.pdf` or `.txt`).
- **Returns**: Dictionary with `skills`, `education`, `primary_education`, `roles`, `certifications`, `experience`, `projects`, `raw_text_preview`.

### `careercast.validate_profile(profile_dict) -> tuple[bool, list[str]]`
Validates user profile input fields.
- **Returns**: `(is_valid: bool, errors: list[str])`.

### `careercast.predict_career(profile, top_k=3, models_dir=None) -> dict`
Generates career domain classification probabilities using trained ML models.
- **Parameters**:
  - `profile` (*dict*): Dictionary containing `education`, `skills`, `experience`, `certifications`, `projects`.
  - `top_k` (*int*): Number of top predictions to return (default 3).
- **Returns**:
  ```python
  {
      "prediction": "Data Scientist",
      "confidence": 0.84,
      "confidence_percentage": 84.0,
      "recommendations": [
          {"career": "Data Scientist", "probability": 0.84, "percentage": 84.0},
          ...
      ],
      "feature_text": "Education: B.Tech | Skills: ..."
  }
  ```

### `careercast.recommend_careers(profile, top_k=5) -> dict`
Multi-modal recommendation engine combining ML probability (50%), Skill Alignment (30%), and Sentence-BERT semantic similarity (20%).
- **Returns**:
  ```python
  {
      "prediction": "Data Scientist",
      "confidence_score": 0.84,
      "confidence_percentage": 84.0,
      "skill_alignment": 63.6,
      "semantic_similarity": 0.82,
      "overall_score": 75.4,
      "active_model": "Random Forest",
      "recommendations": [
          {
              "rank": 1,
              "career": "Data Scientist",
              "overall_score": 75.4,
              "confidence_percentage": 84.0,
              "skill_alignment": 63.6,
              "semantic_percentage": 82.0,
              "matched_count": 7,
              "total_required_count": 11
          }
      ],
      "user_skills": ["Python", "SQL", "Machine Learning", ...]
  }
  ```

### `careercast.analyze_skill_gap(candidate_skills, target_career=None, profile=None) -> dict`
Performs skill gap comparison against canonical career requirements and generates actionable suggestions.
- **Returns**:
  ```python
  {
      "status": "success",
      "target_career": "Data Scientist",
      "candidate_skills": ["Python", "SQL"],
      "required_skills": ["Python", "Machine Learning", "SQL", ...],
      "matched_skills": ["Python", "SQL"],
      "missing_skills": ["Machine Learning", "Pandas", ...],
      "matched_count": 2,
      "missing_count": 9,
      "total_required_count": 11,
      "skill_gap_percentage": 81.8,
      "competency_match_score": 18.2,
      "improvement_suggestions": [
          {
              "skill": "Machine Learning",
              "priority": "High",
              "category": "Core Machine Learning",
              "recommended_action": "...",
              "learning_resources": [...],
              "practical_project": "...",
              "estimated_time": "4 - 6 weeks"
          }
      ]
  }
  ```

### `careercast.generate_markdown_report(...) -> str`
### `careercast.generate_json_report(...) -> str`
### `careercast.generate_pdf_report(profile, rec_result, gap_result, output_path=None, comparison_results=None) -> bytes`
Generates comprehensive career assessment reports in Markdown, JSON, and professional vector PDF format.

### `careercast.compute_cohort_analytics(records) -> dict`
### `careercast.compute_career_comparison(candidate_skills, target_careers=None, profile=None) -> list[dict]`
Computes cohort analytics and multi-career comparison evaluations.

---

## 2. FastAPI REST Endpoints

Base URL: `http://127.0.0.1:8000`

### GET `/health`
Health check and system status.
- **Response**:
  ```json
  {
    "status": "healthy",
    "service": "Career Prediction & Skill Gap REST API",
    "version": "1.0.0",
    "models_loaded": true,
    "active_model": "Random Forest",
    "endpoints": ["GET /health", "POST /predict", "POST /recommend", "POST /skill-gap-report"]
  }
  ```

### POST `/predict`
Predicts career domain and top probabilities.
- **Request Body**:
  ```json
  {
    "education": "B.Tech",
    "skills": "Python, SQL, Machine Learning",
    "experience": 2.0,
    "top_k": 3
  }
  ```
- **Response**: `200 OK` (PredictionResponse)

### POST `/recommend`
Multi-modal ranked recommendations combining ML confidence, S-BERT embeddings, and skill alignment.
- **Request Body**:
  ```json
  {
    "education": "B.Tech",
    "skills": "AWS, Docker, Kubernetes, Linux, Terraform",
    "experience": 2.5,
    "top_k": 5
  }
  ```
- **Response**: `200 OK` (RecommendationResponse)

### POST `/skill-gap-report`
Detailed skill gap breakdown with actionable improvement roadmap.
- **Request Body**:
  ```json
  {
    "skills": "Python, SQL",
    "target_career": "Data Scientist",
    "education": "B.Tech",
    "experience": 2.0
  }
  ```
- **Response**: `200 OK` (SkillGapResponse)
