# Quickstart Guide — CareerCast

Get started with **CareerCast** in less than 5 minutes.

---

## 1. Python API Quickstart

```python
import careercast

# 1. Parse a resume (PDF or raw text)
resume_text = """
Arjun Sharma
Email: arjun@example.com
Education: B.Tech in Computer Science
Experience: 3 years
Skills: Python, Machine Learning, SQL, Deep Learning, Pandas, Scikit-learn, TensorFlow
"""
parsed = careercast.parse_resume(resume_text)
print("Parsed Skills:", parsed["skills"])

# 2. Predict career category & probability distribution
profile = {
    "education": "B.Tech",
    "skills": parsed["skills"],
    "experience": 3.0
}
prediction = careercast.predict_career(profile, top_k=3)
print(f"Predicted Career: {prediction['prediction']} ({prediction['confidence_percentage']}%)")

# 3. Generate multi-modal Top-K recommendations
recommendations = careercast.recommend_careers(profile, top_k=5)
print(f"Top Recommendation: {recommendations['prediction']} (Score: {recommendations['overall_score']}%)")

# 4. Perform skill gap analysis
gap_report = careercast.analyze_skill_gap(
    candidate_skills=profile["skills"],
    target_career="Data Scientist"
)
print(f"Competency Match: {gap_report['competency_match_score']}%")
print(f"Matched Skills: {gap_report['matched_skills']}")
print(f"Missing Skills: {gap_report['missing_skills']}")

# 5. Generate and export assessment reports
pdf_bytes = careercast.generate_pdf_report(
    profile=profile,
    rec_result=recommendations,
    gap_result=gap_report,
    output_path="my_assessment.pdf"
)
print("Saved assessment report to 'my_assessment.pdf'")
```

---

## 2. Command-Line Interface (CLI) Quickstart

### Parse a Resume File
```bash
careercast parse --file sample_resumes/sample_data_scientist_resume.txt
careercast parse --file sample_resumes/sample_data_scientist_resume.txt --json
```

### Predict Career Domain
```bash
careercast predict --skills "Python, Machine Learning, SQL, Pandas" --experience 2.0 --top-k 3
```

### Get Ranked Recommendations
```bash
careercast recommend --skills "AWS, Docker, Kubernetes, Linux, Terraform, CI/CD" --top-k 3
```

### Analyze Skill Gaps & Actionable Roadmap
```bash
careercast gap --skills "Python, SQL" --career "Data Scientist"
```

### Export PDF Report
```bash
careercast report --skills "Python, SQL, Machine Learning" --career "Data Scientist" --format pdf --output assessment.pdf
```

---

## 3. REST API Quickstart

Start the API server:
```bash
careercast serve --port 8000
```

### POST `/predict`
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"skills": "Python, Machine Learning, SQL", "education": "B.Tech", "experience": 2.0, "top_k": 3}'
```

### POST `/recommend`
```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"skills": "AWS, Docker, Kubernetes, Linux", "education": "B.Tech", "experience": 2.0, "top_k": 5}'
```

### POST `/skill-gap-report`
```bash
curl -X POST http://127.0.0.1:8000/skill-gap-report \
  -H "Content-Type: application/json" \
  -d '{"skills": "Python, SQL", "target_career": "Data Scientist"}'
```

---

## 4. Interactive Streamlit Dashboard

Launch the UI:
```bash
careercast ui --port 8501
```

Explore:
1. **Candidate Assessment**: Select candidate presets or upload resumes to review ML probability distributions, skill gap pills, and download PDF reports.
2. **Multi-Career Comparison**: Compare probabilities, alignments, matching vs missing skills across 2-7 career domains side-by-side.
3. **Cohort Analytics**: Review aggregate candidate distributions, skill frequency rankings, bottleneck analysis, and export cohort CSV records.
