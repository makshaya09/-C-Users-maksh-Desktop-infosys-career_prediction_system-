# AI-Based Career Prediction and Recommendation System

## College Major Project — Milestone 1 & Milestone 2 Complete

---

### 📌 Project Title
**AI-Based Career Prediction and Recommendation System**

---

### 📖 Project Overview
This project is an end-to-end AI-powered career counseling platform that extracts candidate qualifications and skills from resumes using NLP, classifies career suitability using tuned ensemble machine learning models, calculates semantic skill alignment using **Sentence-BERT**, and generates ranked **Top-K career recommendations** with matched/missing skill breakdowns and confidence scores.

- **Milestone 1**: SpaCy NER Resume Parsing (Skills, Education, Roles) + User Profile Form Validation + Baseline Logistic Regression Classifier + Evaluation Report & Confusion Matrix.
- **Milestone 2**: Ensemble ML Classifiers (**Random Forest**, **XGBoost**) with **GridSearchCV 5-Fold Cross-Validation** + Dynamic Model Comparison/Selection + **Sentence-BERT Semantic Skill Embeddings** + **Skill Alignment Engine** (Matched $\checkmark$ and Missing $\bullet$ Skills) + **Top-K Recommendation System** + **SemEval & LinkedIn Transition Benchmark Evaluation** (Top-1, Top-3, Top-5, MRR, NDCG@5).

---

### 🛠️ Technology Stack
- **Programming Language**: Python 3.10+ / Python 3.14
- **Web Framework**: Flask
- **NLP & Text Extraction**: SpaCy (`en_core_web_sm`), PyPDF2, Regular Expressions
- **Semantic Embeddings**: Sentence-BERT (`sentence-transformers`, `all-MiniLM-L6-v2`), PyTorch
- **Machine Learning**: Scikit-learn (`RandomForestClassifier`, `LogisticRegression`, `GridSearchCV`, `StratifiedKFold`, `TfidfVectorizer`), XGBoost (`XGBClassifier`)
- **Data Manipulation**: Pandas, NumPy, Scipy
- **Persistence & Serialization**: Joblib, JSON
- **Visualization**: Matplotlib, Seaborn
- **Testing**: Pytest
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5

---

### 📁 Project Structure

```
career_prediction_system/
├── app.py                            # Flask application with web routes and JSON APIs
├── requirements.txt                  # Python dependencies
├── README.md                         # Complete project documentation
│
├── data/
│   ├── career_dataset.csv            # 840-sample career dataset across 7 domains
│   ├── generate_dataset.py           # Reproducible dataset generator
│   ├── linkedin_transitions_sample.csv  # Curated LinkedIn transition dataset
│   └── semeval_career_benchmark_sample.csv # SemEval career benchmark dataset
│
├── models/
│   ├── logistic_regression_model.pkl # Trained baseline model
│   ├── tfidf_vectorizer.pkl          # Fitted TF-IDF feature extractor
│   ├── label_encoder.pkl             # Target label encoder
│   ├── random_forest/                # Tuned Random Forest model & metrics
│   │   ├── random_forest_model.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── label_encoder.pkl
│   │   └── rf_metrics.json
│   ├── xgboost/                      # Tuned XGBoost model & metrics
│   │   ├── xgboost_model.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── label_encoder.pkl
│   │   └── xgb_metrics.json
│   └── sentence_transformer/         # Sentence-BERT embedding metadata
│       └── sbert_status.json
│
├── nlp/
│   ├── __init__.py                   # NLP package initializer
│   ├── resume_parser.py              # SpaCy NER parser (skills, education, roles)
│   └── preprocessing.py              # Text cleaning & profile feature concatenation
│
├── ml/
│   ├── __init__.py                   # ML package initializer
│   ├── config.py                     # Central configuration (weights, paths, Top-K)
│   ├── train_model.py                # Baseline Logistic Regression trainer
│   ├── random_forest.py              # Random Forest + GridSearchCV trainer
│   ├── xgboost_model.py              # XGBoost + GridSearchCV trainer
│   ├── model_comparison.py           # Multi-model comparator & dynamic selector
│   ├── skill_embeddings.py           # Sentence-BERT semantic embedding encoder
│   ├── fine_tune_sbert.py            # Sentence-BERT fine-tuning pipeline interface
│   ├── recommendation_engine.py      # Top-K ranking, skill alignment, matched/missing
│   ├── predict.py                    # Inference pipeline
│   └── evaluation.py                 # SemEval & LinkedIn benchmark evaluation
│
├── results/
│   ├── model_comparison.json         # Comparison table & winning model
│   ├── evaluation_report.json        # Benchmark accuracy, MRR, NDCG@5 metrics
│   └── recommendation_results.json   # Detailed sample evaluation records
│
├── templates/
│   ├── base.html                     # Base layout with navbar, flash messages, footer
│   ├── index.html                    # Landing page with workflow cards
│   ├── upload_resume.html            # Drag-and-drop resume upload (PDF/TXT)
│   ├── profile.html                  # Profile ingestion form with live & server validation
│   ├── result.html                   # Top-K recommendation cards with matched/missing skills
│   └── report.html                   # Model comparison table & benchmark report
│
├── static/
│   ├── css/style.css                 # Custom modern stylesheet
│   └── js/script.js                  # Client validation and drag-and-drop upload
│
├── sample_resumes/                   # Sample resumes for testing
│   ├── sample_data_scientist_resume.txt
│   ├── sample_cloud_engineer_resume.txt
│   └── sample_web_developer_resume.txt
│
└── tests/
    ├── test_parser.py                # Resume parser tests
    ├── test_model.py                 # Baseline model tests
    ├── test_validation.py            # Form validation tests
    ├── test_routes.py                # Flask routes & API tests
    └── test_milestone2.py            # Milestone 2 ML, embeddings, alignment & benchmark tests
```

---

### 🧠 Recommendation Engine & Scoring Mechanism

For any candidate profile, the system computes multi-modal scores across all supported careers:

1. **ML Classification Probability ($P_{\text{ML}}$)**: Generated via `predict_proba()` from the winning model (Random Forest / XGBoost / Logistic Regression).
2. **Skill Alignment ($S_{\text{align}}$)**:
   $$\text{Exact Match \%} = \frac{\text{Matched Skills}}{\text{Total Required Skills}} \times 100$$
   $$\text{Skill Alignment} = (0.60 \times \text{Exact Match \%}) + (0.40 \times \text{Semantic Similarity} \times 100)$$
3. **Sentence-BERT Semantic Similarity ($S_{\text{sem}}$)**: Cosine similarity between dense embeddings of user skills and career job descriptions.
4. **Overall Recommendation Score**:
   $$\text{Overall Score} = (0.40 \times P_{\text{ML}} \times 100) + (0.35 \times S_{\text{align}}) + (0.25 \times S_{\text{sem}} \times 100)$$

All scoring weights and the default Top-K ($K=5$) are configurable in `ml/config.py`.

---

### 📊 Benchmark Evaluation Metrics

The recommendation engine is evaluated on:
- **Curated LinkedIn Career Transition Dataset**
- **SemEval Career Benchmark Dataset**

Metrics computed:
- **Top-1 Accuracy**: Percentage of times ground-truth career is ranked #1.
- **Top-3 Accuracy**: Percentage of times ground-truth career appears in Top 3.
- **Top-5 Accuracy**: Percentage of times ground-truth career appears in Top 5.
- **Mean Reciprocal Rank (MRR)**: $\text{MRR} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}$
- **NDCG@5**: Normalized Discounted Cumulative Gain assessing ranking quality.

---

### 🚀 Setup & Execution Guide (Windows / VS Code)

#### 1. Open Workspace
```powershell
cd c:\Users\maksh\Desktop\infosys\career_prediction_system
```

#### 2. Activate Virtual Environment
```powershell
venv\Scripts\activate
```

#### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

#### 4. Download SpaCy Language Model
```powershell
python -m spacy download en_core_web_sm
```

#### 5. Train All Models & Run Benchmark Evaluations
```powershell
python ml/train_model.py
python ml/random_forest.py
python ml/xgboost_model.py
python ml/model_comparison.py
python ml/evaluation.py
```

#### 6. Run the Flask Web Application
```powershell
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

### 📡 REST API Documentation

#### `POST /api/recommend`
Generates Top-K career recommendations.

**Request**:
```json
{
  "skills": ["Python", "SQL", "Machine Learning", "Pandas"],
  "experience": 2,
  "education": "B.Tech",
  "top_k": 5
}
```

**Response**:
```json
{
  "status": "success",
  "prediction": "Data Scientist",
  "confidence_score": 0.91,
  "skill_alignment": 87.5,
  "semantic_similarity": 0.89,
  "overall_score": 89.2,
  "active_model": "Random Forest",
  "recommendations": [
    {
      "rank": 1,
      "career": "Data Scientist",
      "confidence_score": 0.91,
      "skill_alignment": 87.5,
      "semantic_similarity": 0.89,
      "overall_score": 89.2,
      "matched_skills": ["Python", "SQL", "Machine Learning", "Pandas"],
      "missing_skills": ["TensorFlow", "Statistics", "Deep Learning"]
    }
  ]
}
```

#### `GET /api/model-comparison`
Returns multi-model comparison table (Accuracy, Precision, Recall, F1) and winning model.

---

### 🧪 Running Tests
To run all test suites (Milestone 1 + Milestone 2):
```powershell
pytest tests/ -v
```
