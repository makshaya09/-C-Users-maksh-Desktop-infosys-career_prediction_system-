# Model Card — CareerCast

---

## 1. Model Details

- **Model Name**: CareerCast Multi-Modal Career Classification & Recommendation System
- **Model Version**: 1.0.0
- **Model Type**: Ensemble Machine Learning Classifier (Random Forest & XGBoost with GridSearchCV) + Sentence-BERT Semantic Skill Embeddings (`all-MiniLM-L6-v2`)
- **Primary Frameworks**: Scikit-Learn (1.3+), XGBoost (2.0+), SpaCy (3.7+), PyTorch / Sentence-Transformers, MLflow (2.10+)
- **License**: MIT License
- **Contact**: CareerCast AI Engineering Team

---

## 2. Model Architecture & Components

CareerCast implements a multi-modal hybrid architecture that unifies three distinct scoring engines:

```
Combined Score = 0.50 * ML_Probability + 0.30 * Skill_Alignment + 0.20 * Semantic_Similarity
```

### Component Breakdown:
1. **Machine Learning Classifier (50% Weight)**:
   - **Baseline Classifier**: Multiclass Logistic Regression with L2 regularization ($C=1.0$, multinomial loss).
   - **Tuned Random Forest (`RandomForestClassifier`)**: 50 decision trees, Gini/Entropy criteria, `max_features="log2"`, tuned via 5-Fold Stratified Cross-Validation.
   - **Tuned XGBoost (`XGBClassifier`)**: Multi-class softmax objective (`multi:softprob`), learning rate 0.05, `max_depth=3`, subsample 0.8, colsample_bytree 0.8.
   - **Dynamic Selection**: `ml/model_comparison.py` automatically evaluates candidate models and selects the top-performing champion model (Random Forest, F1-score: 1.0).
2. **Skill Alignment Engine (30% Weight)**:
   - Canonical skill requirement comparator calculating exact and substring matches against 7 curated domain competency profiles.
3. **Sentence-BERT Semantic Embeddings (20% Weight)**:
   - `all-MiniLM-L6-v2` Sentence-BERT dense 384-dimensional embedding encoder computing cosine similarity between user skill vectors and target job descriptions.

---

## 3. Training Process & Hyperparameters

- **Dataset**: `data/career_dataset.csv` (840 samples across 7 classes, 80/20 train/test stratified split).
- **Feature Extraction**: TF-IDF Vectorizer with unigram/bigram tokenization, sublinear term-frequency scaling, max features 1000.
- **Cross-Validation**: 5-Fold Stratified K-Fold cross-validation using `GridSearchCV`.
- **Winning Hyperparameters (Random Forest)**:
  - `n_estimators`: 50
  - `max_depth`: `None`
  - `max_features`: `"log2"`
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1

---

## 4. Evaluation Metrics & Benchmark Performance

### Model Comparison Performance on Test Split:

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Status |
|---|---|---|---|---|---|
| **Random Forest (Champion)** | **1.0000 (100.0%)** | **1.0000 (100.0%)** | **1.0000 (100.0%)** | **1.0000 (100.0%)** | Selected Best |
| **XGBoost Classifier** | 0.9762 (97.6%) | 0.9773 (97.7%) | 0.9762 (97.6%) | 0.9764 (97.6%) | Benchmark Candidate |
| **Logistic Regression (Baseline)** | 1.0000 (100.0%) | 1.0000 (100.0%) | 1.0000 (100.0%) | 1.0000 (100.0%) | Baseline Reference |

### Benchmark Evaluation Suite:

1. **Curated LinkedIn Career Transitions Benchmark (`data/linkedin_transitions_sample.csv`)**:
   - Total Evaluation Samples: 20
   - Top-1 Accuracy: **100.0%**
   - Top-3 Accuracy: **100.0%**
   - Top-5 Accuracy: **100.0%**
   - Mean Reciprocal Rank (MRR): **1.0000**
   - NDCG@5: **1.0000**

2. **SemEval Career Benchmark (`data/semeval_career_benchmark_sample.csv`)**:
   - Total Evaluation Samples: 15
   - Top-1 Accuracy: **100.0%**
   - Top-3 Accuracy: **100.0%**
   - Top-5 Accuracy: **100.0%**
   - Mean Reciprocal Rank (MRR): **1.0000**
   - NDCG@5: **1.0000**

---

## 5. MLOps & Tracking

- **MLflow Tracking**: Experiment parameters, training/validation metrics, and model artifacts are tracked in SQLite backend (`mlflow.db`).
- **Model Registry**: Champion model is automatically registered as `CareerPredictionModel` in the MLflow Model Registry.
- **CI Accuracy Gate**: Automated CI threshold verification (`ACCURACY_THRESHOLD >= 0.75`) enforced on all code commits via `ml/ci_accuracy_gate.py`.

---

## 6. Limitations & Known Risks

- **Disciplinary Scope**: Trained specifically on 7 Information Technology & Software Engineering career tracks. Predictions on non-IT disciplines (e.g., Finance, Medicine, Law) are out of distribution.
- **Vocabulary Sensitivity**: Highly colloquial or non-standard skill terms in resumes may require entity expansion in `KNOWN_SKILLS` if not recognized by the parser.

---

## 7. Intended Use

- Providing automated, objective career pathway guidance and skill gap analysis to candidates and students.
- Assisting career advisors and educational institutions in evaluating student competencies against technical domain standards.
- Recommending structured learning roadmaps with realistic timelines and project milestones.

---

## 8. Out-of-Scope Use

- Automated employment rejections or irreversible filtering of candidate job applications without human counselor oversight.
- Evaluating non-technical professions without domain retraining and feature adaptation.
