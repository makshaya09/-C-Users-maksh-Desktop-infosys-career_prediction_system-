"""
Configuration settings for Milestone 2: Advanced ML & Recommendation Engine.
Centralizes model paths, Top-K settings, recommendation scoring weights, and hyperparameter grids.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Sub-model directories
RF_MODEL_DIR = os.path.join(MODELS_DIR, "random_forest")
XGB_MODEL_DIR = os.path.join(MODELS_DIR, "xgboost")
SBERT_MODEL_DIR = os.path.join(MODELS_DIR, "sentence_transformer")

os.makedirs(RF_MODEL_DIR, exist_ok=True)
os.makedirs(XGB_MODEL_DIR, exist_ok=True)
os.makedirs(SBERT_MODEL_DIR, exist_ok=True)

# Recommendation Engine Configuration
DEFAULT_TOP_K = 5

# Recommendation Scoring Weights (must sum to 1.0)
# Overall Score = (ML_PROB_WEIGHT * ML_prob) + (SKILL_ALIGNMENT_WEIGHT * Skill_Align) + (SEMANTIC_SIMILARITY_WEIGHT * Semantic_Sim)
ML_PROB_WEIGHT = 0.40
SKILL_ALIGNMENT_WEIGHT = 0.35
SEMANTIC_SIMILARITY_WEIGHT = 0.25

# Skill Alignment internal weights: exact match vs semantic similarity
EXACT_MATCH_WEIGHT = 0.60
SEMANTIC_MATCH_WEIGHT = 0.40

# Sentence-BERT Model Configuration
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Hyperparameter search grids for cross-validation
RF_PARAM_GRID = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

XGB_PARAM_GRID = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
