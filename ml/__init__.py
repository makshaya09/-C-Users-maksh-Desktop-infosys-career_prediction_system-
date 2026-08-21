# ML Module for Career Prediction & Recommendation System
from .train_model import train_baseline_model
from .evaluate import evaluate_model
from .predict import predict_career
from .random_forest import train_random_forest
from .xgboost_model import train_xgboost
from .model_comparison import run_model_comparison, get_best_model_artifacts
from .skill_embeddings import (
    encode_skills,
    encode_job_description,
    calculate_skill_similarity,
    get_career_semantic_similarity,
    get_all_career_similarities
)
from .recommendation_engine import (
    recommend_careers,
    calculate_skill_alignment,
    parse_user_skills
)
from .evaluation import (
    evaluate_linkedin_transitions,
    evaluate_semeval_benchmark,
    run_full_evaluation_suite,
    calculate_mrr,
    calculate_ndcg_at_k
)

__all__ = [
    "train_baseline_model",
    "evaluate_model",
    "predict_career",
    "train_random_forest",
    "train_xgboost",
    "run_model_comparison",
    "get_best_model_artifacts",
    "encode_skills",
    "encode_job_description",
    "calculate_skill_similarity",
    "get_career_semantic_similarity",
    "get_all_career_similarities",
    "recommend_careers",
    "calculate_skill_alignment",
    "parse_user_skills",
    "evaluate_linkedin_transitions",
    "evaluate_semeval_benchmark",
    "run_full_evaluation_suite",
    "calculate_mrr",
    "calculate_ndcg_at_k"
]
