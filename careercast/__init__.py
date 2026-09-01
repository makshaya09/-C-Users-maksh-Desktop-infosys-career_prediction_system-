"""
CareerCast — AI-Powered Career Prediction, Recommendation & Skill Gap Analysis System.
Unified Python Library & API.
"""

import sys
import os

__version__ = "1.0.0"
__author__ = "CareerCast AI Team"

# Ensure project root is available for internal package resolution
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Expose primary NLP functions
from nlp.resume_parser import (
    parse_resume,
    extract_text,
    extract_skills,
    extract_education,
    extract_roles,
    extract_experience_years,
    extract_projects,
    extract_entities
)
from nlp.preprocessing import clean_text, combine_profile_features

# Expose Profile Validation
from app import validate_profile_data as validate_profile

# Expose ML Prediction & Inference
from ml.predict import (
    predict_career,
    load_model_artifacts
)

# Expose Recommendation Engine & Alignment
from ml.recommendation_engine import (
    recommend_careers,
    calculate_skill_alignment,
    parse_user_skills,
    CAREER_REQUIRED_SKILLS
)
from ml.model_comparison import (
    get_best_model_artifacts,
    run_model_comparison
)
from ml.skill_embeddings import (
    encode_skills,
    encode_job_description,
    calculate_skill_similarity,
    get_career_semantic_similarity,
    get_all_career_similarities
)

# Expose Skill Gap Analysis
from skill_gap.analyzer import (
    analyze_skill_gap,
    get_required_skills_for_career
)
from skill_gap.suggestions import (
    generate_improvement_suggestions,
    SKILL_SUGGESTIONS_DB
)
SKILL_IMPROVEMENT_SUGGESTIONS = SKILL_SUGGESTIONS_DB


# Expose Report Generation
from careercast.reports.generator import (
    generate_markdown_report,
    generate_json_report,
    generate_pdf_report
)

# Expose Cohort Analytics & Multi-Career Comparison
from careercast.analytics.cohort import (
    compute_cohort_analytics,
    compute_career_comparison,
    get_sample_cohort
)

# Expose CLI main entrypoint
from careercast.cli.main import main as cli_main

# Aliases and exports
__all__ = [
    "__version__",
    # NLP
    "parse_resume",
    "extract_text",
    "extract_skills",
    "extract_education",
    "extract_roles",
    "extract_experience_years",
    "extract_projects",
    "extract_entities",
    "clean_text",
    "combine_profile_features",
    # Validation
    "validate_profile",
    # ML & Prediction
    "predict_career",
    "load_model_artifacts",
    "recommend_careers",
    "calculate_skill_alignment",
    "parse_user_skills",
    "CAREER_REQUIRED_SKILLS",
    "get_best_model_artifacts",
    "run_model_comparison",
    "encode_skills",
    "encode_job_description",
    "calculate_skill_similarity",
    "get_career_semantic_similarity",
    "get_all_career_similarities",
    # Skill Gap
    "analyze_skill_gap",
    "get_required_skills_for_career",
    "generate_improvement_suggestions",
    "SKILL_IMPROVEMENT_SUGGESTIONS",
    # Reports
    "generate_markdown_report",
    "generate_json_report",
    "generate_pdf_report",
    # Analytics
    "compute_cohort_analytics",
    "compute_career_comparison",
    "get_sample_cohort",
    # CLI
    "cli_main"
]
