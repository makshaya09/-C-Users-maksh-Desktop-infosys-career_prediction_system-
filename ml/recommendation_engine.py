"""
Top-K Career Recommendation Engine with Skill Alignment and Semantic Scoring.
Milestone 2: Advanced ML & Recommendation Engine.
"""

import os
import sys
from typing import List, Dict, Any, Union
import numpy as np

# Set path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml.config import (
    DEFAULT_TOP_K,
    ML_PROB_WEIGHT,
    SKILL_ALIGNMENT_WEIGHT,
    SEMANTIC_SIMILARITY_WEIGHT,
    EXACT_MATCH_WEIGHT,
    SEMANTIC_MATCH_WEIGHT
)
from ml.model_comparison import get_best_model_artifacts
from ml.skill_embeddings import get_all_career_similarities, get_career_semantic_similarity
from nlp.preprocessing import combine_profile_features


# Canonical Career Skill Requirements Knowledge Base
CAREER_REQUIRED_SKILLS = {
    "Data Scientist": [
        "Python", "Machine Learning", "SQL", "Pandas", "NumPy", "Scikit-learn",
        "Deep Learning", "TensorFlow", "Statistics", "Data Visualization", "Matplotlib"
    ],
    "Data Analyst": [
        "SQL", "Excel", "Tableau", "Power BI", "Python", "Data Analysis",
        "Data Visualization", "Pandas", "Statistics", "MySQL"
    ],
    "Software Developer": [
        "Java", "C++", "Python", "Data Structures", "Algorithms", "Git",
        "OOP", "REST API", "SQL", "Linux", "Docker"
    ],
    "Web Developer": [
        "HTML", "CSS", "JavaScript", "React", "Node.js", "Express",
        "REST API", "Bootstrap", "MongoDB", "TypeScript", "Git"
    ],
    "Machine Learning Engineer": [
        "Python", "TensorFlow", "PyTorch", "Deep Learning", "Machine Learning",
        "Docker", "Scikit-learn", "MLOps", "Computer Vision", "NLP"
    ],
    "Cloud Engineer": [
        "AWS", "Docker", "Kubernetes", "Linux", "Terraform", "CI/CD",
        "Azure", "Cloud Architecture", "Jenkins", "Git"
    ],
    "Cybersecurity Analyst": [
        "Network Security", "Wireshark", "Linux", "Ethical Hacking", "Firewall",
        "Cybersecurity", "Nmap", "Metasploit", "SIEM", "Penetration Testing"
    ]
}


def parse_user_skills(skills_input: Union[str, List[str]]) -> List[str]:
    """Extracts a clean, normalized list of user skill strings."""
    if isinstance(skills_input, list):
        raw_list = skills_input
    else:
        raw_list = str(skills_input or "").split(",")

    clean_skills = []
    for s in raw_list:
        item = str(s).strip()
        if item and item.lower() not in [x.lower() for x in clean_skills]:
            clean_skills.append(item)
    return clean_skills


def calculate_skill_alignment(
    user_skills: Union[str, List[str]],
    career_name: str,
    semantic_sim: float = None
) -> Dict[str, Any]:
    """
    Compares user skills against canonical career skill requirements.
    Calculates exact match %, matched skills, missing skills, and combined alignment score.
    """
    user_skill_list = parse_user_skills(user_skills)
    user_skills_lower = {s.lower(): s for s in user_skill_list}

    required_skills = CAREER_REQUIRED_SKILLS.get(career_name, [])
    matched_skills = []
    missing_skills = []

    for req in required_skills:
        req_lower = req.lower()
        # Direct or substring match (e.g. "React.js" matches "React")
        matched = False
        for u_low, u_orig in user_skills_lower.items():
            if u_low == req_lower or u_low in req_lower or req_lower in u_low:
                matched_skills.append(req)
                matched = True
                break
        if not matched:
            missing_skills.append(req)

    total_req = len(required_skills) if required_skills else 1
    exact_match_percentage = round((len(matched_skills) / total_req) * 100, 1)

    # Compute semantic similarity if not provided
    if semantic_sim is None:
        semantic_sim = get_career_semantic_similarity(user_skill_list, career_name)

    semantic_pct = semantic_sim * 100.0

    # Combined skill alignment score
    alignment_score = round(
        (EXACT_MATCH_WEIGHT * exact_match_percentage) +
        (SEMANTIC_MATCH_WEIGHT * semantic_pct),
        1
    )

    return {
        "career": career_name,
        "exact_match_percentage": exact_match_percentage,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_count": len(matched_skills),
        "total_required_count": total_req,
        "semantic_similarity": round(semantic_sim, 4),
        "alignment_score": alignment_score
    }


def recommend_careers(
    profile: Dict[str, Any],
    top_k: int = DEFAULT_TOP_K
) -> Dict[str, Any]:
    """
    End-to-end Top-K Career Recommendation Engine:
    1. Loads best trained ML model (Random Forest / XGBoost / Logistic Regression).
    2. Generates ML class probabilities via predict_proba().
    3. Computes Sentence-BERT semantic similarities for all careers.
    4. Calculates skill alignment (matched/missing skills, exact match %).
    5. Combines ML confidence, skill alignment, and semantic similarity into an overall ranking score.
    6. Returns Top-K ranked career recommendations.
    """
    user_skills = profile.get("skills", "")
    user_skill_list = parse_user_skills(user_skills)

    # 1. Load winning ML model artifacts
    model, vectorizer, label_encoder, model_name = get_best_model_artifacts()

    # 2. Vectorize profile text
    feature_text = combine_profile_features(profile)
    feature_vec = vectorizer.transform([feature_text])

    # 3. Predict ML probabilities
    ml_probabilities = model.predict_proba(feature_vec)[0]
    classes = label_encoder.classes_

    # Create mapping of career -> ML probability
    career_ml_probs = {
        str(classes[i]): float(ml_probabilities[i])
        for i in range(len(classes))
    }

    # 4. Compute semantic similarities via Sentence-BERT
    semantic_sims = get_all_career_similarities(user_skill_list)

    # 5. Compute combined score for each career
    ranked_candidates = []
    for career in classes:
        career_str = str(career)
        ml_prob = career_ml_probs.get(career_str, 0.0)
        sem_sim = semantic_sims.get(career_str, 0.0)

        # Skill alignment metrics
        align_info = calculate_skill_alignment(user_skill_list, career_str, semantic_sim=sem_sim)

        # Combined Overall Score
        overall_score = (
            (ML_PROB_WEIGHT * (ml_prob * 100)) +
            (SKILL_ALIGNMENT_WEIGHT * align_info["alignment_score"]) +
            (SEMANTIC_SIMILARITY_WEIGHT * (sem_sim * 100))
        )
        overall_score = round(float(np.clip(overall_score, 0.0, 100.0)), 1)

        ranked_candidates.append({
            "career": career_str,
            "confidence_score": round(ml_prob, 4),
            "confidence_percentage": round(ml_prob * 100, 1),
            "skill_alignment": align_info["alignment_score"],
            "exact_match_percentage": align_info["exact_match_percentage"],
            "semantic_similarity": round(sem_sim, 4),
            "semantic_percentage": round(sem_sim * 100, 1),
            "overall_score": overall_score,
            "matched_skills": align_info["matched_skills"],
            "missing_skills": align_info["missing_skills"],
            "matched_count": align_info["matched_count"],
            "total_required_count": align_info["total_required_count"]
        })

    # Sort by overall score descending
    ranked_candidates.sort(key=lambda x: x["overall_score"], reverse=True)

    # Assign ranks
    top_recommendations = []
    for rank_idx, cand in enumerate(ranked_candidates[:top_k], start=1):
        cand_copy = dict(cand)
        cand_copy["rank"] = rank_idx
        top_recommendations.append(cand_copy)

    top_pick = top_recommendations[0] if top_recommendations else {}

    return {
        "prediction": top_pick.get("career", "N/A"),
        "confidence_score": top_pick.get("confidence_score", 0.0),
        "confidence_percentage": top_pick.get("confidence_percentage", 0.0),
        "skill_alignment": top_pick.get("skill_alignment", 0.0),
        "semantic_similarity": top_pick.get("semantic_similarity", 0.0),
        "overall_score": top_pick.get("overall_score", 0.0),
        "active_model": model_name,
        "recommendations": top_recommendations,
        "feature_text": feature_text,
        "user_skills": user_skill_list,
        "top_k": top_k
    }


if __name__ == "__main__":
    test_profile = {
        "education": "B.Tech",
        "skills": ["Python", "Machine Learning", "SQL", "Pandas", "Scikit-learn"],
        "experience": 2,
        "certifications": "IBM Data Science",
        "projects": "Customer Churn Prediction"
    }
    print("Testing Recommendation Engine...")
    result = recommend_careers(test_profile, top_k=5)
    print(f"\nTop Career: {result['prediction']} (Overall Score: {result['overall_score']}%)")
    print(f"Active Classifier: {result['active_model']}")
    print("\nTop 5 Recommendations:")
    for r in result["recommendations"]:
        print(f"  #{r['rank']} {r['career']:<25} | Score: {r['overall_score']:>5.1f}% | Conf: {r['confidence_percentage']:>5.1f}% | Align: {r['skill_alignment']:>5.1f}% | Matched: {len(r['matched_skills'])} | Missing: {len(r['missing_skills'])}")
