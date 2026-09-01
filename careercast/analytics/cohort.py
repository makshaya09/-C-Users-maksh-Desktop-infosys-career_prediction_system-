"""
Cohort Analytics and Multi-Career Comparison Engine.
Provides aggregate metrics, distributions, skill bottlenecks, and comparative career evaluations.
"""

from collections import Counter
from typing import Dict, Any, List, Optional, Union
import numpy as np

from ml.recommendation_engine import (
    CAREER_REQUIRED_SKILLS,
    parse_user_skills,
    calculate_skill_alignment,
    recommend_careers
)
from ml.predict import predict_career
from skill_gap.analyzer import analyze_skill_gap


def compute_cohort_analytics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes aggregated statistical metrics and distributions across a cohort of evaluated candidates.

    Parameters:
    - records: List of candidate assessment dictionaries. Each item may contain:
      - 'name', 'education', 'experience', 'skills', 'prediction', 'confidence', 'competency_match_score', etc.

    Returns:
    - Dictionary with candidate stats, career distribution, skill frequencies,
      missing skill bottlenecks, confidence distribution, and total count.
    - Gracefully handles empty dataset.
    """
    if not records:
        return {
            "total_candidates": 0,
            "average_experience": 0.0,
            "average_confidence": 0.0,
            "average_competency_score": 0.0,
            "career_distribution": {},
            "top_candidate_skills": {},
            "top_missing_skills": {},
            "education_distribution": {},
            "confidence_bins": {"< 50%": 0, "50-70%": 0, "70-90%": 0, "90-100%": 0},
            "records_summary": []
        }

    total_candidates = len(records)
    experiences = []
    confidences = []
    competency_scores = []
    career_counts = Counter()
    education_counts = Counter()
    skill_counts = Counter()
    missing_skill_counts = Counter()
    confidence_bins = {"< 50%": 0, "50-70%": 0, "70-90%": 0, "90-100%": 0}
    records_summary = []

    for item in records:
        # Experience
        exp = item.get("experience", 0)
        try:
            exp_val = float(exp)
            experiences.append(exp_val)
        except (ValueError, TypeError):
            pass

        # Confidence
        conf = item.get("confidence_percentage", item.get("confidence", 0))
        try:
            conf_val = float(conf)
            if conf_val <= 1.0 and conf_val > 0.0:  # normalize 0.85 -> 85
                conf_val *= 100.0
            confidences.append(conf_val)

            # Binning
            if conf_val < 50:
                confidence_bins["< 50%"] += 1
            elif conf_val < 70:
                confidence_bins["50-70%"] += 1
            elif conf_val < 90:
                confidence_bins["70-90%"] += 1
            else:
                confidence_bins["90-100%"] += 1
        except (ValueError, TypeError):
            pass

        # Competency score
        comp = item.get("competency_match_score", item.get("competency_score", 0))
        try:
            comp_val = float(comp)
            competency_scores.append(comp_val)
        except (ValueError, TypeError):
            pass

        # Career prediction
        career = str(item.get("prediction", item.get("target_career", "Unassigned"))).strip()
        if career:
            career_counts[career] += 1

        # Education
        edu = str(item.get("education", "Unknown")).strip()
        if edu:
            education_counts[edu] += 1

        # Skills
        raw_skills = item.get("skills", item.get("candidate_skills", []))
        parsed_s = parse_user_skills(raw_skills)
        for s in parsed_s:
            skill_counts[s] += 1

        # Missing skills
        raw_missing = item.get("missing_skills", [])
        if isinstance(raw_missing, list):
            for ms in raw_missing:
                missing_skill_counts[ms] += 1

        records_summary.append({
            "Name": item.get("name", "Candidate"),
            "Education": edu,
            "Experience (Yrs)": exp,
            "Predicted Career": career,
            "Confidence (%)": round(conf_val, 1) if 'conf_val' in locals() else 0.0,
            "Skills Count": len(parsed_s)
        })

    avg_exp = round(float(np.mean(experiences)), 1) if experiences else 0.0
    avg_conf = round(float(np.mean(confidences)), 1) if confidences else 0.0
    avg_comp = round(float(np.mean(competency_scores)), 1) if competency_scores else 0.0

    return {
        "total_candidates": total_candidates,
        "average_experience": avg_exp,
        "average_confidence": avg_conf,
        "average_competency_score": avg_comp,
        "career_distribution": dict(career_counts.most_common()),
        "top_candidate_skills": dict(skill_counts.most_common(12)),
        "top_missing_skills": dict(missing_skill_counts.most_common(10)),
        "education_distribution": dict(education_counts.most_common()),
        "confidence_bins": confidence_bins,
        "records_summary": records_summary
    }


def compute_career_comparison(
    candidate_skills: Union[str, List[str]],
    target_careers: Optional[List[str]] = None,
    profile: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Compares candidate skills against multiple target careers side-by-side.

    Parameters:
    - candidate_skills: Candidate skill list or comma-separated string
    - target_careers: List of career names to compare (defaults to all canonical careers)
    - profile: Optional full profile dictionary for ML prediction context

    Returns:
    - List of comparative career assessment dictionaries sorted by overall score descending.
    """
    parsed_skills = parse_user_skills(candidate_skills)
    careers_to_compare = target_careers if target_careers else list(CAREER_REQUIRED_SKILLS.keys())

    # Obtain model prediction probabilities across all careers
    prof_dict = dict(profile) if profile else {}
    if not prof_dict.get("skills"):
        prof_dict["skills"] = parsed_skills

    pred_probs = {}
    try:
        pred_res = predict_career(prof_dict, top_k=len(CAREER_REQUIRED_SKILLS))
        for r in pred_res.get("recommendations", []):
            pred_probs[r["career"]] = {
                "probability": r["probability"],
                "percentage": r["percentage"]
            }
    except Exception:
        pass

    comparisons = []
    for c_name in careers_to_compare:
        req_skills = CAREER_REQUIRED_SKILLS.get(c_name, [])
        align = calculate_skill_alignment(parsed_skills, c_name)
        gap = analyze_skill_gap(parsed_skills, target_career=c_name, profile=prof_dict)

        prob_info = pred_probs.get(c_name, {"probability": 0.0, "percentage": 0.0})
        ml_pct = prob_info["percentage"]
        align_score = align["alignment_score"]
        sem_pct = align["semantic_similarity"] * 100.0

        # Overall composite score matching recommendation engine formula
        overall_score = round(
            (0.50 * ml_pct) + (0.30 * align_score) + (0.20 * sem_pct),
            1
        )

        comparisons.append({
            "career": c_name,
            "overall_score": overall_score,
            "probability": prob_info["probability"],
            "probability_percentage": ml_pct,
            "confidence_percentage": ml_pct,
            "alignment_score": align_score,
            "semantic_similarity": align["semantic_similarity"],
            "semantic_percentage": round(sem_pct, 1),
            "matched_skills": gap["matched_skills"],
            "missing_skills": gap["missing_skills"],
            "matched_count": gap["matched_count"],
            "missing_count": gap["missing_count"],
            "total_required_count": gap["total_required_count"],
            "competency_match_score": gap["competency_match_score"],
            "skill_gap_percentage": gap["skill_gap_percentage"],
            "candidate_skills": parsed_skills,
            "required_skills": req_skills
        })

    comparisons.sort(key=lambda x: x["overall_score"], reverse=True)
    for idx, comp in enumerate(comparisons, 1):
        comp["rank"] = idx

    return comparisons


def get_sample_cohort() -> List[Dict[str, Any]]:
    """Provides a realistic sample cohort dataset for immediate demonstration."""
    return [
        {
            "name": "Arjun Sharma",
            "education": "M.Tech in AI",
            "experience": 3.0,
            "skills": "Python, Machine Learning, Deep Learning, TensorFlow, Pandas, Scikit-learn, SQL",
            "prediction": "Data Scientist",
            "confidence_percentage": 91.2,
            "competency_match_score": 63.6,
            "missing_skills": ["NumPy", "Statistics", "Data Visualization", "Matplotlib"]
        },
        {
            "name": "Priya Patel",
            "education": "B.Tech in CS",
            "experience": 2.0,
            "skills": "HTML, CSS, JavaScript, React, Node.js, Express, MongoDB, REST API",
            "prediction": "Web Developer",
            "confidence_percentage": 94.5,
            "competency_match_score": 72.7,
            "missing_skills": ["Bootstrap", "TypeScript", "Git"]
        },
        {
            "name": "Rohan Verma",
            "education": "B.Tech in IT",
            "experience": 2.5,
            "skills": "AWS, Docker, Kubernetes, Linux, Terraform, CI/CD, Git",
            "prediction": "Cloud Engineer",
            "confidence_percentage": 88.0,
            "competency_match_score": 70.0,
            "missing_skills": ["Azure", "Cloud Architecture", "Jenkins"]
        },
        {
            "name": "Neha Gupta",
            "education": "B.Tech in CS",
            "experience": 3.5,
            "skills": "Network Security, Wireshark, Linux, Ethical Hacking, Nmap, Firewall, SIEM",
            "prediction": "Cybersecurity Analyst",
            "confidence_percentage": 92.3,
            "competency_match_score": 70.0,
            "missing_skills": ["Metasploit", "Penetration Testing", "Cybersecurity"]
        },
        {
            "name": "Vikram Singh",
            "education": "BCA",
            "experience": 1.5,
            "skills": "Java, C++, Data Structures, Algorithms, Git, OOP, SQL",
            "prediction": "Software Developer",
            "confidence_percentage": 85.6,
            "competency_match_score": 63.6,
            "missing_skills": ["REST API", "Linux", "Docker", "Python"]
        },
        {
            "name": "Ananya Roy",
            "education": "B.Sc CS",
            "experience": 1.0,
            "skills": "SQL, Excel, Tableau, Power BI, Python, Data Analysis",
            "prediction": "Data Analyst",
            "confidence_percentage": 89.4,
            "competency_match_score": 60.0,
            "missing_skills": ["Data Visualization", "Pandas", "Statistics", "MySQL"]
        }
    ]
