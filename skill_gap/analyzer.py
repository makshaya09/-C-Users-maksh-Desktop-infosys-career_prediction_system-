"""
Skill Gap Analysis Module.
Analyzes candidate skills against canonical career requirements, identifies matched
and missing skills, calculates gap percentages, and generates actionable improvement roadmaps.
"""

import os
import sys
from typing import Dict, Any, List, Union, Optional

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml.recommendation_engine import (
    CAREER_REQUIRED_SKILLS,
    parse_user_skills,
    recommend_careers
)
from ml.predict import predict_career
from skill_gap.suggestions import generate_improvement_suggestions


def get_required_skills_for_career(career_name: str) -> List[str]:
    """
    Retrieves the canonical required skills list for a specified career category.
    Handles exact and case-insensitive career matching.
    """
    career_clean = str(career_name or "").strip()
    if not career_clean:
        return []

    # Direct match
    if career_clean in CAREER_REQUIRED_SKILLS:
        return list(CAREER_REQUIRED_SKILLS[career_clean])

    # Case-insensitive lookup
    for c_name, skills in CAREER_REQUIRED_SKILLS.items():
        if c_name.lower() == career_clean.lower():
            return list(skills)

    # Substring lookup
    for c_name, skills in CAREER_REQUIRED_SKILLS.items():
        if career_clean.lower() in c_name.lower() or c_name.lower() in career_clean.lower():
            return list(skills)

    # Fallback default required skills for unknown custom career
    return ["Core Domain Knowledge", "Problem Solving", "Technical Communication", "Git", "Project Execution"]


def analyze_skill_gap(
    candidate_skills: Union[str, List[str]],
    target_career: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Performs full skill gap analysis for a candidate profile against a target career.

    Parameters:
    - candidate_skills: List of skills or comma-separated string
    - target_career: Target career name (e.g. 'Data Scientist'). If None/empty,
      it will be automatically predicted using the ML recommendation engine.
    - profile: Optional full candidate profile dictionary.

    Returns:
    {
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
                "recommended_action": "...",
                "learning_resources": [...],
                "practical_project": "...",
                "estimated_time": "..."
            },
            ...
        ]
    }
    """
    parsed_skills = parse_user_skills(candidate_skills)

    # If target_career is not supplied, predict target career using ML model
    resolved_career = str(target_career or "").strip()
    if not resolved_career:
        profile_data = dict(profile) if profile else {}
        if not profile_data.get("skills"):
            profile_data["skills"] = parsed_skills
        try:
            rec_result = recommend_careers(profile_data, top_k=1)
            resolved_career = rec_result.get("prediction", "Software Developer")
        except Exception:
            try:
                pred_result = predict_career(profile_data, top_k=1)
                resolved_career = pred_result.get("prediction", "Software Developer")
            except Exception:
                resolved_career = "Software Developer"

    required_skills = get_required_skills_for_career(resolved_career)
    total_required = len(required_skills) if required_skills else 1

    candidate_skills_lower = {s.lower(): s for s in parsed_skills}

    matched_skills: List[str] = []
    missing_skills: List[str] = []

    for req in required_skills:
        req_lower = req.lower()
        matched = False
        for c_low, c_orig in candidate_skills_lower.items():
            # Exact match or substring match (e.g., 'React.js' matches 'React', 'C++' matches 'C++')
            if c_low == req_lower or c_low in req_lower or req_lower in c_low:
                matched_skills.append(req)
                matched = True
                break
        if not matched:
            missing_skills.append(req)

    matched_count = len(matched_skills)
    missing_count = len(missing_skills)

    competency_match_score = round((matched_count / total_required) * 100.0, 1)
    skill_gap_percentage = round((missing_count / total_required) * 100.0, 1)

    # Generate actionable competency improvement suggestions for missing skills
    improvement_suggestions = generate_improvement_suggestions(missing_skills, target_career=resolved_career)

    return {
        "status": "success",
        "target_career": resolved_career,
        "candidate_skills": parsed_skills,
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_count": matched_count,
        "missing_count": missing_count,
        "total_required_count": total_required,
        "skill_gap_percentage": skill_gap_percentage,
        "competency_match_score": competency_match_score,
        "improvement_suggestions": improvement_suggestions
    }


if __name__ == "__main__":
    test_skills = ["Python", "SQL", "Pandas"]
    res = analyze_skill_gap(test_skills, target_career="Data Scientist")
    print(f"Target Career: {res['target_career']}")
    print(f"Matched Skills ({res['matched_count']}): {res['matched_skills']}")
    print(f"Missing Skills ({res['missing_count']}): {res['missing_skills']}")
    print(f"Skill Gap Percentage: {res['skill_gap_percentage']}%")
    print(f"Top Suggestion: {res['improvement_suggestions'][0]['skill']} -> {res['improvement_suggestions'][0]['recommended_action']}")
