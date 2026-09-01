"""
Skill Gap Analysis Package.
Exports analyzer and improvement suggestion generators.
"""

from skill_gap.analyzer import analyze_skill_gap, get_required_skills_for_career
from skill_gap.suggestions import (
    generate_improvement_suggestions,
    get_suggestion_for_skill,
    SKILL_SUGGESTIONS_DB
)

__all__ = [
    "analyze_skill_gap",
    "get_required_skills_for_career",
    "generate_improvement_suggestions",
    "get_suggestion_for_skill",
    "SKILL_SUGGESTIONS_DB"
]
