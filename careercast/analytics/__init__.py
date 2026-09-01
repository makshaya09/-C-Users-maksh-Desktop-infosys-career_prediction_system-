"""
CareerCast Cohort Analytics Package.
Provides aggregation, distribution metrics, skill frequency analysis, and comparative career evaluations.
"""

from .cohort import (
    compute_cohort_analytics,
    compute_career_comparison,
    get_sample_cohort
)

__all__ = [
    "compute_cohort_analytics",
    "compute_career_comparison",
    "get_sample_cohort"
]
