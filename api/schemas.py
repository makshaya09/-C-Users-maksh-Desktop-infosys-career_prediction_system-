"""
Pydantic Data Schemas and Validation Models for FastAPI REST Service.
"""

from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field, field_validator


class ProfileInput(BaseModel):
    """Candidate profile input schema with field-level validation."""
    name: Optional[str] = Field(default="", description="Full name of the candidate")
    email: Optional[str] = Field(default="", description="Contact email address")
    education: Optional[Union[str, List[str]]] = Field(
        default="B.Tech",
        description="Education degree or degrees list"
    )
    degree_major: Optional[str] = Field(default="", description="Major or specialization")
    skills: Union[str, List[str]] = Field(
        ...,
        description="Technical skills as a list of strings or comma-separated string"
    )
    experience: Union[int, float] = Field(
        default=0.0,
        ge=0.0,
        le=60.0,
        description="Years of professional experience (0 to 60)"
    )
    certifications: Optional[Union[str, List[str]]] = Field(
        default="",
        description="Professional certifications"
    )
    projects: Optional[Union[str, List[str]]] = Field(
        default="",
        description="Key projects completed"
    )
    preferred_career: Optional[str] = Field(
        default="",
        description="Optional career preference"
    )

    @field_validator("skills")
    @classmethod
    def validate_skills_not_empty(cls, v):
        if isinstance(v, list):
            clean = [str(s).strip() for s in v if str(s).strip()]
            if not clean:
                raise ValueError("At least one technical skill is required.")
            return clean
        elif isinstance(v, str):
            if not v.strip():
                raise ValueError("At least one technical skill is required.")
            return v.strip()
        raise ValueError("Skills must be a non-empty string or list of strings.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Jane Doe",
                "email": "jane.doe@example.com",
                "education": "B.Tech in Computer Science",
                "skills": ["Python", "Machine Learning", "Pandas", "Scikit-learn", "SQL"],
                "experience": 2.5,
                "certifications": "IBM Data Science Professional",
                "projects": "Customer Churn Prediction and Recommender System"
            }
        }
    }


class PredictionRequest(ProfileInput):
    """Request schema for career prediction endpoint."""
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top predictions to return")


class PredictionItem(BaseModel):
    """Individual career prediction item with probability."""
    career: str = Field(..., description="Predicted career category name")
    probability: float = Field(..., description="Model classification probability (0.0 - 1.0)")
    percentage: float = Field(..., description="Probability expressed as a percentage (0.0 - 100.0)")


class PredictionResponse(BaseModel):
    """Response schema for career prediction endpoint."""
    status: str = Field(default="success", description="Response status")
    prediction: str = Field(..., description="Top predicted career domain")
    confidence: float = Field(..., description="Confidence probability of top prediction")
    confidence_percentage: float = Field(..., description="Confidence percentage")
    recommendations: List[PredictionItem] = Field(..., description="Top ranked predictions")
    feature_text: Optional[str] = Field(default="", description="Unified input feature representation")


class RecommendationRequest(ProfileInput):
    """Request schema for career recommendation endpoint."""
    top_k: int = Field(default=5, ge=1, le=10, description="Number of top recommendations to return")


class RecommendationItem(BaseModel):
    """Detailed career recommendation with multi-modal scoring metrics."""
    rank: int = Field(..., description="Recommendation ranking position (1-based)")
    career: str = Field(..., description="Recommended career category")
    overall_score: float = Field(..., description="Combined recommendation score (0.0 - 100.0)")
    confidence_score: float = Field(..., description="ML classification probability")
    confidence_percentage: float = Field(..., description="ML confidence percentage")
    skill_alignment: float = Field(..., description="Skill alignment score")
    exact_match_percentage: float = Field(..., description="Exact required skill match percentage")
    semantic_similarity: float = Field(..., description="Sentence-BERT semantic similarity")
    semantic_percentage: float = Field(..., description="Semantic similarity percentage")
    matched_skills: List[str] = Field(default_factory=list, description="Skills possessed matching career")
    missing_skills: List[str] = Field(default_factory=list, description="Required skills missing from profile")
    matched_count: int = Field(..., description="Number of matched skills")
    total_required_count: int = Field(..., description="Total required canonical skills for this career")


class RecommendationResponse(BaseModel):
    """Response schema for career recommendation endpoint."""
    status: str = Field(default="success", description="Response status")
    prediction: str = Field(..., description="Top recommended career")
    confidence_score: float = Field(..., description="ML confidence score")
    confidence_percentage: float = Field(..., description="ML confidence percentage")
    skill_alignment: float = Field(..., description="Skill alignment score")
    semantic_similarity: float = Field(..., description="Semantic similarity score")
    overall_score: float = Field(..., description="Overall combined score")
    active_model: str = Field(..., description="Active ML model utilized")
    recommendations: List[RecommendationItem] = Field(..., description="Top-K ranked career recommendations")
    user_skills: List[str] = Field(..., description="Parsed candidate skill list")


class SkillGapRequest(BaseModel):
    """Request schema for skill gap analysis and report generation."""
    skills: Union[str, List[str]] = Field(
        ...,
        description="Candidate skills list or comma-separated string"
    )
    target_career: Optional[str] = Field(
        default=None,
        description="Target career name (e.g. 'Data Scientist', 'Cloud Engineer'). If omitted, predicted automatically."
    )
    education: Optional[str] = Field(default="B.Tech", description="Education level")
    experience: Optional[Union[int, float]] = Field(default=0.0, ge=0.0, le=60.0, description="Years of experience")
    certifications: Optional[Union[str, List[str]]] = Field(default="", description="Certifications")
    projects: Optional[Union[str, List[str]]] = Field(default="", description="Projects")

    @field_validator("skills")
    @classmethod
    def validate_skills_not_empty(cls, v):
        if isinstance(v, list):
            clean = [str(s).strip() for s in v if str(s).strip()]
            if not clean:
                raise ValueError("At least one candidate skill is required for gap analysis.")
            return clean
        elif isinstance(v, str):
            if not v.strip():
                raise ValueError("At least one candidate skill is required for gap analysis.")
            return v.strip()
        raise ValueError("Skills must be a non-empty string or list of strings.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "skills": ["Python", "SQL", "Pandas"],
                "target_career": "Data Scientist"
            }
        }
    }


class CompetencySuggestion(BaseModel):
    """Structured actionable competency improvement suggestion."""
    skill: str = Field(..., description="Missing skill name")
    priority: str = Field(..., description="Priority level (High, Medium, Normal)")
    category: Optional[str] = Field(default="Domain Competency", description="Skill category")
    recommended_action: str = Field(..., description="Actionable learning advice")
    learning_resources: List[str] = Field(..., description="Curated learning documentation and courses")
    practical_project: str = Field(..., description="Concrete project idea to demonstrate competency")
    estimated_time: str = Field(..., description="Estimated time to achieve proficiency")


class SkillGapResponse(BaseModel):
    """Response schema for skill gap analysis and report generation."""
    status: str = Field(default="success", description="Status code")
    target_career: str = Field(..., description="Target career domain analyzed")
    candidate_skills: List[str] = Field(..., description="Skills possessed by the candidate")
    required_skills: List[str] = Field(..., description="Canonical skills required for this career")
    matched_skills: List[str] = Field(..., description="Skills matched with career requirements")
    missing_skills: List[str] = Field(..., description="Skills needed to bridge the competency gap")
    matched_count: int = Field(..., description="Count of matched skills")
    missing_count: int = Field(..., description="Count of missing skills")
    total_required_count: int = Field(..., description="Total canonical skills required")
    skill_gap_percentage: float = Field(..., description="Skill gap percentage (0.0 - 100.0%)")
    competency_match_score: float = Field(..., description="Competency match score (0.0 - 100.0%)")
    improvement_suggestions: List[CompetencySuggestion] = Field(
        ...,
        description="Actionable improvement roadmaps for each missing skill"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default="healthy", description="Service health state")
    service: str = Field(default="Career Prediction & Skill Gap REST API", description="Service name")
    version: str = Field(default="1.0.0", description="API version")
    models_loaded: bool = Field(default=True, description="Whether ML models are loaded in memory")
    active_model: str = Field(..., description="Currently active classification model")
    endpoints: List[str] = Field(..., description="Available REST API endpoints")
