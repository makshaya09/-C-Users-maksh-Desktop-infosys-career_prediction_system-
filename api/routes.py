"""
FastAPI REST Route Handlers for Prediction, Recommendation, and Skill Gap Analysis.
"""

import os
import sys
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

# Ensure root directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
    SkillGapRequest,
    SkillGapResponse,
    HealthResponse
)
from ml.predict import predict_career
from ml.recommendation_engine import recommend_careers
from ml.model_comparison import get_best_model_artifacts
from skill_gap.analyzer import analyze_skill_gap

router = APIRouter(tags=["Career AI Endpoints"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check and model status",
    description="Returns the operating status, loaded model name, and available REST routes."
)
async def health_check() -> HealthResponse:
    """Returns system status, active model information, and available endpoints."""
    try:
        _, _, _, model_name = get_best_model_artifacts()
    except Exception:
        model_name = "Logistic Regression (Fallback)"

    return HealthResponse(
        status="healthy",
        service="Career Prediction & Skill Gap REST API",
        version="1.0.0",
        models_loaded=True,
        active_model=model_name,
        endpoints=[
            "GET  /health",
            "POST /predict",
            "POST /recommend",
            "POST /skill-gap-report",
            "GET  /docs (Swagger UI)",
            "GET  /redoc (ReDoc UI)"
        ]
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict candidate career domain",
    description="Predicts the most suitable career domain and top probabilities using trained ML models."
)
async def predict_endpoint(payload: PredictionRequest) -> PredictionResponse:
    """
    Generates career predictions and probabilities for a candidate profile.
    Reuses existing ML prediction logic from ml.predict.
    """
    profile_dict = payload.model_dump()
    top_k = payload.top_k

    try:
        raw_result = predict_career(profile_dict, top_k=top_k)
        return PredictionResponse(
            status="success",
            prediction=raw_result["prediction"],
            confidence=raw_result["confidence"],
            confidence_percentage=raw_result["confidence_percentage"],
            recommendations=raw_result["recommendations"],
            feature_text=raw_result.get("feature_text", "")
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error in career prediction: {str(exc)}"
        )


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Top-K Career recommendations with multi-modal scoring",
    description="Calculates ML probabilities, Sentence-BERT semantic similarity, and skill alignment to produce ranked recommendations."
)
async def recommend_endpoint(payload: RecommendationRequest) -> RecommendationResponse:
    """
    Generates Top-K ranked career recommendations combining ML confidence,
    Sentence-BERT semantic similarity, and skill alignment.
    """
    profile_dict = payload.model_dump()
    top_k = payload.top_k

    try:
        raw_result = recommend_careers(profile_dict, top_k=top_k)
        return RecommendationResponse(
            status="success",
            prediction=raw_result["prediction"],
            confidence_score=raw_result["confidence_score"],
            confidence_percentage=raw_result["confidence_percentage"],
            skill_alignment=raw_result["skill_alignment"],
            semantic_similarity=raw_result["semantic_similarity"],
            overall_score=raw_result["overall_score"],
            active_model=raw_result["active_model"],
            recommendations=raw_result["recommendations"],
            user_skills=raw_result["user_skills"]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in recommendation engine: {str(exc)}"
        )


@router.post(
    "/skill-gap-report",
    response_model=SkillGapResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate comprehensive Skill Gap Analysis and actionable improvement roadmap",
    description="Compares candidate skills against target career requirements, detects matched & missing skills, computes gap %, and generates actionable learning recommendations."
)
async def skill_gap_report_endpoint(payload: SkillGapRequest) -> SkillGapResponse:
    """
    Generates a structured Skill Gap Analysis report with matched/missing skills,
    gap percentages, and actionable competency improvement suggestions.
    """
    try:
        profile_dict = payload.model_dump()
        result = analyze_skill_gap(
            candidate_skills=payload.skills,
            target_career=payload.target_career,
            profile=profile_dict
        )
        return SkillGapResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing skill gap: {str(exc)}"
        )
