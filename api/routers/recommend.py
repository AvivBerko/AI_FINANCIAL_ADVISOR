"""POST /api/recommend — runs the 4-stage ML pipeline."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import RecommendRequest, RecommendResponse
from api.services.recommend_service import run_recommend

router = APIRouter()


@router.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    try:
        return run_recommend(req)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"ML artifacts missing: {e}. Run `python pipeline.py` to regenerate "
                "the models and preprocessor."
            ),
        )
