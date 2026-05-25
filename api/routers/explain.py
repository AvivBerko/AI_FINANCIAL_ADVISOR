"""POST /api/explain — structured rationale for an allocation."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import ExplainRequest, ExplainResponse
from api.services.explain_service import build_explanation

router = APIRouter()


@router.post("/api/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    try:
        return build_explanation(req)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"ML artifacts missing: {e}. Run `python pipeline.py` to regenerate "
                "the models and preprocessor."
            ),
        )
