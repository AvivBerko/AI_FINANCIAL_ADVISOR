"""Adapter between RecommendRequest (PascalCase HTTP) and src.inference.recommend (lowercase)."""
from __future__ import annotations

import numpy as np

from api.schemas import RecommendRequest, RecommendResponse


# PascalCase HTTP → lowercase keys src.inference.recommend() expects.
# Mirror of src.inference._INFERENCE_TO_CSV but in the other direction.
_HTTP_TO_INFERENCE = {
    "Age": "age",
    "Income": "income",
    "InvestmentHorizon": "horizon",
    "RiskTolerance": "risk_tolerance",
    "ExperienceYears": "experience",
}


def _to_inference_dict(req: RecommendRequest) -> dict:
    payload = req.model_dump(exclude={"seed"})
    return {_HTTP_TO_INFERENCE.get(k, k): v for k, v in payload.items()}


def run_recommend(req: RecommendRequest) -> RecommendResponse:
    """Run the full 4-stage pipeline and flatten the per-class portfolio into a ticker→weight map."""
    # Import lazily so tests can monkeypatch before module-load.
    from src.inference import recommend

    user = _to_inference_dict(req)

    if req.seed is not None:
        np.random.seed(req.seed)

    result = recommend(user)

    flat = _flatten_portfolio(result["portfolio"], result["allocation"])
    return RecommendResponse(
        portfolio=flat,
        risk_profile=result["risk_profile"],
        allocation=result["allocation"],
        total_etfs=result["total_etfs"],
        etf_counts=result["etf_counts"],
    )


def _flatten_portfolio(portfolio: dict[str, list[str]], allocation: dict[str, float]) -> dict[str, float]:
    """Convert per-class lists + class allocation % into a flat ticker→weight map (sum to 1.0).

    Inside each class, tickers are equal-weighted. Class weight is split evenly across its tickers.
    Final pass renormalizes to exactly 1.0 to absorb float drift.
    """
    class_to_pct = {
        "equity": allocation.get("equity_pct", 0.0),
        "bond": allocation.get("bond_pct", 0.0),
        "alternative": allocation.get("alt_pct", 0.0),
    }
    flat: dict[str, float] = {}
    for cls, tickers in portfolio.items():
        if not tickers:
            continue
        per = class_to_pct.get(cls, 0.0) / len(tickers)
        for t in tickers:
            flat[t] = flat.get(t, 0.0) + per

    total = sum(flat.values())
    if total == 0:
        return flat
    return {t: round(w / total, 6) for t, w in flat.items()}
