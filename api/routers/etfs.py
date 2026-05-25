"""GET /api/etfs/{ticker} — ETF metadata lookup."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.schemas import ETFDetail
from api.services.etf_service import lookup_etf

router = APIRouter()


@router.get("/api/etfs/{ticker}", response_model=ETFDetail)
def get_etf(ticker: str, request: Request) -> ETFDetail:
    etf_df = request.app.state.etf_df
    detail = lookup_etf(ticker, etf_df)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker {ticker} not in supported universe",
        )
    return detail
