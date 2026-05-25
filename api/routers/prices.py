"""POST /api/current-prices — live yfinance prices, partial failures surface per ticker."""
from __future__ import annotations

from fastapi import APIRouter

from api.schemas import CurrentPricesRequest, CurrentPricesResponse
from api.services.prices_service import fetch_current_prices

router = APIRouter()


@router.post("/api/current-prices", response_model=CurrentPricesResponse)
def current_prices(req: CurrentPricesRequest) -> CurrentPricesResponse:
    return fetch_current_prices(req.tickers)
