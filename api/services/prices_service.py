"""Live-price fetcher. The only module in api/ that imports yfinance."""
from __future__ import annotations

import yfinance as yf

from api.schemas import CurrentPricesResponse, TickerPrice


def fetch_current_prices(tickers: list[str]) -> CurrentPricesResponse:
    """Batch-fetch live prices via yfinance. Partial failures surface per-ticker; the batch always returns 200."""
    out: dict[str, TickerPrice] = {}
    for t in tickers:
        out[t] = _fetch_one(t)
    return CurrentPricesResponse(prices=out)


def _fetch_one(ticker: str) -> TickerPrice:
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "fast_info", None)
        price = None
        currency = None
        if info is not None:
            price = getattr(info, "last_price", None)
            currency = getattr(info, "currency", None)
        if price is None or price != price:
            hist = t.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if price is None or price != price:
            return TickerPrice(price=None, error=f"No price data for {ticker}")
        return TickerPrice(price=float(price), currency=currency)
    except Exception as e:
        return TickerPrice(price=None, error=f"{type(e).__name__}: {e}")
