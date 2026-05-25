"""Live-price + holdings + P&L helpers backed by yfinance.

Three jobs:
  1. fetch_current_prices(tickers) — live spot prices, partial failures per ticker
  2. snapshot_holdings(allocation, portfolio, investment_capital, prices) —
     convert a recommendation into a {ticker: {shares, buying_price, bought_at}} dict
  3. compute_pnl(holdings, current_prices) — turn that snapshot + today's
     prices into total / per-ticker gain-loss for the dashboard

Tickers without a usable price are skipped (returned with `price=None` and an
error string) — the dashboard surfaces this rather than crashing.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import yfinance as yf


# ---------------------------------------------------------------------------
# 1. Live-price fetch
# ---------------------------------------------------------------------------
def fetch_current_prices(tickers: list[str]) -> dict[str, Optional[float]]:
    """Batch-fetch latest closing prices.

    Returns `{ticker: price_or_None}`. Per-ticker failures are silent —
    callers receive `None` so they can decide how to render the gap.
    """
    out: dict[str, Optional[float]] = {}
    for t in tickers:
        out[t] = _fetch_one(t)
    return out


def _fetch_one(ticker: str) -> Optional[float]:
    try:
        tk = yf.Ticker(ticker)
        fi = getattr(tk, "fast_info", None)
        price = None
        if fi is not None:
            price = getattr(fi, "last_price", None)
        if price is None or price != price:  # NaN check
            hist = tk.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if price is None or price != price:
            return None
        return float(price)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. Snapshot holdings at portfolio creation
# ---------------------------------------------------------------------------
def per_ticker_weights(
    portfolio: dict[str, list[str]],
    allocation: dict[str, float],
) -> dict[str, float]:
    """Equal-weight-within-class flatten — same rule the rest of the app uses.

    Class % is split evenly across the N tickers in that class. The final
    weights are renormalised to sum to 1.0 to absorb float drift.
    """
    class_to_pct = {
        "equity":      allocation.get("equity_pct", 0.0),
        "bond":        allocation.get("bond_pct",   0.0),
        "alternative": allocation.get("alt_pct",    0.0),
    }
    weights: dict[str, float] = {}
    for cls, tickers in portfolio.items():
        if not tickers:
            continue
        per = class_to_pct.get(cls, 0.0) / len(tickers)
        for t in tickers:
            weights[t] = weights.get(t, 0.0) + per

    total = sum(weights.values())
    if total <= 0:
        return weights
    return {t: w / total for t, w in weights.items()}


def snapshot_holdings(
    portfolio: dict[str, list[str]],
    allocation: dict[str, float],
    investment_capital: float,
    current_prices: dict[str, Optional[float]],
) -> dict[str, dict]:
    """Return `{ticker: {shares, buying_price, bought_at}}`.

    Tickers whose `current_prices[t]` is None are recorded with shares=0 and
    `buying_price=None` so the dashboard can flag them — the dollar
    investment for those slots is effectively skipped (un-invested cash).
    """
    weights = per_ticker_weights(portfolio, allocation)
    now_iso = datetime.now(timezone.utc).isoformat()
    holdings: dict[str, dict] = {}
    for ticker, weight in weights.items():
        price = current_prices.get(ticker)
        dollars = float(investment_capital) * float(weight)
        if price is None or price <= 0:
            holdings[ticker] = {
                "shares":       0.0,
                "buying_price": None,
                "dollars":      round(dollars, 2),
                "bought_at":    now_iso,
                "error":        "no price available at snapshot time",
            }
            continue
        shares = dollars / price
        holdings[ticker] = {
            "shares":       round(shares, 6),
            "buying_price": round(float(price), 4),
            "dollars":      round(dollars, 2),
            "bought_at":    now_iso,
        }
    return holdings


# ---------------------------------------------------------------------------
# 3. P&L over a snapshot
# ---------------------------------------------------------------------------
def compute_pnl(
    holdings: dict[str, dict],
    current_prices: dict[str, Optional[float]],
) -> dict:
    """Aggregate + per-ticker P&L. All currency values are plain floats."""
    per_ticker: list[dict] = []
    total_cost = 0.0
    total_value = 0.0
    for ticker, h in holdings.items():
        shares       = float(h.get("shares") or 0.0)
        buying_price = h.get("buying_price")
        cost         = float(h.get("dollars") or 0.0)
        current      = current_prices.get(ticker)

        if buying_price is None or current is None or shares <= 0:
            per_ticker.append({
                "ticker":        ticker,
                "shares":        shares,
                "buying_price":  buying_price,
                "current_price": current,
                "cost_basis":    cost,
                "current_value": None,
                "pnl":           None,
                "pnl_pct":       None,
            })
            total_cost += cost  # cost still counts even if currentless
            continue

        value = shares * float(current)
        pnl = value - cost
        pnl_pct = (pnl / cost) if cost > 0 else None
        total_cost  += cost
        total_value += value
        per_ticker.append({
            "ticker":        ticker,
            "shares":        round(shares, 6),
            "buying_price":  round(float(buying_price), 4),
            "current_price": round(float(current),      4),
            "cost_basis":    round(cost,                2),
            "current_value": round(value,               2),
            "pnl":           round(pnl,                 2),
            "pnl_pct":       round(pnl_pct, 6) if pnl_pct is not None else None,
        })

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) if total_cost > 0 else None
    return {
        "per_ticker":    per_ticker,
        "total_cost":    round(total_cost,    2),
        "total_value":   round(total_value,   2),
        "total_pnl":     round(total_pnl,     2),
        "total_pnl_pct": round(total_pnl_pct, 6) if total_pnl_pct is not None else None,
    }
