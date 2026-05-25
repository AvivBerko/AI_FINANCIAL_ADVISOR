"""ETF metadata lookup. Reads from the persisted combined_etf_with_morning_star.csv."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from api.schemas import ETFDetail
from src.allocations import get_category_mapping


_SECTOR_COLS = {
    "sector_weightings_realestate": "real_estate",
    "sector_weightings_consumer_cyclical": "consumer_cyclical",
    "sector_weightings_basic_materials": "basic_materials",
    "sector_weightings_consumer_defensive": "consumer_defensive",
    "sector_weightings_technology": "technology",
    "sector_weightings_communication_services": "communication_services",
    "sector_weightings_financial_services": "financial_services",
    "sector_weightings_utilities": "utilities",
    "sector_weightings_industrials": "industrials",
    "sector_weightings_energy": "energy",
    "sector_weightings_healthcare": "healthcare",
}


def lookup_etf(ticker: str, etf_df: pd.DataFrame) -> Optional[ETFDetail]:
    """Return ETFDetail for `ticker`, or None if not in the supported universe."""
    row = etf_df[etf_df["Fund Symbol"].str.upper() == ticker.upper()]
    if row.empty:
        return None
    r = row.iloc[0]

    sectors: dict[str, float] = {}
    for csv_col, out_name in _SECTOR_COLS.items():
        v = r.get(csv_col)
        if v is None or pd.isna(v):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            sectors[out_name] = round(fv, 6)

    return ETFDetail(
        ticker=str(r["Fund Symbol"]),
        name=_opt_str(r.get("Fund Name")),
        asset_class=get_category_mapping(r.get("Category", "")),
        category=_opt_str(r.get("Category")),
        expense_ratio=_opt_float(r.get("expense_ratio")),
        aum=_opt_float(r.get("Assets Under Management (AUM)")),
        morningstar_rating=_opt_float(r.get("custom_star_rating")),
        top_holdings=[],
        sector_breakdown=sectors,
    )


def _opt_str(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


def _opt_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
