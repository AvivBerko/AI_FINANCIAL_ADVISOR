"""Build a structured rationale for an allocation.

Three sections:
  1. Risk-class prediction + top feature contributions (feature_importances_ × value).
  2. Equity/bond/alt split with a templated explanation.
  3. Per-ticker reasoning (asset class, aum_rank, category, star rating).

Reuses src.inference._get_artifacts() so no second copy of the models is loaded.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from api.schemas import (
    EquityBondSplitExplanation,
    ExplainRequest,
    ExplainResponse,
    FeatureContribution,
    PerTickerExplanation,
    RiskClassExplanation,
)
from api.services.recommend_service import _HTTP_TO_INFERENCE


_HORIZON_BAND_REASONING = {
    "long": "long investment horizon supports higher equity exposure",
    "medium": "medium horizon balances equity growth with bond stability",
    "short": "short horizon emphasizes capital preservation through bonds",
}


def _horizon_band(years: int) -> str:
    if years > 10:
        return "long"
    if years >= 5:
        return "medium"
    return "short"


def build_explanation(req: ExplainRequest) -> ExplainResponse:
    from src.inference import FEATURE_COLUMNS, _get_artifacts

    profile_payload = req.profile.model_dump()
    inference_user = {_HTTP_TO_INFERENCE.get(k, k): v for k, v in profile_payload.items()}

    preprocessor, classifier, alloc_regressor, _, label_info, etf_by_class = _get_artifacts()

    X_raw = pd.DataFrame([{col: inference_user[col] for col in FEATURE_COLUMNS}])
    X_processed = preprocessor.transform(X_raw)

    risk_section = _risk_class_section(classifier, preprocessor, X_raw, X_processed, label_info, inference_user)
    split_section = _split_section(alloc_regressor, X_processed, req.profile.InvestmentHorizon, risk_section.predicted_class, req.allocation)
    per_ticker = _per_ticker_section(req.allocation, etf_by_class)

    return ExplainResponse(
        risk_class=risk_section,
        equity_bond_split=split_section,
        per_ticker=per_ticker,
    )


def _risk_class_section(classifier, preprocessor, X_raw, X_processed, label_info, user) -> RiskClassExplanation:
    winner = label_info["winner"]
    inv_label_map = label_info["inv_label_map"]
    raw_pred = classifier.predict(X_processed)
    if winner in ("XGBoost", "MLP"):
        predicted = inv_label_map[str(raw_pred[0])]
    else:
        predicted = str(raw_pred[0])

    importances: Optional[np.ndarray] = getattr(classifier, "feature_importances_", None)
    contribs: list[FeatureContribution] = []
    if importances is not None:
        try:
            feature_names = list(preprocessor.get_feature_names_out())
            x_vec = np.asarray(X_processed)[0]
            contribs_raw = list(zip(feature_names, importances, x_vec))
            contribs_raw.sort(key=lambda t: float(t[1]), reverse=True)
            for name, imp, val in contribs_raw[:5]:
                clean = name.split("__", 1)[-1]
                contribs.append(
                    FeatureContribution(
                        feature=clean,
                        importance=round(float(imp), 6),
                        value=_jsonable(val),
                    )
                )
        except Exception:
            contribs = []

    if not contribs:
        for k in ("risk_tolerance", "age", "horizon", "experience", "income"):
            if k in user:
                contribs.append(
                    FeatureContribution(feature=k, importance=0.0, value=user[k])
                )

    return RiskClassExplanation(predicted_class=predicted, top_features=contribs)


def _split_section(alloc_regressor, X_processed, horizon: int, predicted_class: str, allocation: dict[str, float]) -> EquityBondSplitExplanation:
    pred = alloc_regressor.predict(X_processed)[0]
    eq_raw = float(pred[0])
    bd_raw = float(pred[1])
    alt_raw = max(0.01, 1.0 - eq_raw - bd_raw)
    total = eq_raw + bd_raw + alt_raw
    eq = round(eq_raw / total, 4)
    bd = round(bd_raw / total, 4)
    alt = round(alt_raw / total, 4)

    band = _horizon_band(horizon)
    horizon_text = _HORIZON_BAND_REASONING[band]
    reasoning = (
        f"Stage-2 regressor predicts ~{int(round(eq * 100))}% equity / "
        f"~{int(round(bd * 100))}% bond / ~{int(round(alt * 100))}% alternative for a "
        f"{predicted_class.lower()} risk profile; {horizon_text}."
    )
    return EquityBondSplitExplanation(
        equity_pct=eq, bond_pct=bd, alt_pct=alt, reasoning=reasoning
    )


def _per_ticker_section(allocation: dict[str, float], etf_by_class: dict[str, pd.DataFrame]) -> list[PerTickerExplanation]:
    rows: list[PerTickerExplanation] = []
    aum_rank_by_class: dict[str, dict[str, int]] = {}
    for cls, df in etf_by_class.items():
        if df is None or df.empty or "symbol" not in df.columns or "AUM" not in df.columns:
            aum_rank_by_class[cls] = {}
            continue
        ranks = df["AUM"].rank(ascending=False, method="min").astype(int)
        aum_rank_by_class[cls] = dict(zip(df["symbol"].astype(str), ranks.astype(int)))

    symbol_to_class: dict[str, str] = {}
    symbol_to_row: dict[str, pd.Series] = {}
    for cls, df in etf_by_class.items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            sym = str(r["symbol"])
            symbol_to_class[sym] = cls
            symbol_to_row[sym] = r

    for ticker, weight in allocation.items():
        cls = symbol_to_class.get(ticker, "unknown")
        rank = aum_rank_by_class.get(cls, {}).get(ticker)
        row = symbol_to_row.get(ticker)
        category = str(row["category"]) if row is not None and "category" in row else None
        star = float(row["star_rating"]) if row is not None and "star_rating" in row and not pd.isna(row["star_rating"]) else None
        reasoning = _ticker_reasoning(cls, rank, category, star)
        rows.append(
            PerTickerExplanation(
                ticker=ticker,
                weight=round(float(weight), 6),
                asset_class=cls,
                aum_rank=rank,
                category=category,
                star_rating=star,
                reasoning=reasoning,
            )
        )
    return rows


def _ticker_reasoning(asset_class: str, aum_rank: Optional[int], category: Optional[str], star: Optional[float]) -> str:
    parts = [f"selected from the {asset_class} pool"]
    if category:
        parts.append(f"category '{category}'")
    if aum_rank is not None:
        parts.append(f"AUM rank #{aum_rank}")
    if star is not None:
        parts.append(f"{star:.1f}-star quality")
    return ", ".join(parts) + "."


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        return round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float):
        return round(v, 6)
    return v
