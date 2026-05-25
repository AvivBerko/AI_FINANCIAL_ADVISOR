"""Pydantic v2 request/response models for the FastAPI wrapper.

The 12-field profile schema is the canonical HTTP contract for the
Next.js orchestrator. Field names are PascalCase (matching the locked
DEC-investor-profile-schema-frozen contract); the recommend_service
maps them to the lowercase keys src.inference.recommend() expects.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Locked-contract enum literals — must match exact strings used at training
# time. See DEC-investor-profile-schema-frozen.
# ---------------------------------------------------------------------------
Gender = Literal["Male", "Female"]
Education = Literal["High school", "Bachelor", "Master"]
MaritalStatus = Literal["Single", "Married"]
InvestmentGoal = Literal["Child education", "Retirement", "Home purchase", "Other"]
RiskTolerance = Literal["Low", "Medium", "High"]
FinancialInvolvement = Literal["Low", "Medium", "High"]


class InvestorProfile(BaseModel):
    """The 12 frozen investor profile fields (PascalCase HTTP contract)."""

    model_config = ConfigDict(extra="forbid")

    Age: int = Field(..., ge=18, le=100)
    Gender: Gender
    Education: Education
    MaritalStatus: MaritalStatus
    HouseholdSize: int = Field(..., ge=1, le=20)
    Income: float = Field(..., ge=0)
    InvestmentGoal: InvestmentGoal
    InvestmentHorizon: int = Field(..., ge=1, le=60)
    InvestmentCapital: float = Field(..., ge=0)
    RiskTolerance: RiskTolerance
    FinancialInvolvement: FinancialInvolvement
    ExperienceYears: int = Field(..., ge=0, le=80)


# ---------------------------------------------------------------------------
# /api/recommend
# ---------------------------------------------------------------------------
class RecommendRequest(InvestorProfile):
    """12-field profile + optional reproducibility seed."""

    seed: Optional[int] = Field(
        default=None,
        description="If set, makes the stochastic ETF selection deterministic. Useful for tests.",
    )


class RecommendResponse(BaseModel):
    """Ticker → weight allocation (weights sum to 1.0)."""

    portfolio: dict[str, float] = Field(
        ...,
        description="Flat ticker→weight map summing to 1.0",
        examples=[{"SPY": 0.40, "QQQ": 0.35, "AGG": 0.25}],
    )
    risk_profile: str
    allocation: dict[str, float] = Field(
        ..., description="Asset-class split: {equity_pct, bond_pct, alt_pct}"
    )
    total_etfs: int
    etf_counts: dict[str, int]


# ---------------------------------------------------------------------------
# /api/current-prices
# ---------------------------------------------------------------------------
class CurrentPricesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tickers: list[str] = Field(..., min_length=1, max_length=200)


class TickerPrice(BaseModel):
    price: Optional[float]
    currency: Optional[str] = None
    error: Optional[str] = None


class CurrentPricesResponse(BaseModel):
    prices: dict[str, TickerPrice]


# ---------------------------------------------------------------------------
# /api/explain
# ---------------------------------------------------------------------------
class ExplainRequest(BaseModel):
    """12-field profile + the allocation map to explain.

    Allocation can be either a flat ticker→weight map (typical Next.js call)
    or a per-class map {equity: [...], bond: [...], alternative: [...]} if
    the caller has the structured form already.
    """

    model_config = ConfigDict(extra="forbid")

    profile: InvestorProfile
    allocation: dict[str, float] = Field(
        ..., description="Flat ticker→weight map summing to 1.0"
    )


class FeatureContribution(BaseModel):
    feature: str
    importance: float
    value: object  # str | int | float — pydantic handles


class RiskClassExplanation(BaseModel):
    predicted_class: str
    top_features: list[FeatureContribution]


class EquityBondSplitExplanation(BaseModel):
    equity_pct: float
    bond_pct: float
    alt_pct: float
    reasoning: str


class PerTickerExplanation(BaseModel):
    ticker: str
    weight: float
    asset_class: str
    aum_rank: Optional[int] = None
    category: Optional[str] = None
    star_rating: Optional[float] = None
    reasoning: str


class ExplainResponse(BaseModel):
    risk_class: RiskClassExplanation
    equity_bond_split: EquityBondSplitExplanation
    per_ticker: list[PerTickerExplanation]


# ---------------------------------------------------------------------------
# /api/etfs/{ticker}
# ---------------------------------------------------------------------------
class ETFDetail(BaseModel):
    ticker: str
    name: Optional[str] = None
    asset_class: Optional[str] = None
    category: Optional[str] = None
    expense_ratio: Optional[float] = None
    aum: Optional[float] = None
    morningstar_rating: Optional[float] = None
    top_holdings: list[str] = Field(default_factory=list)
    sector_breakdown: dict[str, float] = Field(default_factory=dict)
