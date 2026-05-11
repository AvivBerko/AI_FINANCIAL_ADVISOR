"""
Data validation layer for the training pipeline.

Exposes:
  - ValidationReport     : dataclass for human-readable validation results
  - validate_investor_data(df) -> (clean_df, ValidationReport)
  - validate_etf_data(df)      -> (clean_df, ValidationReport)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.validation_rules import (
    INVESTOR_FEATURE_COLUMNS, VALID_VALUES, NUMERIC_RANGES,
    ETF_REQUIRED_COLUMNS, ETF_DROP_IF_MISSING, ETF_IMPUTE_COLUMNS,
    ASSET_CLASS_FLOORS, KNOWN_CATEGORY_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    name: str
    n_input: int
    n_output: int
    n_dropped: int
    dropped_reasons:    dict        = field(default_factory=dict)
    imputed:            dict        = field(default_factory=dict)   # ETF only — counts
    imputation_detail:  list        = field(default_factory=list)   # ETF only — per-cell rows
    warnings:           list        = field(default_factory=list)
    critical_issues:    list        = field(default_factory=list)
    asset_class_counts: dict        = field(default_factory=dict)   # ETF only

    def to_markdown(self) -> str:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        title = f"# {self.name.capitalize()} Data Validation Report"
        pct = (self.n_dropped / self.n_input * 100) if self.n_input else 0.0

        lines = [
            title,
            f"Generated: {ts}",
            "",
            "## Summary",
            f"- Input rows:    {self.n_input}",
            f"- Output rows:   {self.n_output}",
            f"- Dropped:       {self.n_dropped} ({pct:.2f}%)",
            "",
        ]

        if self.dropped_reasons:
            lines.append("## Drop reasons")
            for reason, count in sorted(self.dropped_reasons.items()):
                lines.append(f"- {reason}: {count}")
            lines.append("")

        if self.asset_class_counts:
            lines.append("## Asset class breakdown (post-validation)")
            for cls, count in self.asset_class_counts.items():
                floor = ASSET_CLASS_FLOORS.get(cls, 0)
                status = "OK" if count >= floor else "BELOW FLOOR"
                lines.append(f"- {cls}: {count}  (floor: {floor})  {status}")
            lines.append("")

        if self.imputed:
            lines.append("## Imputations (counts)")
            for col, count in sorted(self.imputed.items()):
                lines.append(f"- {col}: {count}")
            lines.append("")

        if self.imputation_detail:
            lines.append("## Imputation detail")
            for row in self.imputation_detail:
                lines.append(
                    f"- {row['ticker']} | {row['column']} | "
                    f"{row['original']} -> {row['imputed']:.4f} ({row['source']})"
                )
            lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        if self.critical_issues:
            lines.append("## Critical issues")
            for c in self.critical_issues:
                lines.append(f"- {c}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Investor validator
# ---------------------------------------------------------------------------
def validate_investor_data(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Returns (cleaned_df, report). Bad rows are DROPPED with reason logged.
    Missing required columns RAISES ValueError.
    """
    rpt = ValidationReport(name="investor", n_input=len(df), n_output=0, n_dropped=0)

    # Schema check
    missing = [c for c in INVESTOR_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        rpt.critical_issues.append(f"Missing columns: {missing}")
        raise ValueError(f"Investor data missing required columns: {missing}")

    # Build a boolean mask of "rows to drop". Increment reason counts per check.
    drop_mask = pd.Series(False, index=df.index)

    # NaN in any feature column
    nan_mask = df[INVESTOR_FEATURE_COLUMNS].isna().any(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan:
        rpt.dropped_reasons['nan_in_feature'] = n_nan
    drop_mask |= nan_mask

    # Categorical out of allowed set
    for col, allowed in VALID_VALUES.items():
        bad = ~df[col].isin(allowed) & df[col].notna()
        n_bad = int(bad.sum())
        if n_bad:
            rpt.dropped_reasons[f'invalid_categorical_{col}'] = n_bad
        drop_mask |= bad

    # Numeric out of range
    for col, (lo, hi) in NUMERIC_RANGES.items():
        as_num = pd.to_numeric(df[col], errors='coerce')
        bad = as_num.notna() & ((as_num < lo) | (as_num > hi))
        n_bad = int(bad.sum())
        if n_bad:
            rpt.dropped_reasons[f'out_of_range_{col}'] = n_bad
        drop_mask |= bad

    # Consistency: ExperienceYears <= Age - 18
    age_num = pd.to_numeric(df['Age'], errors='coerce')
    exp_num = pd.to_numeric(df['ExperienceYears'], errors='coerce')
    consistency_bad = age_num.notna() & exp_num.notna() & (exp_num > (age_num - 18))
    n_cb = int(consistency_bad.sum())
    if n_cb:
        rpt.dropped_reasons['consistency_experience_vs_age'] = n_cb
    drop_mask |= consistency_bad

    clean = df.loc[~drop_mask].copy()
    rpt.n_output = len(clean)
    rpt.n_dropped = rpt.n_input - rpt.n_output
    return clean, rpt
