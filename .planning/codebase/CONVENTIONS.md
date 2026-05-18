# Coding Conventions

**Analysis Date:** 2026-05-18

## Naming Patterns

**Files:**
- `snake_case.py` for all module files: `data_validation.py`, `validation_rules.py`, `train_models.py`, `prepare_data.py`
- `etf_fetcher.py`, `morning_star_rating.py` — no abbreviations, descriptive compound names
- Entry points at project root use plain names: `pipeline.py`, `app.py`

**Functions:**
- `snake_case` throughout: `validate_investor_data`, `get_allocation_mix`, `run_allocations_pipeline`
- Private/internal helpers prefixed with underscore: `_load_artifacts`, `_get_artifacts`, `_is_known_category`, `_map_to_asset_class`, `_to_csv`
- Evaluation helpers prefixed by role: `eval_classifier`, `eval_regressor_mo`

**Variables:**
- `snake_case` for all locals and module-level bindings
- DataFrame variables suffixed `_df` or `_df`: `etf_df`, `investors_df`, `result_df`, `X_processed_df`
- Train/test splits use abbreviated suffixes: `X_train`, `X_test`, `y1_train`, `y2_test`
- Targets named with stage number prefix: `y1_risk_profile`, `y2_allocation`, `y3_total_etfs`
- ML model variables abbreviated: `mo_rf`, `mo_gb`, `rf_best`, `knn_gs`

**Classes:**
- `PascalCase`: `ValidationReport`, `FinnhubETFDataHandler`, `YFinanceETFDataHandler`
- Class names describe the role, not the data: `ValidationReport` (not `ETFValidationResult`)

**Constants:**
- `UPPER_SNAKE_CASE` for module-level config: `CONFLICT_GROUPS`, `INVESTOR_FEATURE_COLUMNS`, `VALID_VALUES`, `NUMERIC_RANGES`, `ETF_IMPUTE_COLUMNS`, `ASSET_CLASS_FLOORS`, `MODEL_DIR`, `DATA_DIR`, `REPORT_DIR`

## Module-Level Docstrings

All `src/` modules except `data_generation.py` and `morning_star_rating.py` carry a module docstring. The pattern is a one-line title, then a blank line, then expanded description or a `Usage:` block. Example from `src/inference.py`:

```python
"""
Core Recommendation Engine — Inference Pipeline
================================================
Takes a structured investor profile (dict) and returns a complete
portfolio recommendation by running through all 4 stages:
  ...
Usage:
    from src.inference import recommend
    result = recommend({...})
"""
```

`pipeline.py` and `src/train_models.py` follow the same style with a prominent `Run from project root:` usage hint.

## Function Docstrings

Docstring style is **plain prose** (not Google, NumPy, or reST). Parameters and returns are described in a loose `Parameters / Returns` block only when the function is part of a public API. Examples:

```python
def validate_investor_data(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Returns (cleaned_df, report). Bad rows are DROPPED with reason logged.
    Missing required columns RAISES ValueError.
    """
```

```python
def recommend(user: dict) -> dict:
    """
    Run the full 4-stage pipeline for a single investor profile.

    Parameters
    ----------
    user : dict
        Investor profile with keys matching FEATURE_COLUMNS.

    Returns
    -------
    dict with keys:
        risk_profile   : str   — Aggressive / Moderate / Conservative
        ...
    """
```

Only `src/inference.py::recommend` and `src/llm.py::chat` use a NumPy-style (`---` underline) parameter block. The rest use inline prose. New functions should match the style of the file they live in.

Private helpers (`_is_known_category`, `_map_to_asset_class`, `_load_artifacts`) often carry single-line docstrings only.

## Type Hints

**Usage is partial — applied selectively to public-API boundaries.**

- `src/data_validation.py` — full annotations on both public functions and most private helpers:
  - `validate_investor_data(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]`
  - `validate_etf_data(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]`
  - `_is_known_category(category) -> bool`
  - `_map_to_asset_class(category) -> str`

- `src/inference.py` — annotated on public API:
  - `validate_input(user: dict) -> None`
  - `recommend(user: dict) -> dict`
  - `print_recommendation(result: dict) -> None`

- `src/llm.py` — annotated on all public functions including complex returns:
  - `chat(...) -> tuple[str | None, dict | None]`
  - `get_initial_explanation(user_inputs: dict, result: dict) -> str`

- `src/allocations.py`, `src/train_models.py`, `src/prepare_data.py`, `src/morning_star_rating.py` — **no type annotations**. These are pipeline scripts (not library modules) so annotations were omitted.

**Rule of thumb:** annotate functions that are `import`ed by other modules. Script-only functions do not require annotations.

## Code Style

**Formatting:**
- No automated formatter is configured (no `.prettierrc`, `pyproject.toml`, or `ruff.toml` found)
- PEP 8 followed manually: 4-space indentation, spaces around `=` in expressions
- Line length is not strictly enforced — long f-strings and comments exceed 79 chars (e.g., `src/train_models.py` lines 335, 344 hit ~100 chars)
- Trailing comments aligned with spaces in constant blocks for readability:
  ```python
  ETF_IMPUTE_COLUMNS = {
      'Assets Under Management (AUM)': 1e9,
      'custom_star_rating':            3,
      'Volatility (Annual STD)':       0.15,
  }
  ```

**Linting:**
- No `.flake8`, `.pylintrc`, or `ruff.toml` config found — no enforced linting
- `warnings.filterwarnings('ignore')` is present at module level in `src/allocations.py`, `src/inference.py`, `src/train_models.py`, and `src/prepare_data.py`

## Import Organization

**Order used in `src/` modules:**

1. Standard library (`os`, `sys`, `json`, `warnings`, `datetime`, `dataclasses`)
2. Third-party (`numpy`, `pandas`, `sklearn`, `xgboost`, `joblib`, `openai`, `streamlit`)
3. Local src imports (`from src.allocations import ...`, `from src.validation_rules import ...`)

Local imports sometimes appear inside functions (deferred imports) in pipeline scripts, and at module level in library modules.

**Path manipulation:**
- `sys.path.append(os.path.dirname(os.path.dirname(__file__)))` appears in `src/prepare_data.py` and `src/inference.py` to make `src.` imports work when scripts are run directly
- `app.py` does not use `sys.path` manipulation (Streamlit sets CWD to project root)

**Path Aliases:** None configured.

## Section Separators

Long scripts and modules use `# ---` dashed separator comments to group logical sections. Three styles in use:

```python
# ---------------------------------------------------------------------------
# Section title
# ---------------------------------------------------------------------------
```

```python
# =============================================================================
# SECTION TITLE (CAPS for top-level stages)
# =============================================================================
```

```python
# --- Sub-heading ---
```

`src/train_models.py` uses the `=====` style for stages and `-----` for sub-sections. `src/data_validation.py` and `src/inference.py` use the `-----` dashed style throughout.

## Error Handling

**Strategy:** Fail loudly for unrecoverable schema/contract errors; silently handle per-record failures.

**Patterns:**
- Schema violations raise immediately with an informative `ValueError` including the offending column/field name:
  ```python
  raise ValueError(f"Investor data missing required columns: {missing}")
  raise ValueError(f"ETF data missing required columns: {missing}")
  raise FileNotFoundError(f"ETF file not found: {etf_file}. Run pipeline.py first.")
  ```
- Per-record errors in the ETF selection loop fall back silently to random sampling:
  ```python
  except Exception as e:
      if len(candidate_df) > 0:
          picked = np.random.choice(candidate_df['symbol'].values)
  ```
  (`src/allocations.py`, lines 234–239)
- API call failures in `src/etf_fetcher.py` catch `Exception`, print a message, and return `None` — callers check for `None` downstream
- Data validation functions never raise on row-level issues; they log to `ValidationReport.dropped_reasons` and return a cleaned DataFrame
- Asset-class floor violations raise `ValueError` (unrecoverable — training cannot proceed)

## Logging

- **No `logging` module used anywhere.** All progress and diagnostic output uses `print()`.
- Progress banners use `"=" * 60` separators with CAPS labels:
  ```python
  print("=" * 60)
  print("PHASE 3: MODEL TRAINING & COMPARISON")
  print("=" * 60)
  ```
- Step-level progress uses `"... [Module] Description..."` prefix inside `run_allocations_pipeline`
- Emoji prefixes (`✅`, `❌`, `⚠️`, `📥`, `🤖`) used in `pipeline.py` and `data_generation.py` for visual scanning in terminal output; not used inside `src/` library modules
- Validation output produced via `ValidationReport.to_markdown()` which is both `print()`-ed and saved to file

## Comments

**When to comment:**
- Long algorithmic blocks (ETF conflict resolution, label generation) use multi-line block comments explaining the design decision above the code, not inline (`src/allocations.py`, lines 147–172)
- Training scripts include rationale comments before each algorithm explaining *why* certain hyperparameters were chosen (`src/train_models.py`, 12–15 line comment blocks before each model)
- One-line `# ---` inline comments explain non-obvious logic (e.g., `# Reduce aggressive for low experience`)
- Constants in `src/validation_rules.py` use trailing `#` comments to explain the source of defaults

**Magic numbers:**
- ETF count bounds clipped to `[7, 13]` — hardcoded in `src/inference.py` line 209 and `src/train_models.py` line 402 without a named constant

## Module Design

**Structure:**
- Library modules (`src/data_validation.py`, `src/inference.py`, `src/llm.py`, `src/allocations.py`) define public functions/classes and are `import`-ed by other modules
- Pipeline scripts (`src/data_generation.py`, `src/prepare_data.py`, `src/train_models.py`) run top-to-bottom when executed directly; they are also callable via `subprocess.run()` from `pipeline.py`
- `src/validation_rules.py` is a pure constants module — no functions, only data structures

**Exports:**
- No `__all__` defined in any module; all public names are implicitly exported
- Public API of `src/inference.py`: `recommend`, `validate_input`, `print_recommendation`
- Public API of `src/data_validation.py`: `validate_investor_data`, `validate_etf_data`, `ValidationReport`
- Public API of `src/llm.py`: `chat`, `get_initial_explanation`, `get_tool_followup`, `apply_adjust_allocation`, `build_system_prompt`

**Singleton lazy-loading pattern** in `src/inference.py`:
```python
_ARTIFACTS = None

def _get_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = _load_artifacts()
    return _ARTIFACTS
```
This caches models in memory after the first `import`, avoiding repeated disk reads on every `recommend()` call.

---

*Convention analysis: 2026-05-18*
