# LLM Layer — Architecture & Design

## Overview

The LLM layer sits on top of the existing 4-stage ML pipeline. It does not collect user data (the Streamlit form handles that) and does not override the ML models. Instead it:

1. Explains the ML-generated portfolio in plain language once it's ready.
2. Hosts an ongoing chat where the user can debate, question, or refine the recommendation.
3. Calls one of two tools when the user wants to adjust the portfolio.

---

## Architectural Decision: Fine-tuning vs. Profile Switching

**The problem:** When a user says "I want to take a bit more risk", switching their `risk_tolerance` from `"Medium"` to `"High"` re-runs the full ML pipeline — changing the risk profile label, re-clipping the PROFILE_BOUNDS in `inference.py`, re-sampling ETF counts, and re-selecting tickers entirely. That is a sledgehammer for a nuanced request.

**Solution: Two separate tools with distinct semantics.**

### Tool A — `adjust_allocation` (for nuance)
Takes **signed deltas** in percentage points and applies them directly on top of the existing ML output. Re-normalizes so the three weights always sum to 1.0. Only the allocation percentages change — ETF tickers remain stable.

```
"I want a bit more risk"           → equity_delta=+0.08, bond_delta=-0.08, alt_delta=0.0
"Reduce my bond exposure slightly" → equity_delta=+0.05, bond_delta=-0.07, alt_delta=+0.02
"Much more aggressive"             → equity_delta=+0.20, bond_delta=-0.15, alt_delta=-0.05
```

**Why deltas and not absolute values?** The LLM does not need to remember the current weights. It only describes *direction and magnitude*. This is less error-prone and more natural for conversational requests.

**When the LLM should call this:** Any qualitative desire to shift risk or exposure — "a bit", "slightly", "more aggressive", "reduce bonds".

### Tool B — `update_investor_profile` (for factual corrections)
Accepts a partial dict of investor profile fields and re-runs the full `recommend()` pipeline from scratch. Resets all allocation overrides.

**When the LLM should call this:** Only when the user explicitly corrects a factual input — "my horizon is actually 30 years, not 10", "change my risk tolerance to High", "I actually earn $120k".

**Decision rule encoded in the system prompt:**
- Qualitative preference tweak → `adjust_allocation`
- Explicit factual correction to a form field → `update_investor_profile`

---

## `st.session_state` Structure

```python
# Populated when form is submitted
st.session_state["user_inputs"]           = {}     # dict: the 12 form fields verbatim
st.session_state["ml_result"]             = {}     # dict: raw frozen output from recommend()

# The "live" view shown in the UI (may diverge from ml_result if Tool A was called)
st.session_state["current_allocation"]    = {}     # {equity_pct, bond_pct, alt_pct}
st.session_state["current_portfolio"]     = {}     # {equity: [...], bond: [...], alternative: [...]}
st.session_state["risk_profile"]          = ""     # str: Aggressive / Moderate / Conservative

# LLM chat
st.session_state["chat_history"]          = []     # Full OpenAI messages format (includes tool artifacts)
st.session_state["display_messages"]      = []     # Only {role, content} pairs shown in the UI

# UI flags
st.session_state["portfolio_ready"]       = False  # Controls whether the chat section renders
st.session_state["allocation_overridden"] = False  # Shows a banner when Tool A has fired
```

**Why separate `ml_result` from `current_allocation`?**
`ml_result` is the frozen ML baseline — it never changes unless Tool B fires. `current_allocation` is what the UI renders. When Tool A fires, only `current_allocation` is mutated. This lets the LLM show: *"Your ML-recommended allocation was 55% equity, but we've adjusted it to 63% based on your preference."*

**Why separate `chat_history` from `display_messages`?**
`chat_history` contains the full OpenAI messages sequence including tool_call assistant messages and tool result messages (role: "tool"), which the API requires for context but should never be rendered in the UI. `display_messages` contains only what the user sees.

---

## Tool Definitions (OpenAI function calling schema)

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "adjust_allocation",
            "description": (
                "Fine-tunes the current portfolio's asset allocation weights "
                "by applying signed percentage-point deltas. Use this when the user "
                "wants a nuanced shift in risk/exposure (e.g., 'a bit more equity', "
                "'reduce bonds slightly'). The three deltas must sum to 0."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equity_delta": {
                        "type": "number",
                        "description": "Change to equity weight, e.g. +0.08 means +8 percentage points."
                    },
                    "bond_delta": {
                        "type": "number",
                        "description": "Change to bond weight, e.g. -0.05 means -5 percentage points."
                    },
                    "alt_delta": {
                        "type": "number",
                        "description": "Change to alternatives weight."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief human-readable reason for the adjustment."
                    }
                },
                "required": ["equity_delta", "bond_delta", "alt_delta", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_investor_profile",
            "description": (
                "Updates one or more investor profile fields and re-runs the full ML pipeline "
                "to generate a new baseline portfolio. Use ONLY when the user explicitly corrects "
                "a factual input (e.g., 'my horizon is actually 30 years', 'change my risk "
                "tolerance to High'). Do NOT use for soft preference adjustments — use "
                "adjust_allocation instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "updates": {
                        "type": "object",
                        "description": (
                            "Partial dict of investor profile fields to update. "
                            "Valid keys: age, Gender, Education, MaritalStatus, HouseholdSize, "
                            "income, InvestmentGoal, horizon, InvestmentCapital, "
                            "risk_tolerance, FinancialInvolvement, experience."
                        )
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this profile change is being made."
                    }
                },
                "required": ["updates", "reason"]
            }
        }
    }
]
```

---

## File Layout

```
src/llm.py          ← OpenAI client, system prompt builder, chat(), tool dispatch helpers
app.py              ← Streamlit UI: form + portfolio display + chat interface
```

## Environment

Set your OpenAI key before running:
```bash
export OPENAI_API_KEY="sk-..."
# or add to .env (already gitignored via python-dotenv)
```
