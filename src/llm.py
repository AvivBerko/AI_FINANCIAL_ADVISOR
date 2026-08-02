"""
LLM Advisory Layer — OpenAI gpt-4o-mini
========================================
Provides:
  - Initial portfolio explanation (called once after ML pipeline runs)
  - Ongoing chat with tool-calling support for portfolio adjustments

Three tools:
  adjust_allocation      — apply signed deltas to equity/bond/alt weights (nuance requests)
  set_allocation         — pin one asset class to an exact target percent (explicit requests)
  update_investor_profile — update form fields and re-run full ML pipeline (factual corrections)
"""

import json
import os

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "adjust_allocation",
            "description": (
                "Fine-tunes the current portfolio's asset allocation weights by applying "
                "signed percentage-point deltas. Use this when the user wants a nuanced "
                "shift in risk or exposure (e.g., 'a bit more equity', 'reduce bonds "
                "slightly', 'I want to take a bit more risk'). The three deltas should "
                "roughly sum to 0. Each delta is a float between -0.40 and +0.40."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equity_delta": {
                        "type": "number",
                        "description": "Change to equity weight, e.g. +0.08 means +8 percentage points.",
                    },
                    "bond_delta": {
                        "type": "number",
                        "description": "Change to bond weight, e.g. -0.05 means -5 percentage points.",
                    },
                    "alt_delta": {
                        "type": "number",
                        "description": "Change to alternatives weight.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief human-readable reason for the adjustment, shown to the user.",
                    },
                },
                "required": ["equity_delta", "bond_delta", "alt_delta", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_allocation",
            "description": (
                "Pins one asset class to an exact target percentage of the portfolio. Use "
                "this when the user names a specific number for one class (e.g., 'make "
                "bonds 20%', 'set equity to 60%', 'I want exactly 15% in alternatives'). "
                "The other two classes absorb the remainder proportionally to their "
                "current relative weights. Do NOT use this for vague/relative requests "
                "('a bit more', 'reduce slightly') — use adjust_allocation for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_class": {
                        "type": "string",
                        "enum": ["equity", "bond", "alt"],
                        "description": "Which asset class the target percentage applies to.",
                    },
                    "target_pct": {
                        "type": "number",
                        "description": "Target weight as a fraction, e.g. 0.20 means 20%.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief human-readable reason for the adjustment, shown to the user.",
                    },
                },
                "required": ["asset_class", "target_pct", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_investor_profile",
            "description": (
                "Updates one or more of the investor's factual profile fields and re-runs "
                "the full ML pipeline to generate a new baseline portfolio. Use ONLY when "
                "the user explicitly corrects a factual input (e.g., 'my horizon is "
                "actually 30 years', 'change my risk tolerance to High', 'I earn $120k "
                "not $80k'). Do NOT use for soft preference adjustments — use "
                "adjust_allocation instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "updates": {
                        "type": "object",
                        "description": (
                            "Partial dict of investor profile fields to update. "
                            "Valid keys: age, Gender, Education, MaritalStatus, "
                            "HouseholdSize, income, InvestmentGoal, horizon, "
                            "InvestmentCapital, risk_tolerance, FinancialInvolvement, "
                            "experience. Only include the fields that are changing."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this profile change is being made.",
                    },
                },
                "required": ["updates", "reason"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
def build_system_prompt(
    user_inputs: dict,
    allocation: dict,
    portfolio: dict,
    risk_profile: str,
) -> str:
    eq_tickers  = ", ".join(portfolio.get("equity",      []))
    bd_tickers  = ", ".join(portfolio.get("bond",        []))
    alt_tickers = ", ".join(portfolio.get("alternative", []))

    return f"""You are a professional, friendly AI financial advisor embedded in a personal ETF \
portfolio recommendation tool.

## Current Investor Profile
- Age: {user_inputs['age']} | Gender: {user_inputs['Gender']} | Marital Status: {user_inputs['MaritalStatus']}
- Education: {user_inputs['Education']} | Household Size: {user_inputs['HouseholdSize']}
- Annual Income: ${user_inputs['income']:,} | Investment Capital: ${user_inputs['InvestmentCapital']:,}
- Investment Goal: {user_inputs['InvestmentGoal']} | Horizon: {user_inputs['horizon']} years
- Risk Tolerance: {user_inputs['risk_tolerance']} | Financial Involvement: {user_inputs['FinancialInvolvement']}
- Years of Experience: {user_inputs['experience']}

## Current Portfolio
- Risk Profile: {risk_profile}
- Allocation: Equity {allocation['equity_pct']*100:.1f}% | Bonds {allocation['bond_pct']*100:.1f}% | Alternatives {allocation['alt_pct']*100:.1f}%
- Equity ETFs: {eq_tickers}
- Bond ETFs: {bd_tickers}
- Alternative ETFs: {alt_tickers}

## Your Role
You explain, justify, and refine this portfolio in conversation with the investor.
- Be concise and professional. Avoid jargon without explanation.
- When the user wants to tweak risk or weights softly ("a bit more", "slightly less", \
"I want to be a bit more aggressive"), call `adjust_allocation` with deltas that roughly sum to 0.
- When the user names an exact target percentage for one asset class ("make bonds 20%", \
"set equity to 60%"), call `set_allocation` with that class and target_pct.
- When the user explicitly corrects a profile fact ("my horizon is 30 years, not 10", \
"change my risk tolerance to High"), call `update_investor_profile`.
- Never fabricate financial data or performance figures. Only reference the ETFs and \
allocations shown above.
- Do not provide legal or tax advice."""


# ---------------------------------------------------------------------------
# Initial explanation (called once after form submit)
# ---------------------------------------------------------------------------
def get_initial_explanation(user_inputs: dict, result: dict) -> str:
    """Generate a friendly explanation of the freshly generated ML portfolio."""
    system = build_system_prompt(
        user_inputs,
        result["allocation"],
        result["portfolio"],
        result["risk_profile"],
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": "Please explain my portfolio recommendation."},
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Chat turn
# ---------------------------------------------------------------------------
def chat(
    chat_history: list,
    user_inputs: dict,
    allocation: dict,
    portfolio: dict,
    risk_profile: str,
) -> tuple[str | None, dict | None]:
    """
    Send the full chat history to the LLM and return (response_text, tool_call_info).

    Returns
    -------
    (text, None)       — normal assistant response
    (None, tool_info)  — the model wants to call a tool; tool_info keys:
                           id, name, arguments (dict), assistant_msg (dict for chat_history)
    """
    system = build_system_prompt(user_inputs, allocation, portfolio, risk_profile)
    messages = [{"role": "system", "content": system}] + chat_history

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=600,
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        # Serialise the assistant tool-call message so it can be stored in session_state
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,  # keep as raw JSON string
                    },
                }
            ],
        }
        return None, {
            "id":            tc.id,
            "name":          tc.function.name,
            "arguments":     json.loads(tc.function.arguments),
            "assistant_msg": assistant_msg,
        }

    return msg.content, None


# ---------------------------------------------------------------------------
# Follow-up after tool execution
# ---------------------------------------------------------------------------
def get_tool_followup(
    chat_history: list,
    user_inputs: dict,
    allocation: dict,
    portfolio: dict,
    risk_profile: str,
) -> str:
    """
    Called after a tool result has been appended to chat_history.
    Returns the LLM's natural-language explanation of the change.
    chat_history should end with: [..., user_msg, assistant_tool_call_msg, tool_result_msg]
    """
    system = build_system_prompt(user_inputs, allocation, portfolio, risk_profile)
    messages = [{"role": "system", "content": system}] + chat_history

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=400,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Allocation math helper
# ---------------------------------------------------------------------------
def apply_adjust_allocation(
    current_allocation: dict,
    equity_delta: float,
    bond_delta: float,
    alt_delta: float,
) -> dict:
    """Apply deltas and re-normalise so the result always sums to 1.0."""
    eq  = max(0.01, current_allocation["equity_pct"] + equity_delta)
    bd  = max(0.01, current_allocation["bond_pct"]   + bond_delta)
    alt = max(0.01, current_allocation["alt_pct"]    + alt_delta)
    total = eq + bd + alt
    return {
        "equity_pct": round(eq  / total, 4),
        "bond_pct":   round(bd  / total, 4),
        "alt_pct":    round(alt / total, 4),
    }


def apply_set_allocation(
    current_allocation: dict,
    asset_class: str,
    target_pct: float,
) -> dict:
    """Pin one asset class to an exact target; the other two split the remainder
    proportionally to their current relative weights (even split if both are 0)."""
    keys = {"equity": "equity_pct", "bond": "bond_pct", "alt": "alt_pct"}
    target_key = keys[asset_class]
    other_keys = [k for k in keys.values() if k != target_key]

    target = min(1.0, max(0.0, target_pct))
    remaining = 1.0 - target

    other_sum = sum(current_allocation[k] for k in other_keys)
    if other_sum <= 0:
        split = {k: remaining / 2 for k in other_keys}
    else:
        split = {
            k: remaining * (current_allocation[k] / other_sum)
            for k in other_keys
        }

    result = {target_key: target, **split}
    total = sum(result.values())
    return {k: round(v / total, 4) for k, v in result.items()}
