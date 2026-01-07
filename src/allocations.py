import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION: MUTUALLY EXCLUSIVE GROUPS
# =============================================================================
# If the code picks ONE of these, it banishes the others for that specific user.
CONFLICT_GROUPS = [
    ['VOO', 'SPY', 'IVV', 'SPLG'],         # S&P 500 Clones
    ['QQQ', 'QQQM', 'TQQQ'],               # Nasdaq 100 Clones
    ['VTI', 'ITOT', 'SCHB', 'VTV'],        # Total US Market (Broad)
    ['BND', 'AGG', 'BNDX', 'BIV'],         # Broad Bond Funds
    ['GLD', 'IAU', 'GLDM', 'SLV'],         # Precious Metals
    ['VEA', 'IEFA', 'EFA', 'IDEV'],        # Developed Markets
    ['VWO', 'IEMG', 'EEM'],                # Emerging Markets
    ['VNQ', 'IYR', 'XLRE']                 # US Real Estate
]
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_profile_probabilities(investor):
    """Convert investor features to P(Aggressive/Moderate/Conservative)"""
    age, horizon, risk_tol, income, experience = (
        investor.get('age'), investor.get('horizon'), investor.get('risk_tolerance'),
        investor.get('income'), investor.get('experience')
    )

    # Age-based probabilities
    if pd.isna(age):
        age_probs = [0.33, 0.33, 0.33]
    elif age < 35:
        age_probs = [0.6, 0.3, 0.1]
    elif age <= 50:
        age_probs = [0.3, 0.5, 0.2]
    else:
        age_probs = [0.1, 0.3, 0.6]

    # Horizon-based
    if pd.isna(horizon):
        horizon_probs = [0.33, 0.33, 0.33]
    elif horizon > 10:
        horizon_probs = [0.7, 0.2, 0.1]
    elif horizon >= 5:
        horizon_probs = [0.4, 0.4, 0.2]
    else:
        horizon_probs = [0.1, 0.4, 0.5]

    # Risk tolerance mapping
    risk_map = {'High': [0.7, 0.2, 0.1], 'Medium': [0.2, 0.6, 0.2], 'Low': [0.1, 0.3, 0.6]}
    risk_probs = risk_map.get(risk_tol, [0.33, 0.33, 0.33])

    # Income mapping
    income_map = {'High': [0.6, 0.3, 0.1], 'Medium': [0.3, 0.5, 0.2], 'Low': [0.2, 0.4, 0.4]}
    income_probs = income_map.get(income, [0.33, 0.33, 0.33])

    # Combine
    exp_factor = min(1.0, max(0.5, 1.0 - experience / 50)) if pd.notna(experience) else 1.0
    final_probs = (np.array(age_probs) + np.array(horizon_probs) +
                   np.array(risk_probs) + np.array(income_probs)) / 4
    final_probs[0] *= exp_factor  # Reduce aggressive for low experience

    # Normalize
    if final_probs.sum() > 0:
        final_probs = final_probs / final_probs.sum()
    else:
        final_probs = [0.33, 0.33, 0.34]

    return dict(zip(['Aggressive', 'Moderate', 'Conservative'], final_probs))


def get_allocation_mix(profile):
    """
    Generates a unique, CONTINUOUS allocation mix for every investor.
    Instead of picking from 3 presets, we add random variation (noise) to a base profile.
    """
    # 1. Define the "Ideal" Center for each profile
    targets = {
        'Aggressive': {'equity': 0.80, 'bond': 0.10, 'alternative': 0.10},
        'Moderate': {'equity': 0.55, 'bond': 0.35, 'alternative': 0.10},
        'Conservative': {'equity': 0.35, 'bond': 0.55, 'alternative': 0.10}
    }

    # Default to Moderate if profile is missing
    base = targets.get(profile, targets['Moderate'])

    # 2. Create continuous variations (Gaussian Noise)
    # Scale=0.05 means ~68% of investors will be within +/- 5% of the target.
    # We use less variation (0.02) for alternatives since the allocation is small.
    raw_eq = np.random.normal(loc=base['equity'], scale=0.05)
    raw_bd = np.random.normal(loc=base['bond'], scale=0.05)
    raw_alt = np.random.normal(loc=base['alternative'], scale=0.02)

    # 3. Clip to ensure positive values (no negative allocations)
    # We enforce a tiny minimum (1%) so no category disappears entirely
    eq = max(0.01, raw_eq)
    bd = max(0.01, raw_bd)
    alt = max(0.01, raw_alt)

    # 4. Normalize so they sum exactly to 1.0 (100%)
    total_weight = eq + bd + alt

    return {
        'equity': eq / total_weight,
        'bond': bd / total_weight,
        'alternative': alt / total_weight
    }


def get_num_etfs(asset_class, profile):
    """Determine how many ETFs per asset class."""
    if asset_class == 'Equity':
        return np.random.choice([4, 5, 6], p=[0.2, 0.6, 0.2])
    elif asset_class == 'Bond':
        if profile == 'Aggressive':
            return np.random.choice([2, 3], p=[0.6, 0.4])
        else:
            return np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
    else:  # Alternative
        return 1 if np.random.random() > 0.3 else 2


def get_category_mapping(category):
    """
    Map ETF categories -> asset classes.
    (Your original logic)
    """
    category = str(category).lower()

    if any(x in category for x in ['large', 'mid', 'small', 'growth', 'value', 'blend']):
        return 'equity'
    elif any(x in category for x in ['bond', 'corporate', 'treasury', 'government']):
        return 'bond'
    elif any(x in category for x in ['real estate', 'reit', 'commodities', 'gold']):
        return 'alternative'
    else:
        return 'equity'



# =============================================================================
# HOW CONFLICT RESOLUTION WORKS
# =============================================================================
#
# THE PROBLEM:
# In the original version, a user could randomly get both 'VOO' and 'SPY'.
# Since these are identical S&P 500 funds, holding both is redundant/silly.
#
# THE SOLUTION:
# We implemented a "Pick One, Ban the Rest" logic using two parts:
#
# 1. THE "NO-FLY LIST" (CONFLICT_GROUPS):
#    - We grouped identical ETFs into families (e.g., [VOO, SPY, IVV]).
#    - If one member of the family is picked, the others become "illegal"
#      for this specific investor.
#
# 2. THE "NUKE" MECHANISM (inside assign_portfolio...):
#    - We create a temporary copy of the ETF list (`candidate_df`) for each user.
#    - STEP A: Algorithm picks a winner (e.g., 'VOO').
#    - STEP B: We immediately remove 'VOO' from the pool (can't pick it twice).
#    - STEP C: We check if 'VOO' has siblings in `CONFLICT_GROUPS`.
#    - STEP D: If yes, we identify the siblings ('SPY', 'IVV') and REMOVE them
#      from `candidate_df` instantly.
#    - Result: The algorithm literally cannot see 'SPY' anymore for this user.
#
# CRITICAL DETAIL:
# We use `candidate_df = full_df.copy()` at the start of the loop.
# This ensures that banning SPY for User A does not ban it for User B.
# =============================================================================
def assign_portfolio_numeric_aum(investor_row, etf_by_class):
    """
    Selects specific ETFs iteratively to avoid conflicts (e.g., No VOO + SPY).
    """
    portfolio = {}
    asset_mapping = {
        'equity': 'num_equity_etfs',
        'bond': 'num_bond_etfs',
        'alternative': 'num_alt_etfs'
    }

    for asset_class, n_etfs_key in asset_mapping.items():
        # Get the pool of available ETFs
        full_df = etf_by_class.get(asset_class)
        if full_df is None or len(full_df) == 0: continue

        # Working copy so we can remove conflicts as we go
        candidate_df = full_df.copy()

        n_slots = int(investor_row.get(n_etfs_key, 0))
        selected_tickers = []

        for _ in range(n_slots):
            if len(candidate_df) == 0: break  # No more options

            # --- Recalculate Weights for current pool ---
            try:
                max_aum = candidate_df['AUM'].max()
                aum_power = np.power(candidate_df['AUM'] / max_aum, 1.5)

                if asset_class == 'equity':
                    # Fix: Handle volatility being 0 to avoid division by zero
                    vol = candidate_df['volatility'] + 0.01
                    quality = candidate_df['star_rating'] / vol
                else:
                    quality = candidate_df['star_rating']

                weights = aum_power * quality

                # Safety check for weights
                if weights.sum() == 0:
                    weights = None
                else:
                    weights = weights / weights.sum()  # Normalize

                # Pick ONE symbol
                picked_symbol = np.random.choice(candidate_df['symbol'].values, p=weights)
                selected_tickers.append(picked_symbol)

                # --- CONFLICT RESOLUTION ---
                # 1. Remove the picked symbol itself so we don't pick it twice
                candidate_df = candidate_df[candidate_df['symbol'] != picked_symbol]

                # 2. Check if this symbol triggers a group exclusion
                for group in CONFLICT_GROUPS:
                    if picked_symbol in group:
                        # Identify siblings to ban (e.g. if VOO picked, ban SPY, IVV)
                        siblings_to_ban = [s for s in group if s != picked_symbol]
                        # Remove them from the candidate pool
                        candidate_df = candidate_df[~candidate_df['symbol'].isin(siblings_to_ban)]

            except Exception as e:
                # Fallback if math fails (just pick random)
                if len(candidate_df) > 0:
                    picked = np.random.choice(candidate_df['symbol'].values)
                    selected_tickers.append(picked)
                    candidate_df = candidate_df[candidate_df['symbol'] != picked]

        portfolio[asset_class] = selected_tickers

    return portfolio


# =============================================================================
# MAIN EXECUTOR (Called by main.py)
# =============================================================================

def run_allocations_pipeline(etf_df, investors_df):
    """
    The orchestrator that takes two DataFrames and returns the final mapped portfolio.
    """
    print("... [Allocations] Preparing ETF Dictionary...")

    # 1. Clean & Prepare ETF Data
    etf_features = etf_df.copy()

    col_map = {
        'Fund Symbol': 'symbol', 'Category': 'category',
        'Assets Under Management (AUM)': 'AUM',
        'custom_star_rating': 'star_rating', 'Volatility (Annual STD)': 'volatility'
    }
    etf_features.rename(columns=col_map, inplace=True)

    # Ensure numeric types
    etf_features['AUM'] = pd.to_numeric(etf_features['AUM'], errors='coerce').fillna(1e9)
    etf_features['star_rating'] = pd.to_numeric(etf_features['star_rating'], errors='coerce').fillna(3)
    etf_features['volatility'] = pd.to_numeric(etf_features['volatility'], errors='coerce').fillna(0.15)

    # Map to Asset Class
    etf_features['asset_class'] = etf_features['category'].apply(get_category_mapping)

    # Create the Dictionary
    etf_by_class = {}
    for cls in ['equity', 'bond', 'alternative']:
        class_mask = etf_features['asset_class'] == cls
        class_df = etf_features[class_mask].copy()

        print(f"    -> Found {len(class_df)} {cls.capitalize()} ETFs")
        if len(class_df) == 0 and cls == 'alternative':
            print(
                f"⚠️ NOTE: No ETFs matched the 'alternative' keywords: ['real estate', 'reit', 'commodities', 'gold']")

        if len(class_df) > 0:
            etf_by_class[cls] = class_df

    print(f"... [Allocations] Processing {len(investors_df)} Investors...")

    # 2. Prepare Investor Data
    inv_df = investors_df.copy()
    inv_rename = {
        'Age': 'age', 'InvestmentHorizon': 'horizon', 'RiskTolerance': 'risk_tolerance',
        'Income': 'income', 'ExperienceYears': 'experience'
    }
    inv_df.rename(columns=inv_rename, inplace=True)

    # 3. Run Logic Pipeline
    inv_df['profile_probs'] = inv_df.apply(get_profile_probabilities, axis=1)
    inv_df['risk_profile'] = inv_df['profile_probs'].apply(
        lambda p: np.random.choice(['Aggressive', 'Moderate', 'Conservative'], p=list(p.values()))
    )

    # --- CONTINUOUS ALLOCATION APPLIED HERE ---
    inv_df['allocation'] = inv_df['risk_profile'].apply(get_allocation_mix)

    inv_df['num_equity_etfs'] = inv_df.apply(lambda r: get_num_etfs('Equity', r['risk_profile']), axis=1)
    inv_df['num_bond_etfs'] = inv_df.apply(lambda r: get_num_etfs('Bond', r['risk_profile']), axis=1)
    inv_df['num_alt_etfs'] = inv_df.apply(lambda r: get_num_etfs('Alternative', r['risk_profile']), axis=1)
    inv_df['total_etfs'] = inv_df['num_equity_etfs'] + inv_df['num_bond_etfs'] + inv_df['num_alt_etfs']

    # 4. Assign Portfolios
    print("... [Allocations] Assigning Specific ETFs...")
    inv_df['portfolio_etfs'] = inv_df.apply(lambda row: assign_portfolio_numeric_aum(row, etf_by_class), axis=1)

    return inv_df