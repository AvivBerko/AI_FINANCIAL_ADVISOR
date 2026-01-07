import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt


np.random.seed(42)
N = 5000

# --- Feature 1: Age ---
ages = []
for _ in range(N):
    u = np.random.uniform(0, 1)
    if u < 0.3:
        ages.append(np.random.randint(18, 35))
    elif u < 0.75:
        ages.append(np.random.randint(35, 55))
    else:
        ages.append(np.random.randint(55, 91))

# --- Feature 2: Gender ---
genders = np.random.choice(["Male", "Female"], size=N, p=[0.6, 0.4])

# --- Feature 3: Education ---
educations = np.random.choice(
    ["High school", "Bachelor", "Master"],
    size=N,
    p=[0.35, 0.45, 0.20]
)

# --- Feature 4: Marital Status ---
maritals = np.random.choice(
    ["Married", "Single", "Divorced", "Widowed"],
    size=N,
    p=[0.72, 0.20, 0.06, 0.02]
)

# --- Feature 5: Investment Goal (correlated with age)
investment_goals = []
for age in ages:
    if age < 30:
        goals_probs = [0.30, 0.30, 0.25, 0.15]
    elif age < 50:
        goals_probs = [0.55, 0.20, 0.15, 0.10]
    else:
        goals_probs = [0.75, 0.10, 0.05, 0.10]
    investment_goals.append(np.random.choice(["Retirement", "Home purchase", "Child education", "Other"], p=goals_probs))

# --- Feature 6: Household Size (correlated with marital & goal) ---
household_size = []
for marital, goal in zip(maritals, investment_goals):
    if marital == "Single":
        household_size.append(np.random.choice([1, 2], p=[0.85, 0.15]))
    elif marital == "Married":
        if goal == "Child education":
            household_size.append(np.random.choice([3, 4], p=[0.7, 0.3]))
        else:
            household_size.append(np.random.choice([2, 3, 4], p=[0.4, 0.35, 0.25]))
    else:
        household_size.append(np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))

# --- Feature 7: Income (age, education, household size) ---
incomes = []
for age, edu, hsize in zip(ages, educations, household_size):
    if age < 25:
        base_income = np.random.randint(20000, 40000)
    elif age < 35:
        base_income = np.random.randint(30000, 70000)
    elif age < 50:
        base_income = np.random.randint(50000, 150000)
    else:
        base_income = np.random.randint(40000, 120000)
    if edu == "Bachelor":
        base_income *= np.random.uniform(1.10, 1.30)
    elif edu == "Master":
        base_income *= np.random.uniform(1.30, 1.55)
    income = base_income / (1 + 0.15 * max(hsize - 1, 0))
    incomes.append(int(min(income, 300000)))


# --- Feature 8: Investment Time Horizon (goal, age, household size) ---
time_horizon = []
for goal, age, hsize in zip(investment_goals, ages, household_size):
    if goal == "Retirement":
        if age > 50:
            horizon = np.random.randint(3, 15)
        else:
            horizon = np.random.randint(5, 40)
    elif goal == "Child education":
        horizon = np.random.randint(3, 18)
    elif goal == "Home purchase":
        horizon = np.random.randint(2, 10)
    else:
        horizon = np.random.randint(1, 10)
    time_horizon.append(horizon)

# --- Feature 9: Investment Capital Size (income, household size, age) ---
capitals = []
for income, hsize, age in zip(incomes, household_size, ages):
    if income < 50000:
        base_capital = np.random.randint(2000, 30000)
    elif income < 100000:
        base_capital = np.random.randint(10000, 80000)
    elif income < 150000:
        base_capital = np.random.randint(30000, 250000)
    else:
        base_capital = np.random.randint(80000, 1000000)
    # Older and smaller household, generally higher investable capital
    adj = 1 + 0.15 * (hsize - 1) - 0.12 * ((age - 35) / 80)  # Age/ household correction
    capital = base_capital / max(adj, 0.7)
    capitals.append(int(capital))

# --- Feature 10 financial involvement below, then use it here.
fin_involve_levels = ["Low", "Medium", "High"]
fin_involve_probs = [0.20, 0.48, 0.32]
financial_involvement = np.random.choice(fin_involve_levels, size=N, p=fin_involve_probs)

# --- Feature 11: Risk Tolerance (SCORE-BASED WITH QUANTILE BINNING) ---
# STRATEGY: Calculate a continuous risk score based on investor characteristics,
# then use pd.qcut to bin into 3 equal groups. This preserves correlations while
# ensuring perfect class balance.

def calculate_risk_score(age, income, capital, horizon, edu, fin_inv, marital):
    """
    Calculate a continuous risk score (0-100) based on investor characteristics.
    Higher score = More aggressive investor profile.
    """
    score = 0
    
    # 1. Age Component (0-25 points) - Younger = More aggressive
    if age < 30:
        score += 25
    elif age < 40:
        score += 20
    elif age < 50:
        score += 15
    elif age < 60:
        score += 10
    else:
        score += 5
    
    # 2. Income Component (0-20 points) - Higher income = More capacity for risk
    if income >= 150000:
        score += 20
    elif income >= 100000:
        score += 15
    elif income >= 70000:
        score += 10
    elif income >= 40000:
        score += 5
    else:
        score += 0
    
    # 3. Capital Component (0-20 points) - Larger portfolios can absorb volatility
    if capital >= 500000:
        score += 20
    elif capital >= 200000:
        score += 15
    elif capital >= 80000:
        score += 10
    elif capital >= 30000:
        score += 5
    else:
        score += 0
    
    # 4. Horizon Component (0-15 points) - Long horizons allow aggressive strategies
    if horizon >= 20:
        score += 15
    elif horizon >= 10:
        score += 10
    elif horizon >= 5:
        score += 5
    else:
        score += 0
    
    # 5. Education Component (0-10 points) - Financial literacy enables risk understanding
    edu_scores = {"Master": 10, "Bachelor": 6, "High school": 3}
    score += edu_scores.get(edu, 3)
    
    # 6. Financial Involvement Component (0-10 points) - Active investors can manage risk
    fin_inv_scores = {"High": 10, "Medium": 5, "Low": 0}
    score += fin_inv_scores.get(fin_inv, 0)
    
    # 7. Marital Status Modifier (-5 to +5 points) - Dependents affect risk capacity
    marital_scores = {"Single": 5, "Married": 0, "Divorced": -2, "Widowed": -5}
    score += marital_scores.get(marital, 0)
    
    return score

# Calculate risk scores for all investors
print("\n=== Calculating Risk Scores ===")
risk_scores = []
for age, income, cap, horizon, edu, fin_inv, marital in zip(
    ages, incomes, capitals, time_horizon, educations, financial_involvement, maritals
):
    score = calculate_risk_score(age, income, cap, horizon, edu, fin_inv, marital)
    risk_scores.append(score)

# Create temporary DataFrame for binning
temp_df = pd.DataFrame({'risk_score': risk_scores})

# Add tiny random noise to break ties (ensures unique scores for perfect binning)
# Noise is small enough (0-0.01) to not affect the ranking meaningfully
temp_df['risk_score_jittered'] = temp_df['risk_score'] + np.random.uniform(0, 0.01, size=len(temp_df))

# Use quantile cut to create 3 equal-sized bins
# This guarantees exactly 33% in each category
temp_df['risk_profile'] = pd.qcut(
    temp_df['risk_score_jittered'],
    q=3,  # 3 quantiles = 3 equal groups
    labels=['Low', 'Medium', 'High']  # Bottom 33%, Middle 33%, Top 33%
)

# Extract the risk_tolerance list
risk_tolerance = temp_df['risk_profile'].tolist()

# Verify balance
print("\n=== Risk Tolerance Distribution (Score-Based) ===")
unique, counts = np.unique(risk_tolerance, return_counts=True)
for cat, count in zip(unique, counts):
    print(f"{cat}: {count} ({count/N*100:.2f}%)")
print(f"Score Range: {min(risk_scores):.1f} - {max(risk_scores):.1f}")


# --- Feature 12: Experience (Age-constrained and edjucation)
experience_years = []
for age, edu in zip(ages, educations):
    max_years = max(age - 18, 0)
    if edu == "High school":
        max_exp = max_years
    elif edu == "Bachelor":
        max_exp = max(max_years - 4, 0)
    else:
        max_exp = max(max_years - 6, 0)
    experience_years.append(np.random.randint(0, max_exp + 1))

# --- Compose dataframe ---
df = pd.DataFrame({
    "Age": ages,
    "Gender": genders,
    "Education": educations,
    "MaritalStatus": maritals,
    "HouseholdSize": household_size,
    "Income": incomes,
    "InvestmentGoal": investment_goals,
    "InvestmentHorizon": time_horizon,
    "InvestmentCapital": capitals,
    "RiskTolerance": risk_tolerance,
    "FinancialInvolvement": financial_involvement,
    "ExperienceYears": experience_years
})

# --- Validation: Verify Class Balance ---
print("\n" + "="*60)
print("FINAL VALIDATION: Risk Tolerance Distribution")
print("="*60)
risk_dist = df['RiskTolerance'].value_counts(sort=False)
for category in ['High', 'Medium', 'Low']:
    count = risk_dist.get(category, 0)
    pct = count / len(df) * 100
    status = "✅" if 32.5 <= pct <= 33.5 else "⚠️"
    print(f"{status} {category:10s}: {count:5d} ({pct:5.2f}%)")

# --- Logical Consistency Check ---
print("\n" + "="*60)
print("Logical Consistency Verification")
print("="*60)

# Check: Young + High Income should trend Aggressive (High risk)
young_rich = df[(df['Age'] < 35) & (df['Income'] > 100000)]
if len(young_rich) > 0:
    print(f"Young (<35) + High Income (>100k): {len(young_rich)} investors")
    print(f"  Risk Profile Distribution:")
    for cat in ['High', 'Medium', 'Low']:
        count = (young_rich['RiskTolerance'] == cat).sum()
        pct = count / len(young_rich) * 100
        print(f"    {cat}: {pct:.1f}%")

# Check: Old + Low Income should trend Conservative (Low risk)
old_poor = df[(df['Age'] > 60) & (df['Income'] < 50000)]
if len(old_poor) > 0:
    print(f"\nOld (>60) + Low Income (<50k): {len(old_poor)} investors")
    print(f"  Risk Profile Distribution:")
    for cat in ['High', 'Medium', 'Low']:
        count = (old_poor['RiskTolerance'] == cat).sum()
        pct = count / len(old_poor) * 100
        print(f"    {cat}: {pct:.1f}%")

print("\n" + "="*60)
print("Summary Statistics")
print("="*60)
print(f"Total Investors: {len(df)}")
print(f"Age Range: {df['Age'].min()} - {df['Age'].max()}")
print(f"Income Range: ${df['Income'].min():,} - ${df['Income'].max():,}")
print(f"Capital Range: ${df['InvestmentCapital'].min():,} - ${df['InvestmentCapital'].max():,}")
print("="*60)

# --- Save and view ---
import os
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent.parent / 'data' / 'processed'
output_dir.mkdir(parents=True, exist_ok=True)

csvfilename = output_dir / f"synthetic_investor_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
df.to_csv(csvfilename, index=False)

print(f"\n✅ Data saved to: {csvfilename}")
print("\nFirst 10 rows:")
print(df.head(10))

