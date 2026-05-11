# Model Selection & Hyperparameter Tuning — Decision Log

**Project:** AI-Based Personal Financial Advisor  
**Author:** Aviv Berkovich  
**Stack:** scikit-learn, XGBoost, OpenAI gpt-4o-mini

---

## 0. Context: What the ML Pipeline Is Solving

The system takes a 12-field investor profile and produces a personalized ETF portfolio.
This is broken into three separate ML problems, each with a different output type:

| Stage | Problem Type | Output |
|-------|-------------|--------|
| 1 | Multi-class Classification | Risk Profile: Aggressive / Moderate / Conservative |
| 2 | Multi-output Regression | Equity % and Bond % allocation |
| 3 | Regression | Total number of ETFs in portfolio |

Each stage was treated as an independent supervised learning problem with its own
algorithm selection and evaluation strategy.

---

## 1. Data & Feature Engineering

### 1.1 Why Synthetic Data

No labeled real-world dataset of investor profiles with corresponding portfolio
allocations exists publicly. Real brokerage data is proprietary and contains
privacy-sensitive information. Synthetic data was generated to:

- Control class balance precisely (see Section 1.3)
- Encode known financial planning heuristics as ground-truth relationships
- Generate a dataset large enough (5,000 samples) for meaningful train/test splits

The trade-off is that the model learns a synthetic distribution. It cannot capture
behavioral biases, market panic responses, or edge cases that only appear in real
investor data.

### 1.2 The 12 Input Features

Features were chosen to reflect the standard inputs used in real financial advisory
questionnaires (e.g., Fidelity, Vanguard onboarding flows):

| Feature | Type | Rationale |
|---------|------|-----------|
| `age` | Numeric | Primary driver of investment horizon and risk capacity |
| `Gender` | Categorical | Demographic segmentation variable |
| `Education` | Categorical | Proxy for financial literacy and income ceiling |
| `MaritalStatus` | Categorical | Affects household expenses and risk capacity |
| `HouseholdSize` | Numeric | Larger households = higher fixed costs = lower risk capacity |
| `income` | Numeric | Determines ability to absorb portfolio losses |
| `InvestmentGoal` | Categorical | Retirement vs. Home Purchase implies different horizons |
| `horizon` | Numeric | Time to liquidation — most direct risk capacity signal |
| `InvestmentCapital` | Numeric | Portfolio size; larger portfolios can diversify more |
| `risk_tolerance` | Categorical | Self-reported tolerance; used as both feature and label source |
| `FinancialInvolvement` | Categorical | Active investors understand and tolerate volatility better |
| `experience` | Numeric | Years of investing experience; correlated with age, upper-bounded by it |

Features were designed with **intentional correlations** to mimic real-world behavior:

- `income` is driven by `age` and `Education` (older + more educated = higher income)
- `horizon` is driven by `InvestmentGoal` and `age` (retirement goal + young age = long horizon)
- `HouseholdSize` is driven by `MaritalStatus` and `InvestmentGoal` (married + child education = larger household)
- `experience` is upper-bounded by `age` minus years spent in education

These correlations make the dataset realistic, but they also mean some features
carry partially redundant information — a consideration when interpreting feature
importance scores.

### 1.3 Risk Profile Label Generation — Quantile Binning

The `risk_tolerance` label was not assigned randomly. A continuous **risk score (0–100)**
was computed for each investor from 7 components:

```python
def calculate_risk_score(age, income, capital, horizon, edu, fin_inv, marital):
    score = 0
    # Age (0-25 pts): younger investors have longer runways, score higher
    # Income (0-20 pts): higher income = more capacity to absorb losses
    # Capital (0-20 pts): larger portfolios can absorb drawdowns
    # Horizon (0-15 pts): longer horizons allow recovery from volatility
    # Education (0-10 pts): proxy for financial literacy
    # Financial Involvement (0-10 pts): active investors understand risk
    # Marital Status (-5 to +5 pts): dependents reduce risk capacity
    return score
```

The score was then binned into 3 equal groups using `pd.qcut`:

```python
temp_df['risk_profile'] = pd.qcut(
    temp_df['risk_score_jittered'],
    q=3,
    labels=['Low', 'Medium', 'High']
)
```

**Why `pd.qcut` instead of fixed thresholds (`pd.cut`)?**
`pd.cut` with fixed boundaries (e.g., 0–33, 34–66, 67–100) would produce unequal
class sizes depending on the score distribution. `pd.qcut` divides by percentile,
guaranteeing exactly 33.3% per class. This is critical because:

1. A balanced dataset prevents any classifier from gaming accuracy by predicting
   the majority class
2. Macro F1 and accuracy become equivalent for balanced classes, making results
   easier to interpret
3. The stratified train/test split (`stratify=y1_risk_profile`) then preserves
   this balance in both the training and test sets

### 1.4 Feature Preprocessing: StandardScaler vs OneHotEncoder

```python
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),          # 6 features
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),  # 6 → 13 features
])
```

**Numerical features → StandardScaler**

The 6 numerical features (`age`, `HouseholdSize`, `income`, `horizon`,
`InvestmentCapital`, `experience`) have very different scales:
- `age` ranges 18–100
- `income` ranges $20,000–$300,000
- `InvestmentCapital` ranges $2,000–$1,000,000

Without scaling, distance-based algorithms (KNN) and gradient-based algorithms
(Logistic Regression, MLP) are dominated by the largest-magnitude features.
StandardScaler centers each feature to mean=0 and scales to std=1, making all
features contribute equally to these algorithms. Tree-based methods (RF, XGBoost)
are scale-invariant by design, but scaling them does no harm.

**Categorical features → OneHotEncoder(drop='first')**

Categorical features cannot be passed as raw strings to sklearn models. OneHotEncoder
converts each category into a binary column. `drop='first'` removes one category per
feature to avoid the dummy variable trap (perfect multicollinearity), where the
dropped category is recoverable from the remaining columns. This reduces the feature
count from what would be 18 binary columns to 13.

**Result: 12 raw features → 19 processed features**

| Group | Raw Count | Processed Count |
|-------|-----------|-----------------|
| Numerical (scaled) | 6 | 6 |
| Categorical (OHE, drop first) | 6 | 13 |
| **Total** | **12** | **19** |

The preprocessor is fit only on the training set and then applied to the test set,
preventing data leakage.

---

## 2. Stage 1: Risk Profile Classification

### 2.1 Why Five Algorithms

A single algorithm choice is an unverifiable assumption. Testing five algorithms
covering different learning paradigms gives an empirical answer to "what works on
this data" rather than relying on intuition:

| Algorithm | Learning Paradigm | Key Characteristic |
|-----------|------------------|-------------------|
| KNN | Instance-based | No training, purely geometric; good baseline |
| Logistic Regression | Linear | Fast, interpretable; tests whether the problem is linearly separable |
| Random Forest | Ensemble (bagging) | Handles non-linearity, robust to noise, provides feature importance |
| XGBoost | Ensemble (boosting) | Sequentially corrects errors; often best on tabular data |
| MLP | Neural Network | Learns arbitrary non-linear mappings; tests whether deep patterns exist |

Starting with KNN as a geometric baseline and Logistic Regression as a linear baseline
tells us the minimum complexity required. If RF and XGBoost don't outperform Logistic
Regression by much, the problem is essentially linear and a simpler model is
preferable. If MLP significantly outperforms tree models, deep feature interactions
may be important.

### 2.2 KNN: GridSearchCV Tuning

```python
knn_params = {'n_neighbors': [5, 11, 21], 'weights': ['uniform', 'distance']}
knn_gs = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5,
                      scoring='accuracy', n_jobs=-1)
knn_gs.fit(X_train, y1_train)
knn_best = knn_gs.best_estimator_
```

**Why k=5, 11, 21?**

- `k=5`: Small neighborhoods, high sensitivity to local structure; tends to overfit
- `k=11`: Medium smoothing, balances bias and variance
- `k=21`: Large neighborhoods, higher bias but more robust to outliers

Odd values are used to avoid tied votes in binary sub-decisions. The search covers
a wide range so the cross-validation can find the right bias-variance trade-off for
this specific dataset size (4,000 training samples).

**Why `weights=['uniform', 'distance']`?**

- `uniform`: Each of the k neighbors votes equally — simple majority rule
- `distance`: Closer neighbors vote with higher weight (weight = 1/distance) —
  more geographically meaningful, especially useful when the decision boundary
  is complex

**Why `cv=5`?**

5-fold cross-validation splits the training set into 5 equal parts, trains on 4
and validates on 1, rotating through all 5 combinations. The final score is the
average of 5 held-out evaluations. This gives a more reliable estimate of
generalization than a single validation split. With 4,000 training samples,
each fold uses 800 samples for validation — large enough for stable estimates
without being computationally prohibitive (a 10-fold would give the same statistical
benefit at more cost).

**GridSearchCV: 6 combinations × 5 folds = 30 model fits.** The best combination
is automatically selected and stored in `knn_gs.best_estimator_`.

### 2.3 Logistic Regression: No Tuning

```python
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
```

Logistic Regression was used with default regularization (`C=1.0`, L2 penalty).
The only modification is `max_iter=1000` because the default 100 iterations
often fails to converge on datasets with many features. No hyperparameter search
was performed because:

1. Its primary role is as a **linear baseline** — testing whether the classes are
   linearly separable in the 19-dimensional feature space
2. Tuning the regularization strength `C` was considered, but the dataset is
   large enough (4,000 samples) that regularization has minimal effect
3. The result is interpretable regardless of tuning — if it underperforms tree
   methods, the problem has non-linear structure

### 2.4 Random Forest: GridSearchCV Tuning

```python
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [8, 12, None],
    'min_samples_split': [5, 10],
}
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
rf_gs.fit(X_train, y1_train)
```

**Why Random Forest?**

Random Forest is an ensemble of decision trees trained on bootstrap samples of the
data (bagging), with each split considering a random subset of features. Its
strengths for this problem:

- Handles mixed feature types (scaled numerics + OHE categoricals) without issues
- Robust to correlated features (age/income are correlated by design; RF's random
  feature subsampling reduces the chance that correlated features dominate every tree)
- Provides feature importance scores — useful for explaining which investor
  characteristics matter most
- Non-parametric: no assumption about the distribution of the data

**Parameter grid rationale:**

`n_estimators: [100, 200]`
More trees reduce variance at the cost of training time. 100 is the standard
minimum for stable estimates. 200 was included to test whether doubling the
ensemble size meaningfully improves performance. Beyond ~200 trees, returns
diminish rapidly for this dataset size.

`max_depth: [8, 12, None]`
Controls how deep each tree can grow:
- `max_depth=8`: Forces early stopping; prevents individual trees from memorizing
  training data; introduces more bias
- `max_depth=12`: Allows more complex trees while still preventing unbounded growth
- `max_depth=None`: Trees grow until leaves are pure or hit `min_samples_split`;
  highest variance, lowest bias — may overfit

`min_samples_split: [5, 10]`
Minimum samples required to split an internal node:
- `min_samples_split=5`: More splits allowed; finer decision boundaries
- `min_samples_split=10`: Coarser splits; smoother boundaries; more regularization

**12 combinations × 5 folds = 60 model fits.**

Why search `max_depth` and `min_samples_split` together? These two parameters
interact — a shallow tree (`max_depth=8`) with loose split requirements
(`min_samples_split=5`) produces different trees than a deep tree with strict
split requirements (`min_samples_split=10`). GridSearchCV finds the combination,
not each parameter in isolation.

### 2.5 XGBoost: Fixed Parameters

```python
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=42, eval_metric='mlogloss', verbosity=0)
```

XGBoost was used with sensible default hyperparameters rather than a grid search.
The reason: a full GridSearchCV over XGBoost's hyperparameter space
(`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`)
would require hundreds of fits and exceed the time budget for a student project.
The chosen values follow widely-cited best-practice starting points:

- `n_estimators=200`: Boosting benefits from more rounds; 200 is a common
  starting point before diminishing returns
- `max_depth=6`: XGBoost trees are typically shallower than RF trees because
  each tree corrects the residuals of the previous one; deep trees in boosting
  overfit aggressively
- `learning_rate=0.1`: Standard starting rate; lower rates improve generalization
  but require proportionally more `n_estimators`

**Note:** XGBoost requires integer-encoded labels. The mapping
`{'Aggressive': 0, 'Conservative': 1, 'Moderate': 2}` was applied, and predictions
were decoded back to strings using the inverse map before evaluation.

### 2.6 MLP: Fixed Architecture

```python
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                    max_iter=500, random_state=42)
```

The MLP was included to test whether deeper feature interactions exist that
tree-based models miss. Architecture choices:

- `(128, 64)`: Two hidden layers with decreasing width — standard pattern for
  tabular classification. The first layer expands representation; the second
  compresses toward the 3 output classes.
- `activation='relu'`: ReLU is the standard default for hidden layers; avoids
  vanishing gradient, computationally efficient.
- `max_iter=500`: The network is given sufficient iterations to converge.

No architecture search was performed. With only 19 input features and 3 classes,
a 2-layer network is likely sufficient — more layers would require regularization
tuning (dropout, weight decay) that is outside the scope of this project.

**Note:** MLP also requires integer-encoded labels due to a scikit-learn /
Python 3.14 compatibility issue with `early_stopping` on string labels.

### 2.7 Winner Selection: Why Macro F1, Not Accuracy

```python
best_name_s1 = max(stage1_results, key=lambda k: stage1_results[k]['macro_f1'])
```

Despite the dataset being perfectly balanced (33.3% per class), Macro F1 was
chosen as the selection metric over accuracy for principled reasons:

- **Accuracy** rewards getting the most common predictions right. On a balanced
  3-class problem, a model that predicts "Moderate" for every investor achieves
  33% accuracy — the same as random guessing.
- **Macro F1** computes precision and recall per class, then averages unweighted
  across classes. A model that ignores one class entirely gets F1=0 for that class,
  dragging the macro average down even if it performs well on the other two.
- Macro F1 therefore rewards a model that correctly identifies all three risk
  profiles, not just the easiest ones to distinguish.

In the case of balanced classes, macro F1 and accuracy tend to agree, but macro
F1 is the more robust and generalizable choice if the class balance ever changes.

---

## 3. Stage 2: Asset Allocation Regression

### 3.1 Why Multi-Output Regression

The stage predicts two continuous values simultaneously: `equity_pct` and `bond_pct`.
These are not independent — they are constrained to sum with `alt_pct` to 1.0.
Two approaches were considered:

1. **Two separate regressors:** One for equity, one for bonds. Simple but ignores
   the correlation between outputs.
2. **MultiOutputRegressor wrapper:** Trains one base estimator per target but on
   the same feature matrix. The outputs are still predicted independently per tree.
3. **Native multi-output support:** XGBoost and some other models support
   multi-output natively. This was not used because scikit-learn's
   `MultiOutputRegressor` provides a unified interface across all estimators.

`MultiOutputRegressor` was chosen because it is the scikit-learn standard, works
with any base estimator, and is sufficient given that the two targets share the
same input features.

The `alt_pct` (alternatives allocation) was not predicted by ML — it is computed
as `max(0.01, 1.0 - equity_raw - bond_raw)` in inference, ensuring allocations
always sum to 1.0.

### 3.2 Algorithms Compared

**Multi-Output Random Forest:**

```python
mo_rf = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, max_depth=15,
                          min_samples_split=10, min_samples_leaf=5,
                          random_state=42))
```

Random Forest regression averages the outputs of many trees, which reduces
variance in predictions. Parameters were set conservatively:
- `max_depth=15`: Deeper than Stage 1 classifier because regression targets
  (continuous percentages) have more fine-grained structure
- `min_samples_split=10, min_samples_leaf=5`: Prevents trees from fitting
  individual noise points in the continuous target

**Multi-Output Gradient Boosting:**

```python
mo_gb = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=100, max_depth=5,
                              learning_rate=0.1, random_state=42))
```

Gradient Boosting sequentially trains shallow trees where each tree fits the
residual errors of the previous ensemble. It is applied here through
`MultiOutputRegressor`, meaning two separate GB models are trained (one per output).
Gradient Boosting often outperforms Random Forest on regression tasks with
structured data but is more sensitive to overfitting.

**Winner selection:** `min(avg_mae)` — lowest average MAE across both targets.

### 3.3 Why R² Is Low and Why That's Acceptable

The Stage 2 models achieve R² ≈ 0.06, which appears poor. This is a direct
consequence of the label generation process:

- Labels were generated by `allocations.py`, which uses `np.random.choice` with
  stochastic sampling. Two investors with identical profiles can receive different
  allocations depending on the random seed.
- This adds **irreducible noise** to the target variable. No model, regardless of
  complexity, can predict the outcome of a random process.
- R² measures explained variance. If most of the target variance is noise, R²
  will be low even for a well-calibrated model.

The practical implication: the model predicts the **mean allocation** for an
investor type reasonably well (MAE ≈ 14–15%), but cannot predict the specific
stochastic draw. Post-hoc `PROFILE_BOUNDS` clamping in inference ensures
predictions remain within domain-valid ranges per risk class.

---

## 4. Stage 3: ETF Count Regression

### 4.1 Why a Simple Random Forest Was Sufficient

```python
etf_regressor = RandomForestRegressor(
    n_estimators=100, max_depth=10,
    min_samples_split=10, min_samples_leaf=5,
    random_state=42)
```

Stage 3 predicts the total number of ETFs in the portfolio, which is clipped to
the integer range [7, 13]. The prediction task has low effective complexity:

- The output range spans only 7 values (7 through 13)
- The mapping from investor profile to ETF count follows a simple, monotonic
  pattern (more aggressive investors tend to get more diversified equity selections)
- A single Random Forest without hyperparameter search was sufficient to achieve
  reasonable performance (MAE < 1.1 ETFs)

No additional algorithms were tested for Stage 3 because the stage is not the
primary value driver of the system — it informs Stage 4 (how many ETFs to select)
but the specific tickers are determined by rule-based logic. Over-engineering
Stage 3 would not meaningfully improve the final portfolio output.

**Prediction post-processing:**

```python
y3_pred_rounded = np.clip(np.round(y3_pred), 7, 13).astype(int)
```

Continuous regression output is rounded to the nearest integer and clipped to [7, 13].
This domain constraint is enforced at inference time, not during training, because
training with clipped targets would bias the model toward the boundaries.

---

## 5. The Circular Label Problem

This is the most important limitation of the entire ML pipeline, and understanding
it explains why the model accuracy is lower than expected.

**The problem:** Labels for all three stages were generated by running
`allocations.py` on the synthetic investor data. The `allocations.py` module
uses `np.random.choice` to assign ETFs with AUM-weighted probabilities. This means
the assignment process is stochastic — the same investor profile will produce a
different portfolio each time it is run.

When this noisy output is used as the training label, the model must learn a
mapping from `(investor profile) → (one particular stochastic draw)` rather than
`(investor profile) → (deterministic optimal portfolio)`. The irreducible noise
floor is baked into the labels.

**Why it explains Stage 1 accuracy (~46%):**

The risk profile (`Aggressive/Moderate/Conservative`) is itself derived from the
stochastic allocation process. The model achieves 46% test accuracy on a 3-class
balanced problem where random guessing scores 33%. This 13-percentage-point lift
above chance represents the true learnable signal — the consistent patterns across
the stochastic draws.

**What would fix it:**
1. Use deterministic labels (remove stochastic sampling from `allocations.py`)
2. Generate labels by running the allocation pipeline multiple times per investor
   and using the modal (most common) outcome
3. Replace synthetic labels with real investor data

---

## 6. Summary of Design Decisions

| Decision | Choice Made | Key Reason |
|----------|-------------|------------|
| Synthetic data generation | 5,000 investors | No real labeled data available |
| Class balance strategy | `pd.qcut` quantile binning | Guarantees exact 33/33/33 split |
| Feature scaling | StandardScaler for numerics | Required for KNN, LR, MLP |
| Categorical encoding | OneHotEncoder(drop='first') | Avoids dummy variable trap |
| Stage 1 algorithm selection | 5 algorithms compared | Empirical rather than assumed best |
| KNN tuning | GridSearchCV k∈[5,11,21], weights | Covers range of bias-variance trade-offs |
| RF tuning | GridSearchCV depth, estimators, split | Most impactful RF parameters |
| XGBoost | Fixed sensible defaults | Full grid too expensive for scope |
| Winner metric | Macro F1 | Robust to class-specific failure modes |
| Stage 2 approach | MultiOutputRegressor | Standard multi-target sklearn interface |
| Stage 2 bounds | Post-hoc PROFILE_BOUNDS clamping | Corrects noisy regressor outputs |
| Stage 3 approach | Single RF, no tuning | Low complexity task, not the value driver |
| ETF selection | Rule-based Stage 4 | Domain knowledge (conflicts) not ML-learnable |
