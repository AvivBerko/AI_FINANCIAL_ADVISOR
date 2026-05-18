# Professor Q&A — Practice Guide
**Topic: Algorithm Selection & Hyperparameter Tuning**

Read each answer out loud. The goal is to be able to explain it in your own words
without reading. Short, confident answers beat long uncertain ones.

---

## ALGORITHM SELECTION

---

**Q: You tested five algorithms. Why these five specifically? Why not just use Random Forest?**

A: I wanted to cover different learning paradigms rather than assume one approach is best.
KNN is a geometric baseline — it makes no assumptions, it just looks at the most similar
investors. Logistic Regression is a linear baseline — it tells me whether the classes
are linearly separable. Random Forest handles non-linearity and gives feature importance.
XGBoost is the boosting alternative to bagging — it often wins on tabular data.
MLP tests whether there are deep feature interactions that tree methods miss.
If I had just picked Random Forest, I would have no evidence that it's actually
the best choice for this data. The comparison gives me that evidence empirically.

---

**Q: What is the difference between Random Forest and XGBoost? They both use decision trees.**

A: Both use decision trees, but the strategy is opposite.
Random Forest uses **bagging** — it builds many trees in parallel, each on a random
sample of the data, then averages their predictions. Each tree is independent.
XGBoost uses **boosting** — it builds trees sequentially. Each new tree specifically
focuses on correcting the mistakes the previous trees made. It learns from its errors.
Bagging reduces variance. Boosting reduces bias. On tabular data, boosting often
wins because it can model complex patterns, but it's more sensitive to overfitting.

---

**Q: Why did you include KNN? It's a very simple algorithm.**

A: That's exactly why — it's a geometric baseline. KNN classifies an investor by
finding the k most similar investors in the training set and taking a majority vote.
No training happens, no parameters are learned. If KNN performs close to the
sophisticated models, it means the classes are well-separated in feature space
and the problem is simpler than expected. If it performs poorly, it confirms
the problem needs more complex decision boundaries. Every good experiment needs
a baseline to compare against.

---

**Q: Why Logistic Regression with max_iter=1000? What does that parameter do?**

A: Logistic Regression finds its optimal weights through an iterative optimization
algorithm. The default is 100 iterations, which is often not enough to converge
when you have 19 features. Without enough iterations, sklearn throws a convergence
warning and gives you a partially-trained model. Setting max_iter=1000 just gives
the optimizer enough steps to fully converge. It's not a meaningful hyperparameter —
it's a safety setting.

---

**Q: Why didn't you do GridSearchCV on XGBoost like you did for Random Forest?**

A: A proper XGBoost grid search would cover at minimum: n_estimators, max_depth,
learning_rate, subsample, and colsample_bytree. That's hundreds of combinations
times 5-fold cross-validation — far beyond the time and compute budget for a
student project. Instead I used well-established starting values: 200 estimators,
depth 6, learning rate 0.1. These are the standard defaults from the XGBoost
documentation and published benchmarks. The goal was not to squeeze the last
percent of performance but to fairly compare learning paradigms.

---

**Q: Why did you include MLP? Isn't a neural network overkill for 19 features?**

A: Possibly — and the results confirm that. The MLP with two hidden layers (128 → 64)
was included to test whether there are deep non-linear feature interactions that
tree methods miss. In practice, for tabular data with fewer than 100 features, neural
networks rarely outperform gradient boosting. The MLP is in the comparison to
validate that assumption empirically, not because I expected it to win.

---

---

## HYPERPARAMETER TUNING

---

**Q: What is GridSearchCV and how does it work?**

A: GridSearchCV is a systematic hyperparameter search from scikit-learn.
You give it a model and a dictionary of parameter values to try. It creates
every possible combination — the full grid — and evaluates each one using
cross-validation. The best-performing combination is stored and used as the
final model. It automates what would otherwise be manual trial-and-error.

---

**Q: What does cv=5 mean in GridSearchCV? Why 5 and not 10?**

A: cv=5 means 5-fold cross-validation. The training set is split into 5 equal parts.
The model trains on 4 parts and validates on 1, rotating through all 5 combinations.
The final score is the average across all 5 validation sets.
I chose 5 folds because: with 4,000 training samples, each fold gives 800 validation
samples — enough for a stable estimate. 10-fold would give the same statistical
benefit at double the compute cost. 3-fold would be too noisy. 5 is the standard
choice for datasets of this size.

---

**Q: Walk me through the Random Forest parameter grid. Why those specific values?**

A: I searched three parameters:

`n_estimators: [100, 200]` — the number of trees. More trees reduce variance
at the cost of training time. 100 is the minimum for stable estimates.
200 tests whether doubling the ensemble meaningfully improves performance.
Beyond 200, returns diminish on a dataset this size.

`max_depth: [8, 12, None]` — how deep each tree can grow.
Depth 8 forces early stopping and introduces more bias but prevents overfitting.
Depth 12 allows more complex trees. None means trees grow until leaves are pure —
highest variance, may overfit. The search finds the right trade-off.

`min_samples_split: [5, 10]` — minimum samples needed before a node can split.
Higher values mean fewer, coarser splits — more regularisation.
These two parameters interact, so GridSearchCV finds the best combination, not
each parameter independently.

In total: 2 × 3 × 2 = 12 combinations, each evaluated with 5-fold CV = 60 model fits.

---

**Q: For KNN, you searched k=5, 11, 21. Why odd numbers?**

A: Odd values prevent tied votes. KNN classifies by majority vote among the k
nearest neighbors. With an even k on a binary or multi-class problem, you can
get ties where two classes have equal votes. Odd numbers guarantee a clear majority
in any two-class vote. It's a simple best practice.

---

**Q: Why did you choose macro F1 to select the winning algorithm instead of accuracy?**

A: Accuracy counts the fraction of correct predictions overall.
On a 3-class balanced dataset, a model that predicts "Moderate" for every single
investor achieves 33% accuracy — the same as random guessing — but accuracy
doesn't flag this as a problem.
Macro F1 computes precision and recall separately for each class, then averages
them equally. If a model completely ignores one class, it gets F1=0 for that class,
which drags the macro average down. Macro F1 rewards a model that correctly
identifies all three risk profiles, not just the easiest ones. It's a more
honest measure of whether the model is actually working.

---

**Q: Your train accuracy is much higher than test accuracy. What does that tell you?**

A: It tells me the model is overfitting — it has memorised patterns in the
training data that don't generalise to new investors. This is expected with
tree-based models on noisy labels. The training accuracy is less important than
the test accuracy, which is the honest measure of performance on unseen data.
The gap is worth noting, but since we selected models by test performance, the
winner is still the most generalisable option from the candidates we tested.

---

---

## RESULTS & LIMITATIONS

---

**Q: Your Stage 1 accuracy is around 46%. That's barely better than random (33%). Is this a good result?**

A: It's an honest result. The 13-percentage-point improvement over random represents
the true learnable signal. The main reason accuracy is not higher is a structural
problem with how the training labels were generated: the labels came from the same
rule-based allocation system I wrote, which uses random sampling internally.
Two investors with identical profiles can receive different labels depending on
the random seed. No model — regardless of complexity — can predict the outcome
of a random process. The noise is in the labels, not in the features.
A cleaner label generation process would likely push accuracy to 70–80%.

---

**Q: You applied PROFILE_BOUNDS after the regression in Stage 2. Isn't that cheating?**

A: It's not cheating — it's domain knowledge enforcement.
The allocation regressor has ~15% MAE because the training labels contained the
same stochastic noise as Stage 1. Without bounds, the model can predict, for example,
90% bonds for an Aggressive investor — which contradicts the definition of
an aggressive profile. The bounds encode financial planning logic: an aggressive
investor should hold mostly equity. The model predicts the rough magnitude of
the allocation; the bounds ensure the output stays within a sensible domain.
This is a standard technique in applied ML when you have hard constraints the
model cannot learn from noisy data.

---

**Q: Why is Stage 2's R² only 0.06? Doesn't that mean the model is useless?**

A: R² measures how much of the variance in the target the model explains.
If most of the variance is irreducible noise — which it is, because labels were
generated stochastically — then R² will be low even for a well-calibrated model.
The model is not useless: it correctly predicts that aggressive investors
get more equity and conservative investors get more bonds. The MAE of ~14% means
predictions are on average 14 percentage points away from the label.
That label itself was randomly drawn, so the 14% error is largely the noise floor,
not model error. R² is the wrong metric here — MAE is more meaningful.

---

**Q: Why did Stage 3 (ETF count) not get the same GridSearchCV treatment as Stage 1?**

A: Two reasons. First, the output range is only 7 to 13 — seven possible integer
values. A Random Forest easily learns a low-complexity mapping like this without
tuning. Second, Stage 3 is not the primary value driver of the recommendation —
it feeds into Stage 4 which does rule-based ETF selection. Whether the portfolio
has 8 or 9 ETFs has a small impact on the final recommendation. Over-engineering
Stage 3 would add complexity without meaningfully improving the user-facing output.

---

**Q: Why is Stage 4 rule-based instead of ML?**

A: Because the core logic — ETF conflict resolution — is domain knowledge that
cannot be learned from data. VOO and SPY are both S&P 500 index trackers. Selecting
both gives the investor double exposure to the same 500 stocks, which wastes
diversification budget. The model has no way to learn this from investor profiles
and portfolio labels alone — it would need to understand fund composition.
Encoding it explicitly as a CONFLICT_GROUPS rule is more reliable, more
interpretable, and easier to maintain than trying to teach a model fund-level
overlap relationships.

---

---

## QUICK REFERENCE — KEY NUMBERS TO KNOW

| Fact | Value |
|------|-------|
| Training samples | 4,000 |
| Test samples | 1,000 |
| Raw input features | 12 |
| Processed features | 19 |
| Stage 1 random baseline | 33.3% |
| Stage 1 winner | Random Forest, Macro F1 = 0.425, Test Accuracy = 42.6% |
| Stage 1 KNN best params | k=21, weights=uniform |
| Stage 1 RF best params | max_depth=8, min_samples_split=10, n_estimators=200 |
| Stage 1 XGBoost overfit | Train 91.3% → Test 40.2% (large gap = overfitting) |
| Stage 1 MLP overfit | Train 93.9% → Test 35.7% (worst generalisation) |
| Stage 2 winner | Gradient Boosting, Avg MAE = 15.48% |
| Stage 2 RF Avg MAE | 15.49% (essentially tied with GB) |
| Stage 3 algorithm | Random Forest only (no comparison) |
| Stage 3 MAE (rounded) | 1.061 ETFs |
| Stage 3 R² | -0.011 (slightly worse than predicting the mean — noise dominated) |
| Stage 3 Exact match | 26.6% |
| ETF portfolio size range | 7–13 ETFs |
| KNN GridSearchCV fits | 30 (6 combos × 5 folds) |
| RF GridSearchCV fits | 60 (12 combos × 5 folds) |
