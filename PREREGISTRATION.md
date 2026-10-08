# Pre-registration — Random seeds v3

**Committed:** see the git timestamp of the commit that adds this file. It is committed before any experiment code exists and before any file in `results/` exists.
**Root seed:** `np.random.SeedSequence(20261008)`. It never changes after this commit. Every random stream is `SeedSequence(20261008, spawn_key=(experiment, stream))`.

Each prediction has a concrete, checkable number or direction. After the runs, an "Outcome" line (HELD / FAILED / PARTIAL + the number) is appended under each prediction. The predictions themselves are never edited.

## Shared design

- **Synthetic population** (E1 synthetic arm, E5): one pool of 10,000 rows from `make_classification(n_features=20, n_informative=5, n_redundant=5, n_clusters_per_class=2, flip_y=0.05, class_sep=0.8)`. Subsamples are drawn from this pool without replacement.
- **Real data:** `sklearn.datasets.load_breast_cancer` (569 rows, UCI Breast Cancer Wisconsin Diagnostic, CC BY 4.0).
- **Splits:** `train_test_split(test_size=0.3, stratify=y)` unless the experiment varies stratification.
- **Spread** always means the sample standard deviation (ddof = 1).
- **CIs:** 95% percentile bootstrap over seeds, 2,000 resamples, for spreads, ratios and rates.
- **Compute:** CPU only, `joblib` across seeds. Target full run under 15 minutes on a 16-thread laptop.

## E1 — Split lottery and boundary map

Design: breast cancer n ∈ {100, 150, 250, 400, 569}; synthetic n ∈ {100, 250, 500, 1000, 2500, 5000}. For each n: 5 subsample seeds × 200 split seeds. Models with fixed internals: scaled `LogisticRegression`, `RandomForestClassifier(n_estimators=100, random_state=0)`, scaled `SVC` (RBF, defaults), scaled `KNeighborsClassifier` (k = 5). At n = 569 the subsample is the full dataset, so the 5 subsamples coincide. The opening scene is breast cancer, n = 150, subsample index 0.

- `seed_spread(n)` = mean over models of each model's std across the 200 split seeds.
- `model_spread(n)` = std of the 4 models' mean accuracies.
- `R(n) = seed_spread / model_spread`, computed per subsample; the line is the median over subsamples, the band is min–max.
- `flip_rate` = share of split seeds where the worse-on-average model of the top pair strictly beats the better one (ties reported separately).

Predictions:
- P1.1 At n = 150 (breast cancer), median R ≥ 2.0.
- P1.2 Breast cancer: R stays above 1 at every tested n (no crossing up to n = 569). Synthetic: R crosses 1 between n = 250 and n = 1000.
- P1.3 Flip rate at n = 150 (breast cancer, subsample 0) is ≥ 20%.
- **Falsifier:** R < 1 at every tested n on both datasets. If so, the title's claim fails and the article says so.

## E2 — Imbalance

Design: one `make_classification` dataset per n ∈ {400, 1000, 4000} (20 features, 5 informative, 5 redundant, `weights=[0.95]`, `flip_y=0.01`). Scaled `LogisticRegression`, deterministic. 300 split seeds, each used with `stratify=None` and `stratify=y`. Track the test positive count k and recall. Unstratified runs with k = 0 have undefined recall; they are counted and reported, not silently dropped.

- P2.1 Stratification reduces the std of the test positive count k to exactly 0 at every n (by construction).
- P2.2 Recall std ratio (stratified / unstratified) is within [0.85, 1.05] at every n.
- P2.3 Stratified recall std is within ±25% of the binomial floor √(r(1−r)/k) at every n, where r is mean recall and k the stratified test positive count.

## E3 — Init lottery audit

Design: full breast cancer, one fixed stratified split. Only `MLPClassifier(random_state=s)` varies, over 200 seeds, default architecture (100 ReLU units, Adam, lr 0.001). Four conditions: A = raw inputs, `max_iter=200` (baseline); B = raw, `max_iter=1000`; C = standardized, `max_iter=200`; D = standardized, `max_iter=1000` (the fix). Captured per run: `n_iter_`, final `loss_`, `ConvergenceWarning`, train and test accuracy.

Classes, fixed in advance:
- **not converged** = a `ConvergenceWarning` was raised;
- **converged, poor minimum** = no warning and train accuracy < 0.90;
- **converged, good minimum** = no warning and train accuracy ≥ 0.90.

- P3.1 In condition A, ≥ 50% of runs with test accuracy below 0.80 are flagged not converged.
- P3.2 After the fix (D), the test-accuracy range (max − min) is ≥ 5× smaller than in A.

## E4 — k-means

Design: `make_blobs(n_samples=1500, centers=8, cluster_std=1.5, center_box=(-10, 10))`, fixed. `KMeans(n_clusters=8)` over 300 seeds × `n_init` ∈ {1, 10} × `init` ∈ {k-means++, random}; the same 300 seeds in every configuration. Metric: ARI against the generating labels. A run is in a **bad local optimum** if its inertia is more than 1% above the lowest inertia among all 1,200 runs.

- P4.1 `n_init=10` cuts ARI variance by ≥ 10× versus `n_init=1` (k-means++).
- P4.2 With k-means++ and `n_init=1`, ≥ 20% of runs are in a bad local optimum; with `n_init=10`, ≤ 5%.

## E5 — Variance decomposition

Design: subsamples of the synthetic population at n = 300 and n = 3,000. Fully crossed: model {`RandomForestClassifier(n_estimators=100)`, scaled `MLPClassifier(max_iter=500)`, `GradientBoostingClassifier(subsample=0.8)`} × 30 split seeds × 30 model-internal seeds = 2,700 fits per size. Analysis: balanced three-factor sum-of-squares decomposition by hand (no replication, so the three-way interaction is the residual). Shares = SS_effect / SS_total for model, split seed, internal seed, the three two-way interactions, and residual.

- P5.1 At n = 300, model explains ≤ 30% of the accuracy variance; everything seed-driven (split, internal seed, all interactions and residual) explains ≥ 70%.
- P5.2 At n = 3,000, the model share rises to ≥ 50%.

## E6 — The fix

Design: the E1 opening setting (breast cancer, n = 150, subsample 0). The top pair is the two models with the highest mean accuracy over the 200 split seeds. The cherry-picked seed is the split seed with the largest win for the worse-on-average model; it is labelled a deliberate worst case. Procedures on the same 150 rows: (i) that single seed; (ii) naive paired t-test over 10×5 repeated stratified k-fold (shown as what not to do); (iii) corrected resampled t-test (Nadeau & Bengio 2003) over the same 50 folds; (iv) 5×2cv paired t-test (Dietterich 1998). Each reports the mean difference, a 95% CI and p.

- P6.1 The single-seed "winner" is not significant under the corrected resampled t-test (p > 0.05).
- P6.2 Under the corrected test the mean difference points toward the better-on-average model, not the cherry-picked winner.

## Analysis choices fixed in advance

- Metrics: accuracy (E1, E5, E6), recall (E2), ARI (E4).
- 95% CIs by bootstrap over seeds (2,000 resamples).
- No seeds excluded post hoc. Failed or non-converged fits are counted as runs and reported.
- If a design parameter has to change after this commit (for example a runtime problem), the change and its reason go into `article/DECISIONS.md` and the article.

---

## Outcomes (appended 2026-10-08 after the full run; predictions above unedited)

Source: `results/*.txt`, full run, 83.6 min with 12 workers (`results/env.txt`). **11 of 15 held.**

| Prediction | Outcome | Observed |
|---|---|---|
| P1.1 median R at n = 150 ≥ 2.0 | HELD | 2.42 |
| P1.2a breast cancer R > 1 at every n | HELD | smallest median R 1.48 (n = 569) |
| P1.2b synthetic crosses 1 between n = 250 and 1,000 | HELD | between 250 and 500; log-interpolated n = 300 |
| P1.3 flip rate at n = 150 (subsample 0) ≥ 20% | **FAILED** | 8.0%. Subsample 0 was the tamest of the 5; the per-subsample rates were 8.0% to 30.0% |
| Falsifier: R < 1 everywhere | not triggered | — |
| P2.1 stratified k std = 0 | HELD | 0.00 at every n |
| P2.2 recall std ratio in [0.85, 1.05] | HELD | 0.93, 0.99, 1.00 |
| P2.3 within ±25% of the binomial floor | HELD | 0.91x, 0.87x, 0.90x |
| P3.1 ≥ 50% of runs below 0.80 are not converged | **FAILED** | no run fell below 0.80 (minimum 0.877) |
| P3.2 range shrinks ≥ 5x from A to D | **FAILED** | 3.0x |
| P4.1 n_init = 10 cuts ARI variance ≥ 10x | HELD | 24.4x |
| P4.2 bad optimum ≥ 20% (n_init = 1), ≤ 5% (n_init = 10) | HELD | 52.0% and 0.7% |
| P5.1 n = 300: model share ≤ 30% | **FAILED** | 54.1%, driven by an MLP that trailed by about 0.12 |
| P5.2 n = 3,000: model share ≥ 50% | HELD | 80.8% |
| P6.1 single-seed winner not significant (corrected test) | HELD | p = 0.180 |
| P6.2 corrected estimate favours the better-on-average model | HELD | +0.0167 |

**Not pre-registered (labelled exploratory):** `experiments/e5_exploratory.py` decomposes only the two close models (RandomForest and GradBoost) from E5's saved fits. Model share: 0.9% at n = 300, 77.8% at n = 3,000.

**Deviation:** E5's MLP used `early_stopping=True`, a runtime decision made before any study run (see `article/DECISIONS.md`).
