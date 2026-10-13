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

**Robustness check (appended 2026-10-09).** P5.1 was re-run at n = 300 with the MLP exactly as pre-registered (`MLPClassifier(max_iter=500)`, no early stopping): same seeds, same subsample, same design (`experiments/e5_prereg_mlp.py`, `results/e5_prereg_mlp.txt`). MLP mean accuracy 0.700, RandomForest 0.696, GradBoost 0.688; model share **1.4% → P5.1 HELD under the exact pre-registered spec.** The failure in the main run came from the early-stopping deviation. n = 3,000 was not re-run with the plain MLP (too slow), so P5.2 stands as run.

---

## E7 — The golden-set lottery: an intent-router eval (ADDED 2026-10-09, committed before any E7 code exists or runs)

Motivation (Benjy, 2026-10-09): golden sets in AI engineering are usually 100–500 labelled examples, because labels are expensive. E7 tests the article's thesis in that setting: a router (an intent classifier that sends each query to one of 77 handlers) is evaluated on a small golden set. A "router failure" is a misrouted query.

Design, fixed in advance:
- **Data:** Banking77 (PolyAI; 10,003 train / 3,080 test customer-service queries, 77 intents, CC BY 4.0), downloaded 2026-10-09 from github.com/PolyAI-LDN/task-specific-datasets. SHA-256: train `b06e26ac…cc664b`, test `d12d6e3b…474d`.
- **Two routers**, trained once on the full official train split and both deterministic:
  - **v1 (current):** TF-IDF on word 1–2-grams (`sublinear_tf=True`) + `LogisticRegression(max_iter=2000)`.
  - **v2 (candidate):** TF-IDF on word 1–2-grams + char_wb 2–5-grams + `LinearSVC(C=1.0)`.
- **"Better" and "worse"** are defined by accuracy on all 3,080 test queries, the population proxy.
- **Golden sets:** g ∈ {100, 200, 300, 500, 1,000, 3,000} queries, drawn **with replacement** from the 3,080 test queries (sampling from the query stream). 1,000 golden sets per size, each from a seed in the `SeedSequence(20261008)` stream (7, k).
- **Per golden set:** both failure rates; the measured difference; the "wrong winner" (the worse router strictly better on that set); ties; and McNemar's exact test on discordant queries (p < 0.05 in the correct direction counts as "detected").
- **Golden-set size for 80% power:** the analytic McNemar sample size from the full-test discordance rates.

Predictions:
- P7.1 The two routers' full-test accuracies differ by less than 3 points (a close, version-to-version pair).
- P7.2 At g = 300, the worse router ties or beats the better one on ≥ 15% of golden sets.
- P7.3 At g = 300, McNemar detects the true difference on ≤ 60% of golden sets.
- P7.4 The golden-set size needed for 80% McNemar power is ≥ 500 queries.
- **Falsifier:** if the routers differ by ≥ 5 points, golden-set noise at g = 300 is irrelevant for this pair, and the article says so.

Presentation change (does not affect any prediction): the article now centres on golden-set sizes (100–500 rows). Results for 1,000+ rows move to the "until they don't" section, and the hero becomes E7's g = 300 distribution.

**E7 outcomes (appended 2026-10-09, after the E7 run; the predictions above are unedited).** v1 failure rate 0.143 vs v2 0.088: a **5.5-point gap**. P7.1 **FAILED** (5.5 ≥ 3). P7.2 **FAILED** (worse router ties or wins on 0.1% of 300-query golden sets). P7.3 **FAILED** (McNemar detects on 93.4%). P7.4 **FAILED** (80% power at 205 queries). **The falsifier triggered**: for a 5-point upgrade, a 300-query golden set is enough, and the article says so.

---

## E7b — Small tweaks to the router (ADDED 2026-10-09 after seeing E7, committed before any E7b code exists or runs)

Why: E7's pair turned out to be a large upgrade. Real router updates are usually small tweaks. To avoid choosing a pair after seeing results, all three tweaks below are fixed now and **all three will be reported**, whatever they show.

Design: the same data, golden-set sizes, 1,000 golden sets per size (sampled with replacement) and metrics as E7. Each comparison is **v2 against one tweak of v2**:
- **t1:** v2 with `LinearSVC(C=0.5)` (more regularization).
- **t2:** v2 with `LinearSVC(class_weight="balanced")`.
- **t3:** v2 with char_wb 3–5-grams instead of 2–5.

Seeds come from the `SeedSequence(20261008)` stream (7, 100 + 10·j + k) for tweak j and size index k. "Better" and "worse" are defined by full-test accuracy within each pair.

Predictions:
- P7b.1 At least 2 of the 3 tweaks are within 1.5 points of v2's full-test accuracy.
- P7b.2 For every tweak within 1.5 points, the worse router of the pair ties or wins on ≥ 20% of 300-query golden sets.
- P7b.3 For every tweak within 1.5 points, McNemar detects the difference on ≤ 30% of 300-query golden sets, and 80% power needs ≥ 1,000 queries.
- **Falsifier:** if all 3 tweaks are ≥ 3 points from v2, the "small tweak" scenario isn't small here, and the article says so.

**E7b outcomes (appended 2026-10-09 after the E7b run; the predictions above are unedited).** All three tweaks are 0.2 points from v2 on the full test set. **P7b.1 HELD** (3 of 3 within 1.5 points). **P7b.2 HELD**: the worse router ties or wins on 51.2% (t1), 47.8% (t2) and 51.1% (t3) of 300-query golden sets. **P7b.3 HELD**: McNemar detects on 0.1% / 0.0% / 0.0% at 300 queries, and 80% power needs 22,239 / 9,399 / 9,399 queries. The falsifier did not trigger.

**Tally across all pre-registered predictions: 14 of 22 held as run** (E1–E6: 11 of 15; E7: 0 of 4; E7b: 3 of 3). P5.1 also holds under the exact pre-registered spec (robustness rerun).

---

## E8 — The effect at full strength: tailored datasets (ADDED 2026-10-13, committed before any E8 study code exists or runs)

Why (Benjy, 2026-10-13): the article should first show the situation where seeds beat the model **strongly and clearly**, then draw the boundaries. E1 on breast cancer shows a modest effect (median R = 2.42 at n = 150). E8 builds datasets designed to show the mechanism at full strength, then turns one dial at a time to show where it dies.

**Pilots (disclosed, not results).** The designs below were chosen after 5 pilot runs on separate seed roots (`SeedSequence(777)` to `SeedSequence(781)`), which no study stream uses. Pilot lessons: off-the-shelf kNN (k = 5), unpruned forests and default boosting overfit label noise, so their averages trail the ceiling and cap R at about 2; CV-tuning every model inside each split was 15× slower and less stable (the tuned MLP sometimes picked a degenerate setting); fixed, regularized settings let all 8 families reach the same ceiling. Pilot R values for the designs below: roughly 5 to 8 (T1), 10 to 12 (T2), 12 to 16 (T3), 16 to 49 (T4). Pilot numbers are never quoted as results.

**Data generator** (2 features, balanced classes): class means sit `separation` apart along x0, unit-variance Gaussian noise, an optional `bend` that moves both classes by `bend · (x1² − 1)` along x0 (so the true boundary curves and linear models can't follow it), and an optional `label_noise` share of labels flipped at random (annotators who disagree).

**The 8-model zoo**, every setting fixed now and never tuned on study data:
- `LogisticRegression()` (scaled)
- `GaussianNB()`
- `SVC()` (RBF, scaled)
- `KNeighborsClassifier(n_neighbors=31)` (scaled)
- `DecisionTreeClassifier(max_depth=2)`
- `RandomForestClassifier(n_estimators=100, min_samples_leaf=20)`
- `HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03)`
- `MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=2000)` (scaled)

All internal seeds are fixed at 0; only the split seed moves (as in E1).

**E8a — four tailored designs.** Each has 5 dataset instances (data seeds from stream (8, 1 + design index)) × 200 stratified split seeds (stream (8, 0), shared by all designs):
- **T1 overlapping classes, 300 rows:** separation 2.0 (no model can beat about 84%); 210 train / 90 test.
- **T2 disputed labels, 300 rows:** separation 6.0 (cleanly separated groups) with 20% of labels flipped; 210 / 90.
- **T3 a 100-row golden set, overlapping classes:** separation 2.0; 1,000 train / 100 test.
- **T4 a 100-row golden set, disputed labels:** separation 6.0, 20% flipped; 1,000 / 100.

Metrics:
- `R` is exactly E1's ratio: the mean over the 8 models of each model's std across split seeds, divided by the std of the 8 models' mean accuracies. R is computed per instance; a design's R is the median over its 5 instances.
- Agreement: the share of test rows on which all 8 models predict the same label (mean over splits).
- A 95% bootstrap CI over split seeds for the hero instance's R.

**Hero selection rule, fixed now:**
- The hero design is the design with the largest median R (all four are eligible).
- The hero instance is that design's median-R instance (the 3rd of 5 when sorted), not its most extreme one.
- Left panel: "same data, different models", the 8 models' mean accuracies over the 200 splits.
- Right panel: "same model, different seeds", logistic regression's accuracy on each of the 200 split seeds.
- Each panel shades ±1 std. The bubble shows the ratio of the two drawn stds: logistic regression's std across seeds ÷ the std of the 8 model means. Both that ratio and R are printed.

**E8b — the dials.** One knob at a time, starting from T1's family (separation 2.0, no label noise). Each setting has 3 instances × 200 split seeds (streams (8, 10+), seeds fixed by setting index):
- **D1 bend** ∈ {0, 0.5, 1, 1.5, 2} at n = 300 (70/30).
- **D2 dataset size** n ∈ {150, 300, 1000, 3000} (70/30), for bend = 0 and bend = 1.
- **D3 golden-set size** g ∈ {50, 100, 300, 1000} test rows, with 1,000 training rows and bend 0.

Predictions:
- P8.1 Every E8a design has median R ≥ 4.
- P8.2 The largest median R among the E8a designs is ≥ 10.
- P8.3 D1: median R ≥ 3 at bend 0 and < 1 at bend 2.
- P8.4 D2: with bend 1, median R < 1 at n = 1,000 and at n = 3,000. With bend 0, median R > 1 at every n.
- P8.5 D3: median R at g = 1,000 is lower than at g = 50, and stays > 1 at every g.
- **Falsifier:** if the largest median R among E8a designs is < 5, the "full strength" framing is dropped and the article says so.
