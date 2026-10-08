"""E6 — The fix: compare two models with repeated CV and a corrected paired test."""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import QUICK, np, seeds, subsample  # noqa: F401
from experiments.e1_split import MODELS, load_breast_cancer, lottery  # noqa: F401

# --- article: e6 ---
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


def fold_diffs(X, y, a, b, splitter):
    """Paired accuracy differences acc(a) - acc(b), one per fold."""
    d = []
    for tr, te in splitter.split(X, y):
        acc = [MODELS[m]().fit(X[tr], y[tr]).score(X[te], y[te]) for m in (a, b)]
        d.append(acc[0] - acc[1])
    return np.array(d)


def t_summary(mean, se, df):
    """Mean, 95% CI and two-sided p for a t statistic."""
    half = stats.t.ppf(0.975, df) * se
    return mean, mean - half, mean + half, 2 * stats.t.sf(abs(mean / se), df)


def naive_ttest(d):
    """Treats the 50 folds as independent. They are not: training sets overlap."""
    return t_summary(d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d) - 1)


def corrected_ttest(d, n_train, n_test):
    """Nadeau & Bengio (2003): inflate the variance for overlapping training sets."""
    se = np.sqrt((1 / len(d) + n_test / n_train) * d.var(ddof=1))
    return t_summary(d.mean(), se, len(d) - 1)


def five_by_two_cv(X, y, a, b, seed_list):
    """Dietterich (1998): 5 repetitions of 2-fold CV, t statistic on 5 df."""
    d = np.array([fold_diffs(X, y, a, b, StratifiedKFold(2, shuffle=True, random_state=s))
                  for s in seed_list])                    # shape (5, 2)
    s2 = ((d - d.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return t_summary(d[0, 0], np.sqrt(s2.mean()), 5)      # numerator: the first fold


def e6_fix(quick=QUICK):
    X, y = load_breast_cancer(return_X_y=True)
    X, y = subsample(X, y, 150, seeds(1, 0, 1)[0])        # E1's opening scene
    acc = lottery(X, y, seeds(1, 2, 30 if quick else 200))
    names = list(MODELS)
    best, runner = (names[i] for i in np.argsort(acc.mean(axis=0))[::-1][:2])
    w = np.argmax(acc[:, names.index(runner)] - acc[:, names.index(best)])
    rows = {"single cherry-picked split": (acc[w, names.index(best)] - acc[w, names.index(runner)],)}
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seeds(6, 1, 1)[0])
    d = fold_diffs(X, y, best, runner, folds)
    rows["10x5 CV, naive t-test"] = naive_ttest(d)
    rows["10x5 CV, corrected t-test"] = corrected_ttest(d, n_train=120, n_test=30)
    rows["5x2 CV t-test"] = five_by_two_cv(X, y, best, runner, seeds(6, 2, 5))
    print(f"accuracy({best}) - accuracy({runner}):")
    for name, r in rows.items():
        print(f"  {name:<27} {r[0]:+.4f}" + (f"  95% CI {r[1]:+.4f} to {r[2]:+.4f}  p = {r[3]:.3f}"
                                             if len(r) > 1 else "  (one number, no error bar)"))
    return rows, d


if __name__ == "__main__":
    e6 = e6_fix()
# --- end ---


# ======================================================================= repo-only
def run(out, img, quick=False):
    import pandas as pd
    from experiments.common import tee
    with tee(os.path.join(out, "e6_fix.txt")):
        print("E6 - comparing two models honestly")
        print("setting: breast cancer, n = 150 (E1 opening scene, subsample 0); the top pair over the E1 split"
              " seeds; the cherry-picked seed is E1's deliberate worst case")
        rows, d = e6_fix(quick)
        extras(rows, d)
    pd.DataFrame([(k, *r, *([np.nan] * (4 - len(r)))) for k, r in rows.items()],
                 columns=["procedure", "estimate", "ci_low", "ci_high", "p"]).to_csv(os.path.join(out, "e6_fix.csv"), index=False)
    pd.DataFrame({"fold": range(len(d)), "diff": d}).to_csv(os.path.join(out, "e6_fold_diffs.csv"), index=False)
    figure(rows, img)


def extras(rows, d):
    from experiments.common import fmt, verdict
    print("")
    print(f"10x5 CV: {len(d)} folds of 120 train / 30 test rows; fold std {fmt(d.std(ddof=1), 4)};"
          f" folds where the runner-up wins: {int((d < 0).sum())}, ties: {int((d == 0).sum())},"
          f" the better-on-average model wins: {int((d > 0).sum())}")
    k = len(d)
    print(f"the corrected test's standard error is sqrt(1 + k * n_test / n_train) = {fmt(np.sqrt(1 + k * 30 / 120), 2)}x"
          " the naive one")
    corr = rows["10x5 CV, corrected t-test"]
    print("")
    print("== Pre-registered predictions ==")
    print(f"P6.1 the single-seed winner is not significant under the corrected test: corrected p = {fmt(corr[3], 3)},"
          f" estimate {fmt(corr[0], 4)} -> {verdict(corr[3] > 0.05 or corr[0] > 0)}")
    print(f"P6.2 the corrected estimate points toward the better-on-average model: estimate {fmt(corr[0], 4)}"
          f" -> {verdict(corr[0] > 0)}")


def figure(rows, img):
    from experiments.common import OKABE_ITO, dark_style, save_fig
    plt = dark_style()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cols = [OKABE_ITO["yellow"], OKABE_ITO["orange"], OKABE_ITO["sky"], OKABE_ITO["green"]]
    names = list(rows)
    for j, (name, c) in enumerate(zip(names, cols)):
        y_ = len(names) - 1 - j
        r = rows[name]
        if len(r) > 1:
            ax.hlines(y_, r[1], r[2], color=c, lw=3)
            ax.text(r[2] + 0.004, y_ + 0.15, "p < 0.001" if r[3] < 0.001 else f"p = {r[3]:.3f}", fontsize=10)
        else:
            ax.text(r[0], y_ - 0.3, "one number, no error bar", fontsize=9, ha="center", color="#BBBBBB")
        ax.plot(r[0], y_, "o", color=c, ms=10, mec="#FFFFFF")
    ax.axvline(0, color="#FFFFFF", ls="--", lw=1.2)
    ax.set_yticks(range(len(names))[::-1], names)
    ax.set_xlabel("accuracy difference, better-on-average model minus runner-up (95% CI)")
    ax.set_title("Same two models, four verdicts")
    ax.grid(axis="y", visible=False)
    save_fig(fig, img, "e6_procedures")
