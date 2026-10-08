"""E5 — Who owns the variance? Model x split seed x internal seed, fully crossed."""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import QUICK, np, pmap, seeds, subsample, synthetic_pool  # noqa: F401

# --- article: e5 ---
import warnings

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

E5_MODELS = {  # s = the model-internal seed
    "RandomForest": lambda s: RandomForestClassifier(n_estimators=100, random_state=s),
    "MLP": lambda s: make_pipeline(StandardScaler(), MLPClassifier(
        max_iter=500, early_stopping=True, random_state=s)),
    "GradBoost": lambda s: GradientBoostingClassifier(subsample=0.8, random_state=s),
}


def crossed_cell(X, y, split_seed, internal_seed):
    """One (split seed, internal seed) cell: every model's test accuracy."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [make(internal_seed).fit(Xtr, ytr).score(Xte, yte) for make in E5_MODELS.values()]


def variance_shares(A):
    """Balanced 3-way sum-of-squares split of A[model, split, internal] (no replication)."""
    M, S, I = A.shape
    g = A.mean()
    m, s, i = A.mean(axis=(1, 2)), A.mean(axis=(0, 2)), A.mean(axis=(0, 1))
    ms, mi, si = A.mean(axis=2), A.mean(axis=1), A.mean(axis=0)
    ss = {
        "model": S * I * ((m - g) ** 2).sum(),
        "split seed": M * I * ((s - g) ** 2).sum(),
        "internal seed": M * S * ((i - g) ** 2).sum(),
        "model x split": I * ((ms - m[:, None] - s[None, :] + g) ** 2).sum(),
        "model x internal": S * ((mi - m[:, None] - i[None, :] + g) ** 2).sum(),
        "split x internal": M * ((si - s[:, None] - i[None, :] + g) ** 2).sum(),
    }
    total = ((A - g) ** 2).sum()
    ss["residual"] = total - sum(ss.values())          # the 3-way interaction
    return {k: v / total for k, v in ss.items()}


def e5_decomposition(quick=QUICK):
    n_s = 8 if quick else 30
    split_seeds, internal_seeds = seeds(5, 1, n_s), seeds(5, 2, n_s)
    X, y = synthetic_pool()
    out = {}
    for n, sub_seed in zip([300, 3000], seeds(5, 0, 2)):
        Xs, ys = subsample(X, y, n, sub_seed)
        cells = pmap(crossed_cell, [(Xs, ys, s, i) for s in split_seeds for i in internal_seeds])
        A = np.array(cells).reshape(n_s, n_s, -1).transpose(2, 0, 1)   # [model, split, internal]
        out[n] = A
        print(f"n={n}: " + ", ".join(f"{name} {A[k].mean():.3f}" for k, name in enumerate(E5_MODELS))
              + f"; overall accuracy std {A.std(ddof=1):.4f}")
        print("   " + ", ".join(f"{k} {v:.1%}" for k, v in variance_shares(A).items()))
    return out


if __name__ == "__main__":
    e5 = e5_decomposition()
# --- end ---


# ======================================================================= repo-only
SHARE_KEYS = ["model", "split seed", "internal seed", "model x split", "model x internal",
              "split x internal", "residual"]


def run(out, img, quick=False):
    import pandas as pd
    from experiments.common import tee
    n_s = 8 if quick else 30
    split_seeds, internal_seeds = seeds(5, 1, n_s), seeds(5, 2, n_s)
    with tee(os.path.join(out, "e5_decomposition.txt")):
        print("E5 - variance decomposition")
        print(f"design: synthetic population; models {list(E5_MODELS)} x {n_s} split seeds x {n_s} internal seeds"
              f" = {3 * n_s * n_s} fits per size; sizes [300, 3000]; MLP uses early_stopping=True")
        res = e5_decomposition(quick)
        shares = extras(res)
    rows = [(n, name, split_seeds[s], internal_seeds[i], A[k, s, i])
            for n, A in res.items() for k, name in enumerate(E5_MODELS)
            for s in range(n_s) for i in range(n_s)]
    pd.DataFrame(rows, columns=["n", "model", "split_seed", "internal_seed", "accuracy"]) \
        .to_csv(os.path.join(out, "e5_decomposition.csv"), index=False)
    redraw(out, img)  # the exploratory close-pair split + the combined figure


def extras(res):
    from experiments.common import fmt, pct, verdict
    shares = {}
    for n, A in res.items():
        sh = variance_shares(A)
        shares[n] = sh
        print("")
        print(f"== n = {n} ({round(0.3 * n)} test rows) ==")
        for k, name in enumerate(E5_MODELS):
            print(f"{name:<13} std over all {A.shape[1] * A.shape[2]} seed pairs {fmt(A[k].std(ddof=1))},"
                  f" range {fmt(A[k].min())} to {fmt(A[k].max())}")
        print(f"model share {pct(sh['model'])}; everything seed-driven {pct(1 - sh['model'])}")
        print(f"split seed incl. its interactions: {pct(sh['split seed'] + sh['model x split'] + sh['split x internal'])};"
              f" internal seed incl. its interactions: {pct(sh['internal seed'] + sh['model x internal'] + sh['split x internal'])}")
    print("")
    print("== Pre-registered predictions ==")
    print(f"P5.1 n = 300: model share <= 30% (seed-driven >= 70%): observed {pct(shares[300]['model'])}"
          f" -> {verdict(shares[300]['model'] <= 0.30)}")
    print(f"P5.2 n = 3000: model share >= 50%: observed {pct(shares[3000]['model'])}"
          f" -> {verdict(shares[3000]['model'] >= 0.50)}")
    return shares


def redraw(out, img):
    from experiments import e5_exploratory
    e5_exploratory.run(out, img)
