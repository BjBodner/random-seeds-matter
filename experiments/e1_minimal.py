"""E1, minimal (descriptive re-run, not pre-registered; added 2026-11-06): the article's opening scene as one
self-contained script. Same seeds, data and models as experiments/e1_split.py (first subsample, n = 150, breast
cancer), so it prints the same numbers; no joblib and no imports from this repo, so a reader can paste it and run it.

`python experiments/e1_minimal.py` (about a minute) or `python seeds_matter.py --only e1m` (writes results/e1_minimal.txt).
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- article: e1min ---
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT_SEED = 20261008  # every random number in the post grows from this one


def seeds(experiment, stream, n):
    """n reproducible seeds for one named stream of random numbers."""
    ss = np.random.SeedSequence(ROOT_SEED, spawn_key=(experiment, stream))
    return [int(s) for s in ss.generate_state(n)]


MODELS = {  # each model's own seed is pinned, so only the split moves
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
    "SVC": lambda: make_pipeline(StandardScaler(), SVC()),
    "kNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier()),
}


def opening_scene(n_rows=150, n_splits=200):
    X, y = load_breast_cancer(return_X_y=True)
    keep = np.sort(np.random.default_rng(seeds(1, 0, 1)[0]).choice(len(y), n_rows, replace=False))
    X, y = X[keep], y[keep]                                     # the 150 labelled rows we "have"
    acc = []
    for split_seed in seeds(1, 2, n_splits):                    # 200 splits; nothing else changes
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split_seed)
        acc.append([make().fit(Xtr, ytr).score(Xte, yte) for make in MODELS.values()])
    acc = np.array(acc)                                         # one row per split, one column per model
    for name, a in zip(MODELS, acc.T):
        print(f"{name:<13} mean {a.mean():.3f}  std {a.std(ddof=1):.3f}  min {a.min():.3f}  max {a.max():.3f}")
    swing = acc.std(axis=0, ddof=1).mean()                      # how far one model moves across splits
    gap = acc.mean(axis=0).std(ddof=1)                          # how far apart the models' averages sit
    best, second = np.argsort(acc.mean(axis=0))[::-1][:2]
    print(f"wobble ratio {swing / gap:.2f}; the second-best model wins on {np.mean(acc[:, second] > acc[:, best]):.1%} "
          f"of splits and ties on {np.mean(acc[:, second] == acc[:, best]):.1%}")
    return acc


if __name__ == "__main__":
    opening_scene()
# --- end ---


# ======================================================================= repo-only
def run(out, img=None, quick=False):
    from experiments.common import tee
    with tee(os.path.join(out, "e1_minimal.txt")):
        print("E1 minimal (descriptive re-run, not pre-registered): the opening scene as one self-contained script;"
              " same seeds as E1, subsample 0, n = 150")
        opening_scene(n_splits=30 if quick else 200)
