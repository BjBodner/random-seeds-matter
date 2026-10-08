"""E5 robustness check: n = 300 with the MLP exactly as pre-registered (no early stopping).

The pre-registration specified `MLPClassifier(max_iter=500)`; the main E5 run used `early_stopping=True`
for runtime (see article/DECISIONS.md). At n = 300 the plain MLP is affordable, so this re-runs that size
with the pre-registered spec, the same seeds and the same split/internal-seed design, and re-checks P5.1.

    python experiments/e5_prereg_mlp.py
"""
import os
import sys
import warnings

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.common import REPO, pct, pmap, seeds, subsample, synthetic_pool, tee
from experiments.e5_decomposition import variance_shares

PREREG_MODELS = {
    "RandomForest": lambda s: RandomForestClassifier(n_estimators=100, random_state=s),
    "MLP (pre-registered, no early stopping)": lambda s: make_pipeline(StandardScaler(), MLPClassifier(
        max_iter=500, random_state=s)),
    "GradBoost": lambda s: GradientBoostingClassifier(subsample=0.8, random_state=s),
}


def cell(X, y, split_seed, internal_seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=split_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [make(internal_seed).fit(Xtr, ytr).score(Xte, yte) for make in PREREG_MODELS.values()]


def run(out=os.path.join(REPO, "results")):
    n_s = 30
    split_seeds, internal_seeds = seeds(5, 1, n_s), seeds(5, 2, n_s)
    X, y = synthetic_pool()
    Xs, ys = subsample(X, y, 300, seeds(5, 0, 2)[0])          # the same n = 300 subsample as E5
    cells = pmap(cell, [(Xs, ys, s, i) for s in split_seeds for i in internal_seeds])
    A = np.array(cells).reshape(n_s, n_s, -1).transpose(2, 0, 1)
    sh = variance_shares(A)
    with tee(os.path.join(out, "e5_prereg_mlp.txt")):
        print("E5 robustness check: n = 300 with the MLP exactly as pre-registered (max_iter=500, no early stopping)")
        print("n=300: " + ", ".join(f"{name} {A[k].mean():.3f}" for k, name in enumerate(PREREG_MODELS)))
        print("   " + ", ".join(f"{k} {pct(v)}" for k, v in sh.items()))
        print(f"P5.1 under the exact pre-registered spec (model share <= 30%): observed {pct(sh['model'])} -> "
              f"{'HELD' if sh['model'] <= 0.30 else 'FAILED'}")


if __name__ == "__main__":
    run()
