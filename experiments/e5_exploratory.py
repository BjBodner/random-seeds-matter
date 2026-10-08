"""E5 follow-up (EXPLORATORY, not pre-registered): the decomposition for the two close models only,
plus the combined E5 figure.

E5's pre-registered decomposition includes an MLP that trails the tree models by about 0.12 at n = 300,
so the model factor is mostly "MLP vs the rest". This asks what happens when the candidates are close
(RandomForest vs GradBoost), using the fits E5 already saved. No new fits are run.

    python experiments/e5_exploratory.py      # reads results/e5_decomposition.csv
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments.common import REPO, pct, tee
from experiments.e5_decomposition import E5_MODELS, variance_shares

PAIR = ["RandomForest", "GradBoost"]


def matrix(d, models):
    """A[model, split, internal] from the long CSV rows of one size."""
    return np.stack([d[d.model == m].pivot(index="split_seed", columns="internal_seed", values="accuracy").to_numpy()
                     for m in models])


def run(out=os.path.join(REPO, "results"), img=os.path.join(REPO, "images"), quick=False):
    df = pd.read_csv(os.path.join(out, "e5_decomposition.csv"))
    bars = {}
    with tee(os.path.join(out, "e5_exploratory.txt")):
        print("E5 follow-up (exploratory, not pre-registered): RandomForest vs GradBoost only")
        for n, d in df.groupby("n"):
            A = matrix(d, PAIR)
            sh = variance_shares(A)
            print(f"n={n}: mean accuracy gap (RandomForest - GradBoost) {A[0].mean() - A[1].mean():.3f};"
                  f" model share {pct(sh['model'])}; everything seed-driven {pct(1 - sh['model'])}")
            print(f"   can change the winner (model x split + model x internal + residual): "
                  f"{pct(sh['model x split'] + sh['model x internal'] + sh['residual'])}")
            print("   " + ", ".join(f"{k} {pct(v)}" for k, v in sh.items()))
            A3 = matrix(d, list(E5_MODELS))
            for label, M in (("RF vs GradBoost", A), ("3 models as run", A3)):
                lo, hi = crossed_bootstrap_model_share(M, seed=n)
                print(f"   {label}: model share {pct(variance_shares(M)['model'])}, 95% CI {pct(lo)} to {pct(hi)}"
                      " (crossed bootstrap over split and internal seeds, 2000 resamples)")
            bars[f"3 models, n = {n:,}"] = variance_shares(A3)
            bars[f"RF vs GradBoost, n = {n:,}"] = sh
    prereg = prereg_shares(out)
    order = ["3 models, n = 300", "3 models, n = 3,000", "RF vs GradBoost, n = 300", "RF vs GradBoost, n = 3,000"]
    figure({k: bars[k] for k in order}, img, prereg)


def crossed_bootstrap_model_share(A, n_boot=2000, seed=0):
    """95% CI for the model share, resampling split seeds and internal seeds independently."""
    rng = np.random.default_rng(seed)
    S, I = A.shape[1], A.shape[2]
    vals = [variance_shares(A[:, rng.integers(0, S, S)][:, :, rng.integers(0, I, I)])["model"] for _ in range(n_boot)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def prereg_shares(out):
    """Shares from the robustness run with the MLP exactly as pre-registered (results/e5_prereg_mlp.txt)."""
    import re
    path = os.path.join(out, "e5_prereg_mlp.txt")
    if not os.path.exists(path):
        return None
    line = [ln for ln in open(path, encoding="utf-8") if ln.strip().startswith("model ")][0]
    return {k: float(v) / 100 for k, v in re.findall(r"([a-z][a-z x]+?) (-?[\d.]+)%", line)}


def figure(bars, img, prereg=None):
    from experiments.common import NEUTRAL, OKABE_ITO, dark_style, save_fig
    plt = dark_style()
    keys = ["model", "split seed", "internal seed", "model x split", "model x internal", "split x internal", "residual"]
    colors = dict(zip(keys, [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["sky"], OKABE_ITO["green"],
                             OKABE_ITO["purple"], OKABE_ITO["yellow"], NEUTRAL]))
    dark_text = {"split seed", "internal seed", "split x internal", "residual", "model x split"}
    fig, ax = plt.subplots(figsize=(11, 6.6))
    rows = list(bars.items())
    ys = [5.6, 4.6, 1.4, 0.4]
    if prereg:
        rows = rows[:2] + [("3 models, n = 300", prereg)] + rows[2:]
        ys = [5.6, 4.6, 3.0, 1.4, 0.4]
    for y_, (label, sh) in zip(ys, rows):
        left = 0
        for k in keys:
            v = max(sh[k], 0)
            ax.barh(y_, v, left=left, color=colors[k], edgecolor="#121212", height=0.7,
                    label=k if y_ == ys[0] else None)
            if v >= 0.05:
                ax.text(left + v / 2, y_, f"{100 * v:.0f}%", ha="center", va="center", fontsize=11, fontweight="bold",
                        color="#121212" if k in dark_text else "#FFFFFF")
            left += v
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.text(1.0, 6.22, "as run: RandomForest, MLP with early stopping, GradBoost", ha="right", fontsize=10, color="#BBBBBB")
    if prereg:
        ax.text(1.0, 3.62, "robustness: the MLP exactly as pre-registered (no early stopping)", ha="right", fontsize=10,
                color="#BBBBBB")
    ax.text(1.0, 2.02, "exploratory: only the two close models (RF vs GradBoost)", ha="right", fontsize=10, color="#BBBBBB")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 6.4)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(t * 100)}%" for t in np.linspace(0, 1, 6)])
    ax.set_xlabel("share of total accuracy variance")
    ax.grid(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10)
    ax.set_title("Who owns the variance? It depends on how close your models are")
    save_fig(fig, img, "e5_variance_shares")


if __name__ == "__main__":
    run()
