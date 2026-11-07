"""E10 (descriptive, not pre-registered; added 2026-11-06): the article's "count the disagreements" check as code,
applied to the E7 router data. Reads results/e7_router_correct.csv (one row per test query: did router v1 / v2
route it correctly?); no refitting. Draws 300-query golden sets from the 3,080 test queries.

`python seeds_matter.py --only e10` writes results/e10_mcnemar.txt.
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- article: mcnemar ---
import numpy as np
from scipy.stats import binomtest


def compare(a_right, b_right):
    """Two models scored on the SAME rows. Only the rows where exactly one is right can pick a winner."""
    a_only = int(np.sum(a_right & ~b_right))                    # A right, B wrong
    b_only = int(np.sum(~a_right & b_right))                    # B right, A wrong
    d = a_only + b_only                                         # the disagreements
    p = binomtest(a_only, d, 0.5).pvalue if d else 1.0          # exact McNemar test
    verdict = "too close to call" if abs(a_only - b_only) < 2 * np.sqrt(d) else "real gap"   # the rule of thumb
    return a_only, b_only, p, verdict
# --- end ---


# ======================================================================= repo-only
def run(out, img=None, quick=False):
    import pandas as pd
    from experiments.common import seeds, tee
    df = pd.read_csv(os.path.join(out, "e7_router_correct.csv"))
    v1, v2 = df.iloc[:, 0].to_numpy(bool), df.iloc[:, 1].to_numpy(bool)
    n_sets = 100 if quick else 1000
    with tee(os.path.join(out, "e10_mcnemar.txt")):
        print("E10 descriptive (not pre-registered): the disagreement check on 300-query golden sets drawn from the"
              f" {len(v1):,} E7 test queries, router v2 (A) against v1 (B)")
        res = [compare(v2[i], v1[i]) for i in
               (np.random.default_rng(s).choice(len(v1), 300, replace=False) for s in seeds(10, 0, n_sets))]
        a, b, p, verdict = res[0]
        p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        print(f"golden set 1: compare(v2, v1) -> ({a}, {b}, {p_txt}, '{verdict}'): v2 alone right on {a} queries, v1 alone"
              f" on {b}; a lead of {a - b} on {a + b} disagreements, against a bar of 2 * sqrt({a + b}) = {2 * np.sqrt(a + b):.1f}")
        rule = np.array([r[3] == "real gap" for r in res])
        exact = np.array([r[2] < 0.05 for r in res])
        d = np.array([r[0] + r[1] for r in res])
        print(f"over {n_sets:,} golden sets: the rule of thumb calls a real gap on {rule.mean():.1%}, the exact test"
              f" (p < 0.05) on {exact.mean():.1%}; they agree on {np.mean(rule == exact):.1%} of golden sets")
        print(f"disagreements per golden set: median {np.median(d):.0f} of 300 ({np.median(d) / 3:.1f}%)")
