"""E7/E7b descriptive follow-up (not a pre-registered prediction): how often does a golden set
show a 'headline' difference of 1 point or more between two routers? Reads results/e7_router.csv.

    python experiments/e7_descriptive.py
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from experiments.common import REPO, pct, tee


def run(out=os.path.join(REPO, "results")):
    df = pd.read_csv(os.path.join(out, "e7_router.csv"))
    df["gap_pts"] = 100 * (df.fail_worse - df.fail_better)
    with tee(os.path.join(out, "e7_descriptive.txt")):
        print("E7/E7b descriptive (not pre-registered): size of the measured difference between two routers")
        for (pair, g), d in df[df.g.isin([300, 500])].groupby(["pair", "g"], sort=False):
            big = (d.gap_pts.abs() >= 1.0).mean()
            wrong_big = (d.gap_pts <= -1.0).mean()
            print(f"{pair} | g={g}: |measured gap| >= 1 point on {pct(big)} of golden sets "
                  f"(the worse router ahead by >= 1 point on {pct(wrong_big)}); "
                  f"largest measured gap {d.gap_pts.abs().max():.1f} points")
        print("")
        print("Calculation (McNemar, two-sided alpha 0.05, 80% power; Connor 1987 formula): golden-set size needed")
        print("to detect a true gap in failure rate, given how often the two routers disagree")
        from scipy import stats
        za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
        print("gap (points)  disagree 5%  disagree 10%  disagree 20%")
        for gap in (1, 2, 3, 5):
            d_ = gap / 100
            cells = []
            for pdisc in (0.05, 0.10, 0.20):
                n = ((za * pdisc ** 0.5 + zb * (pdisc - d_ ** 2) ** 0.5) / d_) ** 2 if pdisc > d_ else float("nan")
                cells.append(f"{int(-(-n // 1)):>11,}" if n == n else f"{'n/a':>11}")
            print(f"{gap:<13} " + "  ".join(cells))


if __name__ == "__main__":
    run()
