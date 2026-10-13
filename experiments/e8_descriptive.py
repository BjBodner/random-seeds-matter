"""E8 descriptive follow-up (NOT pre-registered): is the hero ratio an artifact of comparing model
*averages* with single runs? Reads results/e8_tailored.csv; no refitting.

`python seeds_matter.py --only e8d` writes results/e8_descriptive.txt.
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import np  # noqa: E402
from experiments.e8_tailored import HERO_MODEL, MODEL_NAMES, TAILORED, hero_choice, load_cells, seeds_vs_zoo  # noqa: E402


def per_split(acc):
    """Same data AND same split: the spread across the 8 models on each split, pooled over splits."""
    sd = np.sqrt(acc.var(axis=1, ddof=1).mean())           # root-mean-square of the per-split stds
    ties = np.mean((acc == acc[:, :1]).all(axis=1))         # splits on which all 8 scores are identical
    return sd, ties


def run(out, img=None, quick=False):
    from experiments.common import tee
    cells, n_data, _, _ = load_cells(out)
    with tee(os.path.join(out, "e8_descriptive.txt")):
        print("E8 descriptive (not pre-registered): spread across the 8 models on the SAME split")
        print("per-split model spread = root-mean-square over splits of the std across the 8 models on that split")
        for name in TAILORED:
            sds, ratios = [], []
            for j in range(n_data):
                acc = cells[name, j][0]
                sd, _ = per_split(acc)
                sds.append(sd)
                ratios.append(seeds_vs_zoo(acc)[0] / sd)
            print(f"{name:<43} per-split model spread {100 * np.median(sds):.2f} pts; "
                  f"seed spread / per-split model spread {np.median(ratios):.1f}x (median of {n_data} datasets)")
        design, inst, _ = hero_choice(cells, n_data)
        acc = cells[design, inst][0]
        sd, ties = per_split(acc)
        lr = acc[:, MODEL_NAMES.index(HERO_MODEL)]
        print("")
        print(f"hero ({design}, dataset {inst}): per-split model spread {100 * sd:.2f} pts; "
              f"all 8 models tie exactly on {ties:.1%} of splits; {HERO_MODEL} seed spread {100 * lr.std(ddof=1):.2f} pts "
              f"-> {lr.std(ddof=1) / sd:.1f}x even split by split")
        print(f"hero: {HERO_MODEL} averages {100 * lr.mean():.1f}%; one golden set reports anything from "
              f"{100 * lr.min():.1f}% to {100 * lr.max():.1f}%; middle 90% of seeds {100 * np.percentile(lr, 5):.1f}% "
              f"to {100 * np.percentile(lr, 95):.1f}%")


if __name__ == "__main__":
    from experiments.common import out_dirs
    run(out_dirs(False)[0])
