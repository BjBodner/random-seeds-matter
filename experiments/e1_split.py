"""E1 — The split lottery, and *when* it wins: opening scene + boundary map R(n).

`python seeds_matter.py --only e1` writes results/ and images/; running this file directly
just prints the article's numbers and draws the hero figure.
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (QUICK, bootstrap_ci, np, pmap, seeds, subsample,  # noqa: F401
                                synthetic_pool)

# --- article: e1 ---
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODELS = {  # every model's internals are pinned; only the split will move
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=0),
    "SVC": lambda: make_pipeline(StandardScaler(), SVC()),
    "kNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier()),
}
SIZES = {"breast_cancer": [100, 150, 250, 400, 569],
         "synthetic": [100, 250, 500, 1000, 2500, 5000]}


def split_lottery(X, y, split_seed):
    """Accuracy of every model on one stratified 70/30 split."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=split_seed)
    return [make().fit(Xtr, ytr).score(Xte, yte) for make in MODELS.values()]


def lottery(X, y, split_seeds):
    """One row per split seed, one column per model."""
    return np.array(pmap(split_lottery, [(X, y, s) for s in split_seeds]))


def seeds_vs_models(acc):
    """R = typical seed-to-seed swing / typical model-to-model gap, plus the flip rate."""
    seed_spread = acc.std(axis=0, ddof=1).mean()      # how much one model moves
    means = acc.mean(axis=0)
    model_spread = means.std(ddof=1)                  # how far apart the models sit
    best, runner_up = np.argsort(means)[::-1][:2]
    flip = np.mean(acc[:, runner_up] > acc[:, best])  # runner-up strictly wins
    return seed_spread, model_spread, seed_spread / model_spread, flip


def e1_split_lottery(quick=QUICK):
    n_sub, n_split = (2, 30) if quick else (5, 200)
    split_seeds = seeds(1, 2, n_split)
    data = {"breast_cancer": load_breast_cancer(return_X_y=True), "synthetic": synthetic_pool()}
    cells = {}
    for k, (name, (X, y)) in enumerate(data.items()):
        for n in SIZES[name]:
            for j, sub_seed in enumerate(seeds(1, k, n_sub)):
                if n >= len(y) and j > 0:          # full dataset: every subsample is identical
                    cells[name, n, j] = cells[name, n, 0]
                    continue
                cells[name, n, j] = lottery(*subsample(X, y, n, sub_seed), split_seeds)

    acc = cells["breast_cancer", 150, 0]             # the opening scene
    names, means = list(MODELS), acc.mean(axis=0)
    for m, name in enumerate(names):
        print(f"{name:<13} mean {means[m]:.3f}  std {acc[:, m].std(ddof=1):.3f}  "
              f"min {acc[:, m].min():.3f}  max {acc[:, m].max():.3f}")
    seed_spread, model_spread, R, flip = seeds_vs_models(acc)
    lo, hi = bootstrap_ci(lambda a: seeds_vs_models(a)[2], acc, seed=1)
    print(f"n=150: seed spread {seed_spread:.4f}, model spread {model_spread:.4f}, "
          f"R = {R:.2f} (95% CI {lo:.2f} to {hi:.2f}), flip rate {flip:.1%}")
    best, runner = np.argsort(means)[::-1][:2]
    w = np.argmax(acc[:, runner] - acc[:, best])     # the deliberate worst case
    print(f"worst case (cherry-picked): split seed {split_seeds[w]}: {names[runner]} "
          f"{acc[w, runner]:.3f} vs {names[best]} {acc[w, best]:.3f}")

    for name in SIZES:                               # the boundary map
        for n in SIZES[name]:
            Rs = [seeds_vs_models(cells[name, n, j])[2] for j in range(n_sub)]
            print(f"{name:<13} n={n:<5} median R {np.median(Rs):.2f} "
                  f"(min {min(Rs):.2f}, max {max(Rs):.2f})")
    return cells


def plot_boundary_map(cells, path="images/e1b_boundary_map.png"):
    """The hero figure: R(n) per dataset, median line and min-max band over subsamples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    n_sub = 1 + max(j for (_, _, j) in cells)
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor="#121212")
    ax.set_facecolor("#121212")
    for name, color in (("breast_cancer", "#E69F00"), ("synthetic", "#56B4E9")):
        R = np.array([[seeds_vs_models(cells[name, n, j])[2] for j in range(n_sub)]
                      for n in SIZES[name]])
        ax.fill_between(SIZES[name], R.min(axis=1), R.max(axis=1), color=color, alpha=0.25, lw=0)
        ax.plot(SIZES[name], np.median(R, axis=1), "o-", color=color, lw=2.6, ms=6,
                label={"breast_cancer": "Breast cancer (UCI)", "synthetic": "Synthetic"}[name])
    allR = np.array([seeds_vs_models(a)[2] for a in cells.values()])
    ax.axhline(1, color="white", ls="--", lw=1.4)
    ax.set_ylim(min(allR.min(), 0.5) / 1.6, max(allR.max(), 2) * 1.8)
    ax.set(xscale="log", yscale="log", xlabel="dataset size n (log scale)",
           ylabel="R = split-seed spread / model spread (log scale)")
    ax.text(0.02, 0.96, "seeds win (R > 1)", transform=ax.transAxes, va="top", weight="bold", color="white")
    ax.text(0.02, 0.04, "models win (R < 1)", transform=ax.transAxes, weight="bold", color="white")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.4g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticks([100, 250, 500, 1000, 2500, 5000])
    lo_y, hi_y = ax.get_ylim()
    ax.set_yticks([t for t in (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16) if lo_y <= t <= hi_y])
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BBBBBB")
    ax.grid(color="#333333", lw=0.6)
    ax.legend(frameon=False, labelcolor="white", loc="upper right")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#121212")
    plt.close(fig)


if __name__ == "__main__":
    e1_cells = e1_split_lottery()
    plot_boundary_map(e1_cells)
# --- end ---


# ======================================================================= repo-only
def run(out, img, quick=False):
    import pandas as pd
    from experiments.common import fmt, pct, tee, verdict
    with tee(os.path.join(out, "e1_split.txt")):
        print("E1 - the split lottery and the boundary map")
        n_sub, n_split = (2, 30) if quick else (5, 200)
        print(f"design: {n_sub} subsample seeds x {n_split} split seeds per n; test_size 0.3, stratified;"
              f" models {list(MODELS)}; model internals fixed")
        cells = e1_split_lottery(quick)
        extras(cells, n_sub, n_split)
    split_seeds = seeds(1, 2, n_split)
    rows = [(d, n, j, split_seeds[i], name, a[i, m])
            for (d, n, j), a in cells.items() for i in range(a.shape[0]) for m, name in enumerate(MODELS)]
    pd.DataFrame(rows, columns=["dataset", "n", "subsample", "split_seed", "model", "accuracy"]) \
        .to_csv(os.path.join(out, "e1_split.csv"), index=False)
    plot_boundary_map(cells, os.path.join(img, "e1b_boundary_map.png"))
    figures(cells, img)


def load_cells(out):
    """Rebuild the accuracy matrices from results/e1_split.csv (no refitting)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(out, "e1_split.csv"))
    cells = {}
    for (d, n, j), g in df.groupby(["dataset", "n", "subsample"], sort=False):
        order = g.split_seed.drop_duplicates()
        m = g.pivot(index="split_seed", columns="model", values="accuracy").reindex(order)
        cells[d, int(n), int(j)] = m[list(MODELS)].to_numpy()
    return cells


def redraw(out, img):
    cells = load_cells(out)
    plot_boundary_map(cells, os.path.join(img, "e1b_boundary_map.png"))
    figures(cells, img)


def crossing(ns, rs):
    """First n where the median R drops below 1, log-interpolated between grid points."""
    for (n0, r0), (n1, r1) in zip(zip(ns, rs), zip(ns[1:], rs[1:])):
        if r0 >= 1 > r1:
            t = np.log(r0) / (np.log(r0) - np.log(r1))
            return n0, n1, float(np.exp(np.log(n0) + t * (np.log(n1) - np.log(n0))))
    return None


def extras(cells, n_sub, n_split):
    """Everything else the article quotes, plus the pre-registration check."""
    from experiments.common import fmt, pct, verdict
    names = list(MODELS)
    acc = cells["breast_cancer", 150, 0]
    means = acc.mean(axis=0)
    best, runner = np.argsort(means)[::-1][:2]
    _, _, R, flip = seeds_vs_models(acc)
    f_lo, f_hi = bootstrap_ci(lambda a: seeds_vs_models(a)[3], acc, seed=2)
    ties = np.mean(acc[:, runner] == acc[:, best])
    diff = acc[:, runner] - acc[:, best]
    w = int(np.argmax(diff))
    print("")
    print("== Opening scene details (breast cancer, n = 150, subsample 0) ==")
    print(f"test rows per split: {round(0.3 * 150)}")
    print(f"gap between best and worst model means: {fmt(means.max() - means.min())}")
    print(f"accuracy range per model: " + ", ".join(f"{names[m]} {fmt(acc[:, m].max() - acc[:, m].min())}"
                                                  for m in range(len(names))))
    print(f"top pair: {names[best]} (better on average) vs {names[runner]}")
    print(f"flip rate 95% CI {pct(f_lo)} to {pct(f_hi)}; ties {pct(ties)};"
          f" better model strictly wins {pct(1 - flip - ties)}")
    print(f"runner-up wins on {int((diff > 0).sum())} of {n_split} seeds; ties on {int((diff == 0).sum())}")
    print(f"worst-case margin {fmt(diff[w])}; seeds with a margin at least that large: {int((diff >= diff[w]).sum())}")
    print(f"mean paired difference ({names[runner]} - {names[best]}): {fmt(diff.mean(), 4)}")
    sub = [seeds_vs_models(cells["breast_cancer", 150, j]) for j in range(n_sub)]
    print("n=150 R per subsample: " + ", ".join(fmt(s[2], 2) for s in sub)
          + "; flip rate per subsample: " + ", ".join(pct(s[3]) for s in sub))

    print("")
    print("== Boundary map details ==")
    print("dataset        n     seed_spread  model_spread  flip_median")
    med = {}
    for name in SIZES:
        for n in SIZES[name]:
            st = [seeds_vs_models(cells[name, n, j]) for j in range(n_sub)]
            med[name, n] = np.median([s[2] for s in st])
            print(f"{name:<14} {n:<5} {fmt(np.median([s[0] for s in st]), 4):>11}  "
                  f"{fmt(np.median([s[1] for s in st]), 4):>12}  {pct(np.median([s[3] for s in st])):>11}")
        c = crossing(SIZES[name], [med[name, n] for n in SIZES[name]])
        if c is None:
            m = [med[name, n] for n in SIZES[name]]
            where = "above" if min(m) >= 1 else ("below" if max(m) < 1 else "on both sides of")
            print(f"{name}: median R never crosses 1 downward on the tested grid (all medians {where} 1;"
                  f" smallest {fmt(min(m), 2)})")
        else:
            print(f"{name}: median R crosses 1 between n = {c[0]} and n = {c[1]}"
                  f" (log-interpolated n = {int(round(c[2], -1))})")

    print("")
    print("== Pre-registered predictions ==")
    m150 = med["breast_cancer", 150]
    print(f"P1.1 median R at n = 150 (breast cancer) >= 2.0: observed {fmt(m150, 2)} -> {verdict(m150 >= 2.0)}")
    bc_above = all(med["breast_cancer", n] > 1 for n in SIZES["breast_cancer"])
    print(f"P1.2a breast cancer median R > 1 at every n: {verdict(bc_above)}")
    c = crossing(SIZES["synthetic"], [med["synthetic", n] for n in SIZES["synthetic"]])
    ok = c is not None and c[0] >= 250 and c[1] <= 1000
    print(f"P1.2b synthetic crosses 1 between n = 250 and n = 1000: {verdict(ok)}"
          f" (observed: {'between ' + str(c[0]) + ' and ' + str(c[1]) if c else 'no crossing'})")
    print(f"P1.3 flip rate at n = 150 (subsample 0) >= 20%: observed {pct(flip)} -> {verdict(flip >= 0.20)}")
    trig = all(v < 1 for v in med.values())
    print(f"Falsifier (median R < 1 at every tested n on both datasets): {'TRIGGERED' if trig else 'not triggered'}")


def figures(cells, img):
    from experiments.common import MODEL_COLORS, NEUTRAL, OKABE_ITO, dark_style, fmt, save_fig
    plt = dark_style()
    names = list(MODELS)
    acc = cells["breast_cancer", 150, 0]
    means = acc.mean(axis=0)
    order = list(np.argsort(means)[::-1])
    best, runner = order[0], order[1]
    diff = acc[:, runner] - acc[:, best]
    w = int(np.argmax(diff))
    rng = np.random.default_rng(0)

    # (a) 200 parallel universes
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for x, m in enumerate(order):
        a = acc[:, m]
        vp = ax.violinplot(a, positions=[x], widths=0.8, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(MODEL_COLORS[names[m]]); b.set_alpha(0.18); b.set_edgecolor("none")
        ax.scatter(x + rng.uniform(-0.22, 0.22, len(a)), a, s=9, alpha=0.55,
                   color=MODEL_COLORS[names[m]], linewidths=0)
        ax.hlines(a.mean(), x - 0.32, x + 0.32, color="#FFFFFF", lw=2.2)
    xb, xr = order.index(best), order.index(runner)
    ax.plot([xb, xr], [acc[w, best], acc[w, runner]], color=OKABE_ITO["yellow"], lw=1.6, ls="--",
            marker="o", ms=8, mfc="none", mew=1.8)
    ax.annotate("one split seed, cherry-picked as\nthe deliberate worst case",
                xy=(xr + 0.05, acc[w, runner]), xytext=(xr + 0.55, acc[w, runner] - 0.04),
                fontsize=10, color=OKABE_ITO["yellow"], va="center",
                arrowprops=dict(arrowstyle="->", color=OKABE_ITO["yellow"]))
    lo, hi = means.min(), means.max()
    xg = len(order) - 0.35
    ax.annotate("", xy=(xg, lo), xytext=(xg, hi), arrowprops=dict(arrowstyle="|-|", color="#FFFFFF", lw=1.4))
    ax.text(xg + 0.08, (lo + hi) / 2, f"all 4 model\nmeans sit\nwithin {fmt(hi - lo)}", fontsize=10, va="center")
    ax.set_xticks(range(len(order)), [names[m] for m in order])
    ax.set_xlim(-0.6, len(order) + 0.45)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Same 150 rows, same 4 models, {acc.shape[0]} different train/test splits")
    ax.grid(axis="x", visible=False)
    save_fig(fig, img, "e1a_parallel_universes")

    # (c) the cherry-pick, shown inside its distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    step = 1 / round(0.3 * 150)
    edges = np.arange(diff.min() - step / 2, diff.max() + step, step)
    counts, _, patches = ax.hist(diff, bins=edges, color=NEUTRAL, alpha=0.8, edgecolor="#121212")
    for p, left in zip(patches, edges[:-1]):
        if left <= diff[w] < left + step:
            p.set_facecolor(OKABE_ITO["yellow"])
    ax.axvline(0, color=OKABE_ITO["sky"], lw=1.4)
    ax.axvline(diff.mean(), color="#FFFFFF", lw=1.6, ls="--")
    ax.text(diff.mean(), counts.max() * 0.93, " average over\n all seeds", fontsize=10, va="top")
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.annotate(f"deliberate worst case\n(1 of {len(diff)} seeds)", xy=(diff[w], 1.2),
                xytext=(diff[w], counts.max() * 0.55), ha="center", fontsize=10, color=OKABE_ITO["yellow"],
                arrowprops=dict(arrowstyle="->", color=OKABE_ITO["yellow"]))
    ax.set_xlabel(f"accuracy({names[runner]}) - accuracy({names[best]}) on the same split")
    ax.set_ylabel("number of split seeds")
    ax.set_title(f"{names[runner]} vs {names[best]}: the cherry-pick inside its distribution")
    ins = ax.inset_axes([0.05, 0.5, 0.24, 0.38])
    ins.bar([0, 1], [acc[w, best], acc[w, runner]],
            color=[MODEL_COLORS[names[best]], MODEL_COLORS[names[runner]]], width=0.6)
    for x_, v in enumerate([acc[w, best], acc[w, runner]]):
        ins.text(x_, v + 0.005, fmt(v), ha="center", fontsize=9)
    ins.set_xticks([0, 1], [names[best], names[runner]], fontsize=9)
    ins.set_ylim(min(acc[w, best], acc[w, runner]) - 0.08, 1.03)
    ins.set_title("on the worst-case seed", fontsize=9, fontweight="normal")
    ins.set_yticks([])
    ins.spines["left"].set_visible(False)
    ins.tick_params(labelsize=8)
    ins.grid(False)
    save_fig(fig, img, "e1c_worst_case")
