"""E8 — The effect at full strength: tailored datasets where the seed dwarfs the model, then the dials.

Pre-registered in PREREGISTRATION.md (E8) before this file existed. `python seeds_matter.py --only e8`
writes results/ and images/; running this file directly prints the article's E8a numbers.
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import QUICK, np, pmap, seeds  # noqa: F401

# --- article: e8 ---
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ZOO = {  # 8 model families, each with a regularized setting for small, noisy data, fixed in advance
    "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression()),
    "Naive Bayes": lambda: GaussianNB(),
    "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC()),
    "kNN (k=31)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31)),
    "Decision tree": lambda: DecisionTreeClassifier(max_depth=2, random_state=0),
    "Random forest": lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=0),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, random_state=0),
    "Neural net": lambda: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(32,), alpha=1.0, max_iter=2000, random_state=0)),
}

TAILORED = {  # name: (data settings, training rows, test rows)
    "T1 overlapping classes, 300 rows": (dict(separation=2.0), 210, 90),
    "T2 disputed labels, 300 rows": (dict(separation=6.0, label_noise=0.2), 210, 90),
    "T3 100-row golden set, overlapping classes": (dict(separation=2.0), 1000, 100),
    "T4 100-row golden set, disputed labels": (dict(separation=6.0, label_noise=0.2), 1000, 100),
}


def tailored(n, separation, label_noise=0.0, bend=0.0, seed=0):
    """Two classes in 2-D whose means sit `separation` apart: where they overlap, no model can win.
    label_noise flips that share of labels (annotators who disagree); bend curves the true
    boundary to x0 = bend * (x1^2 - 1), which linear models can't follow."""
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.arange(n) % 2)
    X = rng.normal(size=(n, 2))
    X[:, 0] += (y - 0.5) * separation + bend * (X[:, 1] ** 2 - 1)
    flipped = rng.random(n) < label_noise
    return X, np.where(flipped, 1 - y, y)


def best_possible(separation, label_noise=0.0, bend=0.0):
    """The accuracy of a perfect model (the Bayes rate); the bend moves both classes, so it doesn't matter."""
    p = norm.cdf(separation / 2)
    return (1 - label_noise) * p + label_noise * (1 - p)


def zoo_on_split(X, y, test_rows, split_seed):
    """Accuracy of all 8 models on one split, and the share of test rows on which they all agree."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_rows, stratify=y, random_state=split_seed)
    preds = np.array([make().fit(Xtr, ytr).predict(Xte) for make in ZOO.values()])
    return (preds == yte).mean(axis=1), (preds == preds[0]).all(axis=0).mean()


def seeds_vs_zoo(acc):
    """acc: one row per split seed, one column per model. R is defined exactly as in Trap 1."""
    seed_spread = acc.std(axis=0, ddof=1).mean()   # how far the seed moves one model
    model_spread = acc.mean(axis=0).std(ddof=1)    # how far apart the 8 model averages sit
    return seed_spread, model_spread, seed_spread / model_spread


def e8_full_strength(quick=QUICK):
    n_data, n_split = (2, 30) if quick else (5, 200)
    split_seeds = seeds(8, 0, n_split)
    cells = {}
    for k, (name, (settings, n_train, n_test)) in enumerate(TAILORED.items()):
        for j, data_seed in enumerate(seeds(8, 1 + k, n_data)):
            X, y = tailored(n_train + n_test, **settings, seed=data_seed)
            runs = pmap(zoo_on_split, [(X, y, n_test, s) for s in split_seeds])
            cells[name, j] = np.array([r[0] for r in runs]), np.mean([r[1] for r in runs])
        Rs = [seeds_vs_zoo(cells[name, j][0])[2] for j in range(n_data)]
        agree = np.mean([cells[name, j][1] for j in range(n_data)])
        print(f"{name:<43} best possible {best_possible(**settings):.1%}; R = {np.median(Rs):.1f} (median of "
              f"{n_data} datasets, range {min(Rs):.1f} to {max(Rs):.1f}); all 8 models agree on {agree:.1%} of test rows")
    return cells


if __name__ == "__main__":
    e8_cells = e8_full_strength()
# --- end ---


# ======================================================================= repo-only
# E8b: the dials (one knob at a time, from T1's family: separation 2.0, no label noise)
DIALS = ([("D1 bend", dict(bend=b), 210, 90) for b in (0.0, 0.5, 1.0, 1.5, 2.0)]
         + [("D2 size, bend 0", dict(bend=0.0), round(0.7 * n), n - round(0.7 * n)) for n in (150, 300, 1000, 3000)]
         + [("D2 size, bend 1", dict(bend=1.0), round(0.7 * n), n - round(0.7 * n)) for n in (150, 300, 1000, 3000)]
         + [("D3 golden-set size", dict(bend=0.0), 1000, g) for g in (50, 100, 300, 1000)])
MODEL_NAMES = list(ZOO)
HERO_MODEL = "Logistic regression"


def e8_dials(quick=QUICK):
    n_data, n_split = (2, 30) if quick else (3, 200)
    split_seeds = seeds(8, 0, n_split)
    cells = {}
    for i, (dial, settings, n_train, n_test) in enumerate(DIALS):
        for j, data_seed in enumerate(seeds(8, 10 + i, n_data)):
            X, y = tailored(n_train + n_test, separation=2.0, **settings, seed=data_seed)
            runs = pmap(zoo_on_split, [(X, y, n_test, s) for s in split_seeds])
            cells[i, j] = np.array([r[0] for r in runs]), np.mean([r[1] for r in runs])
    return cells


def dial_label(i):
    dial, settings, n_train, n_test = DIALS[i]
    if dial.startswith("D1"):
        return f"bend {settings['bend']:g} (n = {n_train + n_test})"
    if dial.startswith("D2"):
        return f"n = {n_train + n_test} ({n_train} train / {n_test} test)"
    return f"golden set {n_test} rows ({n_train} train)"


def hero_choice(cells, n_data):
    """The pre-registered rule: the design with the largest median R, at its median-R instance."""
    med = {name: np.median([seeds_vs_zoo(cells[name, j][0])[2] for j in range(n_data)]) for name in TAILORED}
    design = max(med, key=med.get)
    Rs = [seeds_vs_zoo(cells[design, j][0])[2] for j in range(n_data)]
    instance = int(np.argsort(Rs)[(n_data - 1) // 2])
    return design, instance, med


def run(out, img, quick=False):
    import pandas as pd
    from experiments.common import tee
    n_data, n_split = (2, 30) if quick else (5, 200)
    split_seeds = seeds(8, 0, n_split)
    with tee(os.path.join(out, "e8_tailored.txt")):
        print("E8a - the effect at full strength: 4 tailored designs, 8 model families")
        print(f"design: {n_data} datasets per design x {n_split} stratified split seeds; models: {MODEL_NAMES};"
              " every model's internal seed fixed at 0; only the split moves")
        cells = e8_full_strength(quick)
        extras(cells, n_data, split_seeds)
    rows = [(name, j, split_seeds[i], m, round(float(acc[i, k]), 4), round(float(agree), 4))
            for (name, j), (acc, agree) in cells.items() for i in range(acc.shape[0]) for k, m in enumerate(MODEL_NAMES)]
    pd.DataFrame(rows, columns=["design", "dataset", "split_seed", "model", "accuracy", "agreement_all8"]) \
        .to_csv(os.path.join(out, "e8_tailored.csv"), index=False)
    with tee(os.path.join(out, "e8_dials.txt")):
        print("E8b - the dials: one knob at a time from T1's family (separation 2.0, no label noise)")
        nd = 2 if quick else 3
        print(f"design: {nd} datasets per setting x {n_split} split seeds; same 8 models")
        dcells = e8_dials(quick)
        extras_dials(dcells, nd)
    rows = [(DIALS[i][0], dial_label(i), j, split_seeds[s], m, round(float(acc[s, k]), 4))
            for (i, j), (acc, _) in dcells.items() for s in range(acc.shape[0]) for k, m in enumerate(MODEL_NAMES)]
    pd.DataFrame(rows, columns=["dial", "setting", "dataset", "split_seed", "model", "accuracy"]) \
        .to_csv(os.path.join(out, "e8_dials.csv"), index=False)
    figures(cells, n_data, dcells, nd, img)


def extras(cells, n_data, split_seeds):
    from experiments.common import bootstrap_ci, verdict
    print("")
    print("== Per design ==")
    for name in TAILORED:
        for j in range(n_data):
            acc, agree = cells[name, j]
            s, m, R = seeds_vs_zoo(acc)
            means = acc.mean(axis=0)
            print(f"{name} | dataset {j}: seed spread {100 * s:.2f} pts, model spread {100 * m:.2f} pts, R = {R:.1f};"
                  f" model means {100 * means.min():.1f}% to {100 * means.max():.1f}%; agreement {agree:.1%}")
    design, inst, med = hero_choice(cells, n_data)
    acc, agree = cells[design, inst]
    s, m, R = seeds_vs_zoo(acc)
    means = acc.mean(axis=0)
    h = MODEL_NAMES.index(HERO_MODEL)
    lr = acc[:, h]
    lo, hi = bootstrap_ci(lambda a: seeds_vs_zoo(a)[2], acc, seed=8)
    print("")
    print("== Hero (pre-registered rule: largest median R, median-R dataset) ==")
    print(f"hero design: {design}; dataset {inst} (median R of its {n_data})")
    print("model means: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in zip(MODEL_NAMES, means)))
    print(f"8 model means span {100 * means.min():.1f}% to {100 * means.max():.1f}% "
          f"(range {100 * np.ptp(means):.1f} pts, std {100 * m:.2f} pts)")
    print(f"{HERO_MODEL} across {len(lr)} split seeds: mean {100 * lr.mean():.1f}%, std {100 * lr.std(ddof=1):.2f} pts,"
          f" min {100 * lr.min():.1f}%, max {100 * lr.max():.1f}% (range {100 * np.ptp(lr):.1f} pts)")
    print(f"hero ratio ({HERO_MODEL} std across seeds / std of the 8 model means): "
          f"{lr.std(ddof=1) / m:.1f}x")
    print(f"R for this dataset (all 8 models): {R:.1f} (95% CI {lo:.1f} to {hi:.1f}, bootstrap over split seeds)")
    best, worst = int(np.argmax(means)), int(np.argmin(means))
    print(f"best model on average {MODEL_NAMES[best]} {100 * means[best]:.1f}%, worst {MODEL_NAMES[worst]}"
          f" {100 * means[worst]:.1f}%; training rows {TAILORED[design][1]}, test rows {TAILORED[design][2]}")
    print("")
    print("== Pre-registered predictions (E8a) ==")
    low = min(med.values())
    print(f"P8.1 every design has median R >= 4: smallest {low:.1f} -> {verdict(low >= 4)}")
    top = max(med.values())
    print(f"P8.2 largest median R >= 10: {top:.1f} -> {verdict(top >= 10)}")
    print(f"Falsifier (largest median R < 5): {'TRIGGERED' if top < 5 else 'not triggered'}")


def dial_R(dcells, i, nd):
    return np.median([seeds_vs_zoo(dcells[i, j][0])[2] for j in range(nd)])


def extras_dials(dcells, nd):
    from experiments.common import verdict
    for i, (dial, *_rest) in enumerate(DIALS):
        Rs = [seeds_vs_zoo(dcells[i, j][0])[2] for j in range(nd)]
        sp = np.median([seeds_vs_zoo(dcells[i, j][0])[0] for j in range(nd)])
        mp = np.median([seeds_vs_zoo(dcells[i, j][0])[1] for j in range(nd)])
        agree = np.mean([dcells[i, j][1] for j in range(nd)])
        print(f"{dial:<19} {dial_label(i):<36} median R {np.median(Rs):.2f} (range {min(Rs):.2f} to {max(Rs):.2f});"
              f" seed spread {100 * sp:.2f} pts, model spread {100 * mp:.2f} pts; agreement {agree:.1%}")
    print("")
    print("== Pre-registered predictions (E8b) ==")
    r0, r2 = dial_R(dcells, 0, nd), dial_R(dcells, 4, nd)
    print(f"P8.3 D1: median R >= 3 at bend 0 ({r0:.2f}) and < 1 at bend 2 ({r2:.2f}) -> {verdict(r0 >= 3 and r2 < 1)}")
    b1 = [dial_R(dcells, i, nd) for i in range(9, 13)]
    b0 = [dial_R(dcells, i, nd) for i in range(5, 9)]
    ok = b1[2] < 1 and b1[3] < 1 and all(r > 1 for r in b0)
    print(f"P8.4 D2: bend 1 has R < 1 at n = 1000 ({b1[2]:.2f}) and n = 3000 ({b1[3]:.2f}); "
          f"bend 0 has R > 1 at every n (smallest {min(b0):.2f}) -> {verdict(ok)}")
    g = [dial_R(dcells, i, nd) for i in range(13, 17)]
    ok = g[3] < g[0] and all(r > 1 for r in g)
    print(f"P8.5 D3: R at g = 1000 ({g[3]:.2f}) < R at g = 50 ({g[0]:.2f}), and > 1 at every g "
          f"(smallest {min(g):.2f}) -> {verdict(ok)}")


# ----------------------------------------------------------------------- figures
def plot_hero(acc, path, design=""):
    """Two panels: same data, different models (8 averages) vs same model, different seeds (200 runs)."""
    import matplotlib
    from matplotlib.patches import ConnectionPatch
    from matplotlib.ticker import FuncFormatter
    from experiments.common import BG, OKABE_ITO, dark_style
    plt = dark_style()
    sky, orange, yellow = OKABE_ITO["sky"], OKABE_ITO["orange"], OKABE_ITO["yellow"]
    means = 100 * acc.mean(axis=0)
    runs = 100 * acc[:, MODEL_NAMES.index(HERO_MODEL)]
    sd_models, sd_seeds = means.std(ddof=1), runs.std(ddof=1)
    ratio = sd_seeds / sd_models

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.56, 1], wspace=0)
    axl = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[2], sharey=axl)
    names = ["Logistic regression", "Naive Bayes", "SVM (RBF)", "kNN", "Decision tree", "Random forest",
             "Gradient boosting", "Neural net"]
    for ax, v, col, title in [(axl, means, sky, "Same data,\ndifferent models"),
                              (axr, runs, orange, "Same model,\ndifferent seeds")]:
        mu, sd = v.mean(), v.std(ddof=1)
        thin = 2 * sd < 0.02 * np.ptp(runs)           # a band thinner than the dots: fill it, no edge lines
        ax.axhspan(mu - sd, mu + sd, color=col, alpha=0.6 if thin else 0.28, lw=0, zorder=1)
        for edge in ([] if thin else (mu - sd, mu + sd)):
            ax.axhline(edge, color=col, lw=3, zorder=2)
        ax.set_title(title, fontsize=36, fontweight="bold", color=col, pad=24, linespacing=1.05)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="#333333", lw=1)
        ax.tick_params(axis="y", labelsize=27, length=0, pad=10)
        ax.tick_params(axis="x", length=0, pad=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}%"))
    x = np.arange(len(means))
    axl.scatter(x, means, s=650, color=sky, edgecolor=BG, linewidth=3, zorder=4)
    axl.set_xticks(x, names, fontsize=21, rotation=38, ha="right", rotation_mode="anchor")
    axl.set_xlim(-0.6, len(means) - 0.4)
    axl.set_xlabel(f"8 models, each dot = its average over the same {len(acc)} splits", fontsize=22, labelpad=12)
    xs = np.arange(1, len(runs) + 1)
    axr.scatter(xs, runs, s=110, color=orange, edgecolor=BG, linewidth=0.8, zorder=4)
    axr.set_xticks([t for t in (1, 50, 100, 150, 200) if t <= len(runs)])
    axr.tick_params(axis="x", labelsize=24)
    axr.set_xlim(-6, len(runs) + 7)
    axr.set_xlabel(f"{HERO_MODEL}, one dot per random seed", fontsize=22, labelpad=12)
    axr.yaxis.tick_right()                     # keep the gap free for the arrow and the bubble
    axr.tick_params(axis="y", labelright=True, labelleft=False)
    axr.spines["left"].set_visible(False)
    axr.spines["right"].set_visible(True)
    axl.set_ylabel("test accuracy", fontsize=30, labelpad=14)
    lo, hi = runs.min(), runs.max()
    pad = 0.06 * (hi - lo)
    axl.set_ylim(lo - pad, hi + pad * 2.6)
    axl.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5 if hi - lo > 15 else 2))

    # the two spreads, labelled with the same precision so the ratio can be checked by eye
    axl.text(len(means) - 0.5, means.mean() + sd_models + 0.35, f"spread ±{sd_models:.2f} pts",
             ha="right", va="bottom", fontsize=29, color=sky, fontweight="bold")
    axr.text(len(runs) + 5, hi + pad * 0.7, f"spread ±{sd_seeds:.2f} pts", ha="right", va="bottom",
             fontsize=29, color=orange, fontweight="bold")

    # the arrow from the small spread to the large one, and the bubble
    start = (len(means) - 0.3, means.mean())
    end = (-5, runs.mean() + 0.6 * sd_seeds)
    fig.add_artist(ConnectionPatch(xyA=start, coordsA=axl.transData, xyB=end, coordsB=axr.transData,
                                   arrowstyle="-|>,head_length=1.6,head_width=0.8", color=yellow, lw=8,
                                   connectionstyle="arc3,rad=-0.3", zorder=20))
    bl, br = axl.get_position(), axr.get_position()
    fig.text((bl.x1 + br.x0) / 2, bl.y0 + 0.80 * (bl.y1 - bl.y0), f"{ratio:.1f}×\nmore", ha="center",
             va="center", fontsize=54, fontweight="bold", color=BG, linespacing=0.92, zorder=21,
             bbox=dict(boxstyle="round,pad=0.42", fc=yellow, ec="none"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ratio


def plot_dials(dcells, nd, path):
    from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator
    from experiments.common import OKABE_ITO, dark_style
    plt = dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.6), sharey=True, gridspec_kw={"wspace": 0.1})
    R = lambda i: dial_R(dcells, i, nd)  # noqa: E731
    sky, verm = OKABE_ITO["sky"], OKABE_ITO["vermillion"]
    style = dict(lw=4.5, ms=14, marker="o")
    ax = axes[0]
    bends = [0, 0.5, 1, 1.5, 2]
    ax.plot(bends, [R(i) for i in range(5)], color=verm, **style)
    ax.set_xticks(bends, ["straight", "0.5", "1", "1.5", "2"])
    ax.set_xlim(-0.2, 2.2)
    ax.set_xlabel("how much the true boundary bends\n(n = 300)", fontsize=21)
    ax.set_title("1. Bend the true boundary", fontsize=23, loc="left", pad=14)
    ax = axes[1]
    ns = [150, 300, 1000, 3000]
    r0, r1 = [R(i) for i in range(5, 9)], [R(i) for i in range(9, 13)]
    ax.plot(ns, r0, color=sky, label="straight boundary", **style)
    ax.plot(ns, r1, color=verm, label="bent boundary (bend 1)", **style)
    ax.legend(loc="upper right", fontsize=18, handlelength=1.6)
    ax.set_xscale("log")
    ax.set_xticks(ns, [f"{n:,}" for n in ns])
    ax.set_xlim(120, 3800)
    ax.set_xlabel("dataset size n\n(70% train / 30% test)", fontsize=21)
    ax.set_title("2. Add data", fontsize=23, loc="left", pad=14)
    ax = axes[2]
    gsz = [50, 100, 300, 1000]
    ax.plot(gsz, [R(i) for i in range(13, 17)], color=sky, **style)
    ax.set_xscale("log")
    ax.set_xticks(gsz, [f"{g:,}" for g in gsz])
    ax.set_xlim(40, 1250)
    ax.set_xlabel("golden-set size\n(1,000 training rows, straight boundary)", fontsize=21)
    ax.set_title("3. Grow the golden set", fontsize=23, loc="left", pad=14)
    allR = [R(i) for i in range(len(DIALS))]
    for ax in axes:
        ax.axhline(1, color="#FFFFFF", ls="--", lw=2)
        ax.set_yscale("log")
        ax.tick_params(labelsize=19)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylim(min(allR) / 1.7, max(allR) * 1.7)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    lo_y, hi_y = axes[0].get_ylim()
    axes[0].set_yticks([t for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64) if lo_y <= t <= hi_y])
    axes[0].set_ylabel("R = seed spread / model spread", fontsize=22)
    axes[0].text(0.97, 0.96, "seeds win (R > 1)", transform=axes[0].transAxes, va="top", ha="right",
                 fontsize=19, fontweight="bold")
    axes[0].text(0.03, 0.04, "models win (R < 1)", transform=axes[0].transAxes, va="bottom", fontsize=19,
                 fontweight="bold")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figures(cells, n_data, dcells, nd, img):
    design, inst, _ = hero_choice(cells, n_data)
    plot_hero(cells[design, inst][0], os.path.join(img, "e8a_seeds_vs_models.png"), design)
    plot_dials(dcells, nd, os.path.join(img, "e8b_dials.png"))


def load_cells(out):
    """Rebuild the accuracy matrices from the CSVs (no refitting)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(out, "e8_tailored.csv"))
    cells = {}
    for (name, j), g in df.groupby(["design", "dataset"], sort=False):
        order = g.split_seed.drop_duplicates()
        m = g.pivot(index="split_seed", columns="model", values="accuracy").reindex(order)
        cells[name, int(j)] = m[MODEL_NAMES].to_numpy(), float(g.agreement_all8.iloc[0])
    dd = pd.read_csv(os.path.join(out, "e8_dials.csv"))
    labels = [dial_label(i) for i in range(len(DIALS))]
    dcells = {}
    for (dial, setting, j), g in dd.groupby(["dial", "setting", "dataset"], sort=False):
        i = next(k for k in range(len(DIALS)) if DIALS[k][0] == dial and labels[k] == setting)
        order = g.split_seed.drop_duplicates()
        m = g.pivot(index="split_seed", columns="model", values="accuracy").reindex(order)
        dcells[i, int(j)] = m[MODEL_NAMES].to_numpy(), float("nan")
    n_data = 1 + max(j for (_, j) in cells)
    nd = 1 + max(j for (_, j) in dcells)
    return cells, n_data, dcells, nd


def redraw(out, img):
    figures(*load_cells(out), img)
