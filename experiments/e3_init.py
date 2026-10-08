"""E3 — The init lottery, audited: bad luck, or a model that never finished training?"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import QUICK, np, pmap, seeds  # noqa: F401

# --- article: e3 ---
import warnings

from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONDITIONS = {  # name: (standardize inputs?, max_iter)
    "A raw, 200 iters": (False, 200), "B raw, 1000 iters": (False, 1000),
    "C scaled, 200 iters": (True, 200), "D scaled, 1000 iters": (True, 1000),
}


def init_lottery(split, init_seed, scaled, max_iter):
    """Data and split frozen; only the MLP's random_state moves."""
    Xtr, Xte, ytr, yte = split
    mlp = MLPClassifier(random_state=init_seed, max_iter=max_iter)
    model = make_pipeline(StandardScaler(), mlp) if scaled else mlp
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(Xtr, ytr)
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    train_acc, test_acc = model.score(Xtr, ytr), model.score(Xte, yte)
    verdict = ("not converged" if warned else
               "converged, good minimum" if train_acc >= 0.90 else "converged, poor minimum")
    return dict(seed=init_seed, n_iter=mlp.n_iter_, loss=mlp.loss_, verdict=verdict,
                train_acc=train_acc, test_acc=test_acc)


def e3_init_lottery(quick=QUICK):
    X, y = load_breast_cancer(return_X_y=True)
    split = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seeds(3, 0, 1)[0])
    init_seeds = seeds(3, 1, 30 if quick else 200)
    runs = {}
    for cond, (scaled, max_iter) in CONDITIONS.items():
        runs[cond] = pmap(init_lottery, [(split, s, scaled, max_iter) for s in init_seeds])
        acc = np.array([r["test_acc"] for r in runs[cond]])
        verdicts = [r["verdict"] for r in runs[cond]]
        print(f"{cond:<21} test accuracy {acc.min():.3f} to {acc.max():.3f} "
              f"(range {acc.max() - acc.min():.3f}, std {acc.std(ddof=1):.3f}); "
              + ", ".join(f"{v}: {verdicts.count(v)}" for v in sorted(set(verdicts))))
    low = [r for r in runs["A raw, 200 iters"] if r["test_acc"] < 0.80]
    print(f"condition A: {len(low)} runs below 0.80 test accuracy; "
          f"{sum(r['verdict'] == 'not converged' for r in low)} of them flagged 'not converged'")
    return runs


if __name__ == "__main__":
    e3 = e3_init_lottery()
# --- end ---


# ======================================================================= repo-only
CLASS_ORDER = ["not converged", "converged, poor minimum", "converged, good minimum"]


def run(out, img, quick=False):
    import pandas as pd
    from experiments.common import tee
    with tee(os.path.join(out, "e3_init.txt")):
        print("E3 - the init lottery, audited")
        print(f"design: breast cancer, one fixed stratified split (398 train / 171 test), "
              f"{30 if quick else 200} MLPClassifier init seeds per condition, default architecture"
              " (100 ReLU units, Adam)")
        runs = e3_init_lottery(quick)
        df = pd.DataFrame([dict(condition=c, **r) for c, rs in runs.items() for r in rs])
        extras(df)
    df.to_csv(os.path.join(out, "e3_init.csv"), index=False)
    figure(df, img)


def extras(df):
    from experiments.common import fmt, pct, verdict
    print("")
    print("condition              mean   median_iters  not_conv  poor_min  good_min")
    ranges = {}
    for cond in CONDITIONS:
        d = df[df.condition == cond]
        ranges[cond] = d.test_acc.max() - d.test_acc.min()
        vc = d.verdict.value_counts()
        print(f"{cond:<22} {fmt(d.test_acc.mean())}  {int(d.n_iter.median()):>12}  "
              + "  ".join(f"{vc.get(v, 0):>8}" for v in CLASS_ORDER))
    a = df[df.condition == "A raw, 200 iters"]
    for v in CLASS_ORDER:
        e = a[a.verdict == v]
        if len(e):
            print(f"condition A, {v}: {len(e)} runs ({pct(len(e) / len(a))}), test accuracy mean "
                  f"{fmt(e.test_acc.mean())}, min {fmt(e.test_acc.min())}, max {fmt(e.test_acc.max())},"
                  f" median final loss {fmt(e.loss.median())}, median iterations {int(e.n_iter.median())}")
    low = a[a.test_acc < 0.80]
    share = (low.verdict == "not converged").mean() if len(low) else float("nan")
    d_ = df[df.condition == "D scaled, 1000 iters"]
    shrink = ranges["A raw, 200 iters"] / ranges["D scaled, 1000 iters"]
    print(f"accuracy range A / D = {fmt(ranges['A raw, 200 iters'])} / {fmt(ranges['D scaled, 1000 iters'])}"
          f" = {fmt(shrink, 1)}x; std A / D = {fmt(a.test_acc.std(ddof=1) / d_.test_acc.std(ddof=1), 1)}x")
    print("")
    print("== Pre-registered predictions ==")
    print(f"P3.1 >= 50% of condition-A runs below 0.80 accuracy are 'not converged': observed "
          f"{pct(share) if len(low) else 'n/a (no runs below 0.80)'} -> {verdict(len(low) > 0 and share >= 0.5)}")
    print(f"P3.2 accuracy range shrinks >= 5x from A to D: observed {fmt(shrink, 1)}x -> {verdict(shrink >= 5)}")


def figure(df, img):
    from experiments.common import OKABE_ITO, dark_style, save_fig
    plt = dark_style()
    colors = dict(zip(CLASS_ORDER, [OKABE_ITO["orange"], OKABE_ITO["purple"], OKABE_ITO["sky"]]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    a = df[df.condition == "A raw, 200 iters"]
    for v in CLASS_ORDER:
        e = a[a.verdict == v]
        if len(e):
            a1.scatter(e.loss, e.test_acc, s=26, color=colors[v], alpha=0.8, linewidths=0, label=f"{v} ({len(e)})")
    a1.set_xscale("log")
    from matplotlib.ticker import LogFormatter
    a1.xaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    a1.xaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5)))
    a1.set_xlabel("final training loss (log scale)")
    a1.set_ylabel("test accuracy")
    a1.set_title("Raw inputs, 200 iterations: who lands where?")
    a1.legend(fontsize=9, loc="lower left")
    rng = np.random.default_rng(0)
    for x, cond in enumerate(CONDITIONS):
        d = df[df.condition == cond]
        for v in CLASS_ORDER:
            e = d[d.verdict == v]
            a2.scatter(x + rng.uniform(-0.25, 0.25, len(e)), e.test_acc, s=10, color=colors[v], alpha=0.6, linewidths=0)
    a2.set_xticks(range(len(CONDITIONS)), [c.replace(", ", "\n") for c in CONDITIONS], fontsize=10)
    a2.set_ylabel("test accuracy")
    a2.set_title("Scaling, not iterations, collapses the lottery")
    from matplotlib.lines import Line2D
    a2.legend(handles=[Line2D([], [], marker="o", ls="", color=colors[v], label=v) for v in CLASS_ORDER],
              fontsize=9, loc="lower right")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    save_fig(fig, img, "e3_init_audit")


def redraw(out, img):
    import pandas as pd
    figure(pd.read_csv(os.path.join(out, "e3_init.csv")), img)
