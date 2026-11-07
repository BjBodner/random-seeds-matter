"""E2 — Imbalance: stratification fixes the counts, not the recall noise."""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import QUICK, bootstrap_ci, np, pmap, seeds  # noqa: F401

# --- article: e2 ---
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def recall_on_split(X, y, split_seed, stratify):
    """Positives in the test set (k) and recall; recall is undefined when k = 0."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=split_seed,
                                          stratify=y if stratify else None)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
    k = int(yte.sum())
    return k, recall_score(yte, model.predict(Xte)) if k else np.nan


def e2_imbalance(quick=QUICK):
    split_seeds = seeds(2, 1, 50 if quick else 300)
    out = {}
    for n, data_seed in zip([400, 1000, 4000], seeds(2, 0, 3)):
        X, y = make_classification(n_samples=n, n_features=20, n_informative=5, n_redundant=5,
                                   weights=[0.95], flip_y=0.01, random_state=data_seed)
        for strat in (False, True):
            res = np.array(pmap(recall_on_split, [(X, y, s, strat) for s in split_seeds]))
            out[n, strat] = res                                   # columns: k, recall
        u, s = out[n, False][:, 1], out[n, True][:, 1]
        ratio = np.nanstd(s, ddof=1) / np.nanstd(u, ddof=1)
        lo, hi = bootstrap_ci(lambda a, b: np.nanstd(a, ddof=1) / np.nanstd(b, ddof=1), s, u, seed=n)
        k, r = out[n, True][0, 0], np.nanmean(s)
        floor = np.sqrt(r * (1 - r) / k)                         # binomial floor
        print(f"n={n}: {int(y.sum())} positives; unstratified k = {out[n, False][:, 0].mean():.2f} "
              f"+/- {out[n, False][:, 0].std(ddof=1):.2f}, stratified k = {k:.0f} "
              f"+/- {out[n, True][:, 0].std(ddof=1):.2f}")
        print(f"   recall std: unstratified {np.nanstd(u, ddof=1):.3f}, stratified {np.nanstd(s, ddof=1):.3f}, "
              f"ratio {ratio:.2f} (95% CI {lo:.2f} to {hi:.2f})")
        print(f"   binomial floor sqrt(r(1-r)/k) = {floor:.3f} at r = {r:.3f}, k = {k:.0f}; "
              f"observed / floor = {np.nanstd(s, ddof=1) / floor:.2f}")
    return out


if __name__ == "__main__":
    e2 = e2_imbalance()
# --- end ---


# ======================================================================= repo-only
SIZES = [400, 1000, 4000]


def run(out, img, quick=False):
    import pandas as pd
    from experiments.common import tee
    split_seeds = seeds(2, 1, 50 if quick else 300)
    with tee(os.path.join(out, "e2_imbalance.txt")):
        print("E2 - imbalance: stratify=y versus stratify=None")
        print(f"design: n in {SIZES}, about 5% positives, {len(split_seeds)} split seeds per setting"
              " (same seeds for both), scaled LogisticRegression, test_size 0.3")
        res = e2_imbalance(quick)
        stats = extras(res)
    rows = [(n, "stratified" if st else "unstratified", split_seeds[i], int(a[i, 0]), a[i, 1])
            for (n, st), a in res.items() for i in range(len(a))]
    df = pd.DataFrame(rows, columns=["n", "setting", "split_seed", "k", "recall"])
    df.to_csv(os.path.join(out, "e2_imbalance.csv"), index=False)
    figure(df, stats, img)


def extras(res):
    from experiments.common import fmt, verdict
    stats = {}
    for n in SIZES:
        u, s = res[n, False], res[n, True]
        k, r = s[0, 0], np.nanmean(s[:, 1])
        floor = np.sqrt(r * (1 - r) / k)
        stats[n] = dict(ratio=np.nanstd(s[:, 1], ddof=1) / np.nanstd(u[:, 1], ddof=1), k=int(k), r=r,
                        floor=floor, obs=np.nanstd(s[:, 1], ddof=1), k_sd=np.std(s[:, 0], ddof=1),
                        undefined=int(np.isnan(u[:, 1]).sum()))
    print("")
    print("unstratified runs with k = 0 (recall undefined, counted not dropped): "
          + ", ".join(f"n={n}: {stats[n]['undefined']}" for n in SIZES))
    print("unstratified runs grouped by test positive count k (groups with >= 15 runs):")
    print("n     k    runs  recall_std  floor")
    for n in SIZES:
        u = res[n, False]
        for k in np.unique(u[:, 0]):
            g = u[u[:, 0] == k, 1]
            if len(g) >= 15 and k > 0:
                rr = np.nanmean(g)
                print(f"{n:<5} {int(k):<4} {len(g):<5} {fmt(np.nanstd(g, ddof=1)):>10}  {fmt(np.sqrt(rr * (1 - rr) / k))}")
    print("")
    print("== Pre-registered predictions ==")
    print(f"P2.1 stratified k std = 0 at every n: {verdict(all(stats[n]['k_sd'] == 0 for n in SIZES))}")
    ok = all(0.85 <= stats[n]["ratio"] <= 1.05 for n in SIZES)
    print(f"P2.2 recall std ratio within [0.85, 1.05] at every n: {verdict(ok)} ("
          + ", ".join(f"n={n}: {fmt(stats[n]['ratio'], 2)}" for n in SIZES) + ")")
    ok = all(abs(stats[n]["obs"] / stats[n]["floor"] - 1) <= 0.25 for n in SIZES)
    print(f"P2.3 stratified recall std within +/-25% of the binomial floor at every n: {verdict(ok)} ("
          + ", ".join(f"n={n}: {fmt(stats[n]['obs'] / stats[n]['floor'], 2)}x" for n in SIZES) + ")")
    return stats


def figure(df, stats, img):
    from experiments.common import OKABE_ITO, dark_style, fmt, save_fig
    plt = dark_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    n = 1000
    d = df[df.n == n]
    bins = np.linspace(0, 1, 41)
    for setting, c in (("unstratified", OKABE_ITO["orange"]), ("stratified", OKABE_ITO["sky"])):
        r = d[d.setting == setting].recall.dropna()
        a1.hist(r, bins=bins, histtype="step", lw=2.2, color=c, label=f"{setting} (std {fmt(r.std(ddof=1))})")
    st = stats[n]
    a1.axvspan(st["r"] - st["floor"], st["r"] + st["floor"], color="#FFFFFF", alpha=0.10, lw=0,
               label=f"mean recall ± predicted wobble (k = {st['k']})")
    a1.set_xlabel("recall on the test set")
    a1.set_ylabel("number of split seeds")
    a1.set_title(f"n = {n:,}: stratifying barely changes the spread")
    a1.set_xlim(0, max(0.8, d.recall.max() + 0.1))
    a1.set_ylim(0, a1.get_ylim()[1] * 1.18)                                    # headroom so the legend never sits on a bar
    a1.legend(loc="upper right", fontsize=9)

    marks = {400: "o", 1000: "s", 4000: "D"}
    cols = {400: OKABE_ITO["purple"], 1000: OKABE_ITO["yellow"], 4000: OKABE_ITO["green"]}   # not the left panel's orange / sky
    for m in SIZES:
        u = df[(df.n == m) & (df.setting == "unstratified")]
        g = u.groupby("k").recall.agg(["std", "count", "mean"])
        g = g[(g["count"] >= 15) & (g.index > 0)]
        a2.scatter(g.index, g["std"], marker=marks[m], s=26, facecolors="none", edgecolors=cols[m], alpha=0.8)
        ks = np.geomspace(stats[m]["k"] / 2, stats[m]["k"] * 2, 50)
        a2.plot(ks, np.sqrt(stats[m]["r"] * (1 - stats[m]["r"]) / ks), color=cols[m], ls="--", lw=1.2)
        a2.scatter([stats[m]["k"]], [stats[m]["obs"]], color=cols[m], marker=marks[m], s=95,
                   edgecolors="#FFFFFF", linewidths=1.2, label=f"n = {m:,}, stratified (k = {stats[m]['k']})")
    a2.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    a2.xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 50, 100, 200]))
    a2.xaxis.set_major_formatter(ScalarFormatter())
    a2.xaxis.set_minor_formatter(NullFormatter())
    a2.set_xlabel("positives in the test set, k (log scale)")
    a2.set_ylabel("std of recall across splits")
    a2.set_title("More positives, not stratification, shrink the noise")
    a2.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.text(0.5, -0.01, "Right panel: hollow markers = unstratified runs grouped by k; filled = stratified; "
             "dashed lines = the wobble that counting alone predicts.", ha="center", va="top", fontsize=11, color="#BBBBBB")
    save_fig(fig, img, "e2_imbalance")
