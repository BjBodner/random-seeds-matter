"""E4 — k-means restarts: one seed can glue two clusters together; n_init > 1 un-glues them."""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import QUICK, bootstrap_ci, np, pmap, seeds  # noqa: F401

# --- article: e4 ---
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

BLOBS = make_blobs(n_samples=1500, centers=8, cluster_std=1.5, center_box=(-10, 10),
                   random_state=seeds(4, 0, 1)[0])


def kmeans_run(seed, init, n_init):
    """ARI against the true blobs, and the k-means objective (inertia)."""
    X, truth = BLOBS
    km = KMeans(n_clusters=8, init=init, n_init=n_init, random_state=seed).fit(X)
    return adjusted_rand_score(truth, km.labels_), km.inertia_


def e4_kmeans(quick=QUICK):
    km_seeds = seeds(4, 1, 50 if quick else 300)
    res = {(init, n_init): np.array(pmap(kmeans_run, [(s, init, n_init) for s in km_seeds]))
           for init in ("k-means++", "random") for n_init in (1, 10)}   # columns: ARI, inertia
    best = min(r[:, 1].min() for r in res.values())
    for (init, n_init), r in res.items():
        bad = np.mean(r[:, 1] > 1.01 * best)          # bad local optimum: inertia > best + 1%
        print(f"{init:<9} n_init={n_init:<2}  ARI mean {r[:, 0].mean():.3f}  std {r[:, 0].std(ddof=1):.3f}"
              f"  min {r[:, 0].min():.3f}  bad local optimum {bad:.1%}")
    for init in ("k-means++", "random"):
        a1, a10 = res[init, 1][:, 0], res[init, 10][:, 0]
        lo, hi = bootstrap_ci(lambda u, v: u.var(ddof=1) / v.var(ddof=1), a1, a10, seed=4)
        print(f"{init}: n_init=10 cuts ARI variance {a1.var(ddof=1) / a10.var(ddof=1):.1f}x "
              f"(95% CI {lo:.1f} to {hi:.1f})")
    return res


if __name__ == "__main__":
    e4 = e4_kmeans()
# --- end ---


# ======================================================================= repo-only
CONFIGS = [("k-means++", 1), ("k-means++", 10), ("random", 1), ("random", 10)]


def run(out, img, quick=False):
    import pandas as pd
    from experiments.common import tee
    km_seeds = seeds(4, 1, 50 if quick else 300)
    with tee(os.path.join(out, "e4_kmeans.txt")):
        print("E4 - k-means restarts")
        print(f"design: make_blobs 1500 points, 8 overlapping clusters (cluster_std 1.5); {len(km_seeds)} seeds x"
              " n_init {1, 10} x init {k-means++, random}; ARI vs generating labels")
        res = e4_kmeans(quick)
        picks = extras(res, km_seeds)
    rows = [(i, n, km_seeds[j], r[j, 0], r[j, 1]) for (i, n), r in res.items() for j in range(len(r))]
    pd.DataFrame(rows, columns=["init", "n_init", "seed", "ari", "inertia"]) \
        .to_csv(os.path.join(out, "e4_kmeans.csv"), index=False)
    figures(res, picks, img)


def extras(res, km_seeds):
    from experiments.common import fmt, pct, verdict
    best = min(r[:, 1].min() for r in res.values())
    print("")
    print(f"lowest inertia over all {sum(len(r) for r in res.values())} runs: {fmt(best, 1)}")
    for cfg in CONFIGS:
        r = res[cfg]
        print(f"{cfg[0]} n_init={cfg[1]}: ARI max {fmt(r[:, 0].max())}, ARI variance {r[:, 0].var(ddof=1):.2e}")
    a = res["k-means++", 1]
    lo_i, hi_i = int(np.argmin(a[:, 0])), int(np.argmax(a[:, 0]))
    print(f"k-means++, n_init=1: unluckiest seed {km_seeds[lo_i]} ARI {fmt(a[lo_i, 0])}; "
          f"luckiest seed {km_seeds[hi_i]} ARI {fmt(a[hi_i, 0])} (hand-picked extremes, shown as such)")
    r1, r10 = res["k-means++", 1][:, 0].var(ddof=1), res["k-means++", 10][:, 0].var(ddof=1)
    b1 = np.mean(res["k-means++", 1][:, 1] > 1.01 * best)
    b10 = np.mean(res["k-means++", 10][:, 1] > 1.01 * best)
    print("")
    print("== Pre-registered predictions ==")
    print(f"P4.1 n_init=10 cuts ARI variance >= 10x (k-means++): observed {fmt(r1 / r10, 1)}x -> {verdict(r1 / r10 >= 10)}")
    print(f"P4.2 bad-optimum share: n_init=1 >= 20% (observed {pct(b1)}), n_init=10 <= 5% (observed {pct(b10)})"
          f" -> {'HELD' if b1 >= 0.2 and b10 <= 0.05 else 'PARTIAL' if (b1 >= 0.2) or (b10 <= 0.05) else 'FAILED'}")
    return km_seeds[lo_i], km_seeds[hi_i]


def _match(labels, truth):
    """Relabel clusters to their majority true blob so colours line up across panels."""
    out, used = np.empty_like(labels), set()
    for c in np.argsort(-np.bincount(labels)):
        counts = np.bincount(truth[labels == c], minlength=8)
        for t in np.argsort(-counts):
            if t not in used:
                used.add(t)
                out[labels == c] = t
                break
    return out


def figures(res, picks, img):
    from experiments.common import NEUTRAL, OKABE_ITO, dark_style, save_fig
    plt = dark_style()
    X, truth = BLOBS
    pal = [OKABE_ITO[k] for k in ("orange", "sky", "green", "yellow", "blue", "vermillion", "purple")] + [NEUTRAL]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True)
    panels = [("ground truth", truth, None)]
    for name, s in zip(("unlucky seed", "lucky seed"), picks):
        km = KMeans(n_clusters=8, init="k-means++", n_init=1, random_state=s).fit(X)
        panels.append((f"{name} (ARI {adjusted_rand_score(truth, km.labels_):.2f})", _match(km.labels_, truth), km))
    for ax, (title, lab, km) in zip(axes, panels):
        ax.scatter(X[:, 0], X[:, 1], c=[pal[i] for i in lab], s=5, alpha=0.7, linewidths=0)
        if km is not None:
            ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker="X", s=140,
                       color="#FFFFFF", edgecolors="#121212", linewidths=1.2)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("k-means++ with n_init=1: same data, two seeds (hand-picked extremes)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, img, "e4a_kmeans_scatter")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    cols = dict(zip(CONFIGS, [OKABE_ITO["orange"], OKABE_ITO["sky"], OKABE_ITO["purple"], OKABE_ITO["green"]]))
    rng = np.random.default_rng(0)
    for x, cfg in enumerate(CONFIGS):
        a = res[cfg][:, 0]
        ax.scatter(x + rng.uniform(-0.25, 0.25, len(a)), a, s=9, alpha=0.55, color=cols[cfg], linewidths=0)
        ax.hlines(a.mean(), x - 0.32, x + 0.32, color="#FFFFFF", lw=2)
    ax.set_xticks(range(4), [f"{i}\nn_init={n}" for i, n in CONFIGS])
    ax.set_ylabel("adjusted Rand index vs truth")
    ax.set_title("Restarts turn a lottery into a near-constant")
    ax.grid(axis="x", visible=False)
    save_fig(fig, img, "e4b_ari_distributions")
