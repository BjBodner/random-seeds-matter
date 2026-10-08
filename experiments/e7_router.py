"""E7 — The golden-set lottery: two intent routers, judged on golden sets of 100 to 3,000 queries.

Data: Banking77 (PolyAI, CC BY 4.0), 77 customer-service intents. Downloaded on first run to
$BANKING77_DIR (default data/banking77, gitignored).
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import QUICK, np, seeds  # noqa: F401

# --- article: e7 ---
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

BANKING77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv"
GOLDEN_SIZES = [100, 200, 300, 500, 1000, 3000]
ROUTERS = {  # two versions of an intent router: 77 routes, deterministic once trained
    "v1 (word TF-IDF + LogReg)": lambda: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=2000)),
    "v2 (word+char TF-IDF + LinearSVC)": lambda: make_pipeline(make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        LinearSVC(C=1.0, random_state=0)),
}


def load_banking77(split):
    path = os.path.join(os.environ.get("BANKING77_DIR", "data/banking77"), f"{split}.csv")
    if not os.path.exists(path):                    # first run: download (CC BY 4.0, ~1 MB)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.read_csv(BANKING77.format(split)).to_csv(path, index=False)
    return pd.read_csv(path)


def golden_set_lottery(better, worse, g, seed_list):
    """Score both routers on golden sets of g queries drawn from the query stream."""
    rows = []
    for s in seed_list:
        idx = np.random.default_rng(s).integers(0, len(better), g)
        a, b = better[idx], worse[idx]               # per-query success (True = routed right)
        n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
        p = stats.binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0   # exact McNemar
        rows.append((1 - a.mean(), 1 - b.mean(), p < 0.05 and n10 > n01))
    return np.array(rows)                            # failure(better), failure(worse), detected


def e7_router_lottery(quick=QUICK):
    train, test = load_banking77("train"), load_banking77("test")
    correct = {}
    for name, make in ROUTERS.items():
        router = make().fit(train.text, train.category)
        correct[name] = (router.predict(test.text) == test.category).to_numpy()
        print(f"{name}: failure rate on all {len(test)} test queries {1 - correct[name].mean():.3f}")
    worse_name, better_name = sorted(correct, key=lambda k: correct[k].mean())
    better, worse = correct[better_name], correct[worse_name]
    gap = worse.mean() - better.mean()
    p10, p01 = np.mean(better & ~worse), np.mean(~better & worse)
    print(f"true gap {-gap:.3f} in failure rate; discordant queries {p10 + p01:.1%}")
    for k, g in enumerate(GOLDEN_SIZES):
        r = golden_set_lottery(better, worse, g, seeds(7, k, 200 if quick else 1000))
        diff = r[:, 1] - r[:, 0]                    # > 0: the golden set picks the right router
        print(f"g={g:<5} worse router wins {np.mean(diff < 0):.1%}, ties {np.mean(diff == 0):.1%}, "
              f"McNemar detects {r[:, 2].mean():.1%}; measured gap {diff.mean():.3f} +/- {diff.std(ddof=1):.3f}")
    z = stats.norm.ppf(0.975) * np.sqrt(p10 + p01) + stats.norm.ppf(0.8) * np.sqrt(p10 + p01 - (p10 - p01) ** 2)
    print(f"golden-set size for 80% McNemar power: {int(np.ceil((z / (p10 - p01)) ** 2))} queries")
    return correct


if __name__ == "__main__":
    e7 = e7_router_lottery()
# --- end ---


# ======================================================================= repo-only
def run(out, img, quick=False):
    from experiments.common import tee
    with tee(os.path.join(out, "e7_router.txt")):
        print("E7 - the golden-set lottery: two intent routers on Banking77 (77 routes, CC BY 4.0)")
        print(f"design: routers trained once on the 10,003-query train split; golden sets of {GOLDEN_SIZES} queries "
              f"drawn with replacement from the test split; {200 if quick else 1000} golden sets per size")
        correct = e7_router_lottery(quick)
        rows = extras(correct, quick)
    pd.DataFrame(correct).to_csv(os.path.join(out, "e7_router_correct.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(out, "e7_router.csv"), index=False)
    figures(out, img)


def extras(correct, quick):
    from experiments.common import fmt, pct, verdict
    worse_name, better_name = sorted(correct, key=lambda k: correct[k].mean())
    better, worse = correct[better_name], correct[worse_name]
    print("")
    print(f"better router (on all test queries): {better_name}; worse: {worse_name}")
    print(f"accuracy {fmt(better.mean())} vs {fmt(worse.mean())}; gap {pct(better.mean() - worse.mean())} points of accuracy")
    rows, summary = [], {}
    for k, g in enumerate(GOLDEN_SIZES):
        seed_list = seeds(7, k, 200 if quick else 1000)
        r = golden_set_lottery(better, worse, g, seed_list)
        diff = r[:, 1] - r[:, 0]
        summary[g] = dict(wrong=np.mean(diff < 0), ties=np.mean(diff == 0), detect=r[:, 2].mean())
        rows += [dict(g=g, seed=s, fail_better=a, fail_worse=b, detected=bool(d)) for s, (a, b, d) in zip(seed_list, r)]
        print(f"g={g}: wrong or tied {pct(summary[g]['wrong'] + summary[g]['ties'])}; "
              f"measured failure-rate range {fmt(diff.min())} to {fmt(diff.max())}")
    p10, p01 = np.mean(better & ~worse), np.mean(~better & worse)
    z = 1.959964 * np.sqrt(p10 + p01) + 0.841621 * np.sqrt(p10 + p01 - (p10 - p01) ** 2)
    need = int(np.ceil((z / (p10 - p01)) ** 2))
    gap_pts = 100 * (better.mean() - worse.mean())
    print("")
    print("== Pre-registered predictions ==")
    print(f"P7.1 full-test accuracy gap < 3 points: observed {gap_pts:.1f} -> {verdict(gap_pts < 3)}")
    w300 = summary[300]["wrong"] + summary[300]["ties"]
    print(f"P7.2 g = 300: worse router ties or wins on >= 15% of golden sets: observed {pct(w300)} -> {verdict(w300 >= 0.15)}")
    print(f"P7.3 g = 300: McNemar detects on <= 60% of golden sets: observed {pct(summary[300]['detect'])} -> "
          f"{verdict(summary[300]['detect'] <= 0.60)}")
    print(f"P7.4 size for 80% power >= 500: observed {need} -> {verdict(need >= 500)}")
    print(f"Falsifier (gap >= 5 points): {'TRIGGERED' if gap_pts >= 5 else 'not triggered'}")
    return rows


def figures(out, img):
    from experiments.common import BG, NEUTRAL, OKABE_ITO, dark_style, save_fig
    from experiments.e1_split import load_cells, plot_boundary_map
    plt = dark_style()
    df = pd.read_csv(os.path.join(out, "e7_router.csv"))
    correct = pd.read_csv(os.path.join(out, "e7_router_correct.csv"))
    worse_name, better_name = sorted(correct.columns, key=lambda k: correct[k].mean())
    true_gap = correct[better_name].mean() - correct[worse_name].mean()

    # hero: 1,000 golden sets of 300 queries
    d = df[df.g == 300]
    diff = 100 * (d.fail_worse - d.fail_better)
    step = 100 / 300
    edges = np.arange(np.floor(diff.min()) - step / 2, diff.max() + step, step)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts, _, patches = ax.hist(diff, bins=edges, color=OKABE_ITO["sky"], edgecolor=BG, alpha=0.9)
    for p, left in zip(patches, edges[:-1]):
        if left + step / 2 < 0:
            p.set_facecolor(OKABE_ITO["orange"])
        elif abs(left + step / 2) < 1e-9:
            p.set_facecolor(NEUTRAL)
    ax.axvline(100 * true_gap, color="#FFFFFF", ls="--", lw=1.8)
    ax.text(100 * true_gap, counts.max() * 1.02, f"  true gap on all\n  3,080 queries: {100 * true_gap:.1f} pts",
            fontsize=13, va="bottom", ha="left")
    wrong = np.mean(diff < 0)
    tie = np.mean(diff == 0)
    ax.text(0.02, 0.97, f"worse router wins\non {100 * wrong:.1f}% of golden sets", transform=ax.transAxes, va="top",
            fontsize=17, fontweight="bold", color=OKABE_ITO["orange"])
    ax.text(0.02, 0.76, f"tie on {100 * tie:.1f}%", transform=ax.transAxes, va="top", fontsize=13, color=NEUTRAL)
    ax.set_xlabel("measured improvement of the better router (failure-rate points) on a 300-query golden set", fontsize=13)
    ax.set_ylabel("number of golden sets (of 1,000)", fontsize=13)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.tick_params(labelsize=12)
    save_fig(fig, img, "e7a_golden_set_lottery")

    # expansion: when does the model take back control?
    sizes = sorted(df.g.unique())
    agg = df.assign(diff=df.fail_worse - df.fail_better).groupby("g")
    wrong = agg["diff"].apply(lambda x: np.mean(x < 0))
    ties = agg["diff"].apply(lambda x: np.mean(x == 0))
    detect = agg["detected"].mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
    plot_boundary_map(load_cells(out), ax=a1)
    a1.set_title("Training-set size: R(n) on the E1 datasets")
    a2.plot(sizes, 100 * (wrong + ties), "o-", color=OKABE_ITO["orange"], lw=2.6, ms=7, label="worse router wins or ties")
    a2.plot(sizes, 100 * detect, "s-", color=OKABE_ITO["sky"], lw=2.6, ms=7, label="McNemar detects the better one")
    a2.axvspan(100, 500, color="#FFFFFF", alpha=0.06, lw=0)
    a2.text(110, 96, "typical golden sets", fontsize=11, color=NEUTRAL, va="top")
    a2.set_xscale("log")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a2.xaxis.set_minor_formatter(NullFormatter())
    a2.set_xticks(sizes)
    a2.set_ylim(0, 100)
    a2.set_xlabel("golden-set size (queries, log scale)")
    a2.set_ylabel("% of 1,000 golden sets")
    a2.set_title("Golden-set size: the E7 routers")
    a2.legend(loc="center right", fontsize=11)
    fig.tight_layout()
    save_fig(fig, img, "e7b_until_they_dont")


def redraw(out, img):
    figures(out, img)
