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
def v2_with(C=1.0, class_weight=None, char=(2, 5)):
    """Router v2 (word + char TF-IDF + LinearSVC), optionally with one small tweak."""
    return lambda: make_pipeline(make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=char, sublinear_tf=True)),
        LinearSVC(C=C, class_weight=class_weight, random_state=0))


ROUTERS = {  # two versions of an intent router: 77 routes, deterministic once trained
    "v1 (word TF-IDF + LogReg)": lambda: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=2000)),
    "v2 (word+char TF-IDF + LinearSVC)": v2_with(),
}
TWEAKS = {"t1: C=0.5": v2_with(C=0.5), "t2: balanced classes": v2_with(class_weight="balanced"),
          "t3: char 3-5-grams": v2_with(char=(3, 5))}


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


def n_for_80_power(better, worse):
    """Golden-set size at which McNemar's test catches this gap 80% of the time."""
    p10, p01 = np.mean(better & ~worse), np.mean(~better & worse)
    z = stats.norm.ppf(0.975) * np.sqrt(p10 + p01) + stats.norm.ppf(0.8) * np.sqrt(p10 + p01 - (p10 - p01) ** 2)
    return int(np.ceil((z / (p10 - p01)) ** 2)) if p10 != p01 else float("inf")


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
    print(f"golden-set size for 80% McNemar power: {n_for_80_power(better, worse)} queries")
    return correct


def e7b_router_tweaks(correct_v2, quick=QUICK):
    """Small tweaks to router v2, each judged against v2 on golden sets of every size."""
    train, test = load_banking77("train"), load_banking77("test")
    out = {}
    for j, (name, make) in enumerate(TWEAKS.items()):
        c = (make().fit(train.text, train.category).predict(test.text) == test.category).to_numpy()
        better, worse = (correct_v2, c) if correct_v2.mean() >= c.mean() else (c, correct_v2)
        out[name] = {g: golden_set_lottery(better, worse, g, seeds(7, 100 + 10 * j + k, 200 if quick else 1000))
                     for k, g in enumerate(GOLDEN_SIZES)}
        r = out[name][300]
        diff = r[:, 1] - r[:, 0]
        print(f"{name}: failure rate {1 - c.mean():.3f} vs v2 {1 - correct_v2.mean():.3f} "
              f"(gap {better.mean() - worse.mean():.3f}); at g=300 worse router wins {np.mean(diff < 0):.1%}, "
              f"ties {np.mean(diff == 0):.1%}, McNemar detects {r[:, 2].mean():.1%}; "
              f"80% power needs {n_for_80_power(better, worse)} queries")
    return out


if __name__ == "__main__":
    e7 = e7_router_lottery()
    e7b = e7b_router_tweaks(e7["v2 (word+char TF-IDF + LinearSVC)"])
# --- end ---


# ======================================================================= repo-only
V2 = "v2 (word+char TF-IDF + LinearSVC)"


def run(out, img, quick=False):
    from experiments.common import tee
    n_sets = 200 if quick else 1000
    with tee(os.path.join(out, "e7_router.txt")):
        print("E7 - the golden-set lottery: two intent routers on Banking77 (77 routes, CC BY 4.0)")
        print(f"design: routers trained once on the 10,003-query train split; golden sets of {GOLDEN_SIZES} queries "
              f"drawn with replacement from the test split; {n_sets} golden sets per size")
        correct = e7_router_lottery(quick)
        rows = extras(correct, quick)
    with tee(os.path.join(out, "e7b_tweaks.txt")):
        print("E7b - small tweaks to router v2 (pre-registered after E7, before running), each vs v2")
        print(f"design: as E7; {n_sets} golden sets per size; seeds stream (7, 100 + 10*j + k)")
        tweaks = e7b_router_tweaks(correct[V2], quick)
        rows_b = extras_b(tweaks, quick)
    pd.DataFrame(correct).to_csv(os.path.join(out, "e7_router_correct.csv"), index=False)
    pd.DataFrame(rows + rows_b).to_csv(os.path.join(out, "e7_router.csv"), index=False)
    figures(out, img)


def extras(correct, quick):
    from experiments.common import fmt, pct, verdict
    worse_name, better_name = sorted(correct, key=lambda k: correct[k].mean())
    better, worse = correct[better_name], correct[worse_name]
    print("")
    print(f"better router (on all test queries): {better_name}; worse: {worse_name}")
    print(f"accuracy {fmt(better.mean())} vs {fmt(worse.mean())}; "
          f"gap {100 * (better.mean() - worse.mean()):.1f} points of accuracy")
    rows, summary = [], {}
    for k, g in enumerate(GOLDEN_SIZES):
        seed_list = seeds(7, k, 200 if quick else 1000)
        r = golden_set_lottery(better, worse, g, seed_list)
        diff = r[:, 1] - r[:, 0]
        summary[g] = dict(wrong=np.mean(diff < 0), ties=np.mean(diff == 0), detect=r[:, 2].mean())
        rows += [dict(pair="v1 vs v2", g=g, seed=s, fail_better=a, fail_worse=b, detected=bool(d))
                 for s, (a, b, d) in zip(seed_list, r)]
        print(f"g={g}: wrong or tied {pct(summary[g]['wrong'] + summary[g]['ties'])}; "
              f"measured failure-rate range {fmt(diff.min())} to {fmt(diff.max())}")
    need = n_for_80_power(better, worse)
    gap_pts = 100 * (better.mean() - worse.mean())
    print("")
    print("== Pre-registered predictions ==")
    print(f"P7.1 full-test accuracy gap < 3 points: observed {gap_pts:.1f} -> {verdict(gap_pts < 3)}")
    w300 = summary[300]["wrong"] + summary[300]["ties"]
    print(f"P7.2 g = 300: worse router ties or wins on >= 15% of golden sets: observed {pct(w300)} -> "
          f"{verdict(w300 >= 0.15)}")
    print(f"P7.3 g = 300: McNemar detects on <= 60% of golden sets: observed {pct(summary[300]['detect'])} -> "
          f"{verdict(summary[300]['detect'] <= 0.60)}")
    print(f"P7.4 size for 80% power >= 500: observed {need} -> {verdict(need >= 500)}")
    print(f"Falsifier (gap >= 5 points): {'TRIGGERED' if gap_pts >= 5 else 'not triggered'}")
    return rows


def extras_b(tweaks, quick):
    from experiments.common import pct, verdict
    train, test = load_banking77("train"), load_banking77("test")
    v2 = (v2_with()().fit(train.text, train.category).predict(test.text) == test.category).to_numpy()
    rows, info = [], {}
    print("")
    print("tweak                  g      worse wins  ties    McNemar detects")
    for j, (name, by_g) in enumerate(tweaks.items()):
        c = (TWEAKS[name]().fit(train.text, train.category).predict(test.text) == test.category).to_numpy()
        better, worse = (v2, c) if v2.mean() >= c.mean() else (c, v2)
        info[name] = dict(gap=100 * (better.mean() - worse.mean()), need=n_for_80_power(better, worse),
                          close300=np.mean(by_g[300][:, 1] - by_g[300][:, 0] <= 0), detect300=by_g[300][:, 2].mean())
        for k, g in enumerate(GOLDEN_SIZES):
            r = by_g[g]
            diff = r[:, 1] - r[:, 0]
            seed_list = seeds(7, 100 + 10 * j + k, len(r))
            rows += [dict(pair=name, g=g, seed=s, fail_better=a, fail_worse=b, detected=bool(d))
                     for s, (a, b, d) in zip(seed_list, r)]
            print(f"{name:<22} {g:<6} {pct(np.mean(diff < 0)):>10}  {pct(np.mean(diff == 0)):>6}  {pct(r[:, 2].mean()):>15}")
    print("")
    print("== Pre-registered predictions (E7b) ==")
    near = [n for n, v in info.items() if v["gap"] < 1.5]
    print(f"P7b.1 >= 2 of 3 tweaks within 1.5 points of v2: {len(near)} of 3 ("
          + ", ".join(f"{n}: {v['gap']:.1f} pts" for n, v in info.items()) + f") -> {verdict(len(near) >= 2)}")
    if near:
        ok2 = all(info[n]["close300"] >= 0.20 for n in near)
        print(f"P7b.2 for tweaks within 1.5 points, worse ties or wins on >= 20% at g = 300: "
              + ", ".join(f"{n}: {pct(info[n]['close300'])}" for n in near) + f" -> {verdict(ok2)}")
        ok3 = all(info[n]["detect300"] <= 0.30 and info[n]["need"] >= 1000 for n in near)
        print(f"P7b.3 for those, McNemar detects <= 30% at g = 300 and 80% power needs >= 1,000: "
              + ", ".join(f"{n}: {pct(info[n]['detect300'])}, {info[n]['need']} queries" for n in near)
              + f" -> {verdict(ok3)}")
    else:
        print("P7b.2 and P7b.3: not applicable (no tweak within 1.5 points)")
    far = all(v["gap"] >= 3 for v in info.values())
    print(f"Falsifier (all 3 tweaks >= 3 points from v2): {'TRIGGERED' if far else 'not triggered'}")
    return rows


def figures(out, img):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    from experiments.common import BG, NEUTRAL, OKABE_ITO, dark_style, save_fig
    from experiments.e1_split import load_cells, plot_boundary_map
    plt = dark_style()
    df = pd.read_csv(os.path.join(out, "e7_router.csv"))
    pairs = list(dict.fromkeys(df.pair))           # v1 vs v2 first, then the three tweaks
    titles = {p: ("v2 vs v1: a real upgrade" if p == "v1 vs v2" else f"v2 vs {p}") for p in pairs}

    # hero: every pre-registered comparison on 300-query golden sets (no pair is hand-picked).
    # Top row: the real upgrade on its own scale. Bottom row: the three tweaks, zoomed to +/-3 points.
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 16})
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.55, wspace=0.12)
    step = 100 / 300
    d300 = df[df.g == 300]
    colors = (OKABE_ITO["orange"], NEUTRAL, OKABE_ITO["sky"])

    def panel(ax, p, lo, hi, title):
        diff = 100 * (d300[d300.pair == p].fail_worse - d300[d300.pair == p].fail_better)
        edges = np.arange(lo - step / 2, hi + step, step)
        _, _, patches = ax.hist(diff.clip(lo, hi), bins=edges, color=colors[2], edgecolor=BG)
        for patch, left in zip(patches, edges[:-1]):
            mid = left + step / 2
            patch.set_facecolor(colors[0] if mid < -1e-9 else colors[1] if abs(mid) < 1e-9 else colors[2])
        ax.axvline(0, color="#FFFFFF", lw=1.0, ls=":")
        wrong, tie = np.mean(diff < 0), np.mean(diff == 0)
        ax.set_title(title, fontsize=17, fontweight="bold", loc="left")
        ax.text(0.02, 0.95, f"worse ahead {100 * wrong:.1f}%\ntie {100 * tie:.1f}%", transform=ax.transAxes,
                va="top", ha="left", fontsize=15, color=OKABE_ITO["orange"], fontweight="bold")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=14)

    top = fig.add_subplot(gs[0, :])
    panel(top, "v1 vs v2", -3, 12, "A real upgrade: v2 vs v1 (true gap 5.5 points)")
    top.set_xlabel("measured improvement of the better router on one 300-query golden set (points)", fontsize=15)
    tweaks = [p for p in pairs if p != "v1 vs v2"]
    for k, p in enumerate(tweaks):
        ax = fig.add_subplot(gs[1, k])
        panel(ax, p, -3, 3, f"Tweak {p.split(':')[0]}: {p.split(': ', 1)[1]}")
        ax.set_xlim(-3.2, 3.2)
        if k == 1:
            ax.set_xlabel("same measure, zoomed to ±3 points (true gap 0.2 points)", fontsize=15)
    fig.legend(handles=[Patch(color=colors[2], label="better router ahead"), Patch(color=colors[1], label="tie"),
                        Patch(color=colors[0], label="worse router ahead")],
               loc="upper center", ncol=3, fontsize=15, bbox_to_anchor=(0.5, 1.02))
    save_fig(fig, img, "e7a_golden_set_lottery")
    plt.rcParams.update({"font.size": 12})

    # expansion: when does the model take back control?
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
    plot_boundary_map(load_cells(out), ax=a1, legend_loc="upper right", legend_fontsize=11)
    a1.set_title("Dataset size (train + test): R(n) on the E1 datasets")
    marks = ["o", "s", "D", "^"]
    cols = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["purple"], OKABE_ITO["green"]]
    for p, m, c in zip(pairs, marks, cols):
        sub = df[df.pair == p]
        wrong_or_tie = (sub.fail_worse - sub.fail_better <= 0).groupby(sub.g).mean()
        a2.plot(wrong_or_tie.index, 100 * wrong_or_tie.values, marker=m, color=c, lw=2.4, ms=7, label=titles[p])
    a2.axvspan(100, 500, color="#FFFFFF", alpha=0.06, lw=0)
    a2.text(110, 97, "typical golden sets", fontsize=11, color=NEUTRAL, va="top")
    a2.set_xscale("log")
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a2.xaxis.set_minor_formatter(NullFormatter())
    a2.set_xticks(GOLDEN_SIZES)
    a2.set_ylim(0, 100)
    a2.set_xlabel("golden-set size (queries, log scale)")
    a2.set_ylabel("% of golden sets: worse router wins or ties")
    a2.set_title("Golden-set size: the E7 routers")
    a2.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    save_fig(fig, img, "e7b_until_they_dont")


def redraw(out, img):
    figures(out, img)
