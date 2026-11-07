"""E9 descriptive views of the saved E1 and E8 results (NOT pre-registered; chosen after seeing the results).
Reads results/e1_split.csv, e8_tailored.csv and e8_dials.csv; no refitting (so it needs only numpy and pandas).

Three questions, in plain words:
  1. Who wins?   On each split, which model finishes first? (Do the same models win every time?)
  2. The ruler.  How big a swing does the size of the test set alone predict, sqrt(a (1 - a) / n)?
  3. Pairing.    How much noise is removed by scoring both models on the SAME split?

`python seeds_matter.py --only e9` writes results/e9_views.txt and images/e9a_same_vs_different.png.
"""
import os
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import np  # noqa: E402

# The same 8 models and 4 designs as experiments/e8_tailored.py (kept here so this file needs no scikit-learn).
MODEL_NAMES = ["Logistic regression", "Naive Bayes", "SVM (RBF)", "kNN (k=31)", "Decision tree",
               "Random forest", "Gradient boosting", "Neural net"]
DESIGN_TEST_ROWS = {"T1 overlapping classes, 300 rows": 90, "T2 disputed labels, 300 rows": 90,
                    "T3 100-row golden set, overlapping classes": 100, "T4 100-row golden set, disputed labels": 100}
HERO_MODEL = "Logistic regression"


def load(out):
    """Accuracy matrices (one row per split, one column per model) from the saved CSVs."""
    import pandas as pd

    def matrices(df, keys):
        cells = {}
        for key, g in df.groupby(keys, sort=False):
            order = g.split_seed.drop_duplicates()
            m = g.pivot(index="split_seed", columns="model", values="accuracy").reindex(order)
            cells[key] = m[MODEL_NAMES].to_numpy() if set(MODEL_NAMES) <= set(m.columns) else m
        return cells

    main = matrices(pd.read_csv(os.path.join(out, "e8_tailored.csv")), ["design", "dataset"])
    dials = matrices(pd.read_csv(os.path.join(out, "e8_dials.csv")), ["dial", "setting", "dataset"])
    e1 = pd.read_csv(os.path.join(out, "e1_split.csv"))
    e1 = e1[(e1.dataset == "breast_cancer") & (e1.n == 150)]
    real = matrices(e1, "subsample")
    return main, dials, real


def R_of(acc):
    """Seed spread divided by model spread, exactly as in experiments/e8_tailored.py."""
    return acc.std(axis=0, ddof=1).mean() / acc.mean(axis=0).std(ddof=1)


def winners(acc):
    """Each model's share of splits on which it is alone at the top, the share of splits with a tie for first,
    how many models were alone on top at least once, and how many won or tied for first at least once."""
    top = acc == acc.max(axis=1, keepdims=True)
    alone = top & (top.sum(axis=1) == 1)[:, None]
    return alone.mean(axis=0), np.mean(top.sum(axis=1) > 1), int((alone.sum(axis=0) > 0).sum()), int(top.any(axis=0).sum())


def head_to_head(a, b):
    """Share of splits on which a is strictly ahead of b, tied with b, and strictly behind b."""
    return np.mean(a > b), np.mean(a == b), np.mean(a < b)


def gaps(acc, shift=1):
    """For every pair of models, the gap 'worse model (lower average) minus better model', two ways:
    both scored on the same split, and scored on different splits (the worse model's score from the
    split `shift` places earlier, so the splits no longer match). Returns two arrays pooled over all pairs and splits."""
    m = acc.shape[1]
    mean = acc.mean(axis=0)
    same, diff = [], []
    for i in range(m):
        for j in range(i + 1, m):
            worse, better = (i, j) if mean[i] <= mean[j] else (j, i)
            same.append(acc[:, worse] - acc[:, better])
            diff.append(np.roll(acc[:, worse], shift) - acc[:, better])
    return np.concatenate(same), np.concatenate(diff)


def shares(g):
    """Share of comparisons on which the worse model looks strictly better, ties, and looks worse."""
    return np.mean(g > 0), np.mean(g == 0), np.mean(g < 0)


def ruler(acc, n_test):
    """Typical swing of one model across splits, and what the size of the test set alone predicts."""
    a = acc.mean(axis=0)
    return acc.std(axis=0, ddof=1).mean(), np.sqrt(a * (1 - a) / n_test).mean()


def plot_pairing(same, diff, path, label=""):
    """One panel: how far apart two models look on the same test set versus on different test sets."""
    from matplotlib.patches import ConnectionPatch
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from experiments.common import BG, OKABE_ITO, dark_style
    plt = dark_style()
    sky, orange, yellow, vermillion = OKABE_ITO["sky"], OKABE_ITO["orange"], OKABE_ITO["yellow"], OKABE_ITO["vermillion"]
    same, diff = 100 * same, 100 * diff
    sd_s, sd_d = same.std(ddof=1), diff.std(ddof=1)
    lim = 5 * np.ceil(np.percentile(np.abs(diff), 99.8) / 5)
    bins = np.arange(-lim - 0.5, lim + 1.5, 1.0)          # one bin per whole accuracy point
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axvspan(0, lim, color=vermillion, alpha=0.13, lw=0, zorder=0)
    ax.axvline(0, color="#FFFFFF", lw=3, zorder=1)
    ax.hist(diff, bins=bins, weights=np.full(len(diff), 100 / len(diff)), color=orange, alpha=0.95, zorder=2)
    ax.hist(same, bins=bins, weights=np.full(len(same), 100 / len(same)), color=sky, alpha=0.95, zorder=3)
    top = 16.0                                             # clip the tall bar of exact ties so both shapes stay visible
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, top)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(axis="both", labelsize=27, length=0, pad=10)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("how far ahead the worse model looks, in accuracy points", fontsize=30, labelpad=16)
    ax.set_ylabel("share of model comparisons", fontsize=30, labelpad=14)
    ax.text(0.0, 1.03, "← better model looks better", transform=ax.transAxes, ha="left", va="bottom", fontsize=30,
            color="#FFFFFF", fontweight="bold")
    ax.text(1.0, 1.03, "worse model looks better →", transform=ax.transAxes, ha="right", va="bottom", fontsize=30,
            color="#FFFFFF", fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc=vermillion, ec="none"))
    ws, ts, wd = shares(same)[0], shares(same)[1], shares(diff)[0]
    ax.text(-0.9, top * 0.955, f"bar cut off: it reaches {ts:.1%} ▸", ha="right", va="center", fontsize=26, color=sky,
            fontweight="bold")
    ax.text(-lim * 0.17, top * 0.71, f"Same golden set\nspread ±{sd_s:.2f} pts\nexact ties: {ts:.1%}\nworse looks better: {ws:.1%}",
            ha="right", va="center", fontsize=32, color=sky, fontweight="bold", linespacing=1.15)
    ax.text(lim * 0.97, top * 0.60, f"Different golden sets\nspread ±{sd_d:.2f} pts\nworse looks better: {wd:.1%}", ha="right",
            va="center", fontsize=32, color=orange, fontweight="bold", linespacing=1.15)
    fig.add_artist(ConnectionPatch(xyA=(-lim * 0.14, top * 0.64), coordsA=ax.transData, xyB=(lim * 0.12, top * 0.50),
                                   coordsB=ax.transData, arrowstyle="-|>,head_length=1.6,head_width=0.8", color=yellow,
                                   lw=8, connectionstyle="arc3,rad=0.35", zorder=20))
    ax.text(lim * 0.52, top * 0.87, f"≈{sd_d / sd_s:.0f}×\nnoisier", ha="center", va="center", fontsize=54,
            fontweight="bold", color=BG, linespacing=0.92, zorder=21,
            bbox=dict(boxstyle="round,pad=0.42", fc=yellow, ec="none"))
    if label:
        fig.text(0.5, -0.02, label, ha="center", va="top", fontsize=24, color="#FFFFFF", fontweight="bold",
                 linespacing=1.35)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return sd_d / sd_s


def run(out, img=None, quick=False):
    from experiments.common import tee
    main, dials, real = load(out)
    n_data = 1 + max(j for (_, j) in main)
    med = {d: np.median([R_of(main[d, j]) for j in range(n_data)]) for d in DESIGN_TEST_ROWS}
    design = max(med, key=med.get)                                     # E8's hero rule: largest median R ...
    Rs = [R_of(main[design, j]) for j in range(n_data)]
    inst = int(np.argsort(Rs)[(n_data - 1) // 2])                      # ... at its median-R dataset
    hero = main[design, inst]
    mean_acc = hero.mean(axis=0)
    best, worst = int(np.argmax(mean_acc)), int(np.argmin(mean_acc))
    names1 = list(real[0].columns)
    real = {k: v.to_numpy() for k, v in real.items()}
    r0 = real[0]

    with tee(os.path.join(out, "e9_views.txt")):
        print("E9 descriptive views (NOT pre-registered; chosen after seeing the results). No refitting: reads the E1 and E8 CSVs.")
        print(f"check: hero = {design}, dataset {inst}; {HERO_MODEL} seed spread {100 * hero[:, 0].std(ddof=1):.2f} pts (E8 reports 3.73)")

        print("\n== 1. Who wins? ==")
        share, tie, n_alone, n_top = winners(hero)
        print(f"hero, {hero.shape[0]} golden sets, {hero.shape[1]} models")
        print("share of golden sets on which each model was alone at the top: "
              + ", ".join(f"{MODEL_NAMES[k]} {share[k]:.1%}" for k in np.argsort(-share)))
        print(f"tie for first place on {tie:.1%} of golden sets; models that were alone on top at least once: {n_alone} of {hero.shape[1]}; "
              f"models that won or tied for first at least once: {n_top} of {hero.shape[1]}")
        w, t, l = head_to_head(hero[:, worst], hero[:, best])
        print(f"worst model on average ({MODEL_NAMES[worst]}, {100 * mean_acc[worst]:.1f}%) versus best on average "
              f"({MODEL_NAMES[best]}, {100 * mean_acc[best]:.1f}%), same golden set: the worst is ahead on {w:.1%}, tied on {t:.1%}, behind on {l:.1%}")
        for name in DESIGN_TEST_ROWS:
            res = [winners(main[name, j]) for j in range(n_data)]
            print(f"{name:<43} median over {n_data} datasets: models alone on top at least once {np.median([r[2] for r in res]):.0f} of 8; "
                  f"won or tied for first at least once {np.median([r[3] for r in res]):.0f} of 8; tie for first on {np.median([r[1] for r in res]):.1%} of splits")
        share1, tie1, a1, _ = winners(r0)
        print(f"real data (breast cancer, 150 rows, opening subsample, {len(r0)} splits): share of splits alone on top: "
              + ", ".join(f"{names1[k]} {share1[k]:.1%}" for k in np.argsort(-share1))
              + f"; tie for first on {tie1:.1%}; models alone on top at least once {a1} of {len(names1)}")
        rf = names1.index("RandomForest")
        print(f"   the weakest model on average, RandomForest ({100 * r0[:, rf].mean():.1f}%), won or tied for first on "
              f"{np.mean(r0[:, rf] == r0.max(axis=1)):.1%} of splits")

        print("\n== 2. The ruler: the swing that test-set size alone predicts, sqrt(a (1 - a) / n), in points ==")
        print("setting                                      test rows  typical swing (observed)  ruler  observed / ruler")
        rows = []
        for name, n_test in DESIGN_TEST_ROWS.items():
            res = [ruler(main[name, j], n_test) for j in range(n_data)]
            rows.append((name, n_test, np.median([r[0] for r in res]), np.median([r[1] for r in res])))
        nd = 1 + max(j for (_, _, j) in dials)
        for dial, setting in dict.fromkeys((d, s) for (d, s, _) in dials):
            if dial in ("D2 size, bend 0", "D3 golden-set size"):
                n_test = int(re.search(r"(\d+) test", setting).group(1)) if "test" in setting else int(re.search(r"golden set (\d+) rows", setting).group(1))
                res = [ruler(dials[dial, setting, j], n_test) for j in range(nd)]
                rows.append((f"{dial}: {setting}", n_test, np.median([r[0] for r in res]), np.median([r[1] for r in res])))
        res = [ruler(real[k], 45) for k in real]
        rows.append(("breast cancer, 150 rows (4 models)", 45, np.median([r[0] for r in res]), np.median([r[1] for r in res])))
        for name, n_test, obs, rul in rows:
            print(f"{name:<44} {n_test:>9}  {100 * obs:>24.2f}  {100 * rul:>5.2f}  {obs / rul:>16.2f}")
        print("rule of thumb at about 80% accuracy: swing = 40 / sqrt(n) points: "
              + ", ".join(f"n = {n} -> {40 / np.sqrt(n):.1f}" for n in (100, 400, 1600, 10000)))
        print("one test row moves accuracy by 100 / n points: "
              + ", ".join(f"n = {n} -> {100 / n:.2f}" for n in (45, 100, 300, 1000)))
        print("the same rule for a rare class: recall wobbles by 40 / sqrt(k) points at about 80% recall, k = rare cases in the test set: "
              + ", ".join(f"k = {k} -> {40 / np.sqrt(k):.1f}" for k in (25, 100, 400)))

        print("\n== 3. Same split or different splits? Gap between two models (worse minus better), pooled over all model pairs ==")
        print("spread = standard deviation of the gap, in points; 'worse looks better' = share of comparisons where the worse model scores strictly higher")
        for name in DESIGN_TEST_ROWS:
            res = [gaps(main[name, j]) for j in range(n_data)]
            s = np.median([r[0].std(ddof=1) for r in res])
            d = np.median([r[1].std(ddof=1) for r in res])
            ws = np.median([shares(r[0])[0] for r in res])
            wd = np.median([shares(r[1])[0] for r in res])
            print(f"{name:<43} same split {100 * s:.2f}, different splits {100 * d:.2f}; {d / s:.1f}x noisier; "
                  f"worse looks better on {ws:.1%} (same) vs {wd:.1%} (different) (median over {n_data} datasets)")
        same, diff = gaps(hero)
        print(f"hero: same split {100 * same.std(ddof=1):.2f}, different splits {100 * diff.std(ddof=1):.2f}; "
              f"{diff.std(ddof=1) / same.std(ddof=1):.1f}x noisier ({len(same)} comparisons: 28 model pairs x {hero.shape[0]} splits)")
        a, b, c = shares(same)
        a2, b2, c2 = shares(diff)
        print(f"hero: the worse model of the pair looks better on {a:.1%}, ties on {b:.1%}, looks worse on {c:.1%} of same-split comparisons; "
              f"on different-split comparisons: better {a2:.1%}, ties {b2:.1%}, worse {c2:.1%}")
        pooled = [gaps(real[k]) for k in real]
        sr = np.median([p[0].std(ddof=1) for p in pooled])
        dr = np.median([p[1].std(ddof=1) for p in pooled])
        print(f"real data (breast cancer, 150 rows, 4 models, 6 pairs, median over {len(real)} subsamples): same split "
              f"{100 * sr:.2f}, different splits {100 * dr:.2f}; {dr / sr:.1f}x noisier")
        lr1, svc1 = r0[:, names1.index("LogReg")], r0[:, names1.index("SVC")]
        w3, t3, l3 = head_to_head(svc1, lr1)
        w4, t4, l4 = head_to_head(np.roll(svc1, 1), lr1)
        print(f"real data, opening subsample, SVC vs LogReg (LogReg is better on average): same split SVC ahead on {w3:.1%}, tied on {t3:.1%}, behind on {l3:.1%}; "
              f"different splits SVC ahead on {w4:.1%}, tied on {t4:.1%}, behind on {l4:.1%}")

        print("\n== 4. The tie rule: only the rows where two models disagree can pick a winner ==")
        print("Among d disagreements, the lead (rows won by A minus rows won by B) is a fair-coin count with a typical size of "
              "sqrt(d) rows if the models are really equal. A lead of less than 2 * sqrt(d) rows is inside the usual 5% bar: call it a tie.")
        print("(A lead has the same odd/even parity as d. This is the 5% bar for one planned comparison, not for the best of several tries.)")
        for d, n in ((15, 300), (100, 1000)):
            band = 2 * np.sqrt(d)
            rows_tie = int(np.ceil(band)) - 1
            if (rows_tie - d) % 2:
                rows_tie -= 1                                      # leads are odd when d is odd and even when d is even
            print(f"d = {d} disagreements on {n} rows: 2 * sqrt({d}) = {band:.1f} rows, so a lead of {rows_tie} rows or fewer "
                  f"({100 * rows_tie / n:.1f} points) is a tie")
        from math import comb
        d, k = 15, 9
        tail = sum(comb(d, j) for j in range(k, d + 1)) / 2 ** d
        print(f"example: 9 wins to 6 on d = 15 (a lead of 3): a fair coin gives {k} or more heads in {d} flips {tail:.1%} of the time")

        print("\n== 3b. How much does the choice of 'different splits' pairing matter? Hero data, every possible shift 1..199 ==")
        ratios, worse_better = [], []
        for s in range(1, hero.shape[0]):
            sm, df_ = gaps(hero, s)
            ratios.append(df_.std(ddof=1) / sm.std(ddof=1))
            worse_better.append(shares(df_)[0])
        print(f"noise ratio across all shifts: mean {np.mean(ratios):.1f}x (5th to 95th percentile {np.percentile(ratios, 5):.1f} to "
              f"{np.percentile(ratios, 95):.1f}; min {np.min(ratios):.1f}, max {np.max(ratios):.1f}); the plotted pairing (shift 1) gives {ratios[0]:.1f}x")
        print(f"worse model looks better across sets: mean {np.mean(worse_better):.1%} (5th to 95th percentile "
              f"{np.percentile(worse_better, 5):.1%} to {np.percentile(worse_better, 95):.1%})")

        print("\n== 5. Which seed? The split versus the model's own seed (E5 data: random forest and gradient boosting, n = 300, 30 split seeds x 30 model seeds) ==")
        import pandas as pd
        e5 = pd.read_csv(os.path.join(out, "e5_decomposition.csv"))
        e5 = e5[(e5.n == 300) & e5.model.isin(["RandomForest", "GradBoost"])]
        cells, all_vals = [], []
        for model, g in e5.groupby("model"):
            m = g.pivot(index="split_seed", columns="internal_seed", values="accuracy").to_numpy()   # rows: splits, columns: model seeds
            split_sd = 100 * np.std(m.mean(axis=1), ddof=1)
            seed_sd = 100 * np.sqrt(np.mean(np.var(m, axis=1, ddof=1)))
            print(f"{model}: change only the split -> typical swing {split_sd:.2f} pts; on one fixed split, change only the model's own seed -> typical swing {seed_sd:.2f} pts")
            cells += [np.var(row) for row in m]
            all_vals.append(m.ravel())
        total = np.var(np.concatenate(all_vals))
        print(f"share of all the movement that sits inside one (model, split) cell, i.e. the same model on the same split with a different seed: {np.mean(cells) / total:.1%}")

        print("\n== 6. The strict (like-for-like) wobble ratio everywhere: one model's swing across splits / how far apart the models land on the SAME split ==")
        print("(The registered ratio R divides by the spread of the models' 200-split averages instead; it is always at least as large. Both are listed.)")
        strict = lambda m: m.std(axis=0, ddof=1).mean() / np.sqrt(np.mean(m.var(axis=1, ddof=1)))  # noqa: E731
        for name in DESIGN_TEST_ROWS:
            s = [strict(main[name, j]) for j in range(n_data)]
            print(f"{name:<43} strict median {np.median(s):.1f} (min {min(s):.1f}, max {max(s):.1f}); registered R median {np.median([R_of(main[name, j]) for j in range(n_data)]):.1f}")
        print(f"hero dataset: strict {strict(hero):.1f}, registered R {R_of(hero):.1f}")
        e1 = pd.read_csv(os.path.join(out, "e1_split.csv"))
        cells_e1 = {}
        for key, g in e1.groupby(["dataset", "n", "subsample"], sort=False):
            order = g.split_seed.drop_duplicates()
            cells_e1[key] = g.pivot(index="split_seed", columns="model", values="accuracy").reindex(order).to_numpy()
        s150 = [strict(cells_e1["breast_cancer", 150, j]) for j in range(5)]
        print("breast cancer, 150 rows, the 5 draws: strict " + ", ".join(f"{v:.2f}" for v in s150)
              + f" (median {np.median(s150):.2f}); registered R " + ", ".join(f"{R_of(cells_e1['breast_cancer', 150, j]):.2f}" for j in range(5)))
        for ds in ("breast_cancer", "synthetic"):
            for n in sorted({k[1] for k in cells_e1 if k[0] == ds}):
                s = [strict(cells_e1[k]) for k in cells_e1 if k[0] == ds and k[1] == n]
                r = [R_of(cells_e1[k]) for k in cells_e1 if k[0] == ds and k[1] == n]
                print(f"{ds:<13} n={n:<5} strict median {np.median(s):.2f} (min {min(s):.2f}, max {max(s):.2f}); registered R median {np.median(r):.2f}")
        for dial, setting in dict.fromkeys((d_, s_) for (d_, s_, _) in dials):
            s = [strict(dials[dial, setting, j]) for j in range(nd)]
            print(f"{dial}: {setting:<34} strict median {np.median(s):.2f} (min {min(s):.2f}, max {max(s):.2f}); registered R median {np.median([R_of(dials[dial, setting, j]) for j in range(nd)]):.2f}")

        print("\n== 6b. The registered predictions about R, re-scored on the strict ratio (they were written for R and held on R) ==")
        med_e1 = {(ds, n): np.median([strict(cells_e1[k]) for k in cells_e1 if k[0] == ds and k[1] == n])
                  for (ds, n, _) in cells_e1}
        sizes_bc = sorted(n for (ds, n) in med_e1 if ds == "breast_cancer")
        sizes_sy = sorted(n for (ds, n) in med_e1 if ds == "synthetic")
        cross = next((f"between n = {a} and n = {b}" for a, b in zip(sizes_sy, sizes_sy[1:])
                      if med_e1["synthetic", a] >= 1 > med_e1["synthetic", b]), "no crossing")
        dmed = {(d_, s_): np.median([strict(dials[d_, s_, j]) for j in range(nd)]) for (d_, s_, _) in dials}
        dkeys = list(dmed)
        e8med = {name: np.median([strict(main[name, j]) for j in range(n_data)]) for name in DESIGN_TEST_ROWS}
        verdict = lambda ok: "holds" if ok else "FAILS"  # noqa: E731
        checks = [
            ("P1.1 median ratio at n = 150 (breast cancer) >= 2.0", f"{med_e1['breast_cancer', 150]:.2f}", med_e1["breast_cancer", 150] >= 2.0),
            ("P1.2a breast cancer ratio > 1 at every n", "smallest " + f"{min(med_e1['breast_cancer', n] for n in sizes_bc):.2f}",
             all(med_e1["breast_cancer", n] > 1 for n in sizes_bc)),
            ("P1.2b synthetic crosses 1 between n = 250 and n = 1,000", cross, cross.startswith("between n = 250") or cross.startswith("between n = 500")),
            ("P8.1 every design has median ratio >= 4", ", ".join(f"{v:.1f}" for v in e8med.values()), all(v >= 4 for v in e8med.values())),
            ("P8.2 the largest design median ratio >= 10", f"{max(e8med.values()):.1f}", max(e8med.values()) >= 10),
            ("P8.3 D1: ratio >= 3 at bend 0 and < 1 at bend 2", f"{dmed[dkeys[0]]:.2f} and {dmed[dkeys[4]]:.2f}", dmed[dkeys[0]] >= 3 and dmed[dkeys[4]] < 1),
            ("P8.4 D2: bend 1 < 1 at n = 1,000 and 3,000; bend 0 > 1 at every n",
             f"bend 1: {dmed[dkeys[11]]:.2f}, {dmed[dkeys[12]]:.2f}; bend 0 smallest {min(dmed[k] for k in dkeys[5:9]):.2f}",
             dmed[dkeys[11]] < 1 and dmed[dkeys[12]] < 1 and all(dmed[k] > 1 for k in dkeys[5:9])),
            ("P8.5 D3: lower at g = 1,000 than at g = 50, and > 1 at every g",
             f"{dmed[dkeys[16]]:.2f} vs {dmed[dkeys[13]]:.2f}; smallest {min(dmed[k] for k in dkeys[13:17]):.2f}",
             dmed[dkeys[16]] < dmed[dkeys[13]] and all(dmed[k] > 1 for k in dkeys[13:17])),
        ]
        for text, value, ok in checks:
            print(f"{text}: strict {value} -> {verdict(ok)}")
        print(f"on the strict ratio, {sum(not c[2] for c in checks)} of these {len(checks)} predictions (all of which held on R) would fail")
    if img:
        plot_pairing(same, diff, os.path.join(img, "e9a_same_vs_different.png"),
                     label="Lab-built dataset: a 100-row golden set where 1 label in 5 is disputed (the hero's dataset, 200 splits).\n"
                           "Each comparison: the worse model's score minus the better model's, for all 28 pairs of the 8 models.")


def redraw(out, img):
    run(out, img)


if __name__ == "__main__":
    from experiments.common import out_dirs
    r, i = out_dirs(False)
    run(r, i)
