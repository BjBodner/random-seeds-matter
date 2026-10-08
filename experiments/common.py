"""Shared plumbing. The region between the `article` markers is shown verbatim in the article."""
# --- article: common ---
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import BrokenProcessPool

ROOT_SEED = 20261008                        # the root of every random stream
QUICK = "--quick" in sys.argv               # small seed counts, for a fast check
N_JOBS = int(os.environ.get("N_JOBS", -1))  # parallel workers (-1 = all cores)


def seeds(experiment, stream, n):
    """n reproducible 32-bit seeds for one named stream of one experiment."""
    ss = np.random.SeedSequence(ROOT_SEED, spawn_key=(experiment, stream))
    return [int(s) for s in ss.generate_state(n)]


def pmap(fn, tasks):
    """fn(*task) for every task, in parallel. Results depend only on the seeds,
    so if a worker process dies (joblib does this on some Windows setups), retry."""
    for _ in range(3):
        try:
            return Parallel(n_jobs=N_JOBS)(delayed(fn)(*t) for t in tasks)
        except BrokenProcessPool:
            print("a worker process died; retrying", file=sys.stderr)
    return Parallel(n_jobs=N_JOBS, backend="threading")(delayed(fn)(*t) for t in tasks)


def subsample(X, y, n, seed):
    """n rows without replacement (sorted, so n = len(y) gives the dataset itself)."""
    if n >= len(y):
        return X, y
    idx = np.sort(np.random.default_rng(seed).choice(len(y), n, replace=False))
    return X[idx], y[idx]


def synthetic_pool(n_total=10_000):
    """A fixed synthetic population (20 features, 5 informative, 5% label noise)."""
    from sklearn.datasets import make_classification
    return make_classification(n_samples=n_total, n_features=20, n_informative=5,
                               n_redundant=5, flip_y=0.05, class_sep=0.8,
                               random_state=seeds(0, 0, 1)[0])


def bootstrap_ci(stat, *arrays, n_boot=2000, seed=0):
    """95% percentile bootstrap CI, resampling seeds (rows) with shared indices."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = [stat(*(a[idx] for a in arrays)) for idx in rng.integers(0, n, (n_boot, n))]
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))
# --- end ---

import contextlib  # noqa: E402

# ---------------------------------------------------------------- repo-only helpers
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def out_dirs(quick):
    """Real results go to results/ and images/; --quick goes to out/quick/ (never committed)."""
    if quick:
        return os.path.join(REPO, "out", "quick", "results"), os.path.join(REPO, "out", "quick", "images")
    return os.path.join(REPO, "results"), os.path.join(REPO, "images")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def tee(path):
    """Print to the console AND capture into results/<name>.txt."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        old = sys.stdout
        sys.stdout = _Tee(old, f)
        try:
            yield
        finally:
            sys.stdout = old


BG, FG, NEUTRAL = "#121212", "#FFFFFF", "#BBBBBB"
OKABE_ITO = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
MODEL_COLORS = {
    "LogReg": OKABE_ITO["orange"], "RandomForest": OKABE_ITO["sky"],
    "SVC": OKABE_ITO["green"], "kNN": OKABE_ITO["purple"],
    "MLP": OKABE_ITO["vermillion"], "GradBoost": OKABE_ITO["yellow"],
}


def dark_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
        "axes.edgecolor": NEUTRAL, "grid.color": "#333333", "grid.linewidth": 0.6,
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 100,
    })
    return plt


def save_fig(fig, img_dir, name):
    import matplotlib.pyplot as plt
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def verdict(ok):
    return "HELD" if ok else "FAILED"
