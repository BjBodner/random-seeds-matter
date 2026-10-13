"""PILOT ONLY (non-study seed root 777). Explore which data designs make seed spread >> model spread."""
import warnings, sys, time
import numpy as np
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

ROOT = 777
def sd(*key, n=1):
    return [int(s) for s in np.random.SeedSequence(ROOT, spawn_key=key).generate_state(n)]

MODELS_DEFAULT = {
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "LDA": lambda: LinearDiscriminantAnalysis(),
    "NaiveBayes": lambda: GaussianNB(),
    "SVC-RBF": lambda: make_pipeline(StandardScaler(), SVC()),
    "kNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1),
    "HistGB": lambda: HistGradientBoostingClassifier(random_state=0),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=0)),
}

def gen(kind, n, seed, **kw):
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.arange(n) % 2)
    if kind == "blobs_flip":   # two separated blobs in d dims, labels flipped with prob eta
        d, sep, eta = kw.get("d", 2), kw.get("sep", 4.0), kw["eta"]
        X = rng.normal(size=(n, d)); X[:, 0] += np.where(y == 1, sep / 2, -sep / 2)
        flip = rng.random(n) < eta
        y = np.where(flip, 1 - y, y)
        return X, y
    if kind == "overlap":      # two gaussians overlapping, bayes error set by sep
        d, sep = kw.get("d", 2), kw["sep"]
        X = rng.normal(size=(n, d)); X[:, 0] += np.where(y == 1, sep / 2, -sep / 2)
        return X, y
    raise ValueError

MODELSETS = {}
def one_split(X, y, s, mset):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
    return [m().fit(Xtr, ytr).score(Xte, yte) for m in MODELSETS[mset].values()]

def metrics(acc):
    K = acc.shape[1]
    per_split_model_sd = acc.std(axis=1, ddof=1)        # same data (split), different models
    per_model_seed_sd = acc.std(axis=0, ddof=1)         # same model, different seeds
    A = np.median(per_model_seed_sd) / np.median(per_split_model_sd)
    B = per_model_seed_sd.mean() / acc.mean(axis=0).std(ddof=1)
    # matched-K range ratio, first K splits vs models on split 0, median over choices
    rr = []
    for j in range(0, acc.shape[0] - K, K):
        blk = acc[j:j + K]
        rr.append(np.median([np.ptp(blk[:, m]) for m in range(K)]) / max(np.median([np.ptp(blk[i]) for i in range(K)]), 1e-9))
    return A, B, np.median(rr), per_split_model_sd, per_model_seed_sd

def run(name, kind, n, mset, n_data=3, n_split=200, **kw):
    models = MODELSETS[mset]
    t = time.time()
    out = []
    for j, dseed in enumerate(sd(1, sum(map(ord, name)) % 1000, n=n_data)):
        X, y = gen(kind, n, dseed, **kw)
        splits = sd(2, j, n=n_split)
        acc = np.array(Parallel(n_jobs=-1)(delayed(one_split)(X, y, s, mset) for s in splits))
        A, B, RR, ps, pm = metrics(acc)
        out.append((A, B, RR, acc.mean(axis=0), np.median(ps), np.median(pm)))
    A = np.median([o[0] for o in out]); B = np.median([o[1] for o in out]); RR = np.median([o[2] for o in out])
    means = np.mean([o[3] for o in out], axis=0)
    print(f"{name:<34} n={n:<4} A(sd ratio per-split)={A:6.2f}  B(E1 R)={B:6.2f}  rangeRatio={RR:6.2f} "
          f"seedSD={np.median([o[5] for o in out]):.4f} modelSD/split={np.median([o[4] for o in out]):.4f} "
          f"means=" + " ".join(f"{k}:{v:.3f}" for k, v in zip(models, means)) + f"  ({time.time()-t:.0f}s)", flush=True)

MODELSETS['default'] = MODELS_DEFAULT
