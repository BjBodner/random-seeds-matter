"""PILOT 2 ONLY (non-study seed root 778)."""
import warnings, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

ROOT = 779
def sd(*key, n=1):
    return [int(s) for s in np.random.SeedSequence(ROOT, spawn_key=key).generate_state(n)]

def tuned(est, grid):
    return lambda: GridSearchCV(est, grid, cv=StratifiedKFold(5))

TUNED = {
    "LogReg": tuned(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), {"logisticregression__C": [0.01, 0.1, 1, 10]}),
    "NaiveBayes": lambda: GaussianNB(),
    "SVM": tuned(make_pipeline(StandardScaler(), SVC()), {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.1]}),
    "kNN": tuned(make_pipeline(StandardScaler(), KNeighborsClassifier()), {"kneighborsclassifier__n_neighbors": [5, 15, 31, 61]}),
    "Tree": tuned(DecisionTreeClassifier(random_state=0), {"max_depth": [1, 2, 4, None], "min_samples_leaf": [1, 10]}),
    "Forest": tuned(RandomForestClassifier(n_estimators=100, random_state=0), {"min_samples_leaf": [1, 5, 20]}),
    "Boosting": tuned(HistGradientBoostingClassifier(random_state=0), {"max_depth": [2, None], "learning_rate": [0.03, 0.1]}),
    "MLP": tuned(make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=0)), {"mlpclassifier__alpha": [1e-3, 1e-1, 1, 10]}),
}
ROBUST = {  # fixed noise-robust settings, no tuning
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "NaiveBayes": lambda: GaussianNB(),
    "SVM": lambda: make_pipeline(StandardScaler(), SVC()),
    "kNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31)),
    "Tree": lambda: DecisionTreeClassifier(max_depth=2, random_state=0),
    "Forest": lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=0),
    "Boosting": lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, random_state=0),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=2000, random_state=0)),
}
MODELSETS = {"tuned": TUNED, "robust": ROBUST}

def gen(kind, n, seed, **kw):
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.arange(n) % 2)
    d = kw.get("d", 2)
    X = rng.normal(size=(n, d))
    if kind == "flip":        # clean clusters, then annotators flip eta of labels
        X[:, 0] += np.where(y == 1, kw["sep"] / 2, -kw["sep"] / 2)
        y = np.where(rng.random(n) < kw["eta"], 1 - y, y)
    elif kind == "overlap":   # optional bend: the class boundary curves as x0 = bend * (x1^2 - 1)
        X[:, 0] += np.where(y == 1, kw["sep"] / 2, -kw["sep"] / 2) + kw.get("bend", 0.0) * (X[:, 1] ** 2 - 1)
    elif kind == "coinzone":  # fraction h of rows in an ambiguous zone with 50/50 labels
        X[:, 0] += np.where(y == 1, kw["sep"] / 2, -kw["sep"] / 2)
        amb = rng.random(n) < kw["h"]
        X[amb, 0] = rng.normal(0, 0.5, amb.sum())       # ambiguous rows sit in the middle
        y = np.where(amb, rng.integers(0, 2, n), y)
    return X, y

def one_split(X, y, s, mset):
    warnings.filterwarnings("ignore")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
    return [m().fit(Xtr, ytr).score(Xte, yte) for m in MODELSETS[mset].values()]

def metrics(acc):
    v_seed = acc.var(axis=0, ddof=1)       # per model, across seeds
    v_model = acc.var(axis=1, ddof=1)      # per split, across models
    A_rms = np.sqrt(v_seed.mean() / v_model.mean())
    A_med = np.median(np.sqrt(v_seed)) / max(np.median(np.sqrt(v_model)), 1e-12)
    B = np.sqrt(v_seed).mean() / acc.mean(axis=0).std(ddof=1)
    return A_rms, A_med, B, np.sqrt(v_seed.mean()), np.sqrt(v_model.mean())

def run(name, kind, n, mset, n_data=2, n_split=100, **kw):
    t = time.time(); out = []
    for j, dseed in enumerate(sd(1, sum(map(ord, name)) % 1000, n=n_data)):
        X, y = gen(kind, n, dseed, **kw)
        acc = np.array(Parallel(n_jobs=-1)(delayed(one_split)(X, y, s, mset) for s in sd(2, j, n=n_split)))
        out.append(metrics(acc) + (acc.mean(axis=0),))
    med = lambda i: np.median([o[i] for o in out])
    means = np.mean([o[5] for o in out], axis=0)
    print(f"[{mset}] {name:<30} n={n:<4} A_rms={med(0):6.2f} A_med={med(1):6.2f} B(R)={med(2):6.2f} seedSD={med(3):.4f} modelSD/split={med(4):.4f} "
          + " ".join(f"{k}:{v:.3f}" for k, v in zip(MODELSETS[mset], means)) + f" ({time.time()-t:.0f}s)", flush=True)
