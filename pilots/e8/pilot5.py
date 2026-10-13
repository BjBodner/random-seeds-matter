from plib5 import *
if __name__ == "__main__":
    K = dict(mset="robust", n_data=2, n_split=100)
    run("T1 overlap sep=2 n=300", "overlap", 300, sep=2.0, **K)
    run("T2 flip .2 sep=6 n=300", "flip", 300, sep=6.0, eta=0.2, **K)
    run("T3 golden overlap sep=2", "overlap", 1100, test_size=100, sep=2.0, **K)
    run("T4 golden flip .2 sep=6", "flip", 1100, test_size=100, sep=6.0, eta=0.2, **K)
    for b in (0.5, 1.0, 2.0):
        run(f"dial bend={b} sep=2 n=300", "overlap", 300, sep=2.0, bend=b, **K)
    for n in (1000, 3000):
        run(f"dial bend=1 sep=2 n={n}", "overlap", n, sep=2.0, bend=1.0, **K)
    run("dial bend=0 sep=2 n=3000", "overlap", 3000, sep=2.0, **K)
    for g in (300, 1000):
        run(f"dial golden size {g} sep=2", "overlap", 1000 + g, test_size=g, sep=2.0, **K)
