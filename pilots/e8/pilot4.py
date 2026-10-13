from plib4 import *
if __name__ == "__main__":
    K = dict(mset="robust", n_data=5, n_split=200)
    run("overlap sep=2.0 n=300", "overlap", 300, sep=2.0, **K)
    run("overlap sep=1.5 n=300", "overlap", 300, sep=1.5, **K)
    run("flip eta=0.2 sep=6 n=300", "flip", 300, sep=6.0, eta=0.2, **K)
    run("GOLDEN overlap sep=2.0 1000+100", "overlap", 1100, test_size=100, sep=2.0, **K)
    run("GOLDEN overlap sep=1.5 1000+100", "overlap", 1100, test_size=100, sep=1.5, **K)
    run("GOLDEN flip eta=0.2 sep=6 1000+100", "flip", 1100, test_size=100, sep=6.0, eta=0.2, **K)
    run("GOLDEN overlap sep=2.0 300+100", "overlap", 400, test_size=100, sep=2.0, **K)
