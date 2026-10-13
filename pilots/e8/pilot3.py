from plib3 import *
if __name__ == "__main__":
    import sys
    part = sys.argv[1]
    if part == "difficulty":
        for sep in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            run(f"overlap sep={sep}", "overlap", 300, "robust", n_data=3, n_split=200, sep=sep)
        run("overlap sep=1.0 d=5", "overlap", 300, "robust", n_data=3, n_split=200, sep=1.0, d=5)
        run("flip eta=0.3 sep=4 (again)", "flip", 300, "robust", n_data=3, n_split=200, sep=4.0, eta=0.3)
    if part == "size":
        for n in (100, 150, 300, 1000, 3000):
            run(f"overlap sep=1.0 bend=0", "overlap", n, "robust", n_data=2, n_split=100, sep=1.0)
    if part == "bend":
        for b in (0.25, 0.5, 1.0, 2.0):
            run(f"overlap sep=1.0 bend={b}", "overlap", 300, "robust", n_data=2, n_split=100, sep=1.0, bend=b)
        for n in (150, 1000, 3000):
            run(f"overlap sep=1.0 bend=1.0", "overlap", n, "robust", n_data=2, n_split=100, sep=1.0, bend=1.0)
