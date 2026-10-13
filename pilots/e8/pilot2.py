from plib2 import *
if __name__ == "__main__":
    for mset in ("robust", "tuned"):
        run("flip eta=0.3 sep=6 d=2", "flip", 300, mset, eta=0.3, sep=6.0)
        run("flip eta=0.3 sep=4 d=2", "flip", 300, mset, eta=0.3, sep=4.0)
        run("coinzone h=0.5 sep=6", "coinzone", 300, mset, h=0.5, sep=6.0)
        run("overlap sep=1.5", "overlap", 300, mset, sep=1.5)
        run("flip eta=0.3 sep=6 d=5", "flip", 300, mset, eta=0.3, sep=6.0, d=5)
