from plib import *
if __name__ == "__main__":
    for eta in (0.1, 0.2, 0.3):
        run(f"blobs_flip eta={eta} sep=4 d=2", "blobs_flip", 300, 'default', eta=eta, sep=4.0, d=2)
    run("blobs_flip eta=0.3 sep=6 d=2", "blobs_flip", 300, 'default', eta=0.3, sep=6.0, d=2)
    run("blobs_flip eta=0.2 sep=4 d=5", "blobs_flip", 300, 'default', eta=0.2, sep=4.0, d=5)
    for sep in (1.0, 1.5):
        run(f"overlap sep={sep} d=2", "overlap", 300, 'default', sep=sep, d=2)
