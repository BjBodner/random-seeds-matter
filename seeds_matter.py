"""Reproduce every number and figure in "Random seeds matter more than your model, until they don't".

    python seeds_matter.py               # full run -> results/ and images/
    python seeds_matter.py --quick       # small seed counts -> out/quick/ (a few minutes)
    python seeds_matter.py --only e1,e3  # a subset
    python seeds_matter.py --n-jobs 4    # limit parallel workers (default: all cores)

Every random stream derives from np.random.SeedSequence(20261008) (see experiments/common.py).
"""
import argparse
import importlib
import os
import platform
import sys
import time

EXPERIMENTS = {
    "e1": "experiments.e1_split",
    "e2": "experiments.e2_imbalance",
    "e3": "experiments.e3_init",
    "e4": "experiments.e4_kmeans",
    "e5": "experiments.e5_decomposition",
    "e6": "experiments.e6_fix",
}


def env_report(path, timings, quick, n_jobs):
    import joblib, matplotlib, numpy, pandas, scipy, sklearn
    from experiments.common import ROOT_SEED
    lines = [
        f"root seed: SeedSequence({ROOT_SEED})",
        f"mode: {'quick' if quick else 'full'}",
        f"python: {platform.python_version()} ({platform.python_implementation()})",
        f"platform: {platform.platform()}",
        f"processor: {platform.processor() or 'unknown'}",
        f"logical CPUs: {os.cpu_count()}; parallel workers requested: {n_jobs}",
    ] + [f"{m.__name__}: {m.__version__}" for m in (numpy, scipy, sklearn, pandas, matplotlib, joblib)]
    lines += [f"runtime {k}: {v:.0f} s" for k, v in timings.items()]
    lines.append(f"runtime (this invocation): {sum(timings.values()) / 60:.1f} min")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--only", default=",".join(EXPERIMENTS), help="comma list, e.g. e1,e3")
    p.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", -1)))
    p.add_argument("--redraw", action="store_true", help="redraw figures from results/*.csv without refitting (e1, e5)")
    a = p.parse_args()
    if a.redraw:
        from experiments.common import out_dirs
        out, img = out_dirs(a.quick)
        for key in [k.strip() for k in a.only.split(",") if k.strip()]:
            mod = importlib.import_module(EXPERIMENTS[key])
            if hasattr(mod, "redraw"):
                print(f"redrawing {key}")
                mod.redraw(out, img)
        return
    os.environ["N_JOBS"] = str(a.n_jobs)        # read by experiments.common at import time
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))
    from experiments.common import out_dirs
    out, img = out_dirs(a.quick)
    timings = {}
    for key in [k.strip() for k in a.only.split(",") if k.strip()]:
        print(f"\n######## {key} ########", flush=True)
        t = time.time()
        importlib.import_module(EXPERIMENTS[key]).run(out=out, img=img, quick=a.quick)
        timings[key] = time.time() - t
    env_report(os.path.join(out, "env.txt"), timings, a.quick, a.n_jobs)
    # shut the worker pool down explicitly; on Windows a lingering pool can stall interpreter exit
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)


if __name__ == "__main__":
    sys.exit(main())
