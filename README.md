# Random seeds matter more than your model, until they don't

Code, results and figures for the Towards Data Science article by **Benjamin Bodner**:
*At golden-set sizes of a few hundred examples, the random split can outvote your model: 7 pre-registered experiments and the fix.*

Every number in the article is printed by this code and captured in `results/`. Every figure is in `images/`. Every random stream derives from one root, `np.random.SeedSequence(20261008)`.

## Run it

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python seeds_matter.py --quick  # small seed counts, writes to out/quick/ (about 15 min on a busy 16-thread laptop)
python seeds_matter.py          # the full run, writes results/ and images/
python seeds_matter.py --only e1,e3 --n-jobs 8
```

- Python 3.12; the pinned versions are in `requirements.txt`. The runtime and machine for the published numbers are in `results/env.txt`.
- E7 downloads Banking77 (PolyAI, CC BY 4.0, about 1 MB) on first run into `data/banking77` (gitignored); set `BANKING77_DIR` to use another folder.
- CPU only. The full run is dominated by E1 (11 dataset sizes × 5 subsamples × 200 splits × 4 models) and E5 (2 × 2,700 fits).
- **Windows note:** joblib 1.6.0's process pool crashed intermittently on the author's Windows laptop (`WinError 6`). `requirements.txt` pins joblib 1.4.2, and `pmap()` in `experiments/common.py` retries a batch if a worker dies. Results depend only on the seeds, so a retry can't change a number.

## What's where

| Experiment | Script | Results | Figures |
|---|---|---|---|
| Shared seeds, parallel map, data | `experiments/common.py` | — | — |
| E1 — the split lottery and the boundary map | `experiments/e1_split.py` | `results/e1_split.txt`, `.csv` | `e1a_parallel_universes.png`, `e1b_boundary_map.png` (standalone boundary map), `e1c_worst_case.png` |
| E2 — stratification vs recall noise | `experiments/e2_imbalance.py` | `results/e2_imbalance.txt`, `.csv` | `e2_imbalance.png` |
| E3 — the init lottery, audited | `experiments/e3_init.py` | `results/e3_init.txt`, `.csv` | `e3_init_audit.png` |
| E4 — k-means restarts | `experiments/e4_kmeans.py` | `results/e4_kmeans.txt`, `.csv` | `e4a_kmeans_scatter.png`, `e4b_ari_distributions.png` |
| E5 — variance decomposition | `experiments/e5_decomposition.py` | `results/e5_decomposition.txt`, `.csv` | `e5_variance_shares.png` |
| E5 robustness — the MLP exactly as pre-registered (n = 300) | `experiments/e5_prereg_mlp.py` (`--only e5r`) | `results/e5_prereg_mlp.txt`, `.csv` | (bar in `e5_variance_shares.png`) |
| E5 follow-up (exploratory) — close pair only, bootstrap CIs | `experiments/e5_exploratory.py` (reads the E5 CSV) | `results/e5_exploratory.txt` | `e5_variance_shares.png` |
| E7 / E7b — the golden-set lottery: an intent router on Banking77 (a real upgrade + 3 small tweaks) | `experiments/e7_router.py` | `results/e7_router.txt`, `results/e7b_tweaks.txt`, `.csv` | `e7a_golden_set_lottery.png` (hero), `e7b_until_they_dont.png` |
| E7 descriptive + McNemar sample-size table | `experiments/e7_descriptive.py` (`--only e7d`) | `results/e7_descriptive.txt` | — |
| E6 — comparing two models honestly | `experiments/e6_fix.py` | `results/e6_fix.txt`, `.csv` | `e6_procedures.png` |

Each script's code between the `# --- article: ... ---` markers is exactly what the article shows. `tools/render_draft.py` pastes it into the article, and `tools/check_code_blocks.py` fails if the two ever drift. Running a script directly prints the article's numbers; `seeds_matter.py` also writes the CSVs, figures and pre-registration checks.

## Integrity

- `PREREGISTRATION.md` was committed before any experiment ran (see `git log`). Every prediction is reported as held or failed in `results/*.txt` and in the article.
- Deviations from the pre-registration are listed in `article/DECISIONS.md`. The main one: E5's MLP uses `early_stopping=True` for runtime. `experiments/e5_prereg_mlp.py` reruns n = 300 with the MLP exactly as pre-registered; that rerun reverses the one E5 prediction that failed.
- Redraw every figure from the CSVs without refitting: `python seeds_matter.py --redraw --only e1,e3,e5,e7`.
- `tools/check_numbers.py` fails the build if any number in the article's prose is missing from `results/`.

## Data

- UCI Breast Cancer Wisconsin (Diagnostic), W. Wolberg, O. Mangasarian, N. Street and W. Street, https://doi.org/10.24432/C5DW2B, CC BY 4.0. It is loaded through `sklearn.datasets.load_breast_cancer` and used only as a methods benchmark.
- Synthetic data from `sklearn.datasets.make_classification` and `make_blobs`.
- Banking77, I. Casanueva et al. (PolyAI), https://github.com/PolyAI-LDN/task-specific-datasets, CC BY 4.0: 77 customer-service intents, 10,003 train and 3,080 test queries.

## License

Code: MIT (see `LICENSE`). Figures and text: © Benjamin Bodner.
