# E8 pilots: design exploration, NOT study results

These scripts chose the E8 designs before E8 was pre-registered (see `PREREGISTRATION.md`, E8, "Pilots"). They are kept for transparency only.
- They use separate seed roots (`SeedSequence(777)` to `SeedSequence(781)`) that no study stream uses.
- Fewer datasets and splits than the study.
- Each `plibN.py` was patched from the previous one, so the code is exploratory rather than clean.
- **None of these numbers is quoted in the article.** `tools/check_numbers.py` only reads `results/`.

| Pilot | What it tried | Lesson |
|---|---|---|
| 1 | Off-the-shelf models (kNN k=5, default forest and boosting) on label-flip and overlap data | Noisy labels pull kNN, forests and boosting below the ceiling; R tops out at about 2–2.6 |
| 2 | CV-tuned models vs fixed regularized settings | Tuning is ~15× slower and less stable (an MLP picked a degenerate alpha); fixed regularized settings work better |
| 3 | Difficulty, bend and size sweeps | Bending the boundary drives R below 1; more data then helps the model only when ceilings differ |
| 4 | Golden-set designs (1,000 train + 100 test), 5 datasets each | The golden-set designs give the largest R |
| 5 | Agreement between models; calibration of the dials | Set the pre-registered thresholds |

Run one from this folder with `PYTHONPATH=. python pilot3.py difficulty` (pilot 3 takes an argument: `difficulty`, `bend` or `size`).
