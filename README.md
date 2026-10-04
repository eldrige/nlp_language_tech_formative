## BiLSTM / BiGRU approach

Code in `src/`:
- `src/preprocessing.py` - shared cleaning, tokenization, vocabulary and padded-sequence pipeline used by this approach.
- `src/bilstm_gru.py` - the BiLSTM/BiGRU model definitions.
- `src/bilstm_experiments.py` - runs the full experiment grid (17 runs: 9 design-choice runs at seed 42, plus seeds 1 and 2 for the 4 final models). Resumable; finished runs are skipped.
- `src/bilstm_report.py` - rebuilds the summary tables and figures from the saved runs.

To reproduce:
```
cd src
python bilstm_experiments.py   # trains all 17 runs, can take hours on CPU
python bilstm_report.py        # builds summary_*.csv and fig_*.png from the runs
```
A ready-to-run Colab version is in `notebooks/colab_bilstm_gru.ipynb` (also works without retraining, using the committed results).

Outputs:
- `results/bilstm_gru/` - per-run artifacts (`runs/`), summary tables (`summary_runs.csv`, `summary_seeds.csv`), error files, and figures (`fig_*.png`).
- `results/predictions/` - hold-out predictions in the shared evaluation format, plus a Zindi submission file.
- `docs/methodology.md` and `docs/bilstm_gru_results.md` - methodology and full results/error analysis.

Headline results (masked setting, keywords removed so the model can't just spot the collection keyword): BiLSTM/BiGRU reach about 0.94 macro-F1 and 98.4% accuracy, versus 0.911 macro-F1 for TF-IDF + logistic regression on the same masked text. In the unmasked (standard) setting both recurrent models and TF-IDF + LR are near-perfect (>=0.979 macro-F1), since the collection keywords alone make the task easy. See `docs/bilstm_gru_results.md` for the full breakdown and error analysis.
