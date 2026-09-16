# GNPS2-AnnEx — Paper Analysis Code

This folder contains the analysis scripts, notebooks, and reports used to produce
the figures and evaluation results in the GNPS2-AnnEx manuscript. It is intended
to be moved into its own standalone repository for the paper's code-availability
statement.

## Contents

- **`compare_topk.py`**, **`MCES.py`** — top-level comparison / MCES utility scripts.
- **`cosine_consistency.md`** — method specification for the cosine-consistency
  metric described in the manuscript.
- **`MSMLS_lib/`** — evaluation against a validated ground-truth MS/MS standards
  library (GNPS2 task `31bb8489077242c69e3502e45f4061e2`, plus a confusion/match-count
  analysis built on task `384d5ca63dae452b95c762743c78dc31`).
  - `scripts/` — figure- and report-generating scripts (confusion matrix, cosine-vs-app
    comparison, match-count reports, mirror plots, molecular-mass report, parameter
    correlation matrix, ground-truth evaluation).
  - `analysis.ipynb` — consolidated notebook reproducing the analyses in `scripts/`.
  - `data/ground_truth/` — the validated standards table
    (`validated_with_GNPS_by_LibraryName.csv`).
  - `outputs/` — generated figures (`new_figures/`) and evaluation reports (`reports/`).
- **`plusrise_dataset/`** — confidence-score analysis on the PlusRise dataset
  (GNPS2 task `b22ff1bc41624e29a28cf6e12fe17ad5`).
  - `scripts/` — figure scripts (entropy analysis, molecular-weight analysis,
    tanimoto-vs-confidence, violin plots, composite figure, match-count comparison).
  - `correlation_analysis.py`, `gemni_corr_analysis.py` — correlation analyses.
  - `analysis.ipynb` — consolidated notebook.
  - `outputs/` — generated figures (`figures/`) and stats reports (`reports/`).

## GNPS2 task IDs referenced

| Task ID | Used in |
|---|---|
| `31bb8489077242c69e3502e45f4061e2` | `MSMLS_lib/scripts/eval_ground_truth.py` |
| `384d5ca63dae452b95c762743c78dc31` | `MSMLS_lib/scripts/{confusion_matrix,cosine_vs_app,match_count_report*,mol_mass_report,param_correlation_matrix}.py` |
| `b22ff1bc41624e29a28cf6e12fe17ad5` | `plusrise_dataset/` scripts and notebook |

## Reproducing the analyses

Most scripts in `MSMLS_lib/scripts/` and `plusrise_dataset/scripts/` expect the
GNPS2-AnnEx scoring library (`bin/normalizer.py`, `bin/scorer.py`, `bin/tanimoto.py`,
etc.) to be importable — i.e. run from the root of the main
[GNPS2-AnnEx](https://github.com/wilhan-nunes/GNPS2-AnnEx) repository with this
folder's contents merged in (or this repo added to `PYTHONPATH`), for example:

```bash
pip install -r requirements.txt   # from the main GNPS2-AnnEx repo
python MSMLS_lib/scripts/eval_ground_truth.py
```

## Data availability note

Large intermediate/raw data (cached `.mgf` spectra, full `taskresult` TSVs, and the
merged GNPS/standards processing tables under `MSMLS_lib/data/processing/`) are
**not** included here due to size. These are regenerable by re-running the loaders
in the main GNPS2-AnnEx repo against the GNPS2 task IDs above, or are available on
request. Only the small ground-truth standards table
(`MSMLS_lib/data/ground_truth/validated_with_GNPS_by_LibraryName.csv`) is included
directly.
