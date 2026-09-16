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
  library (Sigma-Aldrich MSMLS, 185 positive-mode standards). The active analysis
  task is `fe83e1b8c6a0494f85eaf6933f6d4bdf` — the GNPS2 FBMN job cited in the
  manuscript, filtered to the 185 annotated standards (`analysis.ipynb`'s `TASK_ID`,
  and `scripts/eval_ground_truth.py`). Task `384d5ca63dae452b95c762743c78dc31` is
  the *complete, unfiltered* MSMLS dataset that `fe83e1b8...` was derived from; it
  is kept only for provenance and is what the remaining per-script analyses
  (confusion matrix, cosine-vs-app, match-count reports, mol-mass report, parameter
  correlation matrix) were originally run against.
  - `scripts/` — figure- and report-generating scripts (confusion matrix, cosine-vs-app
    comparison, match-count reports, mirror plots, molecular-mass report, parameter
    correlation matrix, ground-truth evaluation).
  - `analysis.ipynb` — consolidated notebook reproducing the analyses in `scripts/`.
  - `data/ground_truth/` — the validated standards table
    (`validated_with_GNPS_by_LibraryName.csv`).
  - `params/MSMLS_mzmine4_feature_finding.mzbatch` — the MZmine4 batch file used for
    feature finding on the raw MSMLS `.mzML` files prior to FBMN/library search.
  - `outputs/` — generated figures (`new_figures/`) and evaluation reports (`reports/`,
    currently empty — regenerate via `scripts/eval_ground_truth.py` against
    `fe83e1b8c6a0494f85eaf6933f6d4bdf`).
- **`plusrise_dataset/`** — confidence-score analysis on the PlusRise dataset
  (GNPS2 task `b22ff1bc41624e29a28cf6e12fe17ad5`).
  - `scripts/` — figure scripts (entropy analysis, molecular-weight analysis,
    tanimoto-vs-confidence, violin plots, composite figure, match-count comparison).
  - `correlation_analysis.py`, `gemni_corr_analysis.py` — correlation analyses.
  - `analysis.ipynb` — consolidated notebook.
  - `outputs/` — generated figures (`figures/`) and stats reports (`reports/`).

## GNPS2 task IDs referenced

| Task ID | Role | Used in |
|---|---|---|
| `fe83e1b8c6a0494f85eaf6933f6d4bdf` | Active — filtered to the 185 annotated MSMLS standards; the ID cited in the manuscript | `MSMLS_lib/analysis.ipynb` (`TASK_ID`), `MSMLS_lib/scripts/eval_ground_truth.py` |
| `b22ff1bc41624e29a28cf6e12fe17ad5` | Active — PlusRise urine dataset | `plusrise_dataset/` scripts and notebook |
| `384d5ca63dae452b95c762743c78dc31` | Provenance only — complete, unfiltered MSMLS dataset that `fe83e1b8...` was filtered from | `MSMLS_lib/scripts/{confusion_matrix,cosine_vs_app,match_count_report*,mol_mass_report,param_correlation_matrix}.py` |

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

The raw MSMLS LC-MS/MS data is deposited in MassIVE under accession
`MSV000102790`. `.mzML` conversion was done with MSConvert, and feature finding
was performed with MZmine4 using the batch parameters in
`MSMLS_lib/params/MSMLS_mzmine4_feature_finding.mzbatch`, producing the `.mgf`/`.csv`
files that were filtered to the 185 positive-mode standards and submitted as the
FBMN job (`fe83e1b8c6a0494f85eaf6933f6d4bdf`).
