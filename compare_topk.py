"""Compare confidence scores between topK=10 and topK=100 result files."""
import sys
import warnings
import logging

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from bin.normalizer import _normalize_library_matches_dataframe
from bin.scorer import _compute_scan_level_confidence

TASK_10 = "b22ff1bc41624e29a28cf6e12fe17ad5"
TASK_100 = "fb5d050524a947dd95fc99dadba37013"

print("Loading and scoring topK=10 ...", flush=True)
df10_raw = pd.read_csv("b22ff1bc41624e29a28cf6e12fe17ad5_result.csv")
df10 = _normalize_library_matches_dataframe(df10_raw)
s10 = _compute_scan_level_confidence(df10, task_id=TASK_10, use_entropy_similarity=False)
print(f"  topK=10: {len(s10)} scans scored")

print("Loading and scoring topK=100 ...", flush=True)
df100_raw = pd.read_csv(
    "fb5d050524a947dd95fc99dadba37013-merged_results_with_gnps.tsv", sep="\t"
)
df100 = _normalize_library_matches_dataframe(df100_raw)
s100 = _compute_scan_level_confidence(df100, task_id=TASK_100, use_entropy_similarity=False)
print(f"  topK=100: {len(s100)} scans scored")

# Save individual results
s10.to_csv("scores_topk10.csv", index=False)
s100.to_csv("scores_topk100.csv", index=False)

# Merge on scan (inner join — only scans present in both)
s10["scan"] = s10["scan"].astype(str)
s100["scan"] = s100["scan"].astype(str)

merged = s10.merge(
    s100,
    on="scan",
    suffixes=("_10", "_100"),
    how="inner",
)
print(f"\n  Shared scans: {len(merged)}")

merged["score_delta"] = merged["confidence_score_100"] - merged["confidence_score_10"]
merged["total_matches_10"] = merged["total_matches_10"].astype(int)
merged["total_matches_100"] = merged["total_matches_100"].astype(int)


def cat(score):
    if score >= 80:
        return "Consistent"
    if score >= 55:
        return "Inconclusive"
    return "Inconsistent"


merged["cat_10"] = merged["confidence_score_10"].apply(cat)
merged["cat_100"] = merged["confidence_score_100"].apply(cat)

merged.to_csv("scores_comparison.csv", index=False)

# ── Summary stats ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONFIDENCE SCORE SUMMARY")
print("=" * 60)

for label, s in [("topK=10", s10), ("topK=100", s100)]:
    print(f"\n  {label}  ({len(s)} scans)")
    print(f"    Mean score  : {s['confidence_score'].mean():.1f}")
    print(f"    Median score: {s['confidence_score'].median():.1f}")
    print(f"    Std dev     : {s['confidence_score'].std():.1f}")
    cats = s["confidence_label"].value_counts()
    for c, n in cats.items():
        print(f"    {c:<25}: {n:>4}  ({100*n/len(s):.1f}%)")

print(f"\n  Delta (100 − 10) across {len(merged)} shared scans")
print(f"    Mean  Δ: {merged['score_delta'].mean():+.1f}")
print(f"    Median Δ: {merged['score_delta'].median():+.1f}")
print(f"    Std Δ  : {merged['score_delta'].std():.1f}")
print(f"    Scans improved  (Δ>0) : {(merged['score_delta'] > 0).sum()}")
print(f"    Scans unchanged (Δ=0) : {(merged['score_delta'] == 0).sum()}")
print(f"    Scans degraded  (Δ<0) : {(merged['score_delta'] < 0).sum()}")

# Category transitions
print("\n  Category transitions (topK=10 → topK=100):")
trans = merged.groupby(["cat_10", "cat_100"]).size().reset_index(name="n")
order = ["Consistent", "Inconclusive", "Inconsistent"]
for _, row in trans.iterrows():
    arrow = "→"
    print(f"    {row['cat_10']:<15} {arrow} {row['cat_100']:<15}: {int(row['n']):>4}")

# Component-level deltas
print("\n  Component score deltas (mean, 100 − 10):")
for col in ["structure_agreement", "tanimoto_score", "mean_cosine", "max_cosine", "shared_peaks_score", "mz_precision"]:
    c10 = col + "_10" if col + "_10" in merged.columns else col
    c100 = col + "_100" if col + "_100" in merged.columns else col
    if c10 in merged.columns and c100 in merged.columns:
        delta = (merged[c100] - merged[c10]).mean()
        print(f"    {col:<28}: {delta:+.4f}")

# Structure agreement breakdown
if "support_fraction_10" in merged.columns:
    sa_col_10, sa_col_100 = "support_fraction_10", "support_fraction_100"
else:
    sa_col_10 = "supporting_matches_10"
    sa_col_100 = "supporting_matches_100"

print("\n  Matches per scan (topK=10 vs topK=100):")
print(f"    Mean total_matches_10 : {merged['total_matches_10'].mean():.1f}")
print(f"    Mean total_matches_100: {merged['total_matches_100'].mean():.1f}")

# Top improved / degraded scans
print("\n  Top 10 most improved scans (Δ):")
top_imp = merged.nlargest(10, "score_delta")[
    ["scan", "confidence_score_10", "confidence_score_100", "score_delta",
     "top_compound_10", "total_matches_10", "total_matches_100"]
]
print(top_imp.to_string(index=False))

print("\n  Top 10 most degraded scans (Δ):")
top_deg = merged.nsmallest(10, "score_delta")[
    ["scan", "confidence_score_10", "confidence_score_100", "score_delta",
     "top_compound_10", "total_matches_10", "total_matches_100"]
]
print(top_deg.to_string(index=False))

print("\nDone. Files written: scores_topk10.csv, scores_topk100.csv, scores_comparison.csv")
