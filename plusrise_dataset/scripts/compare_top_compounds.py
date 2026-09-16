"""
Compare top_compound vs top_cosine_compound in the annex summary CSV.

Usage:
    python compare_top_compounds.py

Outputs:
    compound_comparison_report.txt  (same directory as this script)
"""

import re
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ANNEX_CSV = HERE / "annex_summary_b22ff1bc4162 (6).csv"
RAW_CSV = HERE.parents[1] / "b22ff1bc41624e29a28cf6e12fe17ad5_result.csv"
OUT_TXT = HERE / "compound_comparison_report.txt"


def normalize_name(name: str) -> str:
    """Return a canonical lowercase form of a compound name for comparison."""
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    # strip trailing trade-name/source parentheticals, e.g. "(Seroquel)" or "(NIST14)"
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    # strip leading "spectral match to " prefix
    name = re.sub(r"^spectral match to\s+", "", name)
    # strip collision-energy annotations: "_CE25,38,59" or " - 40.0 eV"
    name = re.sub(r"[\s_-]*ce\s*[\d.,]+.*$", "", name)
    name = re.sub(r"\s*-\s*[\d.]+\s*ev\s*$", "", name)
    # normalize dashes and underscores
    name = name.replace("–", "-").replace("—", "-").replace("_", " ")
    return re.sub(r"\s+", " ", name).strip()


def main():
    df = pd.read_csv(ANNEX_CSV)
    raw = pd.read_csv(RAW_CSV)

    df["top_norm"] = df["top_compound"].apply(normalize_name)
    df["cosine_norm"] = df["top_cosine_compound"].apply(normalize_name)
    df["names_differ"] = df["top_norm"] != df["cosine_norm"]

    # Best cosine hit per scan from the raw result file
    raw_top = (
        raw.sort_values("MQScore", ascending=False)
        .groupby("#Scan#")
        .first()
        .reset_index()[["#Scan#", "Compound_Name", "MQScore", "InChIKey"]]
        .rename(columns={
            "#Scan#": "scan",
            "Compound_Name": "raw_best_compound",
            "MQScore": "raw_best_mqscore",
            "InChIKey": "raw_best_inchikey",
        })
    )
    df = df.merge(raw_top, on="scan", how="left")

    # Genuine overrides: flag is True AND normalized names differ
    true_overrides = df[df["top_cosine_overridden"] & df["names_differ"]].copy()
    # Synonym / salt cases: flag is True but names are the same after normalization
    synonym_only = df[df["top_cosine_overridden"] & ~df["names_differ"]].copy()
    # Not overridden but different DB label for the same structure
    not_overridden_diff = df[~df["top_cosine_overridden"] & df["names_differ"]]

    label_counts = true_overrides["confidence_label"].value_counts()
    detail_cols = [
        "scan", "precursor_mz", "confidence_score", "confidence_label",
        "top_compound", "top_cosine_compound", "max_cosine",
        "supporting_matches", "total_matches",
    ]
    true_overrides_detail = true_overrides[detail_cols].sort_values(
        "confidence_score", ascending=False
    )

    lines = []
    lines.append("=" * 80)
    lines.append("TOP-COMPOUND vs TOP-COSINE-COMPOUND COMPARISON REPORT")
    lines.append(f"Task: b22ff1bc41624e29a28cf6e12fe17ad5")
    lines.append(f"Source: {ANNEX_CSV.name}")
    lines.append(f"Generated: {date.today().isoformat()}")
    lines.append("=" * 80)
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total annotated scans:                    {len(df):>6}")
    lines.append(f"Scans where top_cosine_overridden = True: {int(df['top_cosine_overridden'].sum()):>6}")
    lines.append(f"  - Genuinely different compound")
    lines.append(f"    (names differ after normalization):    {len(true_overrides):>6}")
    lines.append(f"  - Same compound, different label/salt:  {len(synonym_only):>6}")
    lines.append(f"Scans where override = False but names")
    lines.append(f"  differ (same structure, alt. DB name):  {len(not_overridden_diff):>6}")
    lines.append("")
    lines.append(
        f"  Override rate (genuine):  "
        f"{100 * len(true_overrides) / len(df):.1f}% of all annotated scans"
    )
    lines.append("")

    lines.append("CONFIDENCE LABEL BREAKDOWN (genuine overrides only)")
    lines.append("-" * 40)
    for label, count in label_counts.items():
        pct = 100 * count / len(true_overrides)
        lines.append(f"  {label:<28} {count:>4}  ({pct:.1f}%)")
    lines.append("")

    lines.append("CONFIDENCE SCORE STATS (genuine overrides)")
    lines.append("-" * 40)
    stats = true_overrides["confidence_score"]
    lines.append(f"  Mean:    {stats.mean():.1f}")
    lines.append(f"  Median:  {stats.median():.1f}")
    lines.append(f"  Std:     {stats.std():.1f}")
    lines.append(f"  Min:     {stats.min():.1f}")
    lines.append(f"  Max:     {stats.max():.1f}")
    lines.append("")

    lines.append("SYNONYM / SALT CASES (overridden=True, same normalized name)")
    lines.append("-" * 40)
    for _, row in synonym_only.iterrows():
        lines.append(
            f"  Scan {row['scan']}: '{row['top_compound']}' vs '{row['top_cosine_compound']}'"
        )
    lines.append("")

    lines.append("DETAILED TABLE — GENUINE OVERRIDES (sorted by confidence score, desc)")
    lines.append("-" * 80)
    header = (
        f"{'Scan':>7}  {'MZ':>8}  {'Score':>6}  {'Label':<22}  "
        f"{'MaxCosine':>9}  {'Hits':>4}  {'TopCompound':<35}  TopCosineCompound"
    )
    lines.append(header)
    lines.append("-" * 80)
    for _, row in true_overrides_detail.iterrows():
        tc = str(row["top_compound"])[:35]
        tcc = str(row["top_cosine_compound"])[:45]
        lbl = str(row["confidence_label"])[:22]
        lines.append(
            f"{int(row['scan']):>7}  {row['precursor_mz']:>8.3f}  {row['confidence_score']:>6.1f}  "
            f"{lbl:<22}  {row['max_cosine']:>9.4f}  "
            f"{int(row['total_matches']):>4}  {tc:<35}  {tcc}"
        )
    lines.append("")

    lines.append("NOTES ON NORMALIZATION")
    lines.append("-" * 40)
    lines.append("  Names were normalized before comparison:")
    lines.append("  1. Lowercased and Unicode-normalized (NFKD)")
    lines.append("  2. Trailing parentheticals stripped (trade names, sources)")
    lines.append("  3. Leading 'Spectral match to' prefixes removed")
    lines.append("  4. CE (collision energy) annotations stripped (_CE25, - 40.0 eV, etc.)")
    lines.append("  5. Underscores replaced with spaces; whitespace collapsed")
    lines.append("  The 'top_cosine_overridden' flag in the CSV was also consulted directly.")
    lines.append("")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    OUT_TXT.write_text(report_text)
    print(f"Report saved to: {OUT_TXT}")


if __name__ == "__main__":
    main()
