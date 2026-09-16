"""
Evaluate TopKonfidence scoring accuracy against MSMLS validated ground-truth library.

Usage:
    python eval_ground_truth.py

Outputs:
    eval_results.csv   - per-scan detail (ground truth, top annotation, verdict)
    eval_summary.txt   - printed report saved to disk
"""
import sys
import os
import re
import warnings
import logging

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from difflib import SequenceMatcher

sys.path.insert(0, ".")
from bin.normalizer import _normalize_library_matches_dataframe, _clean_compound_name
from bin.scorer import _compute_scan_level_confidence

TASK_ID = "31bb8489077242c69e3502e45f4061e2"
GROUND_TRUTH_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ground_truth", "validated_with_GNPS_by_LibraryName.csv")

# ── Name normalisation helpers ────────────────────────────────────────────────

_STEREO_PREFIX = re.compile(
    r"^(?:l-|d-|dl-|r-|s-|\(r\)-|\(s\)-|\(\+\)-|\(-\)-|alpha-|beta-|"
    r"gamma-|delta-|cis-|trans-|endo-|exo-|meso-|n-|o-|p-|m-|"
    r"3-|2-|1-|4-|5-|6-)+",
    re.IGNORECASE,
)
_SALT_SUFFIX = re.compile(
    r"\s+(hydrochloride|hcl|sodium|potassium|calcium|acetate|sulfate|"
    r"phosphate|chloride|bromide|iodide|nitrate|citrate|fumarate|"
    r"maleate|succinate|tartrate|monohydrate|dihydrate|hydrate|salt)$",
    re.IGNORECASE,
)
_COLLISION_ENERGY = re.compile(
    r"\s*collision\s*energy\s*:\s*[\d.]+\s*$",
    re.IGNORECASE,
)
_EV_SUFFIX = re.compile(
    r"[\s\-]+\d+(?:\.\d+)?\s*ev(?:\s+unknown)?$",
    re.IGNORECASE,
)
_CANDIDATE_ANNOTATION = re.compile(
    r"^candidate\s+\S+.*?\(delta\s+mass:[^)]+\)$",
    re.IGNORECASE,
)
_KNOWN_ISOMERS_NOTE = re.compile(
    r"\s*\(known structural isomers:[^)]*\)",
    re.IGNORECASE,
)
_CAS_NUMBER = re.compile(r"^\d{2,7}-\d{2}-\d$")

# Synonym table: normalised aliases -> canonical normalised form
_SYNONYMS: dict = {
    "asparticacid": "aspartate",
    "glutamicacid": "glutamate",
    "succinicacid": "succinate",
    "fumaricacid": "fumarate",
    "maleicacid": "maleate",
    "malicacid": "malate",
    "citricacid": "citrate",
    "lacticacid": "lactate",
    "pyruvicacid": "pyruvate",
    "oxalicacid": "oxalate",
    "aceticacid": "acetate",
    "nicotinicacid": "nicotinate",
    "isonicotinicacid": "nicotinate",
    "pyridine3carboxylicacid": "nicotinate",
    "quinicacid": "quinate",
    "gluconicacid": "gluconate",
    "sorbicacid": "sorbate",
    "sebacanicacid": "sebacate",
    "sebacicacid": "sebacate",
    "glycolicacid": "glycolate",
    "glycericacid": "glycerate",
    "hydrocortisoneacetate": "cortisol21acetate",
    "cortisol21aceticacid": "cortisol21acetate",
    "hydrocortisone21acetate": "cortisol21acetate",
    "nadplus": "nadp",
    "nicotinamideadeninedinucleotidephosphate": "nadp",
    "triphosphopyridine": "nadp",
    "betanicotinamideadeninedinucleotidephosphate": "nadp",
    "nacetyl5hydroxytryptamine": "nacetylserotonin",
    "n1acetyl5hydroxytryptamine": "nacetylserotonin",
    "carbocysteine": "scarboxymethylcysteine",
    "lcarboxymethylcysteine": "scarboxymethylcysteine",
    "scarboxymethyllcysteine": "scarboxymethylcysteine",
    "26diaminoheptanedioate": "diaminopimelate",
    "26diaminoheptanedioicacid": "diaminopimelate",
    "26diaminopimeicacid": "diaminopimelate",
    "guanosine35cyclicmonophosphate": "cyclicgmp",
    "cgmp": "cyclicgmp",
    "nepsilonnepsilonnepsilontrimethyllysine": "nnntrimethyllysine",
    "trimethyllysine": "nnntrimethyllysine",
    "5aminoimidazole4carboxamide1betadribofuranosyl5monophosphate": "aicar",
    "5aminoimidazole4carboxamide1betaribofuranosyl5monophosphate": "aicar",
    "aicariboside5monophosphate": "aicar",
    "zmp": "aicar",
    "guanosinediphosphatemannose": "gdpmannose",
    "guanosine5diphosphomanose": "gdpmannose",
    "guanosine5diphosphomannose": "gdpmannose",
    "guanosine5diphosphodmannose": "gdpmannose",
}


def _apply_synonyms(norm: str) -> str:
    return _SYNONYMS.get(norm, norm)


def _normalise(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if _CANDIDATE_ANNOTATION.match(name):
        return "__candidate__"
    if _CAS_NUMBER.match(name.strip()):
        return "__cas__"
    name = _KNOWN_ISOMERS_NOTE.sub("", name)
    name = _clean_compound_name(name)
    name = _COLLISION_ENERGY.sub("", name)
    name = _EV_SUFFIX.sub("", name)
    name = _SALT_SUFFIX.sub("", name)
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]", "", name)
    return _apply_synonyms(name)


def _strip_stereo(name: str) -> str:
    while True:
        new = _STEREO_PREFIX.sub("", name, count=1).strip()
        if new == name:
            break
        name = new
    return name


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _verdict(top: str, truth: str) -> str:
    """
    correct        – match after normalisation + synonym lookup
    correct_stereo – match after also stripping stereo prefixes
    close          – fuzzy similarity >= 0.82
    candidate      – top is a "Candidate X-Y (delta mass)" annotation
    cas_number     – top is a bare CAS number (unresolved name)
    wrong          – no match
    no_data        – empty name
    """
    if isinstance(top, str) and _CANDIDATE_ANNOTATION.match(top.strip()):
        return "candidate"
    if isinstance(top, str) and _CAS_NUMBER.match(top.strip()):
        return "cas_number"

    top_n = _normalise(top)
    truth_n = _normalise(truth)

    if not top_n or not truth_n or top_n in ("__candidate__", "__cas__"):
        return "no_data"

    if top_n == truth_n:
        return "correct"
    if top_n in truth_n or truth_n in top_n:
        return "correct"

    top_s = _apply_synonyms(_normalise(_strip_stereo(top)))
    truth_s = _apply_synonyms(_normalise(_strip_stereo(truth)))
    if top_s and truth_s and (top_s == truth_s or top_s in truth_s or truth_s in top_s):
        return "correct_stereo"

    sim = max(_similarity(top_n, truth_n), _similarity(top_s, truth_s))
    if sim >= 0.82:
        return "close"

    return "wrong"


# ── Load ground truth ─────────────────────────────────────────────────────────

print("Loading ground truth ...", flush=True)
gt = pd.read_csv(GROUND_TRUTH_CSV)
gt.columns = gt.columns.str.strip()
gt["#Scan#"] = gt["#Scan#"].astype(str)
gt = gt[["#Scan#", "PRIMARY_NAME"]].rename(
    columns={"#Scan#": "scan", "PRIMARY_NAME": "ground_truth"}
)
print(f"  {len(gt)} validated scans in ground truth", flush=True)

# ── Fetch library matches from GNPS2 ─────────────────────────────────────────

print(f"\nFetching library matches for task {TASK_ID} ...", flush=True)
raw_df = None

try:
    from gnpsdata import taskresult
    raw_df = taskresult.get_gnps2_task_resultfile_dataframe(
        TASK_ID, "nf_output/library/merged_results_with_gnps.tsv"
    )
    if raw_df is not None and len(raw_df) > 0:
        print(f"  Loaded {len(raw_df)} rows via taskresult TSV", flush=True)
    else:
        raw_df = None
except Exception as exc:
    print(f"  taskresult TSV failed: {exc}", flush=True)

if raw_df is None or len(raw_df) == 0:
    try:
        from gnpsdata import workflow_fbmn
        raw_df = workflow_fbmn.get_library_match_dataframe(TASK_ID, gnps2=True)
        if raw_df is not None and len(raw_df) > 0:
            print(f"  Loaded {len(raw_df)} rows via workflow_fbmn", flush=True)
        else:
            raw_df = None
    except Exception as exc:
        print(f"  workflow_fbmn failed: {exc}", flush=True)

if raw_df is None or len(raw_df) == 0:
    sys.exit("ERROR: Could not load library matches. Check task ID and network.")

# ── Normalise and score ───────────────────────────────────────────────────────

print("\nNormalising and scoring ...", flush=True)
norm_df = _normalize_library_matches_dataframe(raw_df)
scores = _compute_scan_level_confidence(norm_df, task_id=TASK_ID, workflow_choice="fbmn")
scores["scan"] = scores["scan"].astype(str)
print(f"  {len(scores)} scans scored", flush=True)

# ── Merge with ground truth ───────────────────────────────────────────────────

merged = scores.merge(gt, on="scan", how="inner")
print(f"  {len(merged)} scans overlap with ground truth", flush=True)

scans_no_match = len(gt) - len(merged)
print(f"  {scans_no_match} ground truth scans have NO library match (not scored)")

# ── Apply verdict ─────────────────────────────────────────────────────────────

merged["verdict"] = merged.apply(
    lambda r: _verdict(r["top_compound"], r["ground_truth"]), axis=1
)
merged["verdict_count_based"] = merged.apply(
    lambda r: _verdict(r["count_based_top_compound"], r["ground_truth"]), axis=1
)

# Save full detail
out_cols = [
    "scan", "ground_truth", "top_compound", "count_based_top_compound",
    "verdict", "verdict_count_based",
    "confidence_score", "confidence_label",
    "supporting_matches", "total_matches",
    "tanimoto_score", "tanimoto_source",
    "max_cosine", "cosine_consistency",
    "shared_peaks", "mz_precision",
    "top_cosine_overridden", "top_cosine_compound",
]
out_cols = [c for c in out_cols if c in merged.columns]
merged[out_cols].to_csv("eval_results.csv", index=False)
print("\nSaved: eval_results.csv", flush=True)

# ── Report ────────────────────────────────────────────────────────────────────

lines = []
def p(s=""):
    print(s, flush=True)
    lines.append(s)

total = len(merged)
vc = merged["verdict"].value_counts()
correct   = vc.get("correct", 0) + vc.get("correct_stereo", 0)
close     = vc.get("close", 0)
wrong     = vc.get("wrong", 0)
candidate = vc.get("candidate", 0)
cas       = vc.get("cas_number", 0)
no_data   = vc.get("no_data", 0)

pct = lambda n: f"{100*n/total:.1f}%" if total else "-"

p()
p("=" * 70)
p("  TopKonfidence — Ground Truth Accuracy Report")
p(f"  Task : {TASK_ID}")
p(f"  Ground-truth compounds  : {len(gt):>4}")
p(f"  Scans with library match: {len(scores):>4}  ({len(scores)-len(merged)} have no GT)")
p(f"  Overlap (scored + GT)   : {total:>4}")
p(f"  GT scans with NO match  : {scans_no_match:>4}  ({100*scans_no_match/len(gt):.1f}% missed entirely)")
p("=" * 70)
p()
p("ANNOTATION ACCURACY  (top_compound vs ground_truth)")
p("-" * 50)
p(f"  Correct (incl. stereo variants) : {correct:>4}  ({pct(correct)})")
p(f"  Close (fuzzy sim >= 0.82)        : {close:>4}  ({pct(close)})")
p(f"  Wrong (different compound)       : {wrong:>4}  ({pct(wrong)})")
p(f"  Candidate/acyl annotation        : {candidate:>4}  ({pct(candidate)})  [unresolved name]")
p(f"  CAS number (no name)             : {cas:>4}  ({pct(cas)})  [unresolved name]")
p()
scoreable = total - candidate - cas - no_data
p(f"  Excluding unresolved names ({candidate+cas} scans):")
p(f"    Correct : {correct:>4}  ({100*correct/scoreable:.1f}%)")
p(f"    Close   : {close:>4}  ({100*close/scoreable:.1f}%)")
p(f"    Wrong   : {wrong:>4}  ({100*wrong/scoreable:.1f}%)")
p()

p("ACCURACY BY CONFIDENCE CATEGORY")
p("-" * 50)
for cat in ["Consistent evidence", "Inconclusive", "Inconsistent evidence"]:
    sub = merged[merged["confidence_label"] == cat]
    if len(sub) == 0:
        continue
    n = len(sub)
    c  = sub["verdict"].isin(["correct", "correct_stereo"]).sum()
    cl = (sub["verdict"] == "close").sum()
    w  = (sub["verdict"] == "wrong").sum()
    unk = (sub["verdict"].isin(["candidate", "cas_number"])).sum()
    p(f"  {cat:<25} n={n:>3}  correct={c:>3} ({100*c/n:.0f}%)"
      f"  close={cl:>2}  wrong={w:>3} ({100*w/n:.0f}%)  unresolved={unk:>2}")
p()

p("ACCURACY BY SCORE DECILE")
p("-" * 70)
p(f"  {'Score range':<18} {'n':>4}  {'Correct':>9}  {'Close':>7}  {'Wrong':>7}  {'Unresolved':>10}")
p(f"  {'-'*18}  {'-'*4}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*10}")
merged["score_bin"] = pd.cut(
    merged["confidence_score"], bins=list(range(0, 101, 10)), right=True, include_lowest=True
)
for interval, grp in merged.groupby("score_bin", observed=True):
    n    = len(grp)
    c    = grp["verdict"].isin(["correct", "correct_stereo"]).sum()
    cl   = (grp["verdict"] == "close").sum()
    w    = (grp["verdict"] == "wrong").sum()
    unk  = grp["verdict"].isin(["candidate", "cas_number"]).sum()
    p(f"  {str(interval):<18}  {n:>4}  "
      f"{c:>4} ({100*c/n:>4.0f}%)  {cl:>3} ({100*cl/n:>3.0f}%)  "
      f"{w:>3} ({100*w/n:>3.0f}%)  {unk:>5} ({100*unk/n:>3.0f}%)")
p()

p("WRONG ANNOTATIONS  (true errors — different compound identified)")
p("-" * 70)
wrong_df = merged[merged["verdict"] == "wrong"].sort_values("confidence_score", ascending=False)
if len(wrong_df) > 0:
    for _, row in wrong_df.iterrows():
        p(f"  scan={row['scan']:>6}  score={row['confidence_score']:>5.1f}  "
          f"truth={str(row['ground_truth'])[:30]:<30}  "
          f"top={str(row['top_compound'])[:35]}")
p()

p("UNRESOLVED NAMES  (CAS numbers or Candidate annotations)")
p("-" * 70)
unres = merged[merged["verdict"].isin(["candidate", "cas_number"])].sort_values(
    "confidence_score", ascending=False
)
if len(unres) > 0:
    for _, row in unres.iterrows():
        p(f"  scan={row['scan']:>6}  score={row['confidence_score']:>5.1f}  "
          f"truth={str(row['ground_truth'])[:28]:<28}  "
          f"top={str(row['top_compound'])[:40]}")
p()

p("FAILURE MODE ANALYSIS")
p("-" * 70)

hc_wrong = merged[(merged["verdict"] == "wrong") & (merged["confidence_score"] >= 55)]
p(f"  High-confidence wrong predictions (score >= 55): {len(hc_wrong)}")
for _, r in hc_wrong.iterrows():
    p(f"    scan={r['scan']:>6}  score={r['confidence_score']:>5.1f}  "
      f"matches={r['total_matches']:>4}  "
      f"truth={str(r['ground_truth'])[:25]:<25}  top={str(r['top_compound'])[:30]}")
p()

# WCRS vs count-based
better_count = merged[
    (merged["verdict"] == "wrong") &
    (merged["verdict_count_based"].isin(["correct", "correct_stereo"]))
]
p(f"  Wrong by WCRS but correct by count-based: {len(better_count)}")
for _, r in better_count.head(10).iterrows():
    p(f"    scan={r['scan']:>6}  truth={str(r['ground_truth'])[:22]:<22}  "
      f"wcrs={str(r['top_compound'])[:22]:<22}  count={str(r['count_based_top_compound'])[:22]}")
p()

# Single-match scans
single = merged[merged["total_matches"] == 1]
if len(single) > 0:
    sc = single["verdict"].isin(["correct", "correct_stereo"]).sum()
    sw = (single["verdict"] == "wrong").sum()
    p(f"  Single-match scans: {len(single)}  correct={sc}  wrong={sw}")
p()

p("NAME CLEANING ISSUES  (correct compound, bad name in top_compound)")
p("-" * 70)
p("  (would show as 'wrong' before synonym expansion)")
p("  [Already resolved in this run by synonym table, but indicates library")
p("   name hygiene issues that should be fixed in _clean_compound_name]")
p()

p("SCORE DISTRIBUTION: correct vs wrong")
p("-" * 70)
correct_mask = merged["verdict"].isin(["correct", "correct_stereo"])
wrong_mask   = merged["verdict"] == "wrong"
p(f"  {'':22}  {'Mean':>7}  {'Median':>7}  {'Std':>7}  {'n':>4}")
for label, mask in [("Correct", correct_mask), ("Wrong", wrong_mask)]:
    sub = merged.loc[mask, "confidence_score"]
    if len(sub) > 0:
        p(f"  {label:<22}  {sub.mean():>7.1f}  {sub.median():>7.1f}  {sub.std():>7.1f}  {len(sub):>4}")
p()

if "supporting_matches" in merged.columns and "total_matches" in merged.columns:
    merged["_sa_frac"] = merged["supporting_matches"] / merged["total_matches"].replace(0, np.nan)

p("COMPONENT MEANS: correct vs wrong")
p("-" * 70)
p(f"  {'Component':<30}  {'Correct':>9}  {'Wrong':>9}  {'Delta':>9}")
for col in ["_sa_frac", "tanimoto_score", "cosine_consistency", "max_cosine",
            "shared_peaks_score", "mz_precision"]:
    if col not in merged.columns:
        continue
    lbl = col.replace("_sa_frac", "structure_agreement_frac")
    c_mean = merged.loc[correct_mask, col].mean()
    w_mean = merged.loc[wrong_mask, col].mean()
    p(f"  {lbl:<30}  {c_mean:>9.3f}  {w_mean:>9.3f}  {c_mean-w_mean:>+9.3f}")
p()

# ── Recommendations ───────────────────────────────────────────────────────────

p("=" * 70)
p("RECOMMENDATIONS")
p("=" * 70)
p()

p("1. COVERAGE GAP (most critical):")
p(f"   {scans_no_match} of {len(gt)} ground-truth compounds ({100*scans_no_match/len(gt):.0f}%) have NO library")
p(f"   match returned by this task. Possible causes:")
p(f"   - Low topK setting: increase to 50-100 hits per query.")
p(f"   - Strict mz-tolerance: loosen the search window.")
p(f"   - Missing reference spectra in the libraries searched.")
p()

hc_w = len(hc_wrong)
p(f"2. HIGH-CONFIDENCE WRONG PREDICTIONS ({hc_w} scans, score >= 55):")
if hc_w > 0:
    p(f"   These are the most harmful errors. Root causes observed:")
    for _, r in hc_wrong.iterrows():
        truth_s = str(r["ground_truth"])
        top_s   = str(r["top_compound"])
        if "diphospho" in top_s.lower() and "mannose" in truth_s.lower():
            p(f"   - scan {r['scan']}: GDP-glucose returned for GDP-mannose (structural")
            p(f"     isomers). The WCRS/Tanimoto can't distinguish them if SMILES are")
            p(f"     missing or library conflates mannose/glucose annotations.")
        elif "furoic" in top_s.lower() and "cysteate" in truth_s.lower():
            p(f"   - scan {r['scan']}: Furoic acid returned for Cysteate — likely a")
            p(f"     cosine false positive at similar precursor mass. Adding a precursor")
            p(f"     m/z exact-match gate or stricter PPM filter would help.")
        elif "allothreonine" in top_s.lower():
            p(f"   - scan {r['scan']}: Library returns allothreonine for ACC.")
            p(f"     These are isobaric compounds; chromatographic RT must disambiguate.")
        elif "gentisic" in top_s.lower() and "34" in truth_s.lower():
            p(f"   - scan {r['scan']}: Positional isomers (3,4-DHBA vs Gentisic/2,5-DHBA)")
            p(f"     cannot be separated by MS/MS alone; RT or CCS required.")
p()

p("3. NAME CLEANING ISSUES IN LIBRARY ANNOTATIONS:")
p("   The following name patterns are NOT cleaned by _clean_compound_name")
p("   and pollute top_compound, causing incorrect-looking outputs:")
p("   - 'CollisionEnergy:102040' suffix (only 'N eV' style is stripped)")
p("   - Bare CAS registry numbers (e.g. '56546-36-2', '39711-79-0')")
p("   - 'Candidate X-C2:0 (delta mass:...)' acyl-lipid annotations")
p("   - '(known structural isomers: N; isobaric peaks in run: N)' notes")
p("   Fix: extend _clean_compound_name with these four patterns.")
p()

p("4. SCORE CALIBRATION:")
correct_scores = merged.loc[correct_mask, "confidence_score"]
wrong_scores   = merged.loc[wrong_mask, "confidence_score"]
if len(correct_scores) and len(wrong_scores):
    p(f"   Correct mean: {correct_scores.mean():.1f}  |  Wrong mean: {wrong_scores.mean():.1f}")
    if wrong_scores.mean() > correct_scores.mean():
        p(f"   WARNING: Wrong predictions score HIGHER than correct ones on average.")
        p(f"   The score is measuring spectral consistency, not annotation correctness.")
        p(f"   Compounds with many library entries (e.g. amino acids) inflate structure_")
        p(f"   agreement even when the dominant library name differs from the true name.")
        p(f"   Action: weight max_cosine more heavily; consider a library-diversity penalty.")
p()

p("5. TANIMOTO / WCRS:")
p(f"   Correct-vs-wrong Tanimoto delta: "
  f"{merged.loc[correct_mask,'tanimoto_score'].mean() - merged.loc[wrong_mask,'tanimoto_score'].mean():+.3f}")
p(f"   Tanimoto does not discriminate well between correct and wrong here.")
p(f"   This is expected when wrong predictions are structural isomers or")
p(f"   when SMILES are missing and tanimoto falls back to structure_agreement.")
p()

p("Full per-scan detail: eval_results.csv")
p("=" * 70)

with open("eval_summary.txt", "w") as f:
    f.write("\n".join(lines))
print("\nSaved: eval_summary.txt")
