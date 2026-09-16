"""
Compare app (multi-metric) annotation vs top-cosine annotation against ground truth.

Key insight on InChIKey resolution:
  - If top_cosine_overridden=False  → app and cosine agreed on the same compound
    → cosine_ik = structure_key (same structure, no separate lookup needed)
  - If top_cosine_overridden=True   → app overrode the cosine; resolve cosine_ik
    from merged results for that scan (with substring-pipe fallback for Massbank names)
"""

import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROC = os.path.join(BASE, "data", "processing")

GT_FILE     = os.path.join(PROC, "Mise en forme table final v3.tsv")
ANNEX_FILE  = os.path.join(PROC, "annex_summary_384d5ca63dae.csv")
MERGED_FILE = os.path.join(PROC, "384d5ca63dae452b95c762743c78dc31-merged_results_with_gnps.tsv")

OUT_STATIC = os.path.join(BASE, "cosine_vs_app_v2.svg")
OUT_HTML   = os.path.join(BASE, "cosine_vs_app_v2.html")

# ── load ───────────────────────────────────────────────────────────────────────
gt     = pd.read_csv(GT_FILE,     sep="\t")
annex  = pd.read_csv(ANNEX_FILE)
merged = pd.read_csv(MERGED_FILE, sep="\t")

# ── InChIKey resolution helpers ───────────────────────────────────────────────
def _strip_suffix(name):
    return re.sub(r"\s*[\||-]\s*\d+\.?\d*\s*(?:eV|ev)\s*$", "", name, flags=re.I).strip()

def resolve_ik(scan_id, name, merged_df):
    """Resolve InChIKey-Planar for a compound name within a given scan."""
    if pd.isna(name):
        return None
    sub     = merged_df[merged_df["#Scan#"] == scan_id]
    sub_ik  = sub.dropna(subset=["InChIKey-Planar"])

    # 1. Exact / case-insensitive match
    for cand in [name, name.upper()]:
        m = sub_ik[sub_ik["Compound_Name"].str.upper() == cand.upper()]
        if len(m):
            return m.iloc[0]["InChIKey-Planar"]

    # 2. Strip energy suffix then repeat
    base = _strip_suffix(name)
    if base != name:
        m = sub_ik[sub_ik["Compound_Name"].str.upper() == base.upper()]
        if len(m):
            return m.iloc[0]["InChIKey-Planar"]
        m = sub_ik[sub_ik["Compound_Name"].str.upper().str.startswith(base.upper())]
        if len(m):
            return m.iloc[0]["InChIKey-Planar"]

    # 3. Name appears as pipe-separated fragment in a Massbank-style compound name
    #    e.g. "2-amino-3-hydroxybutanoic acid" inside
    #         "Massbank:PB000405 Threonine|2-amino-3-hydroxybutanoic acid"
    #    Use regex=False (literal search) — do NOT wrap with re.escape here.
    m = sub_ik[sub_ik["Compound_Name"].str.contains(name, case=False, na=False, regex=False)]
    if len(m):
        return m.iloc[0]["InChIKey-Planar"]

    # 4. Global lookup (different entry in merged with same name)
    gm = merged_df.dropna(subset=["InChIKey-Planar"])
    m  = gm[gm["Compound_Name"].str.upper() == str(name).upper()]
    if len(m):
        return m.iloc[0]["InChIKey-Planar"]

    # 5. Single-InChIKey pool fallback
    pool = sub_ik["InChIKey-Planar"].unique()
    if len(pool) == 1:
        return pool[0]

    return None

# ── resolve GT InChIKey for all 185 features ──────────────────────────────────
gt["gt_ik"] = [
    resolve_ik(r["id_Mzmine"], r["GNPS_Compound_Name"], merged)
    for _, r in gt.iterrows()
]
print(f"GT InChIKey resolved: {gt['gt_ik'].notna().sum()}/185")

# ── merge ─────────────────────────────────────────────────────────────────────
df = gt.merge(
    annex[["scan", "top_compound", "structure_key", "top_cosine_compound",
           "top_cosine_overridden", "confidence_label", "confidence_score"]],
    left_on="id_Mzmine", right_on="scan", how="inner",
)

# ── resolve cosine InChIKey ───────────────────────────────────────────────────
# When NOT overridden: cosine == app (same structure_key), no separate lookup.
# When overridden: resolve independently from merged results.
def cosine_ik(row):
    if not row["top_cosine_overridden"]:
        return row["structure_key"]           # same compound as app
    return resolve_ik(row["id_Mzmine"], row["top_cosine_compound"], merged)

df["cosine_ik"] = df.apply(cosine_ik, axis=1)

# ── evaluate ──────────────────────────────────────────────────────────────────
df["app_correct"]    = df["gt_ik"] == df["structure_key"]
df["cosine_correct"] = df["cosine_ik"].notna() & (df["gt_ik"] == df["cosine_ik"])
df["cosine_has_ans"] = df["cosine_ik"].notna()

# Quadrant labels
def quadrant(row):
    a = row["app_correct"]
    c = row["cosine_correct"]
    h = row["cosine_has_ans"]
    if a and c:   return "Both correct"
    if a and not h: return "App ✓  |  Cosine: no answer"
    if a and not c: return "App ✓  |  Cosine ✗"
    if not a and c: return "App ✗  |  Cosine ✓"
    return "Both wrong"

df["quadrant"] = df.apply(quadrant, axis=1)

print("\n=== QUADRANT COUNTS ===")
print(df["quadrant"].value_counts())

print("\n=== OVERRIDDEN CASES ONLY ===")
ov = df[df["top_cosine_overridden"]].copy()
print(ov[["id_Mzmine","PRIMARY_NAME","top_compound","top_cosine_compound",
           "cosine_ik","gt_ik","app_correct","cosine_correct",
           "confidence_label","confidence_score","quadrant"]].to_string())

# ── accuracy summary ──────────────────────────────────────────────────────────
n = len(df)
app_acc    = df["app_correct"].sum()
cosine_acc = df["cosine_correct"].sum()
print(f"\nApp accuracy:    {app_acc}/{n} = {app_acc/n*100:.1f}%")
print(f"Cosine accuracy: {cosine_acc}/{n} = {cosine_acc/n*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# STATIC FIGURE
# ══════════════════════════════════════════════════════════════════════════════
QUAD_ORDER  = ["Both correct",
               "App ✓  |  Cosine: no answer",
               "App ✗  |  Cosine ✓",
               "Both wrong"]
QUAD_COLOR  = {
    "Both correct":                  "#2ca02c",
    "App ✓  |  Cosine: no answer":   "#1f77b4",
    "App ✗  |  Cosine ✓":            "#ff7f0e",
    "Both wrong":                    "#d62728",
}

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38, width_ratios=[1, 1.6])

ax_mat  = fig.add_subplot(gs[0])   # 2×2 matrix + accuracy bars
ax_list = fig.add_subplot(gs[1])   # detailed case list

# ── Left: 2×2 matrix ─────────────────────────────────────────────────────────
# rows = cosine correct/wrong, cols = app correct/wrong
# Build table from quadrant assignments
mat_vals = {
    (True,  True):  df[(df["app_correct"]) & (df["cosine_correct"])].shape[0],
    (True,  False): df[(df["app_correct"]) & (~df["cosine_correct"])].shape[0],
    (False, True):  df[(~df["app_correct"]) & (df["cosine_correct"])].shape[0],
    (False, False): df[(~df["app_correct"]) & (~df["cosine_correct"])].shape[0],
}
mat = np.array([
    [mat_vals[(True,  True)],  mat_vals[(False, True)]],   # cosine correct
    [mat_vals[(True,  False)], mat_vals[(False, False)]],  # cosine wrong
])

# colour matrix: green = both right, orange/red for mismatches
cell_colors = [
    ["#d4edda", "#fff3cd"],
    ["#d1ecf1", "#f8d7da"],
]
ax_mat.set_xlim(0, 2); ax_mat.set_ylim(0, 2.9)
ax_mat.axis("off")
ax_mat.set_title("A  |  App vs cosine accuracy (2×2)",
                 fontsize=12, fontweight="bold", loc="left", pad=10)

# Draw 2×2 cells
labels = [["Both correct", "App ✗\nCosine ✓"],
          ["App ✓\nCosine ✗\n(or no answer)", "Both wrong"]]
for ri in range(2):
    for ci in range(2):
        x, y = ci, 1 - ri
        fc = cell_colors[ri][ci]
        rect = mpatches.FancyBboxPatch(
            (x + 0.05, y + 0.05), 0.9, 0.85,
            boxstyle="round,pad=0.03", facecolor=fc,
            edgecolor="#ccc", linewidth=1.2)
        ax_mat.add_patch(rect)
        v = mat[ri, ci]
        ax_mat.text(x + 0.50, y + 0.56, str(v),
                    ha="center", va="center", fontsize=22, fontweight="bold")
        ax_mat.text(x + 0.50, y + 0.23, f"{v/n*100:.1f}%",
                    ha="center", va="center", fontsize=11, color="#555")
        ax_mat.text(x + 0.50, y + 0.10, labels[ri][ci],
                    ha="center", va="center", fontsize=8, color="#666", style="italic")

# axis headers
ax_mat.text(0.50, 1.97, "App correct",   ha="center", va="center",
            fontsize=10, fontweight="bold")
ax_mat.text(1.50, 1.97, "App wrong",     ha="center", va="center",
            fontsize=10, fontweight="bold")
ax_mat.text(-0.08, 1.48, "Cosine\ncorrect", ha="center", va="center",
            fontsize=10, fontweight="bold", rotation=90)
ax_mat.text(-0.08, 0.48, "Cosine\nwrong",   ha="center", va="center",
            fontsize=10, fontweight="bold", rotation=90)
ax_mat.axhline(1.0, xmin=0.04, xmax=0.96, color="#aaa", linewidth=1.2)
ax_mat.axvline(1.0, ymin=0.04, ymax=0.72, color="#aaa", linewidth=1.2)

# Accuracy comparison bars
ax_mat.text(0.5, 2.45, "Overall accuracy", ha="center", va="center",
            fontsize=10, fontweight="bold")
bar_y = 2.20
for label, acc, color, xoff in [
    ("App",    app_acc/n*100,    "#1f77b4", 0.30),
    ("Cosine", cosine_acc/n*100, "#ff7f0e", 1.20),
]:
    bw = 0.38 * acc / 100
    rect = mpatches.FancyBboxPatch(
        (xoff, bar_y), bw, 0.18,
        boxstyle="round,pad=0.01", facecolor=color, edgecolor="none")
    ax_mat.add_patch(rect)
    ax_mat.text(xoff - 0.02, bar_y + 0.09, label,
                ha="right", va="center", fontsize=9, color="#333")
    ax_mat.text(xoff + bw + 0.02, bar_y + 0.09,
                f"{acc:.1f}% ({int(acc*n/100)}/{n})",
                ha="left", va="center", fontsize=9, color="#333")

# ── Right: case detail ────────────────────────────────────────────────────────
ax_list.axis("off")
n_ov = len(ov)
ax_list.set_title(f"B  |  Cases where app and cosine DISAGREE\n"
                  f"({n_ov} cases have top_cosine_overridden=True)",
                  fontsize=11, fontweight="bold", loc="left")

ov_sorted = ov.sort_values("quadrant")
headers = ["GT compound", "App annotation", "Top cosine", "Outcome", "Conf. (score)"]
col_x   = [0.00, 0.22, 0.45, 0.68, 0.82]
y0, row_h = 0.95, 0.083

# header row
for h, xp in zip(headers, col_x):
    ax_list.text(xp, y0, h, transform=ax_list.transAxes,
                 fontsize=8, fontweight="bold", va="top", color="#222")
ax_list.plot([0, 1], [y0 - 0.018, y0 - 0.018],
             transform=ax_list.transAxes, color="#aaa", linewidth=0.8)

QUAD_BG = {
    "Both correct":                  "#f0fff4",
    "App ✓  |  Cosine: no answer":   "#eff6ff",
    "App ✗  |  Cosine ✓":            "#fff7ed",
    "Both wrong":                    "#fff0f0",
}
CONF_COLOR = {
    "Consistent evidence":  "#1f77b4",
    "Inconclusive":          "#d97706",
    "Inconsistent evidence": "#dc2626",
}

for ri, (_, row) in enumerate(ov_sorted.iterrows()):
    yi  = y0 - (ri + 1) * row_h - 0.02
    bg  = QUAD_BG.get(row["quadrant"], "#fff")
    ax_list.add_patch(mpatches.FancyBboxPatch(
        (0, yi - row_h * 0.18), 1, row_h * 0.88,
        boxstyle="round,pad=0.004",
        transform=ax_list.transAxes,
        facecolor=bg, edgecolor="none", zorder=0))

    tc  = str(row["top_cosine_compound"]) if pd.notna(row["top_cosine_compound"]) else "—"
    app = str(row["top_compound"]) if pd.notna(row["top_compound"]) else "—"
    outcome_short = (
        "✓✓ Both correct" if row["quadrant"] == "Both correct" else
        "✓ App | – Cosine" if "no answer" in row["quadrant"] else
        "✗ App | ✓ Cosine" if "Cosine ✓" in row["quadrant"] else
        "✗ Both wrong"
    )
    out_color = (
        "#16a34a" if "Both correct" in row["quadrant"] else
        "#2563eb" if "no answer"    in row["quadrant"] else
        "#ea580c" if "Cosine ✓"     in row["quadrant"] else
        "#dc2626"
    )
    conf_c = CONF_COLOR.get(row["confidence_label"], "#666")

    cells = [
        (str(row["PRIMARY_NAME"])[:22],   "#111"),
        (app[:22],                         "#333"),
        (tc[:22],                          "#333"),
        (outcome_short,                    out_color),
        (f"{str(row['confidence_label'])[:12]} ({row['confidence_score']:.0f})", conf_c),
    ]
    for (v, vc), xp in zip(cells, col_x):
        ax_list.text(xp, yi, v, transform=ax_list.transAxes,
                     fontsize=7.2, va="top", color=vc)

# legend
legend_patches = [
    mpatches.Patch(color=QUAD_BG["Both correct"],                  label="Both correct"),
    mpatches.Patch(color=QUAD_BG["App ✓  |  Cosine: no answer"],   label="App ✓ | Cosine: no answer"),
    mpatches.Patch(color=QUAD_BG["App ✗  |  Cosine ✓"],           label="App ✗ | Cosine ✓"),
]
ax_list.legend(handles=legend_patches, loc="lower right",
               fontsize=8, frameon=True, title="Row color")

fig.suptitle("App (multi-metric) vs top-cosine annotation — MSMLS library",
             fontsize=13, fontweight="bold", y=1.01)

plt.savefig(OUT_STATIC, bbox_inches="tight", dpi=180)
plt.savefig(OUT_STATIC.replace(".svg", ".png"), bbox_inches="tight", dpi=180)
print(f"\nSaved static → {OUT_STATIC}")

# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE HTML
# ══════════════════════════════════════════════════════════════════════════════
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["A — 2×2 accuracy matrix", f"B — All {len(ov_sorted)} override cases"],
    specs=[[{"type": "table"}, {"type": "table"}]],
    horizontal_spacing=0.06,
)

# Panel A – compact accuracy table
fig2.add_trace(
    go.Table(
        header=dict(
            values=["", "<b>App correct</b>", "<b>App wrong</b>"],
            fill_color=["white", "#dbeafe", "#fee2e2"],
            font=dict(size=12),
            align="center",
        ),
        cells=dict(
            values=[
                ["<b>Cosine correct</b>", "<b>Cosine wrong / no answer</b>",
                 "", "<b>Accuracy</b>"],
                [f"<b>{mat[0,0]}</b>  ({mat[0,0]/n*100:.1f}%)",
                 f"{mat[1,0]}  ({mat[1,0]/n*100:.1f}%)",
                 "",
                 f"<b>{app_acc/n*100:.1f}%</b>  ({app_acc}/{n})"],
                [f"{mat[0,1]}  ({mat[0,1]/n*100:.1f}%)",
                 f"{mat[1,1]}  ({mat[1,1]/n*100:.1f}%)",
                 "",
                 f"<b>{cosine_acc/n*100:.1f}%</b>  ({cosine_acc}/{n})"],
            ],
            fill_color=[
                ["#f0fdf4", "#fff7ed", "white", "#f8f9fa"],
                ["#d4edda", "#d1ecf1", "white", "#dbeafe"],
                ["#fff3cd", "#f8d7da", "white", "#fff3cd"],
            ],
            font=dict(size=11),
            align="center",
            height=30,
        ),
    ),
    row=1, col=1,
)

# Panel B – override detail table
ov_s = ov_sorted.copy()
row_fill = [QUAD_BG.get(q, "#fff") for q in ov_s["quadrant"]]
fig2.add_trace(
    go.Table(
        header=dict(
            values=["<b>GT compound</b>", "<b>App annotation</b>",
                    "<b>Top cosine</b>", "<b>Outcome</b>",
                    "<b>Confidence (score)</b>"],
            fill_color="#374151",
            font=dict(color="white", size=11),
            align="left",
        ),
        cells=dict(
            values=[
                ov_s["PRIMARY_NAME"].tolist(),
                ov_s["top_compound"].fillna("—").tolist(),
                ov_s["top_cosine_compound"].fillna("—").tolist(),
                ov_s["quadrant"].tolist(),
                [f"{cl} ({sc:.0f})"
                 for cl, sc in zip(ov_s["confidence_label"], ov_s["confidence_score"])],
            ],
            fill_color=[row_fill] * 5,
            font=dict(size=10),
            align="left",
            height=24,
        ),
    ),
    row=1, col=2,
)

fig2.update_layout(
    title=dict(
        text="App (multi-metric) vs top-cosine annotation<br>"
             "<sup>MSMLS library · 185 ground-truth features</sup>",
        font=dict(size=15),
    ),
    height=620,
    plot_bgcolor="white",
    paper_bgcolor="white",
)

fig2.write_html(OUT_HTML)
print(f"Saved interactive → {OUT_HTML}")
