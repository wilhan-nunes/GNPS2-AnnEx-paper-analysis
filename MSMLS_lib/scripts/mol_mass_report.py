"""
Molecular mass × confidence label analysis.

Panels:
  A  – annotated compound MW by confidence label (all 2916 scans, violin+strip)
  B  – ground-truth compound MW by confidence label (185 GT scans, violin+strip)
  C  – query peak count by confidence label (all 2916 scans, violin+strip)
  D  – annotated MW vs query peak count scatter, coloured by label
  E  – override comparison: MW and peak-count violins (overridden=141 vs rest=2775)
  F  – GT disagreement detail table with MW and spectral context
"""

import re, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("WARNING: rdkit not found; MW will be estimated from formula where possible.")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROC = os.path.join(BASE, "data", "processing")

GT_FILE     = os.path.join(PROC, "Mise en forme table final v3.tsv")
ANNEX_FILE  = os.path.join(PROC, "annex_summary_384d5ca63dae.csv")
MERGED_FILE = os.path.join(PROC, "384d5ca63dae452b95c762743c78dc31-merged_results_with_gnps.tsv")

OUT_STATIC = os.path.join(BASE, "mol_mass_report.svg")
OUT_HTML   = os.path.join(BASE, "mol_mass_report.html")

# ── load data ─────────────────────────────────────────────────────────────────
gt     = pd.read_csv(GT_FILE,     sep="\t")
annex  = pd.read_csv(ANNEX_FILE)
merged = pd.read_csv(MERGED_FILE, sep="\t")

# ── helpers ───────────────────────────────────────────────────────────────────
def smiles_to_mw(smi):
    if not RDKIT_OK or pd.isna(smi) or not str(smi).strip():
        return float("nan")
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Descriptors.ExactMolWt(mol) if mol else float("nan")
    except Exception:
        return float("nan")

# Build InChIKey-Planar → first valid SMILES map from merged
ik_to_smiles = (
    merged.dropna(subset=["InChIKey-Planar", "Smiles"])
    .groupby("InChIKey-Planar")["Smiles"].first()
    .to_dict()
)

# ── compute annotated MW (all 2916) ───────────────────────────────────────────
annex["annotated_smiles"] = annex["structure_key"].map(ik_to_smiles)
annex["annotated_mw"]     = annex["annotated_smiles"].apply(smiles_to_mw)
print(f"Annotated MW computed: {annex['annotated_mw'].notna().sum()}/{len(annex)}")

# ── compute GT MW (185 scans) ─────────────────────────────────────────────────
gt["gt_mw"] = gt["SMILES"].apply(smiles_to_mw)
print(f"GT MW computed: {gt['gt_mw'].notna().sum()}/{len(gt)}")

# merge GT MW into annex subset
gt_annex = annex[annex["scan"].isin(gt["id_Mzmine"])].merge(
    gt[["id_Mzmine", "PRIMARY_NAME", "SMILES", "gt_mw", "GNPS_Compound_Name"]],
    left_on="scan", right_on="id_Mzmine", how="left",
)

# ── resolve GT InChIKey for GT disagreement section ───────────────────────────
def _strip_energy_suffix(name):
    return re.sub(r"\s*[\||-]\s*\d+\.?\d*\s*(?:eV|ev)\s*$", "", name, flags=re.I).strip()

def resolve_gt_inchikey(scan_id, gnps_name, merged_df):
    sub = merged_df[merged_df["#Scan#"] == scan_id]
    sub_has = sub.dropna(subset=["InChIKey-Planar"])
    for cand in [gnps_name, str(gnps_name).upper()]:
        m = sub_has[sub_has["Compound_Name"].str.upper() == str(cand).upper()]
        if len(m): return m.iloc[0]["InChIKey-Planar"]
    base = _strip_energy_suffix(gnps_name)
    if base != gnps_name:
        m = sub_has[sub_has["Compound_Name"].str.upper() == base.upper()]
        if len(m): return m.iloc[0]["InChIKey-Planar"]
        m = sub_has[sub_has["Compound_Name"].str.upper().str.startswith(base.upper())]
        if len(m): return m.iloc[0]["InChIKey-Planar"]
    gm = merged_df.dropna(subset=["InChIKey-Planar"])
    m = gm[gm["Compound_Name"].str.upper() == str(gnps_name).upper()]
    if len(m): return m.iloc[0]["InChIKey-Planar"]
    pool = sub_has["InChIKey-Planar"].unique()
    if len(pool) == 1: return pool[0]
    return None

gt_annex["gt_inchikey"] = [
    resolve_gt_inchikey(r["scan"], r["GNPS_Compound_Name"], merged)
    for _, r in gt_annex.iterrows()
]
gt_annex["outcome"] = gt_annex.apply(
    lambda r: "Agree" if (not pd.isna(r["gt_inchikey"]) and not pd.isna(r["structure_key"])
                          and r["gt_inchikey"] == r["structure_key"]) else "Disagree",
    axis=1
)

disagree_gt = gt_annex[gt_annex["outcome"] == "Disagree"].copy()
print(f"\nGT disagreements: {len(disagree_gt)}")
print(disagree_gt[["scan", "PRIMARY_NAME", "top_compound",
                    "gt_mw", "annotated_mw", "query_peak_count",
                    "confidence_label", "confidence_score"]].to_string())

# ── colour palette ─────────────────────────────────────────────────────────────
CONF_ORDER = ["Consistent evidence", "Inconclusive", "Inconsistent evidence"]
CONF_COLOR = {
    "Consistent evidence":  "#1f77b4",
    "Inconclusive":          "#ff7f0e",
    "Inconsistent evidence": "#d62728",
}
OVR_COLOR = {"Not overridden": "#2ca02c", "Overridden": "#9467bd"}

# ── print summary stats ────────────────────────────────────────────────────────
print("\n=== ANNOTATED MW BY LABEL (median) ===")
for lbl in CONF_ORDER:
    sub = annex[annex["confidence_label"] == lbl]["annotated_mw"].dropna()
    print(f"  {lbl}: n={len(sub)}, median={sub.median():.1f}, IQR={sub.quantile(.25):.1f}–{sub.quantile(.75):.1f}")

print("\n=== GT MW BY LABEL (median) ===")
for lbl in CONF_ORDER:
    sub = gt_annex[gt_annex["confidence_label"] == lbl]["gt_mw"].dropna()
    if len(sub):
        print(f"  {lbl}: n={len(sub)}, median={sub.median():.1f}, IQR={sub.quantile(.25):.1f}–{sub.quantile(.75):.1f}")

print("\n=== QUERY PEAK COUNT BY LABEL (median) ===")
for lbl in CONF_ORDER:
    sub = annex[annex["confidence_label"] == lbl]["query_peak_count"].dropna()
    print(f"  {lbl}: n={len(sub)}, median={sub.median():.1f}, IQR={sub.quantile(.25):.1f}–{sub.quantile(.75):.1f}")

annex["override_label"] = annex["top_cosine_overridden"].map({True: "Overridden", False: "Not overridden"})
print("\n=== MW BY OVERRIDE STATUS (median) ===")
for lbl in ["Not overridden", "Overridden"]:
    sub = annex[annex["override_label"] == lbl]["annotated_mw"].dropna()
    print(f"  {lbl}: n={len(sub)}, median={sub.median():.1f}, IQR={sub.quantile(.25):.1f}–{sub.quantile(.75):.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: violin + strip
# ══════════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)

def violin_strip(ax, groups, labels, colors, ylabel="", title="", strip_alpha=0.35, strip_s=6):
    """Draw a violin with strip (jitter) overlay for each group."""
    data_clean = [np.array(g, dtype=float) for g in groups]
    data_clean = [d[~np.isnan(d)] for d in data_clean]
    positions  = list(range(len(labels)))

    if any(len(d) > 1 for d in data_clean):
        parts = ax.violinplot(
            [d if len(d) > 1 else np.array([d[0], d[0]]) for d in data_clean],
            positions=positions, widths=0.55, showmedians=True,
            showextrema=True,
        )
        for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
            pc.set_edgecolor("none")
        for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
            if part in parts:
                parts[part].set_color("#333")
                parts[part].set_linewidth(1.2)

    for i, (d, color) in enumerate(zip(data_clean, colors)):
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(i + jitter, d, alpha=strip_alpha, s=strip_s,
                   color=color, zorder=4, linewidths=0)

        # median line annotation
        if len(d):
            med = np.median(d)
            ax.text(i + 0.28, med, f"{med:.0f}",
                    va="center", ha="left", fontsize=7.5, color="#222")

    ax.set_xticks(positions)
    short_labels = [l.replace("evidence", "ev.").replace("Inconsistent", "Inconsist.") for l in labels]
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

# ══════════════════════════════════════════════════════════════════════════════
# STATIC FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                        height_ratios=[1.0, 1.0, 1.1])

ax_A = fig.add_subplot(gs[0, 0])   # annotated MW by label
ax_B = fig.add_subplot(gs[0, 1])   # GT MW by label
ax_C = fig.add_subplot(gs[0, 2])   # query peak count by label
ax_D = fig.add_subplot(gs[1, 0:2]) # MW vs peak count scatter
ax_E = fig.add_subplot(gs[1, 2])   # override MW comparison
ax_F = fig.add_subplot(gs[2, 0:3]) # GT disagreement table

# ── A: annotated MW by label ──────────────────────────────────────────────────
violin_strip(
    ax_A,
    [annex[annex["confidence_label"] == l]["annotated_mw"].dropna() for l in CONF_ORDER],
    CONF_ORDER,
    [CONF_COLOR[l] for l in CONF_ORDER],
    ylabel="Exact MW (Da)",
    title="A  |  Annotated compound MW\nby confidence label  (n=2,916)",
)

# ── B: GT MW by label ─────────────────────────────────────────────────────────
violin_strip(
    ax_B,
    [gt_annex[gt_annex["confidence_label"] == l]["gt_mw"].dropna() for l in CONF_ORDER],
    CONF_ORDER,
    [CONF_COLOR[l] for l in CONF_ORDER],
    ylabel="Exact MW (Da)",
    title="B  |  Ground-truth compound MW\nby confidence label  (n=185)",
)

# ── C: query peak count by label ──────────────────────────────────────────────
violin_strip(
    ax_C,
    [annex[annex["confidence_label"] == l]["query_peak_count"].dropna() for l in CONF_ORDER],
    CONF_ORDER,
    [CONF_COLOR[l] for l in CONF_ORDER],
    ylabel="Filtered query peak count",
    title="C  |  Query spectrum peak count\nby confidence label  (n=2,916)",
)

# ── D: MW vs peak count scatter ───────────────────────────────────────────────
for lbl in CONF_ORDER:
    sub = annex[annex["confidence_label"] == lbl].dropna(subset=["annotated_mw", "query_peak_count"])
    ax_D.scatter(sub["annotated_mw"], sub["query_peak_count"],
                 alpha=0.25, s=8, color=CONF_COLOR[lbl], label=lbl,
                 linewidths=0, zorder=3)

# Pearson correlation for context
corr_data = annex.dropna(subset=["annotated_mw", "query_peak_count"])
r = np.corrcoef(corr_data["annotated_mw"], corr_data["query_peak_count"])[0, 1]
ax_D.text(0.97, 0.97, f"r = {r:.3f}", transform=ax_D.transAxes,
          ha="right", va="top", fontsize=9, color="#333",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8))

ax_D.set_xlabel("Annotated compound exact MW (Da)", fontsize=9)
ax_D.set_ylabel("Filtered query peak count", fontsize=9)
ax_D.set_title("D  |  MW vs query peak count (all scans)",
               fontsize=10, fontweight="bold", loc="left")
ax_D.legend(title="Confidence", fontsize=8, title_fontsize=8,
            loc="upper right", frameon=True,
            markerscale=2)
ax_D.spines[["top", "right"]].set_visible(False)
ax_D.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax_D.set_axisbelow(True)

# ── E: override comparison — MW violin ────────────────────────────────────────
ovr_groups  = [annex[annex["override_label"] == l]["annotated_mw"].dropna()
               for l in ["Not overridden", "Overridden"]]
ovr_labels  = ["Not overridden\n(n=2,775)", "Overridden\n(n=141)"]
ovr_colors  = [OVR_COLOR["Not overridden"], OVR_COLOR["Overridden"]]
violin_strip(ax_E, ovr_groups, ovr_labels, ovr_colors,
             ylabel="Exact MW (Da)",
             title="E  |  MW: override vs not\n(top_cosine_overridden)")

# ── F: GT disagreement table ──────────────────────────────────────────────────
ax_F.axis("off")
ax_F.set_title("F  |  GT disagreement cases — molecular context",
               fontsize=10, fontweight="bold", loc="left", pad=6)

headers  = ["Scan", "GT compound", "App annotation",
            "GT MW (Da)", "App MW (Da)", "ΔMW",
            "Query peaks", "Shared peaks",
            "Confidence (score)"]
col_x    = [0.00, 0.08, 0.28, 0.47, 0.56, 0.65, 0.73, 0.82, 0.90]
y0       = 0.91
row_h    = 0.15

for h, xp in zip(headers, col_x):
    ax_F.text(xp, y0, h, transform=ax_F.transAxes,
              fontsize=7.5, fontweight="bold", va="top", color="#111")
ax_F.plot([0, 1], [y0 - 0.03, y0 - 0.03],
          transform=ax_F.transAxes, color="#aaa", linewidth=0.8)

disagree_sorted = disagree_gt.sort_values("confidence_score", ascending=False)
for ri, (_, row) in enumerate(disagree_sorted.iterrows()):
    yi    = y0 - (ri + 1) * row_h - 0.04
    bg    = "#fff7ed"
    ax_F.add_patch(mpatches.FancyBboxPatch(
        (0, yi - row_h * 0.2), 1, row_h * 0.85,
        boxstyle="round,pad=0.004", transform=ax_F.transAxes,
        facecolor=bg, edgecolor="none", zorder=0))

    gt_mw  = row["gt_mw"]
    ann_mw = row["annotated_mw"]
    delta  = ann_mw - gt_mw if (not math.isnan(gt_mw) and not math.isnan(ann_mw)) else float("nan")
    delta_str = f"{delta:+.1f}" if not math.isnan(delta) else "—"

    cells = [
        (str(int(row["scan"])),                                        "#555"),
        (str(row["PRIMARY_NAME"])[:22],                                "#111"),
        (str(row["top_compound"])[:22] if pd.notna(row["top_compound"]) else "—", "#444"),
        (f"{gt_mw:.1f}" if not math.isnan(gt_mw) else "—",           "#333"),
        (f"{ann_mw:.1f}" if not math.isnan(ann_mw) else "—",         "#333"),
        (delta_str, "#c05000" if not math.isnan(delta) and abs(delta) > 20 else "#333"),
        (f"{int(row['query_peak_count'])}" if pd.notna(row['query_peak_count']) else "—", "#333"),
        (f"{row['shared_peaks']:.1f}" if pd.notna(row.get('shared_peaks')) else "—", "#333"),
        (f"{row['confidence_label'][:14]}  ({row['confidence_score']:.0f})",
         CONF_COLOR.get(row["confidence_label"], "#666")),
    ]
    for (v, vc), xp in zip(cells, col_x):
        ax_F.text(xp, yi, v, transform=ax_F.transAxes,
                  fontsize=7.5, va="top", color=vc)

fig.suptitle("Molecular mass × spectral peaks analysis\n"
             "TopKonfidence · MSMLS library · task 384d5ca63dae",
             fontsize=13, fontweight="bold", y=1.00)

plt.savefig(OUT_STATIC,                        bbox_inches="tight", dpi=180)
plt.savefig(OUT_STATIC.replace(".svg", ".png"), bbox_inches="tight", dpi=180)
print(f"\nSaved static → {OUT_STATIC}")

# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE HTML (Plotly)
# ══════════════════════════════════════════════════════════════════════════════
CONF_SHORT = {
    "Consistent evidence":  "Consistent",
    "Inconclusive":          "Inconclusive",
    "Inconsistent evidence": "Inconsistent",
}

fig2 = make_subplots(
    rows=2, cols=3,
    subplot_titles=[
        "A — Annotated MW by label (all scans)",
        "B — GT MW by label (185 GT scans)",
        "C — Query peak count by label",
        "D — MW vs query peak count scatter",
        "E — MW: override vs not",
        "",
    ],
    specs=[
        [{"type": "violin"}, {"type": "violin"}, {"type": "violin"}],
        [{"type": "scatter", "colspan": 2}, None, {"type": "violin"}],
    ],
    vertical_spacing=0.18,
    horizontal_spacing=0.08,
    row_heights=[0.48, 0.52],
)

# Panels A, B, C — violin traces
def add_violin(fig, row, col, data, label, color, showlegend):
    fig.add_trace(
        go.Violin(
            y=data.tolist(), name=CONF_SHORT.get(label, label),
            box_visible=True, meanline_visible=True,
            fillcolor=color, line_color=color,
            opacity=0.7, points="all",
            pointpos=0, jitter=0.4,
            marker=dict(size=3, opacity=0.4, color=color),
            legendgroup=label, showlegend=showlegend,
            hovertemplate=f"<b>{label}</b><br>MW: %{{y:.1f}} Da<extra></extra>",
        ),
        row=row, col=col,
    )

for ci, lbl in enumerate(CONF_ORDER):
    color = CONF_COLOR[lbl]
    ann_mw = annex[annex["confidence_label"] == lbl]["annotated_mw"].dropna()
    gt_mw  = gt_annex[gt_annex["confidence_label"] == lbl]["gt_mw"].dropna()
    pc     = annex[annex["confidence_label"] == lbl]["query_peak_count"].dropna()
    add_violin(fig2, 1, 1, ann_mw,  lbl, color, ci == 0)
    add_violin(fig2, 1, 2, gt_mw,   lbl, color, False)
    add_violin(fig2, 1, 3, pc,      lbl, color, False)

fig2.update_yaxes(title_text="Exact MW (Da)", row=1, col=1)
fig2.update_yaxes(title_text="Exact MW (Da)", row=1, col=2)
fig2.update_yaxes(title_text="Query peak count", row=1, col=3)

# Panel D — scatter MW vs peak count
for lbl in CONF_ORDER:
    sub = annex[annex["confidence_label"] == lbl].dropna(subset=["annotated_mw", "query_peak_count"])
    fig2.add_trace(
        go.Scatter(
            x=sub["annotated_mw"].tolist(), y=sub["query_peak_count"].tolist(),
            mode="markers",
            name=CONF_SHORT.get(lbl, lbl),
            marker=dict(size=4, color=CONF_COLOR[lbl], opacity=0.4),
            legendgroup=lbl, showlegend=False,
            hovertemplate=(
                f"<b>{lbl}</b><br>MW: %{{x:.1f}} Da<br>Peaks: %{{y}}<extra></extra>"
            ),
        ),
        row=2, col=1,
    )
fig2.update_xaxes(title_text="Annotated compound exact MW (Da)", row=2, col=1)
fig2.update_yaxes(title_text="Query peak count", row=2, col=1)

# Panel E — override MW violin
for lbl, color in OVR_COLOR.items():
    n_lbl = 2775 if lbl == "Not overridden" else 141
    sub = annex[annex["override_label"] == lbl]["annotated_mw"].dropna()
    fig2.add_trace(
        go.Violin(
            y=sub.tolist(), name=f"{lbl} (n={n_lbl})",
            box_visible=True, meanline_visible=True,
            fillcolor=color, line_color=color,
            opacity=0.7, points="all",
            pointpos=0, jitter=0.4,
            marker=dict(size=3, opacity=0.4, color=color),
            legendgroup=lbl + "_ovr", showlegend=True,
            hovertemplate=f"<b>{lbl}</b><br>MW: %{{y:.1f}} Da<extra></extra>",
        ),
        row=2, col=3,
    )
fig2.update_yaxes(title_text="Exact MW (Da)", row=2, col=3)

# Disagreement table — add as separate figure below
disagree_table = go.Figure(go.Table(
    header=dict(
        values=["<b>Scan</b>", "<b>GT compound</b>", "<b>App annotation</b>",
                "<b>GT MW (Da)</b>", "<b>App MW (Da)</b>", "<b>ΔMW</b>",
                "<b>Query peaks</b>", "<b>Shared peaks</b>",
                "<b>Confidence (score)</b>"],
        fill_color="#374151",
        font=dict(color="white", size=11),
        align="left",
    ),
    cells=dict(
        values=[
            disagree_sorted["scan"].astype(int).tolist(),
            disagree_sorted["PRIMARY_NAME"].tolist(),
            disagree_sorted["top_compound"].fillna("—").tolist(),
            [f"{v:.1f}" if not math.isnan(v) else "—" for v in disagree_sorted["gt_mw"]],
            [f"{v:.1f}" if not math.isnan(v) else "—" for v in disagree_sorted["annotated_mw"]],
            [f"{(a - g):+.1f}" if not (math.isnan(a) or math.isnan(g)) else "—"
             for a, g in zip(disagree_sorted["annotated_mw"], disagree_sorted["gt_mw"])],
            [f"{int(v)}" if pd.notna(v) else "—" for v in disagree_sorted["query_peak_count"]],
            [f"{v:.1f}" if pd.notna(v) else "—" for v in disagree_sorted["shared_peaks"]],
            [f"{cl} ({sc:.0f})"
             for cl, sc in zip(disagree_sorted["confidence_label"], disagree_sorted["confidence_score"])],
        ],
        fill_color=[["#fff7ed"] * len(disagree_sorted)] * 9,
        font=dict(size=10),
        align="left",
        height=28,
    ),
))
disagree_table.update_layout(
    title="F — GT disagreement cases: molecular context",
    height=260,
    margin=dict(t=60, b=10),
    paper_bgcolor="white",
)

fig2.update_layout(
    title=dict(
        text="Molecular mass × spectral peaks analysis<br>"
             "<sup>TopKonfidence · MSMLS library · task 384d5ca63dae</sup>",
        font=dict(size=16),
    ),
    height=820,
    violinmode="group",
    legend=dict(title="Confidence", x=1.02, y=0.98, xanchor="left"),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

# Write both traces to HTML (combined)
import plotly.offline as po

html_parts = [
    po.plot(fig2, include_plotlyjs="cdn", output_type="div"),
    "<hr>",
    po.plot(disagree_table, include_plotlyjs=False, output_type="div"),
]
with open(OUT_HTML, "w") as f:
    f.write("<html><head><meta charset='utf-8'></head><body>")
    f.write("\n".join(html_parts))
    f.write("</body></html>")

print(f"Saved interactive → {OUT_HTML}")
