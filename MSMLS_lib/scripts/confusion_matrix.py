"""
Confusion analysis: TopKonfidence app annotations vs authentic standards (MSMLS lib).

Three outcomes per scan:
  AGREE              — app's top compound (structure_key) matches the standard's InChIKey-Planar
  DISAGREE_IN_POOL   — mismatch, but the standard's structure appears in the GNPS candidates for that scan
  DISAGREE_ABSENT    — mismatch, and the standard's structure is not in the GNPS candidates at all

Stratified by the app's confidence label. Styled to match paper_figures/plusrise_scores
(mw_analysis.py): plain white background, shared ink/grid palette, lowercase bold
panel letters, top/right spines off.
"""

import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROC = os.path.join(BASE, "data", "processing")

GT_FILE     = os.path.join(PROC, "Mise en forme table final v3.tsv")
ANNEX_FILE  = os.path.join(PROC, "annex_summary_384d5ca63dae.csv")
MERGED_FILE = os.path.join(PROC, "384d5ca63dae452b95c762743c78dc31-merged_results_with_gnps.tsv")

OUT_STATIC  = os.path.join(BASE, "confusion_figure_v2.svg")

# ── shared style (mirrors paper_figures/plusrise_scores/mw_analysis.py) ────────
INK      = "#0b0b0b"
INK_2    = "#52514e"
GRID     = "#dcdbd6"
SURFACE  = "#ffffff"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
})


def panel_letter(ax, letter, dx=-0.09, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=16,
             fontweight="bold", va="bottom", ha="left", color=INK)

# ── load ───────────────────────────────────────────────────────────────────────
gt     = pd.read_csv(GT_FILE,     sep="\t")
annex  = pd.read_csv(ANNEX_FILE)
merged = pd.read_csv(MERGED_FILE, sep="\t")

# ── resolve GT InChIKey-Planar per scan ───────────────────────────────────────
# Strategy: for each GT scan, find the InChIKey-Planar that corresponds to the
# ground-truth GNPS_Compound_Name using a cascade of fallbacks.
def _strip_energy_suffix(name):
    return re.sub(r"\s*[\||-]\s*\d+\.?\d*\s*(?:eV|ev)\s*$", "", name, flags=re.I).strip()

def resolve_gt_inchikey(scan_id, gnps_name, merged_df):
    sub = merged_df[merged_df["#Scan#"] == scan_id]
    sub_has_ik = sub.dropna(subset=["InChIKey-Planar"])

    # 1. Exact match, non-null InChIKey
    m = sub_has_ik[sub_has_ik["Compound_Name"] == gnps_name]
    if len(m):
        return m.iloc[0]["InChIKey-Planar"]

    # 2. Case-insensitive match, non-null InChIKey
    m = sub_has_ik[sub_has_ik["Compound_Name"].str.upper() == gnps_name.upper()]
    if len(m):
        return m.iloc[0]["InChIKey-Planar"]

    # 3. Strip " - XX eV" / "CollisionEnergy:..." suffix, then case-insensitive
    base = _strip_energy_suffix(gnps_name)
    if base != gnps_name:
        m = sub_has_ik[sub_has_ik["Compound_Name"].str.upper() == base.upper()]
        if len(m):
            return m.iloc[0]["InChIKey-Planar"]
        # partial prefix match
        m = sub_has_ik[sub_has_ik["Compound_Name"].str.upper().str.startswith(base.upper())]
        if len(m):
            return m.iloc[0]["InChIKey-Planar"]

    # 4. Exact match anywhere in the full merged file (picks up correct InChIKey
    #    from a different library entry with the same name)
    m = merged_df.dropna(subset=["InChIKey-Planar"])
    m = m[m["Compound_Name"].str.upper() == gnps_name.upper()]
    if len(m):
        return m.iloc[0]["InChIKey-Planar"]

    # 5. If the scan pool has exactly one unique InChIKey, all annotations for
    #    this scan agree on structure — use it (handles naming gaps in GNPS).
    pool = sub_has_ik["InChIKey-Planar"].unique()
    if len(pool) == 1:
        return pool[0]

    return None

gt["gt_inchikey"] = [
    resolve_gt_inchikey(row["id_Mzmine"], row["GNPS_Compound_Name"], merged)
    for _, row in gt.iterrows()
]
unresolved_gt = gt["gt_inchikey"].isna().sum()
print(f"GT InChIKey resolved: {len(gt) - unresolved_gt}/{len(gt)}")

# ── build candidate InChIKey pool per scan ────────────────────────────────────
scan_pool = (
    merged[merged["#Scan#"].isin(gt["id_Mzmine"])]
    .groupby("#Scan#")["InChIKey-Planar"]
    .apply(lambda x: set(x.dropna()))
)

# ── merge gt + annex ──────────────────────────────────────────────────────────
df = gt.merge(
    annex[["scan", "top_compound", "structure_key", "confidence_label",
           "confidence_score", "top_cosine_compound", "top_cosine_overridden"]],
    left_on="id_Mzmine", right_on="scan", how="inner",
)

# ── assign outcome ─────────────────────────────────────────────────────────────
CONF_ORDER = ["Consistent evidence", "Inconclusive", "Inconsistent evidence"]
OUTCOME_ORDER  = ["Agree", "Disagree – GT in candidates", "Disagree – GT absent"]

def outcome(row):
    gt_ik  = row["gt_inchikey"]
    app_ik = row["structure_key"]
    if pd.isna(gt_ik) or pd.isna(app_ik):
        return "Unresolved"
    if gt_ik == app_ik:
        return "Agree"
    pool = scan_pool.get(row["id_Mzmine"], set())
    return "Disagree – GT in candidates" if gt_ik in pool else "Disagree – GT absent"

df["outcome"] = df.apply(outcome, axis=1)
df["confidence_label"] = pd.Categorical(df["confidence_label"], categories=CONF_ORDER, ordered=True)

# ── resolve top-cosine InChIKey (for the evidence-score vs cosine-only panel) ──
# Same cascade as resolve_gt_inchikey, plus a pipe-fragment fallback for
# Massbank-style names (e.g. "Threonine|2-amino-3-hydroxybutanoic acid").
def resolve_compound_inchikey(scan_id, name, merged_df):
    if pd.isna(name):
        return None
    sub_has_ik = merged_df[merged_df["#Scan#"] == scan_id].dropna(subset=["InChIKey-Planar"])

    for cand in (name, name.upper()):
        m = sub_has_ik[sub_has_ik["Compound_Name"].str.upper() == cand.upper()]
        if len(m):
            return m.iloc[0]["InChIKey-Planar"]

    base = _strip_energy_suffix(name)
    if base != name:
        m = sub_has_ik[sub_has_ik["Compound_Name"].str.upper() == base.upper()]
        if len(m):
            return m.iloc[0]["InChIKey-Planar"]
        m = sub_has_ik[sub_has_ik["Compound_Name"].str.upper().str.startswith(base.upper())]
        if len(m):
            return m.iloc[0]["InChIKey-Planar"]

    m = sub_has_ik[sub_has_ik["Compound_Name"].str.contains(name, case=False, na=False, regex=False)]
    if len(m):
        return m.iloc[0]["InChIKey-Planar"]

    gm = merged_df.dropna(subset=["InChIKey-Planar"])
    m = gm[gm["Compound_Name"].str.upper() == str(name).upper()]
    if len(m):
        return m.iloc[0]["InChIKey-Planar"]

    pool = sub_has_ik["InChIKey-Planar"].unique()
    if len(pool) == 1:
        return pool[0]
    return None

df["cosine_ik"] = df.apply(
    lambda r: r["structure_key"] if not r["top_cosine_overridden"]
              else resolve_compound_inchikey(r["id_Mzmine"], r["top_cosine_compound"], merged),
    axis=1,
)
df["app_correct"]    = df["gt_inchikey"] == df["structure_key"]
df["cosine_correct"] = df["cosine_ik"].notna() & (df["gt_inchikey"] == df["cosine_ik"])

n_gt          = len(df)
app_correct_n = int(df["app_correct"].sum())
cos_correct_n = int(df["cosine_correct"].sum())
print(f"\nEvidence-based score accuracy: {app_correct_n}/{n_gt} ({app_correct_n/n_gt*100:.1f}%)")
print(f"Top cosine only accuracy:      {cos_correct_n}/{n_gt} ({cos_correct_n/n_gt*100:.1f}%)")

# ── print summary ──────────────────────────────────────────────────────────────
print("\n=== OUTCOME SUMMARY ===")
print(df["outcome"].value_counts())
print()
ct = (df[df["outcome"].isin(OUTCOME_ORDER)]
      .groupby(["confidence_label", "outcome"], observed=True)
      .size()
      .unstack(fill_value=0)
      .reindex(columns=OUTCOME_ORDER, fill_value=0))
print(ct)
print()
print("Disagreement details:")
disagree = df[df["outcome"].str.startswith("Disagree")].copy()
print(disagree[["id_Mzmine","PRIMARY_NAME","top_compound",
                "gt_inchikey","structure_key",
                "confidence_label","confidence_score"]].to_string())

# ── colour palette (aligned with mw_analysis.LABEL_COLOR hues) ─────────────────
OUTCOME_COLORS = {
    "Agree":                        "#008300",
    "Disagree – GT in candidates":  "#eda100",
    "Disagree – GT absent":         "#e34948",
    "Unresolved":                   "#a8a7a2",
}
CONF_COLORS = {
    "Consistent evidence":  "#008300",
    "Inconclusive":          "#eda100",
    "Inconsistent evidence": "#e34948",
}
OUTCOME_SHORT = {
    "Agree":                        "Agree",
    "Disagree – GT in candidates":  "Disagree, in pool",
    "Disagree – GT absent":         "Disagree, absent",
    "Unresolved":                   "Unresolved",
}

present_outcomes = [o for o in OUTCOME_ORDER if o in df["outcome"].values]
all_outcomes = present_outcomes + (["Unresolved"] if "Unresolved" in df["outcome"].values else [])

ct_plot = (df.groupby(["confidence_label", "outcome"], observed=True)
             .size()
             .unstack(fill_value=0)
             .reindex(index=CONF_ORDER, columns=OUTCOME_ORDER, fill_value=0))
ct_plot["Unresolved"] = (
    df[df["outcome"] == "Unresolved"]
    .groupby("confidence_label", observed=True).size()
    .reindex(CONF_ORDER, fill_value=0)
)
matrix     = ct_plot[OUTCOME_ORDER]
matrix_pct = matrix.div(matrix.sum(axis=1), axis=0) * 100

disagree_sorted = disagree.sort_values(["outcome", "confidence_score"], ascending=[True, False])

# ── shared drawers for panels a, b, d (identical across both candidates) ───────
def draw_outcome_bar(ax_bar):
    y = np.arange(len(CONF_ORDER))
    totals = ct_plot.sum(axis=1).reindex(CONF_ORDER).values
    left = np.zeros(len(CONF_ORDER))

    for out_label in all_outcomes:
        counts = ct_plot.get(out_label, pd.Series(0, index=CONF_ORDER)).reindex(CONF_ORDER, fill_value=0).values
        pct = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0) * 100
        ax_bar.barh(y, pct, left=left, height=0.62,
                    color=OUTCOME_COLORS[out_label], label=OUTCOME_SHORT[out_label],
                    edgecolor=SURFACE, linewidth=0.8, zorder=3)
        for yi, p, l, n in zip(y, pct, left, counts):
            if n > 0 and p >= 6:
                ax_bar.text(l + p / 2, yi, str(int(n)), ha="center", va="center",
                            fontsize=8.5, fontweight="bold", color=SURFACE, zorder=4)
        left = left + pct

    for yi, tot in zip(y, totals):
        ax_bar.text(104, yi, f"n={int(tot)}", ha="left", va="center",
                    fontsize=7.5, color=INK_2, clip_on=False)

    ax_bar.set_xlim(0, 100)
    ax_bar.set_xticks([0, 25, 50, 75, 100])
    ax_bar.set_xlabel("Scans (%)", fontsize=8.5)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([c.replace(" evidence", "\nevidence") for c in CONF_ORDER], fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_title("Outcome vs authentic standards,\nby evidence tier",
                      fontsize=10, fontweight="bold", loc="left", pad=22)
    ax_bar.legend(frameon=False, ncol=2, fontsize=7,
                  loc="lower left", bbox_to_anchor=(-0.02, 1.0),
                  columnspacing=1.0, handlelength=1.3, handletextpad=0.5)
    ax_bar.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.tick_params(axis="y", length=0)


def draw_accuracy_heatmap(ax_heat):
    im = ax_heat.imshow(matrix_pct.values, cmap="Greens", aspect="auto", vmin=0, vmax=100)
    for i in range(len(CONF_ORDER)):
        for j in range(len(OUTCOME_ORDER)):
            n   = int(matrix.values[i, j])
            pct = matrix_pct.values[i, j]
            ax_heat.text(j, i, f"{n}\n({pct:.0f}%)", ha="center", va="center",
                         fontsize=8.5, color=SURFACE if pct > 60 else INK)

    ax_heat.set_xticks(range(len(OUTCOME_ORDER)))
    ax_heat.set_xticklabels([OUTCOME_SHORT[o].replace(", ", "\n") for o in OUTCOME_ORDER], fontsize=8)
    ax_heat.set_yticks(range(len(CONF_ORDER)))
    ax_heat.set_yticklabels([c.replace(" evidence", "\nevidence") for c in CONF_ORDER], fontsize=8)
    ax_heat.tick_params(length=0)
    ax_heat.set_title("Per-tier outcome\nbreakdown", fontsize=10, fontweight="bold", loc="left", pad=8)
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Row %", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)


def draw_disagreement_list(ax_list):
    ax_list.axis("off")
    ax_list.set_title(f"Disagreement cases\n(n = {len(disagree_sorted)})",
                       fontsize=10, fontweight="bold", loc="left", pad=8)

    n_cases = len(disagree_sorted)
    row_h = 0.95 / max(n_cases, 1)
    for ri, (_, row) in enumerate(disagree_sorted.iterrows()):
        yi = 0.98 - ri * row_h
        is_pool = "candidates" in row["outcome"]
        marker_color = OUTCOME_COLORS["Disagree – GT in candidates" if is_pool else "Disagree – GT absent"]
        conf_color = CONF_COLORS.get(row["confidence_label"], INK_2)

        ax_list.scatter([0.015], [yi], s=20, color=marker_color, zorder=3,
                         transform=ax_list.transAxes, clip_on=False)
        ax_list.text(0.06, yi, str(row["PRIMARY_NAME"]).title(),
                     transform=ax_list.transAxes, fontsize=7.8, fontweight="bold",
                     color=INK, va="top")
        predicted = str(row["top_compound"]).title().replace("?", "'") if pd.notna(row["top_compound"]) else "—"
        ax_list.text(0.06, yi - row_h * 0.34, f"→ annotation explorer: {predicted}",
                     transform=ax_list.transAxes, fontsize=7.2, color=INK_2, va="top")
        ax_list.text(0.06, yi - row_h * 0.63,
                     f"{row['confidence_label']} ({row['confidence_score']:.0f})"
                     f"  ·  {'recoverable' if is_pool else 'not in pool'}",
                     transform=ax_list.transAxes, fontsize=6.8, color=conf_color, va="top")

    ax_list.set_xlim(0, 1)
    ax_list.set_ylim(0, 1)


# ── evidence-based score vs top-cosine-only: 2x2 win/loss breakdown ────────────
def draw_cosine_comparison_2x2(ax):
    ax.axis("off")
    ax.set_title("Evidence-based score vs\ntop cosine only", fontsize=10,
                 fontweight="bold", loc="left", pad=8)

    both_correct = int((df["app_correct"] & df["cosine_correct"]).sum())
    app_only     = int((df["app_correct"] & ~df["cosine_correct"]).sum())
    cos_only     = int((~df["app_correct"] & df["cosine_correct"]).sum())
    both_wrong   = int((~df["app_correct"] & ~df["cosine_correct"]).sum())
    mat = np.array([[both_correct, cos_only], [app_only, both_wrong]])
    cell_labels = [["Both correct", "Evidence ✗\nCosine ✓"],
                   ["Evidence ✓\nCosine ✗", "Both wrong"]]
    cell_colors = [["#d9f2d9", "#fdecc8"], ["#dce8fb", "#f9d6d6"]]

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.35)
    for ri in range(2):
        for ci in range(2):
            x, y0 = ci, 1 - ri
            ax.add_patch(plt.Rectangle((x + 0.04, y0 + 0.04), 0.92, 0.86,
                                        facecolor=cell_colors[ri][ci], edgecolor=GRID, linewidth=1))
            v = mat[ri, ci]
            ax.text(x + 0.5, y0 + 0.58, str(v), ha="center", va="center",
                    fontsize=15, fontweight="bold", color=INK)
            ax.text(x + 0.5, y0 + 0.36, f"{v / n_gt * 100:.0f}%", ha="center", va="center",
                    fontsize=8, color=INK_2)
            ax.text(x + 0.5, y0 + 0.16, cell_labels[ri][ci], ha="center", va="center",
                    fontsize=6.3, color=INK_2, style="italic")

    ax.text(0.5, 2.18, "Evidence correct", ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
    ax.text(1.5, 2.18, "Evidence wrong", ha="center", va="center", fontsize=7.5, fontweight="bold", color=INK)
    ax.text(-0.09, 1.44, "Cosine\ncorrect", ha="center", va="center", fontsize=7.5,
            fontweight="bold", rotation=90, color=INK)
    ax.text(-0.09, 0.44, "Cosine\nwrong", ha="center", va="center", fontsize=7.5,
            fontweight="bold", rotation=90, color=INK)


def build_figure(out_path):
    fig = plt.figure(figsize=(9.2, 7.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1],
                            wspace=0.45, hspace=0.55, top=0.90, bottom=0.07, left=0.09, right=0.97)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_outcome_bar(ax_a)
    draw_accuracy_heatmap(ax_b)
    draw_cosine_comparison_2x2(ax_c)
    draw_disagreement_list(ax_d)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_letter(ax, letter, dx=-0.14, dy=1.18)

    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(out_path.replace(".svg", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {out_path}")


build_figure(OUT_STATIC)

# ── final summary ──────────────────────────────────────────────────────────────
print("\n=== FINAL COUNTS ===")
for o, n in df["outcome"].value_counts().items():
    print(f"  {o}: {n}  ({n/len(df)*100:.1f}%)")
