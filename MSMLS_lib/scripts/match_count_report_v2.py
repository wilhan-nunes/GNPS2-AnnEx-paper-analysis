"""
Match count × confidence label — expanded analysis with GT subset and PPM deep dive.

Figure layout (3 rows × 3 cols):
  A  total_matches by label — full 2916        B  total_matches — GT subset 185
  C  Inconsistent cause breakdown full vs GT   D  PPM error by label (full + GT overlay)
  E  PPM error by Inconsistent category        F  mz_precision by label (full + GT overlay)
  G  support_fraction by label — full          H  support_fraction — GT subset
  I  Score component heatmap (Inconsistent)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as po
import os

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROC = os.path.join(BASE, "data", "processing")
GT_FILE    = os.path.join(PROC, "Mise en forme table final v3.tsv")
ANNEX_FILE = os.path.join(PROC, "annex_summary_384d5ca63dae.csv")
OUT_STATIC = os.path.join(BASE, "match_count_report_v2.svg")
OUT_HTML   = os.path.join(BASE, "match_count_report_v2.html")

MATCH_CAP  = 10
CONF_ORDER = ["Consistent evidence", "Inconclusive", "Inconsistent evidence"]
CONF_COLOR = {
    "Consistent evidence":  "#1f77b4",
    "Inconclusive":          "#ff7f0e",
    "Inconsistent evidence": "#d62728",
}
CAT_LABELS = [
    "Single hit (n=1)",
    "2–9 hits, full consensus",
    "2–9 hits, fragmented",
    "10 hits (cap), full consensus",
    "10 hits (cap), fragmented",
]
CAT_COLORS = ["#e15759", "#f28e2b", "#edc948", "#76b7b2", "#4e79a7"]

# ── load & derive ─────────────────────────────────────────────────────────────
annex = pd.read_csv(ANNEX_FILE)
gt    = pd.read_csv(GT_FILE, sep="\t")

annex["support_fraction"] = annex["supporting_matches"] / annex["total_matches"]
gt_scans  = set(gt["id_Mzmine"])
gt_annex  = annex[annex["scan"].isin(gt_scans)].copy()

# ── categorise Inconsistent evidence (shared helper) ─────────────────────────
def categorise_incon(df):
    d = df[df["confidence_label"] == "Inconsistent evidence"].copy()
    d["support_fraction"] = d["supporting_matches"] / d["total_matches"]
    single      = d["total_matches"] == 1
    at_cap      = d["total_matches"] == MATCH_CAP
    few_cons    = (~single) & (~at_cap) & (d["support_fraction"] == 1.0)
    few_frag    = (~single) & (~at_cap) & (d["support_fraction"] <  1.0)
    cap_cons    = at_cap  & (d["support_fraction"] == 1.0)
    cap_frag    = at_cap  & (d["support_fraction"] <  1.0)
    masks = [single, few_cons, few_frag, cap_cons, cap_frag]
    d["cause_cat"] = "Other"
    for lbl, mask in zip(CAT_LABELS, masks):
        d.loc[mask, "cause_cat"] = lbl
    return d, masks

incon_full, masks_full = categorise_incon(annex)
incon_gt,   masks_gt   = categorise_incon(gt_annex)

counts_full = [m.sum() for m in masks_full]
counts_gt   = [m.sum() for m in masks_gt]

print("=== Inconsistent evidence breakdown — FULL ===")
for lbl, n in zip(CAT_LABELS, counts_full):
    print(f"  {lbl}: {n} ({n/len(incon_full)*100:.1f}%)")

print("\n=== Inconsistent evidence breakdown — GT subset ===")
for lbl, n in zip(CAT_LABELS, counts_gt):
    print(f"  {lbl}: {n} ({n/len(incon_gt)*100:.1f}%)")

# ── score component medians per Inconsistent category (full) ──────────────────
SCORE_COMPONENTS = {
    "Structure\nagreement": "support_fraction",
    "Tanimoto":             "tanimoto_score",
    "Cosine\nconsistency":  "cosine_consistency",
    "Max cosine":           "max_cosine",
    "Shared peaks":         "shared_peaks_score",
    "M/Z precision":        "mz_precision",
}
comp_matrix = np.full((len(CAT_LABELS), len(SCORE_COMPONENTS)), np.nan)
for ri, (lbl, mask) in enumerate(zip(CAT_LABELS, masks_full)):
    for ci, col in enumerate(SCORE_COMPONENTS.values()):
        comp_matrix[ri, ci] = incon_full[mask][col].median()

# ── violin + strip helper ─────────────────────────────────────────────────────
rng = np.random.default_rng(42)

def violin_strip(ax, groups, labels, colors, ylabel="", title="",
                 strip_alpha=0.28, strip_s=5,
                 overlay_groups=None, overlay_label="GT subset",
                 overlay_color="#222", overlay_s=28, log_y=False):
    """
    Violin + strip. Optionally overlay a second dataset (GT subset) as scatter.
    overlay_groups: list of arrays, same length as groups.
    """
    data_clean = [np.asarray(g, dtype=float) for g in groups]
    data_clean = [d[~np.isnan(d)] for d in data_clean]
    positions  = list(range(len(labels)))

    parts = ax.violinplot(
        [d if len(d) > 1 else np.array([d[0], d[0]]) for d in data_clean],
        positions=positions, widths=0.52, showmedians=True, showextrema=True,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color); pc.set_alpha(0.30); pc.set_edgecolor("none")
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if part in parts:
            parts[part].set_color("#333"); parts[part].set_linewidth(1.1)

    for i, (d, color) in enumerate(zip(data_clean, colors)):
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax.scatter(i + jitter, d, alpha=strip_alpha, s=strip_s,
                   color=color, zorder=3, linewidths=0)
        if len(d):
            med = np.median(d)
            ax.text(i + 0.29, med, f"{med:.1f}",
                    va="center", ha="left", fontsize=7.2, color="#222")

    if overlay_groups is not None:
        first = True
        for i, og in enumerate(overlay_groups):
            oa = np.asarray(og, dtype=float)
            oa = oa[~np.isnan(oa)]
            if len(oa) == 0:
                continue
            jitter = rng.uniform(-0.18, 0.18, len(oa))
            ax.scatter(i + jitter, oa, s=overlay_s, color=overlay_color,
                       edgecolors="white", linewidths=0.6,
                       alpha=0.85, zorder=6, marker="D",
                       label=overlay_label if first else "_nolegend_")
            first = False
        ax.legend(fontsize=7.5, loc="upper right", frameon=True, markerscale=0.9)

    if log_y:
        ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(positions)
    short = [l.replace(" evidence", "\nev.").replace("Inconsistent", "Inconsist.") for l in labels]
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.32, zorder=0)
    ax.set_axisbelow(True)

# ══════════════════════════════════════════════════════════════════════════════
# STATIC FIGURE  —  3 rows × 3 cols
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.60, wspace=0.38,
                        height_ratios=[1.0, 1.0, 1.0])

ax_A = fig.add_subplot(gs[0, 0])   # total_matches full
ax_B = fig.add_subplot(gs[0, 1])   # total_matches GT
ax_C = fig.add_subplot(gs[0, 2])   # Inconsistent breakdown full vs GT
ax_D = fig.add_subplot(gs[1, 0])   # PPM error by label (full + GT overlay)
ax_E = fig.add_subplot(gs[1, 1])   # PPM error by Inconsistent category
ax_F = fig.add_subplot(gs[1, 2])   # mz_precision by label (full + GT overlay)
ax_G = fig.add_subplot(gs[2, 0])   # support_fraction full
ax_H = fig.add_subplot(gs[2, 1])   # support_fraction GT
ax_I = fig.add_subplot(gs[2, 2])   # score component heatmap

# ── A: total_matches — full ────────────────────────────────────────────────────
violin_strip(
    ax_A,
    [annex[annex["confidence_label"] == l]["total_matches"] for l in CONF_ORDER],
    CONF_ORDER, [CONF_COLOR[l] for l in CONF_ORDER],
    ylabel="Total retrieved matches",
    title="A  |  Match count by label\n(all 2,916 scans)",
)
ax_A.axhline(MATCH_CAP, color="#888", linestyle="--", linewidth=1, zorder=2)
ax_A.text(2.48, MATCH_CAP + 0.1, "cap=10", va="bottom", ha="right",
          fontsize=7, color="#888")

# ── B: total_matches — GT subset ──────────────────────────────────────────────
violin_strip(
    ax_B,
    [gt_annex[gt_annex["confidence_label"] == l]["total_matches"] for l in CONF_ORDER],
    CONF_ORDER, [CONF_COLOR[l] for l in CONF_ORDER],
    ylabel="Total retrieved matches",
    title="B  |  Match count by label\n(GT subset, n=185)",
)
ax_B.axhline(MATCH_CAP, color="#888", linestyle="--", linewidth=1, zorder=2)
ax_B.text(2.48, MATCH_CAP + 0.1, "cap=10", va="bottom", ha="right",
          fontsize=7, color="#888")
# annotation
ax_B.text(2, 6.2, "76% hit cap",
          ha="center", va="top", fontsize=7.5, color=CONF_COLOR["Inconsistent evidence"],
          style="italic")

# ── C: Inconsistent breakdown — stacked bars full vs GT ───────────────────────
ax_C.set_title("C  |  Inconsistent evidence driver breakdown\nFull dataset vs GT subset",
               fontsize=9.5, fontweight="bold", loc="left")
ax_C.axis("off")

datasets = [
    ("Full\n(n=1,100)", counts_full, len(incon_full)),
    ("GT subset\n(n=33)",  counts_gt,   len(incon_gt)),
]
bar_w    = 0.30
bar_xs   = [0.22, 0.68]
bar_ybot = 0.08
bar_htot = 0.82

for bx, (ds_label, counts, total) in zip(bar_xs, datasets):
    ax_C.text(bx, bar_ybot - 0.05, ds_label,
              transform=ax_C.transAxes, ha="center", va="top",
              fontsize=8, color="#333")
    bottom = 0.0
    for lbl, n, color in zip(CAT_LABELS, counts, CAT_COLORS):
        h = (n / total) * bar_htot if total else 0
        ax_C.add_patch(mpatches.Rectangle(
            (bx - bar_w / 2, bar_ybot + bottom), bar_w, h,
            transform=ax_C.transAxes,
            facecolor=color, edgecolor="white", linewidth=1.5, zorder=3,
        ))
        if h > 0.04:
            ax_C.text(bx, bar_ybot + bottom + h / 2,
                      f"{n}\n({n/total*100:.0f}%)",
                      transform=ax_C.transAxes,
                      ha="center", va="center", fontsize=7.2,
                      fontweight="bold", color="white", zorder=4)
        bottom += h

legend_patches = [mpatches.Patch(color=c, label=l)
                  for l, c in zip(CAT_LABELS, CAT_COLORS)]
ax_C.legend(handles=legend_patches, loc="lower left",
            bbox_to_anchor=(-0.02, -0.04), fontsize=7, frameon=True,
            title="Category", title_fontsize=7.5, ncol=1)

# highlight the contrast
ax_C.text(0.50, 0.95,
          "Full: 60.5% single-hit\nGT: 76% at 10-match cap",
          transform=ax_C.transAxes, ha="center", va="top",
          fontsize=8, color="#555",
          bbox=dict(boxstyle="round,pad=0.35", fc="#fff9f0", ec="#f28e2b", lw=1))

# ── D: PPM error by label — full + GT overlay ─────────────────────────────────
violin_strip(
    ax_D,
    [annex[annex["confidence_label"] == l]["median_abs_ppm_error"].dropna() for l in CONF_ORDER],
    CONF_ORDER, [CONF_COLOR[l] for l in CONF_ORDER],
    ylabel="Median |PPM error|",
    title="D  |  m/z precision error by label\n(full 2,916 · GT diamonds)",
    log_y=True,
    overlay_groups=[gt_annex[gt_annex["confidence_label"] == l]["median_abs_ppm_error"].dropna()
                    for l in CONF_ORDER],
    overlay_label="GT subset",
    overlay_color="#222",
)
ax_D.axhline(5.0, color="#888", linestyle=":", linewidth=1, zorder=2)
ax_D.text(2.48, 5.0, "PPM_TOL\n= 5", va="center", ha="right",
          fontsize=6.5, color="#888")

# ── E: PPM error by Inconsistent category ────────────────────────────────────
cat_ppm = [incon_full[mask]["median_abs_ppm_error"].dropna() for mask in masks_full]
short_cats = [l.replace("full consensus", "full\nconsensus")
               .replace("fragmented", "fragm.")
               .replace("10 hits (cap),", "10 (cap),")
               .replace("2–9 hits,", "2–9,") for l in CAT_LABELS]

violin_strip(
    ax_E, cat_ppm, short_cats, CAT_COLORS,
    ylabel="Median |PPM error|",
    title="E  |  PPM error per Inconsistent\nevidence category (full dataset)",
    log_y=True,
)
ax_E.axhline(5.0, color="#888", linestyle=":", linewidth=1, zorder=2)
ax_E.tick_params(axis="x", labelsize=7)

# ── F: mz_precision by label — full + GT overlay ─────────────────────────────
violin_strip(
    ax_F,
    [annex[annex["confidence_label"] == l]["mz_precision"].dropna() for l in CONF_ORDER],
    CONF_ORDER, [CONF_COLOR[l] for l in CONF_ORDER],
    ylabel="m/z precision factor [0–1]",
    title="F  |  m/z precision factor by label\n(full · GT diamonds; floor = 0.5)",
    overlay_groups=[gt_annex[gt_annex["confidence_label"] == l]["mz_precision"].dropna()
                    for l in CONF_ORDER],
    overlay_label="GT subset",
    overlay_color="#222",
)
ax_F.set_ylim(0.45, 1.05)
ax_F.axhline(0.5, color="#d62728", linestyle="--", linewidth=1, zorder=2)
ax_F.text(2.48, 0.50, "floor=0.5", va="bottom", ha="right",
          fontsize=7, color="#d62728")

# add pct at floor annotation
pct_floor = (annex[annex["confidence_label"] == "Inconsistent evidence"]["mz_precision"] == 0.5).mean() * 100
ax_F.text(2, 0.48, f"{pct_floor:.0f}% at floor",
          ha="center", va="top", fontsize=7.5, color=CONF_COLOR["Inconsistent evidence"],
          style="italic")

# ── G: support_fraction — full ────────────────────────────────────────────────
violin_strip(
    ax_G,
    [annex[annex["confidence_label"] == l]["support_fraction"].dropna() for l in CONF_ORDER],
    CONF_ORDER, [CONF_COLOR[l] for l in CONF_ORDER],
    ylabel="Support fraction (supporting / total)",
    title="G  |  Structural consensus fraction\n(full 2,916 scans)",
)
ax_G.set_ylim(-0.05, 1.12)

# ── H: support_fraction — GT subset ──────────────────────────────────────────
violin_strip(
    ax_H,
    [gt_annex[gt_annex["confidence_label"] == l]["support_fraction"].dropna() for l in CONF_ORDER],
    CONF_ORDER, [CONF_COLOR[l] for l in CONF_ORDER],
    ylabel="Support fraction",
    title="H  |  Structural consensus fraction\n(GT subset, n=185)",
)
ax_H.set_ylim(-0.05, 1.12)

# ── I: score component heatmap ────────────────────────────────────────────────
comp_labels = list(SCORE_COMPONENTS.keys())
im = ax_I.imshow(comp_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, ax=ax_I, label="Median [0–1]", fraction=0.044, pad=0.04)

for ri in range(len(CAT_LABELS)):
    for ci in range(len(comp_labels)):
        v = comp_matrix[ri, ci]
        txt = f"{v:.2f}" if not np.isnan(v) else "—"
        color = "white" if v < 0.35 or v > 0.80 else "black"
        ax_I.text(ci, ri, txt, ha="center", va="center", fontsize=7.2, color=color)

ax_I.set_xticks(range(len(comp_labels)))
ax_I.set_xticklabels(comp_labels, fontsize=7.5, rotation=30, ha="right")
short_cat = [l.replace(" (n=1)", "").replace("(cap)", "(cap)\n")
              .replace("consensus", "cons.").replace("hits,", "hits,\n")
              .replace("2–9", "2–9") for l in CAT_LABELS]
ax_I.set_yticks(range(len(CAT_LABELS)))
ax_I.set_yticklabels(short_cat, fontsize=7.2)
ax_I.set_title("I  |  Median score components\nper Inconsistent category (full)",
               fontsize=9.5, fontweight="bold", loc="left")

# highlight M/Z precision column (last column) as the dominant bottleneck
for ri in range(len(CAT_LABELS)):
    ax_I.add_patch(mpatches.Rectangle(
        (len(comp_labels) - 1 - 0.5, ri - 0.5), 1, 1,
        fill=False, edgecolor="#333", linewidth=1.8, zorder=5,
    ))

fig.suptitle(
    "Library match count, structural consensus & m/z precision × confidence label\n"
    "TopKonfidence · MSMLS library · task 384d5ca63dae",
    fontsize=12.5, fontweight="bold", y=1.005,
)

plt.savefig(OUT_STATIC,                        bbox_inches="tight", dpi=180)
plt.savefig(OUT_STATIC.replace(".svg", ".png"), bbox_inches="tight", dpi=180)
print(f"\nSaved static → {OUT_STATIC}")

# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE HTML
# ══════════════════════════════════════════════════════════════════════════════
def plotly_violin(data_full, data_gt, label, color, row, col, fig, showlegend, ylabel=""):
    for d, name, opacity, points in [
        (data_full, label.replace(" evidence", " ev."), 0.65, "all"),
        (data_gt,  f"{label.replace(' evidence', ' ev.')} (GT)", 0.90, "all"),
    ]:
        is_gt = "(GT)" in name
        fig.add_trace(
            go.Violin(
                y=d.tolist(), name=name,
                box_visible=True, meanline_visible=True,
                fillcolor=color if not is_gt else "rgba(0,0,0,0)",
                line_color=color,
                opacity=opacity,
                points="all" if is_gt else "outliers",
                pointpos=0, jitter=0.3,
                marker=dict(
                    size=5 if is_gt else 3,
                    opacity=0.85 if is_gt else 0.3,
                    color=color,
                    symbol="diamond" if is_gt else "circle",
                ),
                legendgroup=label + ("_gt" if is_gt else ""),
                showlegend=showlegend and (not is_gt),
            ),
            row=row, col=col,
        )

fig2 = make_subplots(
    rows=3, cols=3,
    subplot_titles=[
        "A — Match count (all 2,916)", "B — Match count (GT 185)", "C — Inconsistent breakdown",
        "D — PPM error by label", "E — PPM error, Inconsistent categories", "F — m/z precision by label",
        "G — Support fraction (all)", "H — Support fraction (GT)", "I — Score components (Inconsistent)",
    ],
    specs=[
        [{"type": "violin"}, {"type": "violin"}, {"type": "bar"}],
        [{"type": "violin"}, {"type": "violin"}, {"type": "violin"}],
        [{"type": "violin"}, {"type": "violin"}, {"type": "heatmap"}],
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

# Panels A, B, G, H — violins (match count + support_fraction)
for ci, lbl in enumerate(CONF_ORDER):
    color = CONF_COLOR[lbl]
    sl = ci == 0
    plotly_violin(
        annex[annex["confidence_label"]==lbl]["total_matches"],
        gt_annex[gt_annex["confidence_label"]==lbl]["total_matches"],
        lbl, color, 1, 1, fig2, sl)
    plotly_violin(
        gt_annex[gt_annex["confidence_label"]==lbl]["total_matches"],
        gt_annex[gt_annex["confidence_label"]==lbl]["total_matches"],
        lbl, color, 1, 2, fig2, False)
    plotly_violin(
        annex[annex["confidence_label"]==lbl]["support_fraction"].dropna(),
        gt_annex[gt_annex["confidence_label"]==lbl]["support_fraction"].dropna(),
        lbl, color, 3, 1, fig2, False)
    plotly_violin(
        gt_annex[gt_annex["confidence_label"]==lbl]["support_fraction"].dropna(),
        gt_annex[gt_annex["confidence_label"]==lbl]["support_fraction"].dropna(),
        lbl, color, 3, 2, fig2, False)
    plotly_violin(
        annex[annex["confidence_label"]==lbl]["median_abs_ppm_error"].dropna(),
        gt_annex[gt_annex["confidence_label"]==lbl]["median_abs_ppm_error"].dropna(),
        lbl, color, 2, 1, fig2, False)
    plotly_violin(
        annex[annex["confidence_label"]==lbl]["mz_precision"].dropna(),
        gt_annex[gt_annex["confidence_label"]==lbl]["mz_precision"].dropna(),
        lbl, color, 2, 3, fig2, False)

# Panel C — stacked horizontal bars
for ds_idx, (ds_name, counts, total) in enumerate([
    ("Full (n=1,100)", counts_full, len(incon_full)),
    ("GT (n=33)",      counts_gt,   len(incon_gt)),
]):
    for lbl, n, color in zip(CAT_LABELS, counts, CAT_COLORS):
        pct = n / total * 100 if total else 0
        fig2.add_trace(
            go.Bar(
                name=lbl, x=[pct], y=[ds_name],
                orientation="h", marker_color=color,
                text=f"{n} ({pct:.0f}%)",
                textposition="inside", insidetextanchor="middle",
                showlegend=(ds_idx == 0),
                legendgroup=lbl + "_cat",
                hovertemplate=f"<b>{lbl}</b><br>{ds_name}: {n} ({pct:.1f}%)<extra></extra>",
            ),
            row=1, col=3,
        )

# Panel E — PPM by Inconsistent category
for lbl, mask, color in zip(CAT_LABELS, masks_full, CAT_COLORS):
    d = incon_full[mask]["median_abs_ppm_error"].dropna()
    fig2.add_trace(
        go.Violin(
            y=d.tolist(), name=lbl,
            box_visible=True, meanline_visible=True,
            fillcolor=color, line_color=color, opacity=0.7,
            points="outliers", marker=dict(size=3, opacity=0.4, color=color),
            legendgroup=lbl + "_e", showlegend=False,
            hovertemplate=f"<b>{lbl}</b><br>PPM: %{{y:.2f}}<extra></extra>",
        ),
        row=2, col=2,
    )

# Panel I — heatmap
fig2.add_trace(
    go.Heatmap(
        z=comp_matrix.tolist(), x=comp_labels,
        y=[l.replace(" (n=1)", "").replace("(cap), ", "(cap)\n") for l in CAT_LABELS],
        colorscale="RdYlGn", zmin=0, zmax=1,
        text=[[f"{comp_matrix[i,j]:.2f}" if not np.isnan(comp_matrix[i,j]) else "—"
               for j in range(len(comp_labels))]
              for i in range(len(CAT_LABELS))],
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Median", x=1.01, len=0.28, y=0.13),
    ),
    row=3, col=3,
)

# axis labels
fig2.update_yaxes(title_text="Total matches", row=1, col=1)
fig2.update_yaxes(title_text="Total matches", row=1, col=2)
fig2.update_xaxes(title_text="% of Inconsistent evidence scans", row=1, col=3)
fig2.update_yaxes(title_text="Median |PPM error|", row=2, col=1)
fig2.update_yaxes(title_text="Median |PPM error|", row=2, col=2)
fig2.update_yaxes(title_text="m/z precision [0–1]", row=2, col=3)
fig2.update_yaxes(title_text="Support fraction", row=3, col=1)
fig2.update_yaxes(title_text="Support fraction", row=3, col=2)

fig2.update_layout(
    barmode="stack",
    violinmode="overlay",
    title=dict(
        text="Library match count, consensus & m/z precision × confidence label<br>"
             "<sup>TopKonfidence · MSMLS library · task 384d5ca63dae</sup>",
        font=dict(size=15),
    ),
    height=1100,
    legend=dict(x=1.04, y=0.98, xanchor="left", font=dict(size=9)),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

with open(OUT_HTML, "w") as f:
    f.write("<html><head><meta charset='utf-8'></head><body>")
    f.write(po.plot(fig2, include_plotlyjs="cdn", output_type="div"))
    f.write("</body></html>")

print(f"Saved interactive → {OUT_HTML}")
