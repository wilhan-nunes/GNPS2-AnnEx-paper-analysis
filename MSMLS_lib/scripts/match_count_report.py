"""
Match count × confidence label analysis.

Key question: does the number of retrieved library matches cause Inconsistent evidence?

Panels:
  A  – total_matches by confidence label (violin + strip)
  B  – support_fraction (supporting/total) by confidence label (violin + strip)
  C  – Breakdown of what actually drives Inconsistent evidence (5 categories)
  D  – Scatter: total_matches vs confidence_score (all scans, coloured by label)
  E  – For each Inconsistent category: which score component is lowest (heatmap)
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
ANNEX_FILE = os.path.join(PROC, "annex_summary_384d5ca63dae.csv")
OUT_STATIC = os.path.join(BASE, "match_count_report.svg")
OUT_HTML   = os.path.join(BASE, "match_count_report.html")

# ── load ───────────────────────────────────────────────────────────────────────
annex = pd.read_csv(ANNEX_FILE)
annex["support_fraction"]    = annex["supporting_matches"] / annex["total_matches"]
annex["disagreeing_matches"] = annex["total_matches"] - annex["supporting_matches"]

CONF_ORDER = ["Consistent evidence", "Inconclusive", "Inconsistent evidence"]
CONF_COLOR = {
    "Consistent evidence":  "#1f77b4",
    "Inconclusive":          "#ff7f0e",
    "Inconsistent evidence": "#d62728",
}
MATCH_CAP = 10   # app retrieves at most 10 matches per scan

# ── categorise Inconsistent evidence causes ────────────────────────────────────
incon = annex[annex["confidence_label"] == "Inconsistent evidence"].copy()

single        = incon["total_matches"] == 1
at_cap        = incon["total_matches"] == MATCH_CAP
few_fragmented  = (~single) & (~at_cap) & (incon["support_fraction"] < 1.0)
few_consensus   = (~single) & (~at_cap) & (incon["support_fraction"] == 1.0)
cap_consensus   = at_cap & (incon["support_fraction"] == 1.0)
cap_fragmented  = at_cap & (incon["support_fraction"] < 1.0)

CAT_LABELS = [
    "Single hit (n=1)",
    "2–9 hits, full consensus",
    "2–9 hits, fragmented",
    "10 hits (cap), full consensus",
    "10 hits (cap), fragmented",
]
CAT_MASKS  = [single, few_consensus, few_fragmented, cap_consensus, cap_fragmented]
CAT_COLORS = ["#e15759", "#f28e2b", "#edc948", "#76b7b2", "#4e79a7"]
cat_counts = [m.sum() for m in CAT_MASKS]

# Assign category column
incon = incon.copy()
incon["cause_cat"] = "Other"
for lbl, mask in zip(CAT_LABELS, CAT_MASKS):
    incon.loc[mask, "cause_cat"] = lbl

print("=== Inconsistent evidence cause breakdown ===")
for lbl, n in zip(CAT_LABELS, cat_counts):
    print(f"  {lbl}: {n}  ({n/len(incon)*100:.1f}%)")

# ── score component medians per category ──────────────────────────────────────
SCORE_COMPONENTS = {
    "Structure\nagreement": "support_fraction",
    "Tanimoto":             "tanimoto_score",
    "Cosine\nconsistency":  "cosine_consistency",
    "Max cosine":           "max_cosine",
    "Shared peaks":         "shared_peaks_score",
    "M/Z precision":        "mz_precision",
}
comp_matrix = np.full((len(CAT_LABELS), len(SCORE_COMPONENTS)), np.nan)
for ri, (lbl, mask) in enumerate(zip(CAT_LABELS, CAT_MASKS)):
    sub = incon[mask]
    for ci, col in enumerate(SCORE_COMPONENTS.values()):
        if col in sub.columns:
            comp_matrix[ri, ci] = sub[col].median()

print("\n=== Median score components per Inconsistent category ===")
print(pd.DataFrame(comp_matrix, index=CAT_LABELS, columns=SCORE_COMPONENTS.keys()).round(3).to_string())

# ── helper: violin + strip ────────────────────────────────────────────────────
rng = np.random.default_rng(42)

def violin_strip(ax, groups, labels, colors, ylabel="", title="",
                 strip_alpha=0.3, strip_s=5, log_scale=False):
    data_clean = [np.array(g, dtype=float) for g in groups]
    data_clean = [d[~np.isnan(d)] for d in data_clean]
    positions  = list(range(len(labels)))

    parts = ax.violinplot(
        [d if len(d) > 1 else np.array([d[0], d[0]]) for d in data_clean],
        positions=positions, widths=0.55, showmedians=True, showextrema=True,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color); pc.set_alpha(0.35); pc.set_edgecolor("none")
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if part in parts:
            parts[part].set_color("#333"); parts[part].set_linewidth(1.2)

    for i, (d, color) in enumerate(zip(data_clean, colors)):
        jitter = rng.uniform(-0.13, 0.13, len(d))
        ax.scatter(i + jitter, d, alpha=strip_alpha, s=strip_s,
                   color=color, zorder=4, linewidths=0)
        if len(d):
            med = np.median(d)
            ax.text(i + 0.30, med, f"{med:.1f}",
                    va="center", ha="left", fontsize=7.5, color="#222")

    if log_scale:
        ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(positions)
    short = [l.replace("evidence", "ev.").replace("Inconsistent", "Inconsist.") for l in labels]
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

# ══════════════════════════════════════════════════════════════════════════════
# STATIC FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 13))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38,
                        height_ratios=[1.05, 1.0])

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[0, 2])
ax_D = fig.add_subplot(gs[1, 0:2])
ax_E = fig.add_subplot(gs[1, 2])

# ── A: total_matches by label ─────────────────────────────────────────────────
violin_strip(
    ax_A,
    [annex[annex["confidence_label"] == l]["total_matches"] for l in CONF_ORDER],
    CONF_ORDER,
    [CONF_COLOR[l] for l in CONF_ORDER],
    ylabel="Total retrieved matches",
    title="A  |  Match count by confidence label\n(capped at 10 per scan)",
)
ax_A.axhline(MATCH_CAP, color="#999", linestyle="--", linewidth=1, zorder=2)
ax_A.text(2.48, MATCH_CAP + 0.05, "cap = 10", va="bottom", ha="right",
          fontsize=7.5, color="#888")

# ── B: support_fraction by label ──────────────────────────────────────────────
violin_strip(
    ax_B,
    [annex[annex["confidence_label"] == l]["support_fraction"] for l in CONF_ORDER],
    CONF_ORDER,
    [CONF_COLOR[l] for l in CONF_ORDER],
    ylabel="Support fraction (supporting / total)",
    title="B  |  Structural consensus fraction\nby confidence label",
)
ax_B.set_ylim(-0.05, 1.12)

# Add annotation: 71% of Inconsistent have fraction == 1.0
pct_perfect = (incon["support_fraction"] == 1.0).mean() * 100
ax_B.text(2, 1.08, f"{pct_perfect:.0f}% have\nfraction=1.0",
          ha="center", va="bottom", fontsize=7.5, color=CONF_COLOR["Inconsistent evidence"],
          style="italic")

# ── C: Inconsistent evidence cause breakdown (horizontal bar) ─────────────────
ax_C.set_title("C  |  What drives 'Inconsistent evidence'?\n(n=1,100 scans)",
               fontsize=10, fontweight="bold", loc="left")
ax_C.axis("off")

total_incon = len(incon)
bar_height  = 0.55
y_pos       = 0.42   # normalized axes coords
# Draw a single stacked bar in axis space
left = 0.0
bar_patches, bar_labels = [], []
for lbl, n, color in zip(CAT_LABELS, cat_counts, CAT_COLORS):
    w = n / total_incon
    ax_C.add_patch(mpatches.Rectangle(
        (left, y_pos - bar_height / 2), w, bar_height,
        transform=ax_C.transAxes,
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3,
    ))
    # label inside if wide enough
    if w > 0.08:
        ax_C.text(left + w / 2, y_pos, f"{n}\n({n/total_incon*100:.0f}%)",
                  transform=ax_C.transAxes,
                  ha="center", va="center", fontsize=8, fontweight="bold",
                  color="white", zorder=4)
    bar_patches.append(mpatches.Patch(color=color, label=f"{lbl}: {n} ({n/total_incon*100:.0f}%)"))
    left += w

ax_C.legend(handles=bar_patches, loc="lower left", bbox_to_anchor=(0, -0.05),
            fontsize=7.5, frameon=True, title="Category", title_fontsize=8,
            ncol=1)

# Annotation: 71% have perfect consensus despite low score
ax_C.text(0.5, 0.90,
          "71% have perfect structural consensus\n"
          "→ low score driven by other metrics",
          transform=ax_C.transAxes,
          ha="center", va="top", fontsize=8.5, color="#444",
          bbox=dict(boxstyle="round,pad=0.4", fc="#fff9f0", ec="#f28e2b", lw=1))

# ── D: Scatter total_matches vs confidence_score ──────────────────────────────
# Jitter x so single-match points don't stack
for lbl in CONF_ORDER:
    sub = annex[annex["confidence_label"] == lbl]
    jitter = rng.uniform(-0.3, 0.3, len(sub))
    ax_D.scatter(sub["total_matches"] + jitter, sub["confidence_score"],
                 alpha=0.2, s=7, color=CONF_COLOR[lbl], label=lbl,
                 linewidths=0, zorder=3)

# Overlay median confidence_score per match count (all scans)
for n_match in range(1, MATCH_CAP + 1):
    sub = annex[annex["total_matches"] == n_match]["confidence_score"]
    if len(sub):
        ax_D.plot(n_match, sub.median(), "D", color="#222",
                  markersize=5, zorder=6, alpha=0.85)

ax_D.plot([], [], "D", color="#222", markersize=5, label="Median per match count")
ax_D.axhline(80, color="#1f77b4", linestyle=":", linewidth=1, alpha=0.7)
ax_D.axhline(55, color="#d62728", linestyle=":", linewidth=1, alpha=0.7)
ax_D.text(10.15, 80, "80", va="center", fontsize=7.5, color="#1f77b4")
ax_D.text(10.15, 55, "55", va="center", fontsize=7.5, color="#d62728")
ax_D.set_xlabel("Total retrieved matches (jittered)", fontsize=9)
ax_D.set_ylabel("Confidence score", fontsize=9)
ax_D.set_title("D  |  Confidence score vs match count (all 2,916 scans)\n"
               "Diamonds = median per match count",
               fontsize=10, fontweight="bold", loc="left")
ax_D.legend(fontsize=8, loc="lower right", frameon=True, markerscale=1.5)
ax_D.set_xticks(range(1, MATCH_CAP + 1))
ax_D.spines[["top", "right"]].set_visible(False)
ax_D.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax_D.set_axisbelow(True)

# ── E: score component heatmap per Inconsistent category ─────────────────────
comp_labels = list(SCORE_COMPONENTS.keys())
im = ax_E.imshow(comp_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, ax=ax_E, label="Median value [0–1]", fraction=0.046, pad=0.04)

for ri in range(len(CAT_LABELS)):
    for ci in range(len(comp_labels)):
        v = comp_matrix[ri, ci]
        txt = f"{v:.2f}" if not np.isnan(v) else "—"
        color = "white" if v < 0.35 or v > 0.80 else "black"
        ax_E.text(ci, ri, txt, ha="center", va="center",
                  fontsize=7.5, color=color)

ax_E.set_xticks(range(len(comp_labels)))
ax_E.set_xticklabels(comp_labels, fontsize=7.5, rotation=30, ha="right")
short_cat = [l.replace("(cap)", "(cap)\n").replace("consensus", "cons.") for l in CAT_LABELS]
ax_E.set_yticks(range(len(CAT_LABELS)))
ax_E.set_yticklabels(short_cat, fontsize=7.5)
ax_E.set_title("E  |  Median score components\nper Inconsistent category",
               fontsize=10, fontweight="bold", loc="left")

fig.suptitle("Retrieved match count × confidence label analysis\n"
             "TopKonfidence · MSMLS library · task 384d5ca63dae",
             fontsize=13, fontweight="bold", y=1.00)

plt.savefig(OUT_STATIC,                        bbox_inches="tight", dpi=180)
plt.savefig(OUT_STATIC.replace(".svg", ".png"), bbox_inches="tight", dpi=180)
print(f"\nSaved static → {OUT_STATIC}")

# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE HTML
# ══════════════════════════════════════════════════════════════════════════════
fig2 = make_subplots(
    rows=2, cols=3,
    subplot_titles=[
        "A — Total matches by label",
        "B — Support fraction by label",
        "C — Inconsistent evidence drivers",
        "D — Confidence score vs match count",
        "E — Score components heatmap (Inconsistent only)",
        "",
    ],
    specs=[
        [{"type": "violin"}, {"type": "violin"}, {"type": "bar"}],
        [{"type": "scatter", "colspan": 2}, None, {"type": "heatmap"}],
    ],
    vertical_spacing=0.20,
    horizontal_spacing=0.09,
    row_heights=[0.46, 0.54],
)

# Panels A & B — violins
def add_violin(fig, row, col, data, label, color, showlegend, hover_suffix=""):
    fig.add_trace(
        go.Violin(
            y=data.tolist(), name=label.replace("evidence", "ev."),
            box_visible=True, meanline_visible=True,
            fillcolor=color, line_color=color,
            opacity=0.7, points="all", pointpos=0, jitter=0.4,
            marker=dict(size=3, opacity=0.35, color=color),
            legendgroup=label, showlegend=showlegend,
            hovertemplate=f"<b>{label}</b><br>%{{y}}{hover_suffix}<extra></extra>",
        ),
        row=row, col=col,
    )

for i, lbl in enumerate(CONF_ORDER):
    color = CONF_COLOR[lbl]
    add_violin(fig2, 1, 1, annex[annex["confidence_label"]==lbl]["total_matches"], lbl, color, i==0, " matches")
    add_violin(fig2, 1, 2, annex[annex["confidence_label"]==lbl]["support_fraction"], lbl, color, False, " support fraction")

fig2.update_yaxes(title_text="Total matches (cap=10)", row=1, col=1)
fig2.update_yaxes(title_text="Supporting / total matches", row=1, col=2)

# Add cap reference line
fig2.add_shape(type="line", x0=-0.5, x1=2.5, y0=10, y1=10,
               line=dict(color="#999", dash="dash", width=1.5), row=1, col=1)

# Panel C — stacked horizontal bar (one bar split into segments)
cum_pct = 0.0
for lbl, n, color in zip(CAT_LABELS, cat_counts, CAT_COLORS):
    pct = n / total_incon * 100
    fig2.add_trace(
        go.Bar(
            name=lbl, x=[pct], y=["Inconsistent\nevidence"],
            orientation="h",
            marker_color=color,
            text=f"{n} ({pct:.0f}%)",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"<b>{lbl}</b><br>{n} scans ({pct:.1f}%)<extra></extra>",
        ),
        row=1, col=3,
    )
fig2.update_layout(barmode="stack")
fig2.update_xaxes(title_text="% of Inconsistent evidence scans", row=1, col=3)

# Panel D — scatter
for lbl in CONF_ORDER:
    sub = annex[annex["confidence_label"] == lbl]
    fig2.add_trace(
        go.Scatter(
            x=(sub["total_matches"] + rng.uniform(-0.3, 0.3, len(sub))).tolist(),
            y=sub["confidence_score"].tolist(),
            mode="markers",
            name=lbl.replace("evidence", "ev."),
            marker=dict(size=4, color=CONF_COLOR[lbl], opacity=0.25),
            legendgroup=lbl, showlegend=False,
            hovertemplate=(
                f"<b>{lbl}</b><br>Matches: %{{x:.0f}}<br>Score: %{{y:.1f}}<extra></extra>"
            ),
        ),
        row=2, col=1,
    )

# Median diamonds
med_x, med_y = [], []
for n_match in range(1, MATCH_CAP + 1):
    sub = annex[annex["total_matches"] == n_match]["confidence_score"]
    if len(sub):
        med_x.append(n_match); med_y.append(sub.median())
fig2.add_trace(
    go.Scatter(x=med_x, y=med_y, mode="markers+lines",
               name="Median per match count",
               marker=dict(symbol="diamond", size=8, color="#222"),
               line=dict(color="#222", width=1.5),
               showlegend=True),
    row=2, col=1,
)

# Threshold lines
for threshold, color, label in [(80, "#1f77b4", "Consistent ≥80"), (55, "#d62728", "Inconsistent <55")]:
    fig2.add_shape(type="line", x0=0.5, x1=10.5, y0=threshold, y1=threshold,
                   line=dict(color=color, dash="dot", width=1.5), row=2, col=1)

fig2.update_xaxes(title_text="Total matches (jittered)", row=2, col=1)
fig2.update_yaxes(title_text="Confidence score", row=2, col=1)

# Panel E — heatmap
hover_heat = [
    [f"<b>{CAT_LABELS[i]}</b><br>{comp_labels[j]}: {comp_matrix[i,j]:.3f}"
     for j in range(len(comp_labels))]
    for i in range(len(CAT_LABELS))
]
fig2.add_trace(
    go.Heatmap(
        z=comp_matrix.tolist(),
        x=comp_labels,
        y=[l.replace("(cap)", "(cap)") for l in CAT_LABELS],
        colorscale="RdYlGn", zmin=0, zmax=1,
        text=[[f"{comp_matrix[i,j]:.2f}" if not np.isnan(comp_matrix[i,j]) else "—"
               for j in range(len(comp_labels))]
              for i in range(len(CAT_LABELS))],
        texttemplate="%{text}",
        hovertext=hover_heat, hoverinfo="text",
        showscale=True,
        colorbar=dict(title="Median [0–1]", x=1.01, len=0.45, y=0.22),
    ),
    row=2, col=3,
)

fig2.update_layout(
    title=dict(
        text="Retrieved match count × confidence label analysis<br>"
             "<sup>TopKonfidence · MSMLS library · task 384d5ca63dae</sup>",
        font=dict(size=15),
    ),
    height=860,
    legend=dict(x=1.04, y=0.80, xanchor="left"),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

with open(OUT_HTML, "w") as f:
    f.write("<html><head><meta charset='utf-8'></head><body>")
    f.write(po.plot(fig2, include_plotlyjs="cdn", output_type="div"))
    f.write("</body></html>")

print(f"Saved interactive → {OUT_HTML}")
