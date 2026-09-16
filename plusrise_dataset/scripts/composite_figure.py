"""Composite multi-panel figure for the PlusRise scoring analysis.

Panels (reusing the exact data builders + drawing logic of the standalone scripts):
  (a) confidence-label distribution + median-score trend across spectral-entropy bins
      -> from entropy_analysis.build_frame / stacked_label_panel
  (b) precursor MW by confidence label (violin + box + Dunn brackets)
      -> from violin_mw_by_label.build_frame + its stats
  (c) MW vs spectral entropy (the left panel of fig_mw_vs_entropy)

The four standalone figures already in this folder are left untouched; this only
adds fig_composite.svg.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

import entropy_analysis as ent
import violin_mw_by_label as vio
from mw_analysis import GRID, INK, INK_2, LABEL_COLOR, LABEL_ORDER, SURFACE

HERE = Path(__file__).parent

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
})


def panel_letter(ax, letter, dx=-0.09, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="bottom", ha="left", color=INK)


# ── panel a: entropy vs confidence ────────────────────────────────────────────

def draw_entropy_panel(ax_bars, ax_line, df):
    df = df.copy()
    df["S_q"] = pd.qcut(df["S"], 6, duplicates="drop")
    q_order = list(df["S_q"].cat.categories)
    q_labels = [f"{c.left:.2f}–{c.right:.2f}" for c in q_order]

    ct, _ = ent.stacked_label_panel(ax_bars, df, "S_q", q_order)
    ax_bars.set_ylabel("Share of scans (%)")
    ax_bars.set_title("Confidence label distribution by spectral entropy",
                      fontsize=12, fontweight="bold", loc="left", pad=24)
    ax_bars.legend(frameon=False, ncol=3, loc="lower left",
                   bbox_to_anchor=(0, 1.005), fontsize=8.5)
    ax_bars.set_xticks(np.arange(len(q_order)))
    ax_bars.set_xticklabels([])

    x = np.arange(len(q_order))
    g = df.groupby("S_q", observed=True)["confidence_score"]
    med = g.median().reindex(q_order)
    q1 = g.quantile(.25).reindex(q_order)
    q3 = g.quantile(.75).reindex(q_order)
    ax_line.fill_between(x, q1, q3, color="#2a78d6", alpha=0.16, linewidth=0)
    ax_line.plot(x, med, color="#2a78d6", linewidth=2, marker="o", markersize=6.5,
                 markeredgecolor=SURFACE, markeredgewidth=1.5,
                 label="median score (IQR band)")
    ax_line.axhline(80, color=INK_2, linewidth=1, linestyle=":", zorder=0)
    ax_line.set_ylabel("Confidence score")
    ax_line.set_xlabel("Spectral entropy S (nats), equal-count bins")
    ax_line.set_xticks(x)
    ax_line.set_xticklabels(q_labels)
    ax_line.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax_line.grid(axis="y", color=GRID, linewidth=0.8)
    ax_line.set_axisbelow(True)


# ── panel b: violin of MW by confidence label ─────────────────────────────────

def draw_violin_panel(ax, df):
    groups = [df.loc[df["confidence_label"] == lab, "mw"].to_numpy() for lab in vio.ORDER]
    pairs = [(0, 1), (1, 2), (0, 2)]
    H, pH = stats.kruskal(*groups)
    eps2 = vio.epsilon_squared(H, len(df), len(groups))
    z = vio.jonckheere_terpstra(groups)[1]
    dunn = sp.posthoc_dunn(df, val_col="mw", group_col="confidence_label",
                           p_adjust="holm").reindex(index=vio.ORDER, columns=vio.ORDER)
    adj = np.array([dunn.loc[vio.ORDER[i], vio.ORDER[j]] for i, j in pairs])

    pos = np.arange(1, len(vio.ORDER) + 1)
    parts = ax.violinplot(groups, positions=pos, widths=0.78,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, lab in zip(parts["bodies"], vio.ORDER):
        body.set_facecolor(LABEL_COLOR[lab])
        body.set_alpha(0.32)
        body.set_edgecolor(LABEL_COLOR[lab])
        body.set_linewidth(1.5)

    bp = ax.boxplot(groups, positions=pos, widths=0.13, patch_artist=True,
                    showfliers=False, medianprops=dict(color=SURFACE, linewidth=2),
                    whiskerprops=dict(color=INK_2, linewidth=1.2),
                    capprops=dict(color=INK_2, linewidth=1.2))
    for patch, lab in zip(bp["boxes"], vio.ORDER):
        patch.set_facecolor(LABEL_COLOR[lab])
        patch.set_edgecolor(LABEL_COLOR[lab])

    rng = np.random.default_rng(0)
    for p, gg, lab in zip(pos, groups, vio.ORDER):
        ax.scatter(p + rng.uniform(-0.06, 0.06, len(gg)), gg, s=4, alpha=0.16,
                   color=LABEL_COLOR[lab], linewidth=0, zorder=1)
        ax.text(p + 0.11, np.median(gg), f"med {np.median(gg):.0f}", va="center",
                ha="left", fontsize=8, color=INK_2)

    top = max(gg.max() for gg in groups)
    y0, step = top + 60, 85
    for lvl, ((i, j), pa) in enumerate(zip(pairs, adj)):
        y = y0 + lvl * step
        x1, x2 = pos[i], pos[j]
        ax.plot([x1, x1, x2, x2], [y, y + 22, y + 22, y], color=INK_2, linewidth=1.1)
        star = ("***" if pa < .001 else "**" if pa < .01 else "*" if pa < .05 else "ns")
        ptxt = f"p = {pa:.1e}" if pa < 0.001 else f"p = {pa:.3f}"
        ax.text((x1 + x2) / 2, y + 30, f"{star}   {ptxt}", ha="center",
                fontsize=8, color=INK_2)
    for p, gg in zip(pos, groups):
        ax.text(p, top + 18, f"n = {len(gg)}", ha="center", fontsize=8.5, color=INK_2)

    ax.set_xlim(0.45, len(vio.ORDER) + 0.72)
    ax.set_ylim(0, y0 + (len(pairs) - 1) * step + 95)
    ax.set_xticks(pos)
    ax.set_xticklabels([l.replace(" evidence", "\nevidence") for l in vio.ORDER])
    ax.set_ylabel("Neutral monoisotopic mass (Da)")
    ax.set_xlabel("Confidence label")
    ax.set_title("Precursor mass by confidence label", fontsize=12,
                 fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.008, f"Kruskal-Wallis H = {H:.1f}, p = {pH:.1e}, ε² = {eps2:.3f} "
                      f"(small) · Jonckheere z = {z:.2f}\nbrackets: Dunn post-hoc, "
                      "Holm-adjusted",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ── panel c: MW vs spectral entropy ───────────────────────────────────────────

def draw_mw_entropy_panel(ax, df):
    edges = [0, 150, 200, 250, 300, 350, 400, 500, 600, 10000]
    mb = pd.cut(df["mw"], bins=edges, right=False)
    for lab in LABEL_ORDER:
        s = df[df["confidence_label"] == lab]
        ax.scatter(s["mw"], s["S"], s=20, alpha=0.5, color=LABEL_COLOR[lab],
                   edgecolor=SURFACE, linewidth=0.4, label=lab)
    lo = stats.linregress(df["mw"], df["S"])
    xs = np.linspace(df["mw"].min(), df["mw"].max(), 100)
    ax.plot(xs, lo.intercept + lo.slope * xs, color=INK, linewidth=2,
            linestyle="--", zorder=5, label="linear fit")
    bmed = df.groupby(mb, observed=True)["S"].median()
    bx = df.groupby(mb, observed=True)["mw"].median()
    ax.plot(bx, bmed, color=INK, linewidth=2.5, marker="o", markersize=7.5,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=6,
            label="median per MW bin")
    r_p, p_p = stats.pearsonr(df["mw"], df["S"])
    r_s, p_s = stats.spearmanr(df["mw"], df["S"])
    ax.text(0.02, 0.97, f"Pearson r = {r_p:.2f} (p={p_p:.1e})\n"
                        f"Spearman ρ = {r_s:.2f} (p={p_s:.1e})\n"
                        f"R² = {lo.rvalue ** 2:.3f}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=GRID))
    ax.set_xlabel("Neutral monoisotopic mass (Da)")
    ax.set_ylabel("Spectral entropy S (nats)")
    ax.set_title("MW vs spectral entropy", fontsize=12, fontweight="bold",
                 loc="left", pad=10)
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.9,
              ncol=1, loc="lower right", fontsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ent_df = ent.build_frame()          # S, eff_peaks, mw, confidence_*, ...
    vio_df = vio.build_frame()          # mw (charge-aware), confidence_*

    fig = plt.figure(figsize=(13, 13.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.1, 2.75], hspace=0.30)
    top = outer[0].subgridspec(2, 1, height_ratios=[2.2, 0.95], hspace=0.12)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.24)

    ax_a_bars = fig.add_subplot(top[0])
    ax_a_line = fig.add_subplot(top[1], sharex=ax_a_bars)
    ax_b = fig.add_subplot(bot[0])
    ax_c = fig.add_subplot(bot[1])

    draw_entropy_panel(ax_a_bars, ax_a_line, ent_df)
    draw_violin_panel(ax_b, vio_df)
    draw_mw_entropy_panel(ax_c, ent_df)

    panel_letter(ax_a_bars, "a", dx=-0.065, dy=1.14)
    panel_letter(ax_b, "b", dx=-0.145, dy=1.10)
    panel_letter(ax_c, "c", dx=-0.135, dy=1.10)

    fig.savefig(HERE / "fig_composite.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_composite.svg")


if __name__ == "__main__":
    main()
