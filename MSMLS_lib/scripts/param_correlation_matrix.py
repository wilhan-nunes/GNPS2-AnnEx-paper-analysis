"""
Publication-quality pairs matrix for the 5 main scoring parameters.
- Diagonal:       per-confidence-label KDE
- Upper triangle: Spearman r (annotated)
- Lower triangle: scatter coloured by confidence label
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import gaussian_kde

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANNEX = os.path.join(BASE, "data", "processing", "annex_summary_384d5ca63dae.csv")

# ── Parameters ────────────────────────────────────────────────────────────────
PARAMS = [
    "adjusted_support_score",
    "tanimoto_score",
    "cosine_consistency",
    "max_cosine",
    "mz_precision",
]
LABELS = [
    "Structure\nagreement",
    "Tanimoto\nsimilarity",
    "Cosine\nconsistency",
    "Max cosine",
    "M/Z\nprecision",
]
WEIGHTS = ["w = 0.40", "w = 0.15", "w = 0.20", "w = 0.15", "multiplier"]

CONF_ORDER  = ["Consistent evidence", "Inconclusive", "Inconsistent evidence"]
CONF_SHORT  = ["Consistent", "Inconclusive", "Inconsistent"]
COLORS      = ["#4daf4a", "#ff7f00", "#e41a1c"]   # green / orange / red (ColorBrewer)
ALPHAS      = [0.22, 0.22, 0.22]
POINT_SIZE  = 4
SCATTER_N   = 600   # downsample per group for lower-triangle scatter

FONT_FAMILY = "sans-serif"
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(ANNEX)
df = df[PARAMS + ["confidence_label"]].dropna()
groups = {lab: df[df["confidence_label"] == lab] for lab in CONF_ORDER}

# ── Figure ────────────────────────────────────────────────────────────────────
N = len(PARAMS)
FIG_SIZE = 7.2   # inches — fits a typical journal single-column (7–8 in)
fig, axes = plt.subplots(N, N, figsize=(FIG_SIZE, FIG_SIZE))
fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.12,
                    wspace=0.08, hspace=0.08)

rng = np.random.default_rng(42)

for row in range(N):
    for col in range(N):
        ax = axes[row, col]
        xparam, yparam = PARAMS[col], PARAMS[row]

        # ── Diagonal: per-label KDE ──────────────────────────────────────────
        if row == col:
            for lab, color in zip(CONF_ORDER, COLORS):
                vals = groups[lab][xparam].values
                x_grid = np.linspace(vals.min() - 0.05, vals.max() + 0.05, 300)
                try:
                    kde = gaussian_kde(vals, bw_method=0.25)
                    ax.plot(x_grid, kde(x_grid), color=color, lw=1.4)
                    ax.fill_between(x_grid, kde(x_grid), alpha=0.18, color=color)
                except Exception:
                    pass
            ax.set_xlim(-0.05, 1.05)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)

        # ── Upper triangle: Spearman r ───────────────────────────────────────
        elif col > row:
            ax.set_axis_off()
            xv = df[xparam].values
            yv = df[yparam].values
            r, pval = stats.spearmanr(xv, yv)
            # colour-code r (blue→red diverging)
            norm_r  = (r + 1) / 2        # 0…1
            r_color = plt.cm.RdBu_r(norm_r)
            fontsize_r = 9 + 5 * abs(r)  # scale with magnitude

            ax.text(0.5, 0.55, f"ρ = {r:+.2f}",
                    ha="center", va="center",
                    fontsize=fontsize_r, fontweight="bold",
                    color=r_color, transform=ax.transAxes)
            # significance stars
            if pval < 0.001:
                stars = "***"
            elif pval < 0.01:
                stars = "**"
            elif pval < 0.05:
                stars = "*"
            else:
                stars = "n.s."
            ax.text(0.5, 0.30, stars, ha="center", va="center",
                    fontsize=7, color="0.45", transform=ax.transAxes)

        # ── Lower triangle: scatter ──────────────────────────────────────────
        else:
            for lab, color, alpha in zip(CONF_ORDER, COLORS, ALPHAS):
                grp = groups[lab]
                n = min(SCATTER_N, len(grp))
                idx = rng.choice(len(grp), size=n, replace=False)
                ax.scatter(grp[xparam].values[idx],
                           grp[yparam].values[idx],
                           s=POINT_SIZE, color=color, alpha=alpha,
                           linewidths=0, rasterized=True)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

        # ── Tick / label visibility ──────────────────────────────────────────
        if row < N - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(LABELS[col], fontsize=7.5, labelpad=3)
            ax.tick_params(axis="x", labelsize=6.5, length=3)

        if col > 0:
            ax.set_yticklabels([])
        else:
            if row == col:
                ax.set_ylabel(LABELS[row], fontsize=7.5, labelpad=3)
            else:
                ax.set_ylabel(LABELS[row], fontsize=7.5, labelpad=3)
            ax.tick_params(axis="y", labelsize=6.5, length=3)

        # remove top/right spines on scatter/KDE panels
        if row != col or col <= row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

# ── Weight annotations on diagonal ───────────────────────────────────────────
for i, (ax, wtxt) in enumerate(zip(axes.diagonal(), WEIGHTS)):
    ax.text(0.97, 0.97, wtxt, ha="right", va="top",
            fontsize=6, color="0.45", transform=ax.transAxes,
            style="italic")

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=c, label=s, alpha=0.85)
    for c, s in zip(COLORS, CONF_SHORT)
]
fig.legend(handles=patches, loc="upper right",
           bbox_to_anchor=(0.97, 0.99),
           frameon=False, fontsize=7.5,
           handlelength=1.2, handleheight=0.9,
           borderpad=0.4, labelspacing=0.3)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("svg", "png"):
    out = os.path.join(BASE, f"param_correlation_matrix.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")

plt.close(fig)
print("Done.")
