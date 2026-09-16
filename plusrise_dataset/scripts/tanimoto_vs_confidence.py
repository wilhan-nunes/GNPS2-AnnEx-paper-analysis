"""Tanimoto score vs confidence score, restyled to match the other panels.

Recreation of the "Tanimoto Score vs Confidence" screenshot:
  x = tanimoto_score        y = confidence_score
  marker SIZE  = number of library hits (total_matches)
  marker COLOR = number of library hits (sequential blue ramp + colorbar)
  marker SHAPE = the basis of the Tanimoto value (tanimoto_source):
                 circle  = measured ECFP4 (>= 2 unique structures)
                 diamond = single_structure (one unique SMILES -> Tanimoto = 1.0
                           by definition, not by measurement)
                 X       = NO structure available at all (missing == total hits);
                           Tanimoto falls back to the support fraction
  red ring     = scan where >= 50% of hits are missing a structure
                 (the X points are the 100%-missing extreme of this)
  red dashed line + band = OLS fit with 95% mean-response CI

Note: the original screenshot showed r = 0.72; it predates the current scoring
formula (Bayesian shrinkage / Gaussian PPM penalty). On this CSV the fit is
weaker — the value printed on the figure is computed from the plotted data.

Output: fig_tanimoto_vs_confidence.svg
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy import stats

from mw_analysis import CSV, GRID, INK, INK_2, SURFACE

HERE = Path(__file__).parent

RED = "#e34948"           # fit line + missing-structure rings (status accent)
HIT_CMAP = LinearSegmentedColormap.from_list(
    "hits_blue", ["#bcd4f2", "#2a78d6", "#0b3b73"])   # sequential, single hue

# marker shape per tanimoto_source; last field is a size bump so the highlighted
# "no structure at all" points read larger than their hit count alone would give.
SOURCE_STYLE = {
    "tanimoto_wcrs":                ("o", "Measured ECFP4 (≥2 structures)", 0),
    "single_structure":             ("D", "Single structure (Tanimoto = 1\nby definition)", 0),
    "structure_agreement_fallback": ("X", "No structure available\n(Tanimoto = support fraction)", 55),
}

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
})


def hit_size(hits):
    """Marker area scales with number of hits (1 -> small, 10 -> large)."""
    return 22 + (np.asarray(hits, float) - 1) * 20  # 22 .. 202


def main():
    df = pd.read_csv(CSV).dropna(subset=["tanimoto_score", "confidence_score"]).copy()
    df["missing_frac"] = df["missing_structures"] / df["total_matches"]
    flagged = df["missing_frac"] >= 0.5
    no_structure = df["missing_structures"] == df["total_matches"]

    lo = stats.linregress(df["tanimoto_score"], df["confidence_score"])
    r = lo.rvalue

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    norm = Normalize(1, 10)

    # Passes: each tanimoto_source (shape) x {plain, flagged}. Flagged (>=50%
    # missing) drawn last with a red ring; the no-structure X points sit on top.
    for src, (marker, _, bump) in SOURCE_STYLE.items():
        shape_mask = df["tanimoto_source"] == src
        for is_flag, edge, lw, al in [
            (False, SURFACE, 0.4, 0.72),
            (True, RED, 1.6, 0.9),
        ]:
            s = df[shape_mask & (flagged if is_flag else ~flagged)]
            if s.empty:
                continue
            z = 7 if marker == "X" else (4 if is_flag else 2)
            ax.scatter(s["tanimoto_score"], s["confidence_score"], marker=marker,
                       s=hit_size(s["total_matches"]) + bump, c=s["total_matches"],
                       cmap=HIT_CMAP, norm=norm, alpha=al,
                       edgecolor=edge, linewidth=lw, zorder=z)

    # OLS fit + 95% CI band for the mean response
    x = df["tanimoto_score"].to_numpy()
    y = df["confidence_score"].to_numpy()
    n = len(x)
    xs = np.linspace(x.min(), x.max(), 200)
    yhat = lo.intercept + lo.slope * xs
    resid = y - (lo.intercept + lo.slope * x)
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))
    sxx = np.sum((x - x.mean()) ** 2)
    tval = stats.t.ppf(0.975, n - 2)
    ci = tval * s_err * np.sqrt(1.0 / n + (xs - x.mean()) ** 2 / sxx)
    ax.fill_between(xs, yhat - ci, yhat + ci, color=RED, alpha=0.14, linewidth=0,
                    zorder=3)
    ax.plot(xs, yhat, color=RED, linewidth=2.2, linestyle="--", zorder=5)

    # ---- colorbar for hit count ----
    sm = plt.cm.ScalarMappable(cmap=HIT_CMAP, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.045)
    cbar.set_label("Library hits (n)", color=INK_2)
    cbar.set_ticks([1, 2, 4, 6, 8, 10])
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.tick_params(color=GRID)

    # ---- shape + ring legend (upper-left, region is empty) ----
    shape_handles = [
        Line2D([0], [0], marker=mk, linestyle="none", markerfacecolor="#7aa9e0",
               markeredgecolor=(RED if mk == "X" else SURFACE),
               markeredgewidth=(1.6 if mk == "X" else 0.6),
               markersize=(10 if mk == "X" else 8.5), label=lbl)
        for mk, lbl, _ in SOURCE_STYLE.values()
    ]
    shape_handles.append(
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#7aa9e0",
               markeredgecolor=RED, markeredgewidth=1.6, markersize=9,
               label="≥ 50% missing structures"))
    leg1 = ax.legend(handles=shape_handles, loc="upper left", frameon=True,
                     facecolor=SURFACE, edgecolor=GRID, framealpha=0.92,
                     fontsize=8.5, labelspacing=1.0, borderpad=0.9,
                     handletextpad=1.0)
    leg1.set_zorder(6)
    ax.add_artist(leg1)

    # ---- size legend (lower-right, sparse region) ----
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#7aa9e0",
               markeredgecolor=SURFACE, markersize=np.sqrt(hit_size(h)),
               label=f"{h} hit" + ("" if h == 1 else "s"))
        for h in (1, 4, 7, 10)
    ]
    leg2 = ax.legend(handles=size_handles, loc="lower right", frameon=True,
                     facecolor=SURFACE, edgecolor=GRID, framealpha=0.92,
                     fontsize=8.5, labelspacing=1.2, borderpad=0.9,
                     handletextpad=1.0, title="Library hits")
    leg2.get_title().set_color(INK_2)
    leg2.set_zorder(6)

    ax.set_xlabel("Tanimoto score")
    ax.set_ylabel("Confidence score")
    ax.set_title("Tanimoto score vs confidence", fontsize=13, fontweight="bold",
                 loc="left", pad=26)
    ax.text(0, 1.012,
            f"OLS r = {r:.2f} (p = {lo.pvalue:.1e}, n = {n}) · size & color = library "
            f"hits · ✕ = no structure available (n = {int(no_structure.sum())})",
            transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.savefig(HERE / "fig_tanimoto_vs_confidence.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_tanimoto_vs_confidence.svg   (r={r:.3f}, n={n}, "
          f"flagged={int(flagged.sum())})")


if __name__ == "__main__":
    main()
