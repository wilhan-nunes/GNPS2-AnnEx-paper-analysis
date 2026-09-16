"""Number of library matches vs consistency score.

Inputs
------
b22ff1bc41624e29a28cf6e12fe17ad5_result.csv          raw GNPS2 library match results
temp/mgf_cache/b22ff1bc41624e29a28cf6e12fe17ad5.mgf  cached query MGF

Scores are recomputed live via bin._compute_scan_level_confidence so the
figure reflects the current scoring formula.

Output: fig_matches_vs_score.svg, matches_vs_score_stats.txt
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

CSV = REPO_ROOT / "b22ff1bc41624e29a28cf6e12fe17ad5_result.csv"
TASK_ID = "b22ff1bc41624e29a28cf6e12fe17ad5"

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdbd6"
SURFACE = "#ffffff"
ACCENT = "#2a78d6"

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    from bin.scorer import _compute_scan_level_confidence

    raw = pd.read_csv(CSV)
    scores = _compute_scan_level_confidence(raw, task_id=TASK_ID, workflow_choice="fbmn")

    df = scores[["total_matches", "confidence_score"]].dropna().copy()
    x = df["total_matches"].to_numpy(dtype=float)
    y = df["confidence_score"].to_numpy(dtype=float)

    rho, p_rho = stats.spearmanr(x, y)
    lo = stats.linregress(x, y)

    # total_matches is a small integer (1-10, capped); group into one violin
    # per match count so the distribution shape/spread at each n is visible
    # instead of an overplotted point cloud.
    match_counts = sorted(df["total_matches"].unique())
    groups = [df.loc[df["total_matches"] == n, "confidence_score"].to_numpy() for n in match_counts]
    pos = np.arange(1, len(match_counts) + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    parts = ax.violinplot(groups, positions=pos, widths=0.78,
                          showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(ACCENT)
        body.set_alpha(0.30)
        body.set_edgecolor(ACCENT)
        body.set_linewidth(1.4)

    bp = ax.boxplot(groups, positions=pos, widths=0.13, patch_artist=True,
                    showfliers=False, medianprops=dict(color=SURFACE, linewidth=2),
                    whiskerprops=dict(color=INK_2, linewidth=1.2),
                    capprops=dict(color=INK_2, linewidth=1.2))
    for patch in bp["boxes"]:
        patch.set_facecolor(ACCENT)
        patch.set_edgecolor(ACCENT)

    rng = np.random.default_rng(0)
    for p, g in zip(pos, groups):
        ax.scatter(p + rng.uniform(-0.06, 0.06, len(g)), g, s=5, alpha=0.18,
                   color=ACCENT, linewidth=0, zorder=1)

    # OLS trend across the (unjittered) match-count positions, for reference
    xs = np.linspace(pos.min(), pos.max(), 100)
    lo_pos = stats.linregress(pos, [np.median(g) for g in groups])
    ax.plot(xs, lo_pos.intercept + lo_pos.slope * xs, color=INK, linewidth=1.8,
            linestyle="--", zorder=5, label="trend across medians")

    for p, g in zip(pos, groups):
        ax.text(p, max(gg.max() for gg in groups) + 4, f"n={len(g)}", ha="center",
                fontsize=8, color=INK_2)

    ax.set_xticks(pos)
    ax.set_xticklabels([str(n) for n in match_counts])
    ax.set_xlabel("Library matches (n)")
    ax.set_ylabel("Consistency score")
    ax.set_title("Library matches vs consistency score", fontsize=13,
                 fontweight="bold", loc="left", pad=20)
    ax.text(0.02, 0.02, f"Spearman ρ = {rho:.2f} (p = {p_rho:.1e}, n = {len(df)})",
            transform=ax.transAxes, fontsize=9, color=INK_2, ha="left", va="bottom")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.savefig(HERE.parent / "fig_matches_vs_score.svg", dpi=200, bbox_inches="tight")
    plt.close(fig)

    out = []
    out.append(f"scans analysed: {len(df)}")
    out.append(f"Spearman rho = {rho:.3f}  p = {p_rho:.3e}")
    r_p, p_p = stats.pearsonr(x, y)
    out.append(f"Pearson  r   = {r_p:.3f}  p = {p_p:.3e}")
    out.append(f"linear fit: score = {lo.slope:.3f} * n_matches + {lo.intercept:.2f}  (R^2 = {lo.rvalue**2:.3f})")
    out.append("")
    out.append("median / mean consistency score per match count:")
    out.append(df.groupby("total_matches")["confidence_score"].agg(["count", "mean", "median"]).round(2).to_string())

    text = "\n".join(out)
    (HERE.parent / "matches_vs_score_stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
