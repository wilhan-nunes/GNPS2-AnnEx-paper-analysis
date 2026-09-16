"""Correlation structure among the scoring components that make up the
per-scan consistency score.

Inputs
------
b22ff1bc41624e29a28cf6e12fe17ad5_result.csv   raw GNPS2 library match results
temp/mgf_cache/b22ff1bc41624e29a28cf6e12fe17ad5.mgf   cached query MGF (peak counts / entropy)

Scores are recomputed live via bin._compute_scan_level_confidence, so the
figure always reflects the current scoring formula rather than a stale export.

Outputs
-------
fig_component_correlations.svg   heatmap of pairwise correlations + component vs score scatter
correlation_analysis_stats.txt   full correlation matrix and component/score correlations
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
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

# Scoring components, in the order they contribute to the final score.
COMPONENTS = {
    "support_fraction": "Structure agreement",
    "tanimoto_score": "Tanimoto",
    "cosine_consistency": "Cosine consistency",
    "max_cosine": "Max cosine",
    "shared_peaks_score": "Shared peaks",
    "library_diversity_score": "Library diversity",
}


def main():
    from bin.scorer import _compute_scan_level_confidence

    raw = pd.read_csv(CSV)
    scores = _compute_scan_level_confidence(raw, task_id=TASK_ID, workflow_choice="fbmn")

    cols = list(COMPONENTS.keys()) + ["confidence_score"]
    df = scores[cols].apply(pd.to_numeric, errors="coerce").dropna()
    labels = list(COMPONENTS.values()) + ["Consistency score"]

    corr = df.corr(method="spearman")
    corr.index = labels
    corr.columns = labels

    # ---- figure: correlation heatmap + two representative scatters -------
    fig = plt.figure(figsize=(14.3, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.5, top=0.86)
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_s1 = fig.add_subplot(gs[0, 1])
    ax_s2 = fig.add_subplot(gs[0, 2])

    n = len(labels)
    im = ax_hm.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax_hm.set_xticks(range(n))
    ax_hm.set_yticks(range(n))
    ax_hm.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_hm.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr.to_numpy()[i, j]
            txt_color = "white" if abs(v) > 0.55 else INK
            ax_hm.text(j, i, f"{v:.2f}", ha="center", va="center",
                       fontsize=7.5, color=txt_color)
    ax_hm.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_hm.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_hm.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax_hm.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=9, color=INK_2)
    cbar.ax.tick_params(labelsize=8)
    ax_hm.text(-0.02, 1.06, "a", transform=ax_hm.transAxes, fontsize=13,
               fontweight="bold", ha="left", va="bottom")

    # Pick the two components most correlated (by |rho|) with the final score
    # for the illustrative scatter panels, excluding the score itself.
    with_score = corr["Consistency score"].drop("Consistency score").abs().sort_values(ascending=False)
    top_two = with_score.index[:2].tolist()
    inv_labels = {v: k for k, v in COMPONENTS.items()}

    for ax, label, panel_letter in zip((ax_s1, ax_s2), top_two, ("b", "c")):
        col = inv_labels[label]
        x = df[col].to_numpy()
        y = df["confidence_score"].to_numpy()
        ax.scatter(x, y, s=14, alpha=0.35, color=ACCENT, edgecolor="none")
        rho, p = stats.spearmanr(x, y)
        lo = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, lo.intercept + lo.slope * xs, color=INK, linewidth=1.8, zorder=5)
        ax.set_xlabel(label)
        ax.set_ylabel("Consistency score")
        ax.text(0.03, 0.03, f"ρ = {rho:.2f}", transform=ax.transAxes,
                fontsize=9, color=INK_2, ha="left", va="bottom")
        ax.text(-0.02, 1.06, panel_letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", ha="left", va="bottom")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        if panel_letter == "b":
            ax.set_xlim(right=1.0)

    fig.suptitle("Scoring component correlations", fontsize=13, fontweight="bold", x=0.02, y=0.98, ha="left")
    fig.savefig(HERE / "fig_component_correlations.svg", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- stats -------------------------------------------------------------
    out = []
    out.append(f"scans scored: {len(scores)}   used in analysis: {len(df)}")
    out.append("")
    out.append("== Spearman correlation matrix (components + consistency score) ==")
    out.append(corr.round(3).to_string())
    out.append("")
    out.append("== component vs consistency score ==")
    for col, label in COMPONENTS.items():
        rho, p = stats.spearmanr(df[col], df["confidence_score"])
        r, pp = stats.pearsonr(df[col], df["confidence_score"])
        out.append(f"{label:22s}  Spearman rho = {rho:6.3f} (p = {p:.2e})   Pearson r = {r:6.3f} (p = {pp:.2e})")

    text = "\n".join(out)
    (HERE / "correlation_analysis_stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
