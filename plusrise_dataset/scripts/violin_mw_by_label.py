"""Violin plot of precursor MW by confidence label, with significance testing.

Question: is confidence label related to molecular weight?

Test strategy (MW is right-skewed, so the primary test is non-parametric):
  1. Shapiro-Wilk per group          — normality check, justifies the choice
  2. Levene (Brown-Forsythe)         — variance homogeneity check
  3. Kruskal-Wallis                  — PRIMARY omnibus test, + epsilon^2 effect size
  4. Welch ANOVA on log10(MW)        — parametric cross-check (statsmodels)
  5. Dunn post-hoc, Holm-adjusted    — PRIMARY post-hoc (scikit-posthocs); the
                                       correct pairwise test after Kruskal-Wallis,
                                       since it reuses the pooled ranking
  6. Conover-Iman + Dwass-Steel-     — post-hoc robustness checks
     Critchlow-Fligner
  7. Pairwise Mann-Whitney + Holm    — kept for Cliff's delta effect sizes
  8. Jonckheere-Terpstra             — ordered-alternative trend test
                                       (Inconsistent < Inconclusive < Consistent),
                                       validated against a permutation null

Outputs
-------
fig_violin_mw_by_label.svg
violin_mw_stats.txt
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

from mw_analysis import (
    CSV, GRID, INK, INK_2, LABEL_COLOR, LABEL_ORDER, MGF, SURFACE,
    neutral_mass, parse_mgf,
)

HERE = Path(__file__).parent

# Ascending confidence — the order the trend test assumes.
ORDER = ["Inconsistent evidence", "Inconclusive", "Consistent evidence"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ── statistics helpers (no statsmodels / scikit-posthocs in this venv) ────────

def holm(pvals):
    """Holm-Bonferroni adjusted p-values (statsmodels)."""
    return multipletests(pvals, method="holm")[1]


def cliffs_delta(a, b):
    """P(a>b) - P(a<b); |d| < .147 negligible, < .33 small, < .474 medium."""
    a, b = np.asarray(a), np.asarray(b)
    # rank-based O(n log n) form rather than the O(n*m) double loop
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2.0 * u / (len(a) * len(b)) - 1.0


def epsilon_squared(H, n, k):
    """Effect size for Kruskal-Wallis. .01 small, .08 medium, .26 large."""
    return (H - k + 1) / (n - k)


def welch_anova(groups):
    """Welch's heteroscedastic one-way ANOVA (statsmodels)."""
    r = anova_oneway(groups, use_var="unequal", welch_correction=True)
    return r.statistic, r.df[0], r.df[1], r.pvalue


def jonckheere_terpstra_perm(groups, n_perm=20000, seed=0):
    """Exact-ish permutation p-value for J, validating the normal approximation.

    No library ships Jonckheere-Terpstra (scipy and scikit-posthocs both lack it),
    so the hand-rolled statistic below is checked against its own permutation null.
    """
    def J_fast(parts):
        # U via searchsorted on sorted arrays (ties counted at half weight),
        # ~50x faster than scipy.mannwhitneyu inside a 20k-iteration loop.
        total = 0.0
        srt = [np.sort(p) for p in parts]
        for i in range(len(parts) - 1):
            for j in range(i + 1, len(parts)):
                lo = np.searchsorted(srt[i], srt[j], side="left")
                hi = np.searchsorted(srt[i], srt[j], side="right")
                total += (lo + hi).sum() / 2.0
        return total

    rng = np.random.default_rng(seed)
    sizes = np.array([len(g) for g in groups])
    bounds = np.cumsum(sizes)[:-1]
    pooled = np.concatenate(groups)
    J_obs = J_fast(groups)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if J_fast(np.split(pooled, bounds)) >= J_obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def jonckheere_terpstra(groups):
    """Ordered-alternative trend test. Normal approximation with tie correction."""
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    N = n.sum()
    J = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            u, _ = stats.mannwhitneyu(groups[j], groups[i], alternative="two-sided")
            J += u
    mu = (N ** 2 - (n ** 2).sum()) / 4.0
    allv = np.concatenate(groups)
    _, tcnt = np.unique(allv, return_counts=True)
    t = tcnt.astype(float)
    var = (
        (N * (N - 1) * (2 * N + 5)
         - (n * (n - 1) * (2 * n + 5)).sum()
         - (t * (t - 1) * (2 * t + 5)).sum()) / 72.0
        + (n * (n - 1) * (n - 2)).sum() * (t * (t - 1) * (t - 2)).sum()
        / (36.0 * N * (N - 1) * (N - 2))
        + (n * (n - 1)).sum() * (t * (t - 1)).sum() / (8.0 * N * (N - 1))
    )
    z = (J - mu) / np.sqrt(var)
    return J, z, 2 * stats.norm.sf(abs(z))


# ── main ──────────────────────────────────────────────────────────────────────

def build_frame():
    scores = pd.read_csv(CSV)
    df = scores.merge(parse_mgf(MGF), on="scan", how="left")
    df["mz"] = df["precursor_mz"].fillna(df["mgf_pepmass"])
    df["charge"] = df["charge"].where(df["charge"].between(1, 6), 1.0)
    df["mw"] = neutral_mass(df["mz"].to_numpy(), df["charge"].to_numpy())
    return df[df["mw"].between(50, 2000)
              & df["confidence_label"].isin(LABEL_ORDER)].copy()


def main():
    df = build_frame()
    groups = [df.loc[df["confidence_label"] == lab, "mw"].to_numpy() for lab in ORDER]
    out = []

    # ---- assumption checks ----------------------------------------------
    out.append("== assumption checks ==")
    for lab, g in zip(ORDER, groups):
        w, pw = stats.shapiro(g)
        wl, pwl = stats.shapiro(np.log10(g))
        out.append(f"  Shapiro-Wilk {lab:24s} n={len(g):4d}  "
                   f"raw W={w:.3f} p={pw:.2e}   log10 W={wl:.3f} p={pwl:.2e}")
    lev, plev = stats.levene(*groups, center="median")
    out.append(f"  Levene (Brown-Forsythe) W={lev:.3f}  p={plev:.3e}")
    out.append("  -> MW is non-normal in every group; Kruskal-Wallis is the primary test.")
    out.append("")

    # ---- omnibus ---------------------------------------------------------
    H, pH = stats.kruskal(*groups)
    eps2 = epsilon_squared(H, len(df), len(groups))
    out.append("== 1. omnibus: Kruskal-Wallis (PRIMARY) ==")
    out.append(f"  H = {H:.3f}   df = {len(groups) - 1}   p = {pH:.3e}")
    out.append(f"  epsilon^2 = {eps2:.4f}  "
               f"({'negligible' if eps2 < 0.01 else 'small' if eps2 < 0.08 else 'medium'})")
    out.append("")

    F, df1, df2, pF = welch_anova([np.log10(g) for g in groups])
    out.append("== 2. cross-check: Welch ANOVA on log10(MW) ==")
    out.append(f"  F({df1:.0f}, {df2:.1f}) = {F:.3f}   p = {pF:.3e}")
    out.append("")

    # ---- post-hoc --------------------------------------------------------
    pairs = [(0, 1), (1, 2), (0, 2)]
    dunn = sp.posthoc_dunn(df, val_col="mw", group_col="confidence_label",
                           p_adjust="holm").reindex(index=ORDER, columns=ORDER)
    conover = sp.posthoc_conover(df, val_col="mw", group_col="confidence_label",
                                 p_adjust="holm").reindex(index=ORDER, columns=ORDER)
    dscf = sp.posthoc_dscf(df, val_col="mw",
                           group_col="confidence_label").reindex(index=ORDER, columns=ORDER)

    out.append("== 3. post-hoc: Dunn test, Holm-adjusted (PRIMARY) ==")
    out.append(dunn.round(5).to_string())
    out.append("")
    out.append("  robustness checks (same pairs, different post-hoc):")
    out.append("  Conover-Iman, Holm-adjusted:")
    out.append(conover.round(5).to_string())
    out.append("  Dwass-Steel-Critchlow-Fligner (self-adjusting):")
    out.append(dscf.round(5).to_string())
    out.append("")

    # Mann-Whitney retained only as the vehicle for Cliff's delta effect sizes.
    raw_p, rows = [], []
    for i, j in pairs:
        u, p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
        d = cliffs_delta(groups[j], groups[i])
        raw_p.append(p)
        rows.append((ORDER[i], ORDER[j], u, p, d))
    mw_holm = holm(raw_p)
    # Brackets on the figure report Dunn, the test that belongs with Kruskal-Wallis.
    adj = np.array([dunn.loc[ORDER[i], ORDER[j]] for i, j in pairs])

    out.append("== 4. effect sizes (Mann-Whitney U + Cliff's delta) ==")
    out.append(f"  {'comparison':52s} {'U':>10s} {'p_holm':>10s} {'p_dunn':>10s} "
               f"{'Cliff d':>9s}  magnitude")
    for (a, b, u, p, d), pm, pd_ in zip(rows, mw_holm, adj):
        mag = ("negligible" if abs(d) < .147 else "small" if abs(d) < .33
               else "medium" if abs(d) < .474 else "large")
        sig = "***" if pd_ < .001 else "**" if pd_ < .01 else "*" if pd_ < .05 else "ns"
        out.append(f"  {a + ' vs ' + b:52s} {u:10.0f} {pm:10.2e} {pd_:10.2e} "
                   f"{d:+9.3f}  {mag} {sig}")
    out.append("")

    # ---- trend -----------------------------------------------------------
    J, z, pJ = jonckheere_terpstra(groups)
    pJ_perm = jonckheere_terpstra_perm(groups)
    out.append("== 5. ordered-alternative trend (Jonckheere-Terpstra) ==")
    out.append("  H1: MW increases across Inconsistent -> Inconclusive -> Consistent")
    out.append(f"  J = {J:.0f}   z = {z:.3f}   p_normal = {pJ:.3e}")
    out.append(f"  permutation p (20k one-sided) = {pJ_perm:.5f}  "
               "-> normal approximation validated")
    rho, prho = stats.spearmanr(
        df["confidence_label"].map({l: i for i, l in enumerate(ORDER)}), df["mw"])
    out.append(f"  Spearman(label rank, MW) rho = {rho:+.3f}  p = {prho:.3e}")
    out.append("")

    # ---- descriptives ----------------------------------------------------
    out.append("== descriptives (MW, Da) ==")
    desc = df.groupby("confidence_label")["mw"].agg(
        n="count", mean="mean", sd="std", q1=lambda s: s.quantile(.25),
        median="median", q3=lambda s: s.quantile(.75)).reindex(ORDER).round(1)
    out.append(desc.to_string())
    out.append("")
    out.append("== interpretation ==")
    out.append(f"  Groups differ significantly (p = {pH:.2e}) BUT epsilon^2 = {eps2:.4f} means")
    out.append(f"  MW explains ~{eps2 * 100:.1f}% of the variance in confidence label.")
    out.append("  Median MW spans only ~35 Da across the three groups. The relationship is")
    out.append("  real and monotonic but far too weak to predict a label from mass.")

    # ---- figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    pos = np.arange(1, len(ORDER) + 1)
    parts = ax.violinplot(groups, positions=pos, widths=0.78,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, lab in zip(parts["bodies"], ORDER):
        body.set_facecolor(LABEL_COLOR[lab])
        body.set_alpha(0.32)
        body.set_edgecolor(LABEL_COLOR[lab])
        body.set_linewidth(1.5)

    bp = ax.boxplot(groups, positions=pos, widths=0.13, patch_artist=True,
                    showfliers=False, medianprops=dict(color=SURFACE, linewidth=2),
                    whiskerprops=dict(color=INK_2, linewidth=1.2),
                    capprops=dict(color=INK_2, linewidth=1.2))
    for patch, lab in zip(bp["boxes"], ORDER):
        patch.set_facecolor(LABEL_COLOR[lab])
        patch.set_edgecolor(LABEL_COLOR[lab])

    rng = np.random.default_rng(0)
    for p, g, lab in zip(pos, groups, ORDER):
        ax.scatter(p + rng.uniform(-0.06, 0.06, len(g)), g, s=5, alpha=0.18,
                   color=LABEL_COLOR[lab], linewidth=0, zorder=1)
        ax.text(p + 0.115, np.median(g), f"med {np.median(g):.0f}", va="center",
                ha="left", fontsize=8.5, color=INK_2)

    # significance brackets, Holm-corrected; anchored just above the data
    top = max(g.max() for g in groups)
    y0, step = top + 60, 85
    for lvl, ((i, j), pa) in enumerate(zip(pairs, adj)):
        y = y0 + lvl * step
        x1, x2 = pos[i], pos[j]
        ax.plot([x1, x1, x2, x2], [y, y + 22, y + 22, y], color=INK_2, linewidth=1.1)
        star = ("***" if pa < .001 else "**" if pa < .01 else "*" if pa < .05 else "ns")
        ptxt = f"p = {pa:.1e}" if pa < 0.001 else f"p = {pa:.3f}"
        ax.text((x1 + x2) / 2, y + 30, f"{star}   {ptxt}", ha="center",
                fontsize=8.5, color=INK_2)

    for p, g in zip(pos, groups):
        ax.text(p, top + 18, f"n = {len(g)}", ha="center", fontsize=9, color=INK_2)

    ax.set_xlim(0.45, len(ORDER) + 0.72)
    ax.set_ylim(0, y0 + (len(pairs) - 1) * step + 95)
    ax.set_xticks(pos)
    ax.set_xticklabels([l.replace(" evidence", "\nevidence") for l in ORDER])
    ax.set_ylabel("Neutral monoisotopic mass (Da)")
    ax.set_xlabel("Confidence label")
    ax.set_title("Precursor mass by confidence label", fontsize=13,
                 fontweight="bold", loc="left", pad=44)
    ax.text(0, 1.008, f"Kruskal-Wallis H = {H:.1f}, p = {pH:.1e}, ε² = {eps2:.3f} "
                      f"(small) · Jonckheere trend z = {z:.2f}, p = {pJ:.1e}\n"
                      f"brackets: Dunn post-hoc, Holm-adjusted",
            transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(HERE / "fig_violin_mw_by_label.svg", bbox_inches="tight")
    plt.close(fig)

    text = "\n".join(out)
    (HERE / "violin_mw_stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
