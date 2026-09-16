"""Spectral-entropy view of the MW/confidence analysis.

Companion to mw_analysis.py. Instead of the raw MGF peak count (which counts
noise peaks), this uses ms_entropy spectral entropy S — computed through the
project's own bin/spectral_entropy_v2.spectral_entropy_for_spectrum, so the
preprocessing (clean_spectrum=True, 0.01 base-peak noise floor, 0.02 Da peak
merging, 100-peak cap) matches what the dashboard scores against.

Three complexity measures are compared:
  n_peaks_raw       every peak in the MGF, noise included
  n_peaks_filtered  after the same noise floor + peak cap the scorer applies
  eff_peaks = e^S   perplexity: the "effective" number of peaks carrying signal

Outputs
-------
fig_entropy_vs_confidence.svg   label distribution + score trend across entropy bins
fig_mw_vs_entropy.svg           MW vs entropy / effective peak count
entropy_analysis_stats.txt
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from bin.spectral_entropy_v2 import (  # noqa: E402
    DEFAULT_MAX_PEAK_NUM,
    DEFAULT_MS2_TOLERANCE_DA,
    DEFAULT_NOISE_THRESHOLD,
    spectral_entropy_for_spectrum,
)
from bin.mgf_loader import _parse_mgf  # noqa: E402
from mw_analysis import (  # noqa: E402
    CSV, GRID, INK, INK_2, LABEL_COLOR, LABEL_ORDER, MGF, SURFACE, neutral_mass,
)

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
})

ENTROPY_BANDS = [
    ("noisy / low quality", 0.0, 0.5),
    ("moderate quality", 0.5, 1.0),
    ("clean / good quality", 1.0, 1.75),
    ("very diffuse / uniform", 1.75, np.inf),
]


def build_frame():
    scores = pd.read_csv(CSV)
    spectra = _parse_mgf(MGF)

    rows = []
    for scan_str, peaks in spectra.items():
        if not peaks:
            rows.append({"scan": int(scan_str), "S": np.nan,
                         "n_peaks_raw": 0, "n_peaks_filtered": 0})
            continue
        S = spectral_entropy_for_spectrum(
            peaks,
            ms2_tolerance_da=DEFAULT_MS2_TOLERANCE_DA,
            noise_threshold=DEFAULT_NOISE_THRESHOLD,
            max_peak_num=DEFAULT_MAX_PEAK_NUM,
        )
        base = max(p[1] for p in peaks)
        filt = [p for p in peaks if base > 0 and p[1] >= DEFAULT_NOISE_THRESHOLD * base]
        rows.append({
            "scan": int(scan_str),
            "S": S,
            "n_peaks_raw": len(peaks),
            "n_peaks_filtered": min(len(filt), DEFAULT_MAX_PEAK_NUM),
        })

    df = scores.merge(pd.DataFrame(rows), on="scan", how="left")
    df["eff_peaks"] = np.exp(df["S"])
    df["mw"] = neutral_mass(df["precursor_mz"].to_numpy(),
                            np.ones(len(df)))  # 99.4% singly charged; see stats file
    df = df[df["mw"].between(50, 2000)
            & df["confidence_label"].isin(LABEL_ORDER)
            & df["S"].notna()].copy()
    df["entropy_band"] = pd.cut(
        df["S"], bins=[b[1] for b in ENTROPY_BANDS] + [np.inf],
        labels=[b[0] for b in ENTROPY_BANDS], right=False,
    )
    return df


def stacked_label_panel(ax, df, group_col, order):
    ct = pd.crosstab(df[group_col], df["confidence_label"]).reindex(
        index=order, columns=LABEL_ORDER, fill_value=0)
    frac = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for lab in LABEL_ORDER:
        vals = frac[lab].fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottom, width=0.72, color=LABEL_COLOR[lab],
               label=lab, edgecolor=SURFACE, linewidth=2)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 6:
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        bottom += vals
    for xi, n in enumerate(ct.sum(axis=1)):
        ax.text(xi, 101.5, f"n={n}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return ct, frac


def main():
    df = build_frame()
    out = []

    # ---- figure 1: confidence across entropy bins -----------------------
    qbins = pd.qcut(df["S"], 6, duplicates="drop")
    df["S_q"] = qbins
    q_order = list(df["S_q"].cat.categories)
    q_labels = [f"{c.left:.2f}–{c.right:.2f}" for c in q_order]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 8), height_ratios=[2.2, 1], sharex=True,
        gridspec_kw={"hspace": 0.18})

    ct_q, frac_q = stacked_label_panel(ax1, df, "S_q", q_order)
    ax1.set_ylabel("Share of scans (%)")
    ax1.set_title("Confidence label distribution by spectral entropy",
                  fontsize=13, fontweight="bold", loc="left", pad=26)
    ax1.legend(frameon=False, ncol=3, loc="lower left",
               bbox_to_anchor=(0, 1.005), fontsize=9)

    x = np.arange(len(q_order))
    g = df.groupby("S_q", observed=True)["confidence_score"]
    med, q1, q3 = g.median().reindex(q_order), g.quantile(.25).reindex(q_order), g.quantile(.75).reindex(q_order)
    ax2.fill_between(x, q1, q3, color="#2a78d6", alpha=0.16, linewidth=0)
    ax2.plot(x, med, color="#2a78d6", linewidth=2, marker="o", markersize=7,
             markeredgecolor=SURFACE, markeredgewidth=1.5, label="median score (IQR band)")
    ax2.axhline(80, color=INK_2, linewidth=1, linestyle=":", zorder=0)
    ax2.set_ylabel("Confidence score")
    ax2.set_xlabel("Spectral entropy S (nats), equal-count bins")
    ax2.set_xticks(x)
    ax2.set_xticklabels(q_labels)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    fig.savefig(HERE / "fig_entropy_vs_confidence.svg", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: MW vs entropy / effective peaks ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    edges = [0, 150, 200, 250, 300, 350, 400, 500, 600, 10000]
    mb = pd.cut(df["mw"], bins=edges, right=False)

    for ax, ycol, ylabel, title in [
        (axes[0], "S", "Spectral entropy S (nats)", "MW vs spectral entropy"),
        (axes[1], "eff_peaks", "Effective peaks  (e^S)", "MW vs effective peak count"),
    ]:
        for lab in LABEL_ORDER:
            s = df[df["confidence_label"] == lab]
            ax.scatter(s["mw"], s[ycol], s=22, alpha=0.5, color=LABEL_COLOR[lab],
                       edgecolor=SURFACE, linewidth=0.4, label=lab)
        lo = stats.linregress(df["mw"], df[ycol])
        xs = np.linspace(df["mw"].min(), df["mw"].max(), 100)
        ax.plot(xs, lo.intercept + lo.slope * xs, color=INK, linewidth=2,
                linestyle="--", zorder=5, label="linear fit")
        bmed = df.groupby(mb, observed=True)[ycol].median()
        bx = df.groupby(mb, observed=True)["mw"].median()
        ax.plot(bx, bmed, color=INK, linewidth=2.5, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=6,
                label="median per MW bin")
        r_p, p_p = stats.pearsonr(df["mw"], df[ycol])
        r_s, p_s = stats.spearmanr(df["mw"], df[ycol])
        ax.text(0.02, 0.97, f"Pearson r = {r_p:.2f} (p={p_p:.1e})\n"
                            f"Spearman ρ = {r_s:.2f} (p={p_s:.1e})\n"
                            f"R² = {lo.rvalue ** 2:.3f}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK_2,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=GRID))
        ax.set_xlabel("Neutral monoisotopic mass (Da)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=8)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, loc="upper left",
               bbox_to_anchor=(0.06, 1.02), fontsize=9)
    fig.savefig(HERE / "fig_mw_vs_entropy.svg", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- stats ----------------------------------------------------------
    out.append(f"scans analysed: {len(df)}")
    out.append(f"entropy: ms_entropy {DEFAULT_NOISE_THRESHOLD} noise floor, "
               f"{DEFAULT_MS2_TOLERANCE_DA} Da merge, cap {DEFAULT_MAX_PEAK_NUM} peaks, clean_spectrum=True")
    out.append("")
    out.append("== how the three complexity measures relate ==")
    for a, b in [("n_peaks_raw", "n_peaks_filtered"), ("n_peaks_raw", "eff_peaks"),
                 ("n_peaks_filtered", "eff_peaks"), ("S", "n_peaks_raw"), ("S", "n_peaks_filtered")]:
        r, p = stats.spearmanr(df[a], df[b])
        out.append(f"  Spearman({a}, {b}) = {r:.3f}  p = {p:.2e}")
    out.append("")
    out.append(df[["n_peaks_raw", "n_peaks_filtered", "eff_peaks", "S"]]
               .describe().round(2).to_string())
    out.append("")
    out.append("== each measure vs confidence score (Spearman) ==")
    for c in ["n_peaks_raw", "n_peaks_filtered", "eff_peaks", "S", "mw"]:
        r, p = stats.spearmanr(df[c], df["confidence_score"])
        out.append(f"  {c:18s} rho = {r:+.3f}  p = {p:.2e}")
    out.append("")
    out.append("== confidence by entropy quantile bin ==")
    out.append(ct_q.to_string())
    out.append("")
    out.append(frac_q.round(1).to_string())
    out.append("")
    out.append(df.groupby("S_q", observed=True)["confidence_score"]
                 .agg(["count", "mean", "median"]).round(2).to_string())
    out.append("")
    out.append("== confidence by named entropy band (bin/mgf_loader._entropy_label) ==")
    band_ct = pd.crosstab(df["entropy_band"], df["confidence_label"]).reindex(
        columns=LABEL_ORDER, fill_value=0)
    out.append(band_ct.to_string())
    out.append("")
    out.append((band_ct.div(band_ct.sum(axis=1).replace(0, np.nan), axis=0) * 100)
               .round(1).to_string())
    out.append("")
    out.append(df.groupby("entropy_band", observed=True)["confidence_score"]
                 .agg(["count", "mean", "median"]).round(2).to_string())
    out.append("")
    out.append("== MW vs entropy ==")
    for c in ["S", "eff_peaks"]:
        r, p = stats.spearmanr(df["mw"], df[c])
        out.append(f"  Spearman(mw, {c}) = {r:+.3f}  p = {p:.2e}")
    out.append("")
    out.append("== partial: does entropy explain the low-MW penalty? ==")
    lowmw = df[df["mw"] < 150]
    hi = df[df["mw"] >= 150]
    out.append(f"  MW<150:  n={len(lowmw)}  median S={lowmw['S'].median():.3f}  "
               f"median eff_peaks={lowmw['eff_peaks'].median():.1f}  "
               f"median score={lowmw['confidence_score'].median():.1f}")
    out.append(f"  MW>=150: n={len(hi)}  median S={hi['S'].median():.3f}  "
               f"median eff_peaks={hi['eff_peaks'].median():.1f}  "
               f"median score={hi['confidence_score'].median():.1f}")
    u, pu = stats.mannwhitneyu(lowmw["S"], hi["S"])
    out.append(f"  Mann-Whitney on S (low vs high MW): U={u:.0f}  p={pu:.2e}")
    # within-entropy-band MW effect: if MW still matters at fixed entropy,
    # entropy is not the mechanism.
    out.append("")
    out.append("  Spearman(mw, score) within each entropy band:")
    for band in df["entropy_band"].cat.categories:
        sub = df[df["entropy_band"] == band]
        if len(sub) > 20:
            r, p = stats.spearmanr(sub["mw"], sub["confidence_score"])
            out.append(f"    {band:24s} n={len(sub):4d}  rho={r:+.3f}  p={p:.2e}")

    text = "\n".join(out)
    (HERE / "entropy_analysis_stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
