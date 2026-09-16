"""Confidence label distribution vs precursor molecular weight, and MW vs peak count.

Inputs
------
annex_summary_b22ff1bc4162 (6).csv   scan-level confidence scores
PlusRise_..._spectra_reformatted.mgf raw MS/MS spectra (PEPMASS, CHARGE, peak list)

Outputs
-------
fig_mw_vs_confidence.svg   label distribution + score trend across MW bins
fig_mw_vs_peaks.svg        MW vs peak count, colored by confidence label
mw_analysis_stats.txt      correlation / test statistics
"""

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
CSV = HERE / "annex_summary_b22ff1bc4162 (6).csv"
MGF = HERE / "PlusRise_FBMN_all_default_libr-b22ff1bc41624e29a28cf6e12fe17ad5-spectra_reformatted.mgf"

PROTON = 1.007276

LABEL_ORDER = ["Consistent evidence", "Inconclusive", "Inconsistent evidence"]
LABEL_COLOR = {
    "Consistent evidence": "#008300",
    "Inconclusive": "#eda100",
    "Inconsistent evidence": "#e34948",
}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdbd6"
SURFACE = "#ffffff"

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


def parse_mgf(path):
    """Return DataFrame: scan, mgf_pepmass, charge, n_peaks."""
    peak_re = re.compile(r"^\d")
    rows, cur = [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                cur = {"scan": None, "mgf_pepmass": np.nan, "charge": np.nan, "n_peaks": 0}
            elif line == "END IONS":
                if cur and cur["scan"] is not None:
                    rows.append(cur)
                cur = None
            elif cur is None:
                continue
            elif line.startswith("SCANS="):
                cur["scan"] = int(line[6:])
            elif line.startswith("PEPMASS="):
                cur["mgf_pepmass"] = float(line[8:].split()[0])
            elif line.startswith("CHARGE="):
                cur["charge"] = float(re.sub(r"[^0-9.\-+]", "", line[7:]) or "nan")
            elif peak_re.match(line):
                cur["n_peaks"] += 1
    return pd.DataFrame(rows)


def neutral_mass(mz, charge):
    z = np.where((charge >= 1) & np.isfinite(charge), charge, 1.0)
    return mz * z - z * PROTON


def main():
    scores = pd.read_csv(CSV)
    spectra = parse_mgf(MGF)
    df = scores.merge(spectra, on="scan", how="left")

    # MGF PEPMASS is authoritative for charge; precursor_mz from the summary is the
    # observed m/z. Fall back to the MGF value where the summary lacks one.
    df["mz"] = df["precursor_mz"].fillna(df["mgf_pepmass"])
    df["charge"] = df["charge"].where(df["charge"].between(1, 6), 1.0)
    df["mw"] = neutral_mass(df["mz"].to_numpy(), df["charge"].to_numpy())
    df = df[df["mw"].between(50, 2000) & df["confidence_label"].isin(LABEL_ORDER)].copy()
    df["n_peaks"] = df["n_peaks"].fillna(df["query_peak_count"])

    # ---- binning --------------------------------------------------------
    edges = [0, 150, 200, 250, 300, 350, 400, 500, 600, 10000]
    labels = ["<150", "150–200", "200–250", "250–300", "300–350",
              "350–400", "400–500", "500–600", "≥600"]
    df["mw_bin"] = pd.cut(df["mw"], bins=edges, labels=labels, right=False)

    ct = pd.crosstab(df["mw_bin"], df["confidence_label"])
    ct = ct.reindex(columns=LABEL_ORDER, fill_value=0)
    frac = ct.div(ct.sum(axis=1), axis=0) * 100
    n_per_bin = ct.sum(axis=1)

    # ---- figure 1: label distribution across MW -------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 8), height_ratios=[2.2, 1], sharex=True,
        gridspec_kw={"hspace": 0.18},
    )

    x = np.arange(len(frac))
    bottom = np.zeros(len(frac))
    for lab in LABEL_ORDER:
        vals = frac[lab].to_numpy()
        ax1.bar(x, vals, bottom=bottom, width=0.72, color=LABEL_COLOR[lab],
                label=lab, edgecolor=SURFACE, linewidth=2)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 6:
                ax1.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                         color="white", fontsize=9, fontweight="bold")
        bottom += vals

    for xi, n in enumerate(n_per_bin):
        ax1.text(xi, 101.5, f"n={n}", ha="center", va="bottom",
                 fontsize=8, color=INK_2)

    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Share of scans (%)")
    ax1.set_title("Confidence label distribution by precursor molecular weight",
                  fontsize=13, fontweight="bold", loc="left", pad=26)
    ax1.legend(frameon=False, ncol=3, loc="lower left",
               bbox_to_anchor=(0, 1.005), fontsize=9)
    ax1.grid(axis="y", color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)

    med = df.groupby("mw_bin", observed=True)["confidence_score"].median().reindex(labels)
    q1 = df.groupby("mw_bin", observed=True)["confidence_score"].quantile(0.25).reindex(labels)
    q3 = df.groupby("mw_bin", observed=True)["confidence_score"].quantile(0.75).reindex(labels)
    ax2.fill_between(x, q1, q3, color="#2a78d6", alpha=0.16, linewidth=0)
    ax2.plot(x, med, color="#2a78d6", linewidth=2, marker="o", markersize=7,
             markeredgecolor=SURFACE, markeredgewidth=1.5, label="median score (IQR band)")
    ax2.axhline(80, color=INK_2, linewidth=1, linestyle=":", zorder=0)
    ax2.text(len(x) - 0.4, 80.8, "80 = consistent", fontsize=8, color=INK_2, ha="right")
    ax2.set_ylabel("Confidence score")
    ax2.set_xlabel("Neutral monoisotopic mass (Da)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.grid(axis="y", color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)

    fig.savefig(HERE / "fig_mw_vs_confidence.svg", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: MW vs peak count -------------------------------------
    d2 = df.dropna(subset=["n_peaks"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for lab in LABEL_ORDER:
        s = d2[d2["confidence_label"] == lab]
        ax.scatter(s["mw"], s["n_peaks"], s=26, alpha=0.55,
                   color=LABEL_COLOR[lab], edgecolor=SURFACE, linewidth=0.5,
                   label=f"{lab} (n={len(s)})")

    lo = stats.linregress(d2["mw"], d2["n_peaks"])
    xs = np.linspace(d2["mw"].min(), d2["mw"].max(), 100)
    ax.plot(xs, lo.intercept + lo.slope * xs, color=INK, linewidth=2,
            linestyle="--", zorder=5, label="linear fit")

    # binned median makes the (weak) central tendency readable through the cloud
    mb = pd.cut(d2["mw"], bins=edges, labels=labels, right=False)
    bmed = d2.groupby(mb, observed=True)["n_peaks"].median().reindex(labels)
    bx = d2.groupby(mb, observed=True)["mw"].median().reindex(labels)
    ok = bmed.notna() & bx.notna()
    ax.plot(bx[ok], bmed[ok], color=INK, linewidth=2.5, marker="o", markersize=9,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=6,
            label="median per MW bin")

    r_p, p_p = stats.pearsonr(d2["mw"], d2["n_peaks"])
    r_s, p_s = stats.spearmanr(d2["mw"], d2["n_peaks"])
    ax.text(0.02, 0.97,
            f"Pearson r = {r_p:.2f}  (p = {p_p:.1e})\n"
            f"Spearman ρ = {r_s:.2f}  (p = {p_s:.1e})\n"
            f"slope = {lo.slope:.3f} peaks/Da",
            transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK_2,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=GRID))

    ax.set_xlabel("Neutral monoisotopic mass (Da)")
    ax.set_ylabel("Peaks in MS/MS spectrum")
    ax.set_title("Peak count vs precursor mass — weak positive trend (R² = "
                 f"{lo.rvalue ** 2:.2f})",
                 fontsize=13, fontweight="bold", loc="left", pad=40)
    ax.legend(frameon=False, ncol=3, loc="lower left",
              bbox_to_anchor=(0, 1.005), fontsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(HERE / "fig_mw_vs_peaks.svg", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- stats ----------------------------------------------------------
    out = []
    out.append(f"scans scored: {len(scores)}   matched to MGF: {df['n_peaks'].notna().sum()}   analysed: {len(df)}")
    out.append(f"charge states used: {df['charge'].value_counts().to_dict()}")
    out.append("")
    out.append("== MW vs confidence score ==")
    rs, ps = stats.spearmanr(df["mw"], df["confidence_score"])
    rp, pp = stats.pearsonr(df["mw"], df["confidence_score"])
    out.append(f"Spearman rho = {rs:.3f}  p = {ps:.3e}")
    out.append(f"Pearson  r   = {rp:.3f}  p = {pp:.3e}")
    groups = [g["mw"].to_numpy() for _, g in df.groupby("confidence_label") if len(g) > 1]
    if len(groups) > 1:
        h, ph = stats.kruskal(*groups)
        out.append(f"Kruskal-Wallis on MW across labels: H = {h:.2f}  p = {ph:.3e}")
    out.append("")
    out.append("median MW per label:")
    out.append(df.groupby("confidence_label")["mw"].describe()[["count", "25%", "50%", "75%"]].to_string())
    out.append("")
    out.append("== label counts per MW bin ==")
    out.append(ct.to_string())
    out.append("")
    out.append("== label share (%) per MW bin ==")
    out.append(frac.round(1).to_string())
    out.append("")
    out.append("== confidence score per MW bin ==")
    out.append(df.groupby("mw_bin", observed=True)["confidence_score"]
                 .agg(["count", "mean", "median"]).round(2).to_string())
    out.append("")
    out.append("== MW vs peak count ==")
    out.append(f"Pearson  r   = {r_p:.3f}  p = {p_p:.3e}")
    out.append(f"Spearman rho = {r_s:.3f}  p = {p_s:.3e}")
    out.append(f"linear fit: n_peaks = {lo.slope:.4f} * MW + {lo.intercept:.2f}  (R^2 = {lo.rvalue**2:.3f})")
    out.append("")
    out.append("== peak count vs confidence score ==")
    rs2, ps2 = stats.spearmanr(d2["n_peaks"], d2["confidence_score"])
    out.append(f"Spearman rho = {rs2:.3f}  p = {ps2:.3e}")

    text = "\n".join(out)
    (HERE / "mw_analysis_stats.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
