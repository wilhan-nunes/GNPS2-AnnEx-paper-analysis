"""
MS/MS mirror plot: L-Norvaline (CCMSLIB00006120370, top) vs L-Valine (CCMSLIB00005885071, bottom).

Peaks fetched live from the GNPS2 metabolomics-usi JSON endpoint (fragment_mz_tolerance=0.1),
matching the query used in metabolomics-usi.gnps2.org/dashinterface for the same two USIs.
Shared/annotated fragments (m/z 55.05, 72.08, 118.09 — common amino-acid immonium/loss ions)
are highlighted; all other peaks are drawn in the neutral stem color.
"""

import json
import os
import urllib.request
import urllib.parse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from rdkit import Chem
from rdkit.Chem import Draw

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_STATIC = os.path.join(BASE, "mirror_plot_norvaline_valine.svg")

USI_TOP    = "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00006120370"  # L-Norvaline
USI_BOTTOM = "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005885071"  # L-Valine

TOP_NAME    = "L-Norvaline"
BOTTOM_NAME = "L-Valine"
TOP_SMILES    = "O=C(O)C(N)CCC"
BOTTOM_SMILES = "CC(C)C(C(=O)O)N"

MZ_MIN, MZ_MAX = 40.0, 150.0
FRAGMENT_TOL = 0.1
ANNOTATE_PEAKS = [55.0541, 72.0806, 118.0861]  # shared fragments called out in the source dashboard query

# ── fetch peaks from GNPS2 USI resolver ─────────────────────────────────────
def fetch_peaks(usi):
    url = "https://metabolomics-usi.gnps2.org/json/?" + urllib.parse.urlencode({"usi1": usi})
    with urllib.request.urlopen(url, timeout=30) as resp:
        d = json.load(resp)
    peaks = np.array(d["peaks"], dtype=float)
    return peaks, d.get("precursor_mz")

top_peaks, top_prec = fetch_peaks(USI_TOP)
bot_peaks, bot_prec = fetch_peaks(USI_BOTTOM)

def filter_range(peaks, lo, hi):
    m = (peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)
    return peaks[m]

top_peaks = filter_range(top_peaks, MZ_MIN, MZ_MAX)
bot_peaks = filter_range(bot_peaks, MZ_MIN, MZ_MAX)

def norm_intensity(peaks):
    # cap stems at 65% so the top ~35% of each half stays clear for header text/labels
    return peaks[:, 1] / peaks[:, 1].max() * 65.0

top_int = norm_intensity(top_peaks)
bot_int = norm_intensity(bot_peaks)

def is_annotated(mz, targets, tol=FRAGMENT_TOL):
    return any(abs(mz - t) <= tol for t in targets)

def label_positions(peaks, intensities, targets, tol=FRAGMENT_TOL):
    """One label per target m/z, placed at the tallest matching peak."""
    out = {}
    for t in targets:
        m = np.abs(peaks[:, 0] - t) <= tol
        if not m.any():
            continue
        idx = np.where(m)[0][np.argmax(intensities[m])]
        out[t] = (peaks[idx, 0], intensities[idx])
    return out

# ── colour palette (consistent with other MSMLS_lib figures) ─────────────
COLOR_TOP        = "#1f77b4"   # Consistent-evidence blue
COLOR_BOTTOM     = "#d62728"   # Inconsistent-evidence red
COLOR_ANNOTATED  = "#2ca02c"   # override/highlight green
COLOR_STEM_ALPHA = 0.55

# ── build figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 6.8))

for mz, inten in zip(top_peaks[:, 0], top_int):
    annotated = is_annotated(mz, ANNOTATE_PEAKS)
    color = COLOR_ANNOTATED if annotated else COLOR_TOP
    alpha = 1.0 if annotated else COLOR_STEM_ALPHA
    ax.plot([mz, mz], [0, inten], color=color, linewidth=1.4 if annotated else 1.0, alpha=alpha, zorder=3 if annotated else 2)

for mz, inten in zip(bot_peaks[:, 0], bot_int):
    annotated = is_annotated(mz, ANNOTATE_PEAKS)
    color = COLOR_ANNOTATED if annotated else COLOR_BOTTOM
    alpha = 1.0 if annotated else COLOR_STEM_ALPHA
    ax.plot([mz, mz], [0, -inten], color=color, linewidth=1.4 if annotated else 1.0, alpha=alpha, zorder=3 if annotated else 2)

# one label per shared fragment, taken from whichever side shows it taller
top_labels = label_positions(top_peaks, top_int, ANNOTATE_PEAKS)
bot_labels = label_positions(bot_peaks, bot_int, ANNOTATE_PEAKS)
for t in ANNOTATE_PEAKS:
    if t in top_labels:
        mz, inten = top_labels[t]
        ax.text(mz, inten + 3, f"{mz:.4f}", ha="center", va="bottom", fontsize=9,
                 color=COLOR_ANNOTATED, rotation=90)
    if t in bot_labels:
        mz, inten = bot_labels[t]
        ax.text(mz, -inten - 3, f"{mz:.4f}", ha="center", va="top", fontsize=9,
                 color=COLOR_ANNOTATED, rotation=90)

ax.axhline(0, color="#333", linewidth=0.8, zorder=1)
ax.set_xlim(MZ_MIN, MZ_MAX)
ax.set_ylim(-82, 82)
ax.set_xlabel("m/z", fontsize=13)
ax.set_ylabel("Relative intensity (%)", fontsize=13)
ax.set_yticks([-65, -32.5, 0, 32.5, 65])
ax.set_yticklabels(["100", "50", "0", "50", "100"])
ax.tick_params(axis="both", labelsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
ax.set_axisbelow(True)

fig.subplots_adjust(top=0.82, bottom=0.16, left=0.09, right=0.98)

fig.text(0.10, 0.955, f"{TOP_NAME}   {USI_TOP.split(':')[-1]}   ·   precursor m/z {top_prec:.4f}",
         ha="left", va="top", fontsize=11, color=COLOR_TOP, fontweight="bold")
fig.text(0.10, 0.045, f"{BOTTOM_NAME}   {USI_BOTTOM.split(':')[-1]}   ·   precursor m/z {bot_prec:.4f}",
         ha="left", va="bottom", fontsize=11, color=COLOR_BOTTOM, fontweight="bold")

handles = [
    plt.Line2D([0], [0], color=COLOR_TOP, lw=2, label=TOP_NAME),
    plt.Line2D([0], [0], color=COLOR_BOTTOM, lw=2, label=BOTTOM_NAME),
    plt.Line2D([0], [0], color=COLOR_ANNOTATED, lw=2, label="Shared fragment"),
]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           ncol=3, fontsize=10, frameon=False)

# ── structure insets (top-right = top spectrum compound, bottom-right = bottom) ──
def mol_image_array(smiles, size=(220, 180)):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=size)
    return np.array(img)

top_struct = mol_image_array(TOP_SMILES)
bot_struct = mol_image_array(BOTTOM_SMILES)

ab_top = AnnotationBbox(
    OffsetImage(top_struct, zoom=0.62), (0.985, 0.97),
    xycoords="axes fraction", box_alignment=(1, 1), frameon=False,
    pad=0.15,
)
ab_bot = AnnotationBbox(
    OffsetImage(bot_struct, zoom=0.62), (0.985, 0.03),
    xycoords="axes fraction", box_alignment=(1, 0), frameon=False,
    pad=0.15,
)
ax.add_artist(ab_top)
ax.add_artist(ab_bot)

plt.savefig(OUT_STATIC, dpi=180)
plt.savefig(OUT_STATIC.replace(".svg", ".png"), dpi=180)
print(f"Saved static → {OUT_STATIC}")
