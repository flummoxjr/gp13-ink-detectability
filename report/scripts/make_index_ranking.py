"""Polished K2b ranking figure -> trackD/report/figures/index_ranking.png.

Data read live from trackD/out/k2b_index/*.json. Two panels sharing row order
(ranked by median structural SNR @ q=0.25, best on top): SNR bars + IQR
whiskers, bandwidth dots + IQR whiskers. PHerc0139 calibrator highlighted;
two tiers shaded; 8.64 um / 116 keV campaign flagged; residual-noise-ref and
fallback-air flagged.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IDX = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\k2b_index"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\report\figures\index_ranking.png"

# palette (validated: ordinal blue pair + categorical blue/orange)
BLUE = "#2a78d6"       # readable-class tier
BLUE_LT = "#86b6ef"    # degraded tier (ordinal step, same hue)
ORANGE = "#eb6834"     # PHerc0139 calibrator
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
WASH = "#f4f3f0"       # tier wash

CAMPAIGN_864 = {"PHerc0268", "PHerc0800", "PHerc1218", "PHerc1447"}

scrolls = {}
for f in os.listdir(IDX):
    if f.startswith("PHerc") and f.endswith(".json"):
        with open(os.path.join(IDX, f)) as fh:
            scrolls[f[:-5]] = json.load(fh)

rows = []
for s, d in scrolls.items():
    m, lo, hi = d["snr_q025_med_iqr"]
    bm, blo, bhi = d["bandwidth_med_iqr"]
    rows.append((s, m, lo, hi, bm, blo, bhi))
rows.sort(key=lambda r: -r[1])            # best first (top row)

names = [r[0] for r in rows]
snr = np.array([[r[1], r[2], r[3]] for r in rows])
bw = np.array([[r[4], r[5], r[6]] for r in rows])
n_deg = sum(1 for r in rows if r[1] < 30)  # degraded tier = median SNR < 30
n_good = len(rows) - n_deg

def label(s):
    t = s.replace("PHerc", "PHerc ")
    tags = []
    if s in CAMPAIGN_864:
        tags.append("\u2020")               # dagger: 8.64um/116keV campaign
    if scrolls[s].get("noise_ref") == "residual":
        tags.append("*")                    # residual noise reference
    if s == "PHerc0800":
        tags.append("\u2021")               # double dagger: fallback air
    return t + (" " + "".join(tags) if tags else "")

labels = [label(s) for s in names]
colors = [ORANGE if s == "PHerc0139" else (BLUE_LT if m < 30 else BLUE)
          for s, m in zip(names, snr[:, 0])]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.linewidth": 0.8,
})

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10.6, 5.8), dpi=200, sharey=True,
    gridspec_kw={"width_ratios": [2.6, 1.0]})
fig.patch.set_facecolor("white")
y = np.arange(len(names))

for ax in (ax1, ax2):
    ax.set_facecolor("white")
    ax.axhspan(n_good - 0.5, len(names) - 0.5, color=WASH, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.7)

ax1.set_ylim(len(names) - 0.5, -0.5)      # row 0 (best) on top

# ---- left: SNR @ q=0.25 ----
err = [snr[:, 0] - snr[:, 1], snr[:, 2] - snr[:, 0]]
ax1.barh(y, snr[:, 0], height=0.58, color=colors, zorder=3)
ax1.errorbar(snr[:, 0], y, xerr=err, fmt="none", ecolor=INK2,
             elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=4)
for yi, (m, lo, hi) in zip(y, snr):
    ax1.text(hi + 3.5, yi, f"{m:.1f}", va="center", ha="left",
             fontsize=7.5, color=INK2)
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=8.5)
ax1.set_xlim(0, 185)
ax1.set_xlabel("structural SNR @ q = 0.25 cyc/px   (papyrus PSD / noise PSD; "
               "median, IQR over ROIs)", fontsize=8)
ax1.tick_params(axis="x", labelsize=7.5)

# 0139 anchor line + label (lower right, over the wash, clear of tier label)
anchor = snr[names.index("PHerc0139"), 0]
ax1.axvline(anchor, color=ORANGE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
ax1.text(anchor + 3, len(names) - 0.75,
         "PHerc 0139 median = 115.5\n(calibrator: letters proven\nlegible in this volume)",
         fontsize=7.2, color=ORANGE, ha="left", va="bottom", linespacing=1.25)

# tier annotations
ax1.text(182, 3.0, "readable-class tier\nSNR 72\u2013160",
         fontsize=7.8, color=INK2, ha="right", va="center", linespacing=1.3)
ax1.text(105, n_good + (n_deg - 1) / 2 - 0.4,
         "degraded tier\nSNR 8\u201324  (3\u00d7 gap)",
         fontsize=7.8, color=INK2, ha="right", va="center", linespacing=1.3)

# ---- right: structural bandwidth ----
berr = [bw[:, 0] - bw[:, 1], bw[:, 2] - bw[:, 0]]
ax2.errorbar(bw[:, 0], y, xerr=berr, fmt="none", ecolor=INK2,
             elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=3)
ax2.scatter(bw[:, 0], y, s=34, color=colors, zorder=4)
ax2.axvline(0.4958, color=MUTED, linewidth=0.9, linestyle=(0, (2, 2)))
ax2.text(0.488, 1.55, "top q bin (0.496) =\nmeasurement ceiling", fontsize=7.0,
         color=MUTED, ha="right", va="center", linespacing=1.2)
ax2.set_xlim(0.34, 0.525)
ax2.set_xticks([0.35, 0.40, 0.45, 0.50])
ax2.set_xlabel("structural bandwidth (cyc/px)\nmax q with papyrus \u2265 2\u00d7 noise",
               fontsize=8)
ax2.tick_params(axis="x", labelsize=7.5)

fig.suptitle("K2b scan-quality index \u2014 13 Grand-Prize scrolls + PHerc 0139 "
             "calibrator, ranked by mid-band structural SNR",
             fontsize=10, x=0.5, y=0.985, color=INK)
fig.text(0.012, 0.040,
         "\u2020 8.64 \u00b5m / 116 keV scan campaign (other ten volumes: "
         "9.362 \u00b5m / 113 keV)     \u2021 air reference from full-z fallback "
         "search (single validated window)",
         fontsize=6.8, color=INK2)
fig.text(0.012, 0.012,
         "* PHerc 0139: noise reference = papyrus high-pass residual (no genuine "
         "air window found); its SNR is not on the same reference as the other "
         "13 \u2014 see text",
         fontsize=6.8, color=INK2)
fig.subplots_adjust(left=0.105, right=0.985, top=0.925, bottom=0.155, wspace=0.06)
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
