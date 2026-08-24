"""T6 visual smoking-gun figure: for tiles (7,5) and (8,5) show the 7.91um
render at k=-1, the same render with letter-mask contours (red) and
saturated-zone contours (cyan), the ink24 TIF, and the resampled ink11 TIF.
Writes trackD/qc/qc_s1a_overlay.png
"""
import os

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QCD = r"D:\vesuvius-data\trackD\w032\qc"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc"
WIN = (-0.03, 0.145)
STRONG_INK = 170
SAT_RAW = 240
KI_M1 = 3  # k=-1 index in -4..4

TILES = [(7, 5), (8, 5)]

fig, axes = plt.subplots(len(TILES), 4, figsize=(21, 5.4 * len(TILES)),
                         squeeze=False)
lo, hi = WIN
for r, (ty, tx) in enumerate(TILES):
    d = np.load(os.path.join(QCD, f"tile_{ty}_{tx}.npz"))
    vals = d["vals"]
    ok = d["ok_pre"] & (d["raw0"] > 5)
    rk = np.where(ok, vals[KI_M1], np.nan)
    v = rk[np.isfinite(rk)]
    v0, v1 = np.percentile(v, [2, 98])
    letters = d["sub_ink"] > STRONG_INK
    raw_u8 = np.nan_to_num((vals[4] - lo) / (hi - lo) * 255.0, nan=255.0)
    sat = raw_u8 >= SAT_RAW
    lsm = gaussian_filter(letters.astype(float), 1.5)
    ssm = gaussian_filter(sat.astype(float), 1.5)

    ax = axes[r][0]
    ax.imshow(rk, cmap="gray", vmin=v0, vmax=v1)
    ax.set_title(f"tile({ty},{tx})  7.91um render k=-1")

    ax = axes[r][1]
    ax.imshow(rk, cmap="gray", vmin=v0, vmax=v1)
    ax.contour(lsm, levels=[0.5], colors="red", linewidths=0.9)
    ax.contour(ssm, levels=[0.5], colors="cyan", linewidths=0.9)
    ax.set_title("render + letters(red) + saturated(cyan)")

    ax = axes[r][2]
    ax.imshow(d["sub_ink"], cmap="inferno", vmin=0, vmax=255)
    ax.set_title("ink24 TIF (ds4)")

    ax = axes[r][3]
    ax.imshow(d["ink11r"], cmap="inferno", vmin=0, vmax=255)
    ax.set_title("ink11 TIF resampled")

    for ax in axes[r]:
        ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
fig.savefig(os.path.join(OUT, "qc_s1a_overlay.png"), dpi=110)
print("wrote qc_s1a_overlay.png")
