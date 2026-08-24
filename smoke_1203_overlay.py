"""Finalize (sigmoid) the smoke-run logits ourselves and render CT + ink overlays.

(Workaround for villa finalize_outputs NameError bug — see trackD/LOG.md.)
"""
import os

import numpy as np
import zarr

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"
MERGED = os.path.join(OUT, "smoke_1203", "merged_logits.zarr")

g = zarr.open(MERGED, mode="r")
print("merged tree:", list(g.keys()) if hasattr(g, "keys") else "array")
arr = g["0"] if hasattr(g, "keys") and "0" in list(g.keys()) else g
# store is full-volume-shaped; only the bbox region was written (216 patches,
# 6x6x6 grid, stride 96, patch 192 from origin (7281, 7007, 11615))
oz, oy, ox = 7281, 7007, 11615
ez, ey, ex = oz + 672, oy + 672, ox + 672
lg = np.asarray(arr[..., oz:ez, oy:ey, ox:ex])
print("logits:", lg.shape, lg.dtype, "min/max:", np.nanmin(lg), np.nanmax(lg))
while lg.ndim > 3:
    lg = lg[0]
prob = 1.0 / (1.0 + np.exp(-lg.astype(np.float32)))
print("prob stats: mean", prob.mean(), "p99", np.percentile(prob, 99),
      "frac>0.5:", float((prob > 0.5).mean()))
np.save(os.path.join(OUT, "smoke_1203_prob.npy"), prob.astype(np.float16))

# fetch matching CT for overlay
import fsspec
ct_url = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
zct = zarr.open(fsspec.get_mapper(ct_url), mode="r")["0"]
# coords store origin: read from the coordinates (bbox was expanded to 7281.. per report)
cz, cy, cx = oz, oy, ox
sz, sy, sx = prob.shape
print("fetching CT", (cz, cy, cx), (sz, sy, sx))
ct = zct[cz:cz + sz, cy:cy + sy, cx:cx + sx]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for i, frac in enumerate([0.35, 0.5, 0.65]):
    z = int(sz * frac)
    axes[0][i].imshow(ct[z], cmap="gray")
    axes[0][i].set_title(f"CT z={cz + z}")
    axes[1][i].imshow(ct[z], cmap="gray")
    pm = np.ma.masked_less(prob[z], 0.5)
    axes[1][i].imshow(pm, cmap="autumn", alpha=0.6, vmin=0.5, vmax=1.0)
    axes[1][i].set_title(f"+ ink prob>0.5 ({float((prob[z] > 0.5).mean()):.4f})")
    for r in range(2):
        axes[r][i].set_xticks([]); axes[r][i].set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "smoke_1203_overlay.png"), dpi=100)
print("wrote smoke_1203_overlay.png")
