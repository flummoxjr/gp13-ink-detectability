"""K3 follow-up — L0 verification imagery of the top HIGH-channel (mid-Z) clusters.

Fetch native-45.5um windows around the top clusters from both energy volumes,
render 74keV / 110keV / offset-corrected ratio side by side.
Output: trackD/out/k3_s2_high_verify.png
"""
import json
import os

import numpy as np
import zarr
import fsspec

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
V74 = f"{BUCKET}/PHercParis4/volumes/20260310170716-45.532um-11.0m-74keV-masked.zarr"
V110 = f"{BUCKET}/PHercParis4/volumes/20260310173927-45.532um-11.0m-110keV-masked.zarr"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"
WIN74 = (-0.058, 0.27)
WIN110 = (-0.04, 0.2)
OFF74, OFF110 = -0.03099, -0.02682  # stage-2 air offsets

with open(os.path.join(OUT, "k3_s2_high_clusters.json")) as f:
    rows = json.load(f)[:5]

z74 = zarr.open(fsspec.get_mapper(V74), mode="r")["0"]
z110 = zarr.open(fsspec.get_mapper(V110), mode="r")["0"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, len(rows), figsize=(3.8 * len(rows), 11), squeeze=False)
S = 60
for i, r in enumerate(rows):
    z, y, x = r["centroid_zyx_L0"]
    a74 = np.asarray(z74[z, max(y - S, 0):y + S, max(x - S, 0):x + S]).astype(np.float32)
    a110 = np.asarray(z110[z, max(y - S, 0):y + S, max(x - S, 0):x + S]).astype(np.float32)
    f74 = a74 / 255 * (WIN74[1] - WIN74[0]) + WIN74[0] - OFF74
    f110 = a110 / 255 * (WIN110[1] - WIN110[0]) + WIN110[0] - OFF110
    pap = (a74 > 0) & (a110 > 0) & (f74 > 0.02) & (f110 > 0.015)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pap, f74 / np.maximum(f110, 1e-4), np.nan)
    axes[0][i].imshow(f74, cmap="gray")
    axes[0][i].set_title(f"#{i + 1} 74keV z={z} n={r['size_vox']}")
    axes[1][i].imshow(f110, cmap="gray")
    axes[1][i].set_title("110keV")
    im = axes[2][i].imshow(ratio, cmap="coolwarm", vmin=0.9, vmax=1.7)
    axes[2][i].set_title("ratio (papyrus)")
    for rr in range(3):
        axes[rr][i].set_xticks([]); axes[rr][i].set_yticks([])
fig.colorbar(im, ax=axes[2], shrink=0.7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "k3_s2_high_verify.png"), dpi=100)
print("wrote k3_s2_high_verify.png")
