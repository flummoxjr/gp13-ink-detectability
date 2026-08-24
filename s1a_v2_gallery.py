"""S1a-v2 step 1 — candidate-tile gallery with a VISUAL GATE (per QC prescription).

Select w032 tiles where BOTH released ink TIFs (2.4um model and 1.129um model)
have prediction coverage and mark letters, with on-prediction background
available. Dump full-res ink crops so a human/agent can verify actual GLYPHS
(elongated strokes) before any statistics are run.

Verifier guidance (qc/s1a_verification.md): usable region ~ds4 rows 8-15,
cols 8-14; ink24 nonzero floor ~28 (0 = no-data); ink11 must be resampled
(L1 = 2.258um/px vs ink24 2.399um/px, global shift ~36px at full res).
"""
import os

import numpy as np

CACHE = r"D:\vesuvius-data\trackD\w032"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"
TILE = 512

ink24 = np.load(os.path.join(CACHE, "ink24_ds4.npy"))
ink11 = np.load(os.path.join(CACHE, "ink11_ds4.npy"))
print("ink24 ds4:", ink24.shape, "ink11 ds4:", ink11.shape)

# Resample ink11 onto ink24's ds4 grid: full-res scale factor 2.258/2.399 px
# (ink11 L1 px are smaller in uv-um -> its image is larger; map by uv microns)
from scipy.ndimage import zoom
# uv microns per ds4 px: ink24 = 4*2.399, ink11 = 4*2.258
zf = (4 * 2.258) / (4 * 2.399)
ink11r = zoom(ink11, zf, order=1)
# pad/crop to ink24 shape
h = min(ink11r.shape[0], ink24.shape[0])
w = min(ink11r.shape[1], ink24.shape[1])
ink11a = np.zeros_like(ink24)
ink11a[:h, :w] = ink11r[:h, :w]
print("ink11 resampled:", ink11r.shape, "->", ink11a.shape)

n_ty, n_tx = ink24.shape[0] // TILE, ink24.shape[1] // TILE
rows = []
for ty in range(8, min(16, n_ty)):
    for tx in range(8, min(15, n_tx)):
        s24 = ink24[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE]
        s11 = ink11a[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE]
        cov24 = (s24 > 0).mean()
        cov11 = (s11 > 0).mean()
        let24 = (s24 >= 200).mean()
        let11 = (s11 >= 150).mean()
        both = ((s24 >= 200) & (s11 >= 150)).mean()
        bg_ok = ((s24 >= 28) & (s24 <= 60)).mean()
        if cov24 > 0.95 and cov11 > 0.6 and both > 0.005 and bg_ok > 0.15:
            rows.append((both, ty, tx, cov24, cov11, let24, let11, bg_ok))

rows.sort(reverse=True)
print(f"{len(rows)} candidate tiles:")
for r in rows[:12]:
    print(f"  tile({r[1]},{r[2]}): both-model letters {r[0]:.4f}, cov24 {r[3]:.2f}, "
          f"cov11 {r[4]:.2f}, let24 {r[5]:.3f}, let11 {r[6]:.3f}, bg {r[7]:.2f}")

# full-res gallery of top 8 (read windows from the full-res TIF via memmap)
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tif = tifffile.imread(os.path.join(CACHE, "ink24.tif"))
top = rows[:8]
if top:
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), squeeze=False)
    for ax, r in zip(axes.ravel(), top):
        _, ty, tx = r[0], r[1], r[2]
        y0, x0 = ty * TILE * 4, tx * TILE * 4
        crop = np.asarray(tif[y0:y0 + TILE * 4:2, x0:x0 + TILE * 4:2])
        ax.imshow(crop, cmap="inferno", vmin=0, vmax=255)
        ax.set_title(f"tile({ty},{tx}) full-res/2")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s1a_v2_gallery.png"), dpi=90)
    print("wrote s1a_v2_gallery.png")
