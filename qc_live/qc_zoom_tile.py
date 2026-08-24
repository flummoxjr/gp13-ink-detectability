"""Zoomed 3-slice look at the highest-f05 full-interior pulled tile."""
import numpy as np
import zarr
import fsspec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VOL = "vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
FP = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\round_1\w5_prob_1280_9984_12032.npy"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\qc_zoom_w5_1280_9984_12032.png"

oz, oy, ox = 1280, 9984, 12032
cz, cy, cx = oz // 4, oy // 4, ox // 4
prob = np.load(FP).astype(np.float32) / 255.0
fs = fsspec.filesystem("s3", anon=True)
ct2 = zarr.open(fs.get_mapper(VOL), mode="r")["2"]
ct = np.asarray(ct2[cz:cz + 64, cy:cy + 64, cx:cx + 64]).astype(np.float32)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for j, zi in enumerate((16, 32, 48)):
    axes[0, j].imshow(ct[zi], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0, j].set_title(f"CT L2 z={oz + zi * 4}")
    axes[1, j].imshow(ct[zi], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1, j].imshow(np.ma.masked_less(prob[zi], 0.5), cmap="autumn", vmin=0, vmax=1,
                      alpha=0.65, interpolation="nearest")
    axes[1, j].set_title(f"+ prob>0.5 (f05={float((prob[zi] > 0.5).mean()):.3f})")
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
plt.savefig(OUT, dpi=100)
print("WROTE", OUT)
