"""Probe PHerc1203 2.4um band (level 5) for a papyrus-dense smoke-test ROI."""
import numpy as np
import zarr
import fsspec

URL = "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
g = zarr.open(fsspec.get_mapper(URL), mode="r")
for lvl in sorted(g.keys(), key=int):
    print("level", lvl, g[lvl].shape, g[lvl].dtype)

L = "5"
arr = g[L]
zmid = arr.shape[0] // 2
sl = arr[zmid]
print(f"L{L} central slice z={zmid}: nonzero {np.mean(sl > 0):.3f}, mean {sl[sl > 0].mean():.1f}")

# densest 32x32 window at L5 (=1024^2 at L0)
from scipy.ndimage import uniform_filter
fill = uniform_filter((sl > 5).astype(np.float32), size=32, mode="constant")
yy, xx = np.unravel_index(np.argmax(fill), fill.shape)
print(f"best window at L5 (y,x)=({yy},{xx}) fill={fill[yy, xx]:.2f}")
scale = 2 ** int(L)
z0, y0, x0 = zmid * scale, yy * scale, xx * scale
print(f"L0 ROI center approx: z={z0}, y={y0}, x={x0}")
print(f'suggested --bbox "{z0 - 192}:{z0 + 192},{y0 - 192}:{y0 + 192},{x0 - 192}:{x0 + 192}"')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(sl, cmap="gray")
ax.add_patch(plt.Rectangle((xx - 16, yy - 16), 32, 32, fill=False, color="red", lw=2))
fig.savefig(r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\probe_1203_band_L5.png", dpi=100)
print("wrote probe_1203_band_L5.png")
