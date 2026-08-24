"""Render the full w032 ink maps at reading scale (whole-segment overview)."""
import os
import numpy as np

CACHE = r"D:\vesuvius-data\trackD\w032"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"

ink24 = np.load(os.path.join(CACHE, "ink24_ds4.npy"))
ink11 = np.load(os.path.join(CACHE, "ink11_ds4.npy"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ds = 4  # additional downsample -> ds16 overall (38.4um/px, whole segment ~9.1x7.9cm)
a = ink24[::ds, ::ds]
b = ink11[::ds, ::ds]
fig, axes = plt.subplots(1, 2, figsize=(24, 15))
axes[0].imshow(a, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("w032 ink @2.4um model (ds16)")
axes[1].imshow(b, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("w032 ink @1.129um model (ds16)")
# mark the ds4-tile grid used in analysis (each 512 ds4 px -> 128 ds16 px)
for ax in axes[:1]:
    for t in range(0, a.shape[0], 128):
        ax.axhline(t, color="cyan", lw=0.3)
    for t in range(0, a.shape[1], 128):
        ax.axvline(t, color="cyan", lw=0.3)
    ax.add_patch(plt.Rectangle((8 * 128, 8 * 128), 7 * 128, 8 * 128, fill=False, color="red", lw=2))
for ax in axes:
    ax.set_xticks(range(0, a.shape[1], 256))
    ax.set_yticks(range(0, a.shape[0], 256))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "s1a_v2_overview.png"), dpi=85)
print("wrote s1a_v2_overview.png", a.shape)
