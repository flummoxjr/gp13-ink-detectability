"""Quicklook PNGs: w035 pred + labels, 1203 segments, at ds6, for visual check."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_map, valid_mask, load_w035_label2d, downsample, SALVAGE

w = load_map("w035_s42").astype(np.float32)
lab = load_w035_label2d(w.shape).astype(np.float32)
a = load_map("1203A_s42").astype(np.float32)
b = load_map("1203B_s42").astype(np.float32)

fig, ax = plt.subplots(2, 2, figsize=(16, 16))
ax[0, 0].imshow(downsample(w, 6), cmap="inferno", vmax=200)
ax[0, 0].set_title("w035 pred seed42 (ds6)")
ax[0, 1].imshow(downsample(lab * 255, 6), cmap="gray")
ax[0, 1].set_title("w035 human ink labels (ds6)")
ax[1, 0].imshow(downsample(a, 6), cmap="inferno", vmax=200)
ax[1, 0].set_title("1203 segA seed42 (ds6)")
ax[1, 1].imshow(downsample(b, 6), cmap="inferno", vmax=200)
ax[1, 1].set_title("1203 segB seed42 (ds6)")
for axx in ax.ravel():
    axx.axis("off")
plt.tight_layout()
plt.savefig(SALVAGE / "verdict_quicklook.png", dpi=90)
print("saved")
