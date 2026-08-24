"""Analyze the ink_9um predictions on PHerc1203's auto-grown segments.

No labels exist here — analysis is statistical + visual, anchored by the two
known reference behaviors: w035 control (letters: sparse bright glyph rows,
frac>half ~0.12) and the ink_3d failure mode (blanket sheet-painting).
"""
import json
import os

import numpy as np
import tifffile

DIR = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_1203"
W035 = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_w035\w035_seed42-075000.tif"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"

ref = tifffile.imread(W035).astype(np.float32)
ref_m = ref[ref > 0]
print(f"w035 reference: max {ref.max():.0f}, frac>50%max {float((ref > 0.5 * ref.max()).mean()):.4f}, "
      f"nonzero mean {ref_m.mean():.1f}")

stats = {}
imgs = {}
for fn in sorted(os.listdir(DIR)):
    if not fn.endswith(".tif"):
        continue
    a = tifffile.imread(os.path.join(DIR, fn)).astype(np.float32)
    m = a > 0
    row = {
        "shape": list(a.shape), "max": float(a.max()),
        "frac_nonzero": round(float(m.mean()), 4),
        "frac_gt_half": round(float((a > 0.5 * max(a.max(), 1)).mean()), 4),
        "nonzero_mean": round(float(a[m].mean()), 1) if m.any() else 0,
        "nonzero_p99": round(float(np.percentile(a[m], 99)), 1) if m.any() else 0,
    }
    stats[fn] = row
    print(fn, row)
    imgs[fn] = a

with open(os.path.join(OUT, "ink9um_1203_stats.json"), "w") as f:
    json.dump(stats, f, indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

keys = [k for k in imgs if "seed42" in k]
fig, axes = plt.subplots(2, len(keys) + 1, figsize=(5.2 * (len(keys) + 1), 11), squeeze=False)
v = np.percentile(ref, 99.9)
axes[0][0].imshow(ref[::4, ::4], cmap="inferno", vmin=0, vmax=v)
axes[0][0].set_title("w035 CONTROL (letters)")
axes[1][0].axis("off")
for i, k in enumerate(keys, start=1):
    a = imgs[k]
    vv = max(np.percentile(a, 99.9), 1)
    axes[0][i].imshow(a[::4, ::4], cmap="inferno", vmin=0, vmax=vv)
    axes[0][i].set_title(k.replace("_hybrid_3d2d-seed42_step-075000", "").replace("auto_grown_", "ag_")[:44], fontsize=8)
    rk = k.replace("075000.tif", "075000_reverse.tif")
    if rk in imgs:
        b = imgs[rk]
        vv2 = max(np.percentile(b, 99.9), 1)
        axes[1][i].imshow(b[::4, ::4], cmap="inferno", vmin=0, vmax=vv2)
        axes[1][i].set_title("(reverse)", fontsize=8)
for row_ in axes:
    for ax in row_:
        ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ink9um_1203_gallery.png"), dpi=85)
print("wrote ink9um_1203_gallery.png")
