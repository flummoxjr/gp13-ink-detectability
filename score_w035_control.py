"""Score the pod's ink_9um w035 predictions against the human ink labels (acceptance gate)."""
import json
import os

import numpy as np
import tifffile
from scipy.stats import rankdata

PRED_DIR = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_w035"
CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"

ink2d = (np.load(os.path.join(CACHE, "w035_ink.npy")) > 0).max(axis=0)
sup2d = (np.load(os.path.join(CACHE, "w035_sup.npy")) > 0).max(axis=0)
surf_nonzero = (np.load(os.path.join(CACHE, "w035_surf.npy")) > 0).mean(axis=0) > 0.9
valid = sup2d & surf_nonzero
print(f"labels: ink px {int((ink2d & valid).sum())}, bg px {int((valid & ~ink2d).sum())}")


def tie_auc(pos, neg, max_n=400_000):
    rng = np.random.default_rng(0)
    if len(pos) > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if len(neg) > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    allv = np.concatenate([pos, neg])
    r = rankdata(allv)
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


results = {}
best = (None, 0.5, None)
for fn in sorted(os.listdir(PRED_DIR)):
    if not fn.endswith(".tif"):
        continue
    p = tifffile.imread(os.path.join(PRED_DIR, fn)).astype(np.float32)
    if p.shape != ink2d.shape:
        print(f"{fn}: shape {p.shape} vs labels {ink2d.shape} — cropping to min")
        h = min(p.shape[0], ink2d.shape[0]); w = min(p.shape[1], ink2d.shape[1])
        pc, ic, vc = p[:h, :w], ink2d[:h, :w], valid[:h, :w]
    else:
        pc, ic, vc = p, ink2d, valid
    auc = tie_auc(pc[ic & vc], pc[vc & ~ic])
    frac_fired = float((pc > 0.5 * pc.max()).mean()) if pc.max() > 0 else 0.0
    results[fn] = {"auc": round(auc, 4), "pred_range": [float(pc.min()), float(pc.max())],
                   "frac_gt_half": round(frac_fired, 4)}
    print(fn, results[fn])
    if abs(auc - 0.5) > abs(best[1] - 0.5):
        best = (fn, auc, pc)

with open(os.path.join(OUT, "ink9um_w035_scores.json"), "w") as f:
    json.dump(results, f, indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fn, auc, pc = best
fig, axes = plt.subplots(1, 3, figsize=(21, 8))
v1 = np.percentile(pc, 99.5)
axes[0].imshow(pc, cmap="inferno", vmin=0, vmax=v1 if v1 > 0 else 1)
axes[0].set_title(f"{fn} (AUC {auc:.3f})")
axes[1].imshow(ink2d, cmap="gray")
axes[1].set_title("human ink labels (max-z)")
ov = np.where(valid, pc, np.nan)
axes[2].imshow(ov, cmap="inferno", vmin=0, vmax=v1 if v1 > 0 else 1)
axes[2].contour(ink2d, levels=[0.5], colors="cyan", linewidths=0.4)
axes[2].set_title("pred within supervision + label contours")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ink9um_w035_control.png"), dpi=90)
print("wrote ink9um_w035_control.png; best:", fn, auc)
