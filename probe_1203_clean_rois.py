"""Find clean parallel-sheet ROIs in PHerc1203's 2.4um band using the RELEASED
m7-L2 surface prediction (coherent-sheet regions), avoiding the damaged zone
the first smoke ROI landed in.

Scoring per candidate window (at surface-pred low-res): surface-voxel fraction
in a healthy band (sheets occupy ~15-45% of volume when parallel and resolved)
+ high z-column consistency (sheets continue through the window) + papyrus fill
from the CT. Picks 6 well-separated ROIs, mapped to band L0 coords.
Outputs: trackD/out/probe_1203_clean_rois.{json,png}
"""
import json
import os

import numpy as np
import zarr
import fsspec
from scipy.ndimage import uniform_filter

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
SURF = f"{BUCKET}/PHerc1203/representations/predictions/surfaces/20260319130212-surface-20260413222639-surface-m7-L2-th0.2.zarr"
CT = f"{BUCKET}/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"
CACHE = r"D:\vesuvius-data\trackD"

gs = zarr.open(fsspec.get_mapper(SURF), mode="r")
keys = []
for k in ["0", "1", "2", "3", "4"]:
    try:
        a = gs[k]
        keys.append((k, a.shape, str(a.dtype)))
    except Exception:
        pass
print("surface pred levels:", keys)

# use a level of the surface pred that's ~40-80MB
lvl = None
for k, shape, dt in keys:
    vox = shape[0] * shape[1] * shape[2] if len(shape) == 3 else np.prod(shape)
    if vox < 3e8:
        lvl = k
        break
print("using surface level", lvl)
cache = os.path.join(CACHE, f"surf1203_L{lvl}.npy")
if os.path.exists(cache):
    sp = np.load(cache)
else:
    sp = np.asarray(gs[lvl][:])
    np.save(cache, sp)
print("surface pred:", sp.shape, sp.dtype, "nonzero frac:", float((sp > 0).mean()))

binary = sp > 127 if sp.dtype == np.uint8 else sp > 0.5

# interior mask from the CT itself (the surface-pred has a border-ring artifact
# at coarse levels that the first pass latched onto): papyrus fill at CT L5,
# downsampled to the surf-pred grid, heavily eroded.
ct_cache = os.path.join(CACHE, "ct1203_L5.npy")
if os.path.exists(ct_cache):
    ct5 = np.load(ct_cache)
else:
    ct5 = np.asarray(zarr.open(fsspec.get_mapper(CT), mode="r")["5"][:])
    np.save(ct_cache, ct5)
print("CT L5:", ct5.shape)
# CT L5 voxel = 32 x L0; surf-pred level `lvl` voxel = (2**lvl)*4 x L0
ct_to_sp = (2 ** int(lvl)) * 4 // 32
if ct_to_sp >= 1:
    ctm = ct5[::ct_to_sp, ::ct_to_sp, ::ct_to_sp] > 5
else:
    ctm = np.repeat(np.repeat(np.repeat(ct5 > 5, 2, 0), 2, 1), 2, 2)
# align shapes
mz = min(ctm.shape[0], sp.shape[0]); my = min(ctm.shape[1], sp.shape[1]); mx = min(ctm.shape[2], sp.shape[2])
interior = np.zeros(sp.shape, bool)
from scipy.ndimage import binary_erosion
interior[:mz, :my, :mx] = binary_erosion(ctm[:mz, :my, :mx], iterations=10)
print("interior frac:", float(interior.mean()))

W = 24  # window at this level
sheet_frac = uniform_filter(binary.astype(np.float32), size=W, mode="constant")
# z-continuity: fraction of (y,x) columns in the window that contain surface in
# most z-slices — approximate via averaging along z then thresholding
col_any = uniform_filter(binary.astype(np.float32), size=(3, W, W), mode="constant")
zcont = uniform_filter((col_any > 0.05).astype(np.float32), size=(W, 1, 1), mode="constant")

inner_ok = uniform_filter(interior.astype(np.float32), size=W, mode="constant") > 0.995
score = np.where(inner_ok & (sheet_frac > 0.10) & (sheet_frac < 0.45), sheet_frac * zcont, 0)

picks = []
s = score.copy()
for _ in range(200):
    if s.max() <= 0 or len(picks) >= 6:
        break
    zyx = np.unravel_index(np.argmax(s), s.shape)
    if all(max(abs(a - b) for a, b in zip(zyx, p)) >= W * 2 for p in picks):
        picks.append(tuple(int(v) for v in zyx))
    z, y, x = zyx
    s[max(z - W, 0):z + W, max(y - W, 0):y + W, max(x - W, 0):x + W] = 0

# surface pred level L in ITS pyramid; its L0 corresponds to band L2 (m7-L2 model
# ran at band level 2). So band-L0 coord = surfpred_coord * (2**int(lvl)) * 4.
scale_to_band_L0 = (2 ** int(lvl)) * 4
rois = []
for (z, y, x) in picks:
    c = [int(z * scale_to_band_L0), int(y * scale_to_band_L0), int(x * scale_to_band_L0)]
    rois.append({"surfpred_zyx": [z, y, x], "band_L0_center": c,
                 "score": round(float(score[z, y, x]), 4),
                 "bbox": f"{c[0]-192}:{c[0]+192},{c[1]-192}:{c[1]+192},{c[2]-192}:{c[2]+192}"})
    print(rois[-1])

with open(os.path.join(OUT, "probe_1203_clean_rois.json"), "w") as f:
    json.dump({"surf_level": lvl, "scale_to_band_L0": scale_to_band_L0, "rois": rois}, f, indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(16, 10), squeeze=False)
for ax, r in zip(axes.ravel(), rois):
    z = r["surfpred_zyx"][0]
    ax.imshow(binary[z], cmap="gray")
    ax.add_patch(plt.Rectangle((r["surfpred_zyx"][2] - W // 2, r["surfpred_zyx"][1] - W // 2),
                               W, W, fill=False, color="red", lw=2))
    ax.set_title(f"surfpred z={z} score={r['score']}")
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "probe_1203_clean_rois.png"), dpi=95)
print("wrote probe_1203_clean_rois.png")
