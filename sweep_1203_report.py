"""Report for the 1203 clean-ROI ink sweep: per-ROI probability stats, the
on-sheet vs off-sheet prediction-density artifact metric, and CT+ink overlays.

Artifact metric: ink predictions should live ON papyrus sheets. Using the
released m7-L2 surface prediction (band-L2 frame = L0/4), compute
  density_on  = frac of surf-positive voxels with prob>THR
  density_off = frac of interior non-surface voxels with prob>THR
A real signal has density_on >> density_off; damage-cavity artifacts invert it.
"""
import glob
import json
import os

import numpy as np
import zarr
import fsspec

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\sweep_1203"
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
CT_URL = f"{BUCKET}/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
SURF_URL = f"{BUCKET}/PHerc1203/representations/predictions/surfaces/20260319130212-surface-20260413222639-surface-m7-L2-th0.2.zarr"
THR = 0.5

ct_z = zarr.open(fsspec.get_mapper(CT_URL), mode="r")["0"]
surf_z = zarr.open(fsspec.get_mapper(SURF_URL), mode="r")["0"]  # band-L2 frame

report = {}
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dirs = sorted(glob.glob(os.path.join(OUT, "roi*")))
fig, axes = plt.subplots(2, max(len(dirs), 1), figsize=(4.5 * max(len(dirs), 1), 9), squeeze=False)

for i, d in enumerate(dirs):
    name = os.path.basename(d)
    merged = os.path.join(d, "merged_logits.zarr")
    coords = glob.glob(os.path.join(d, "coordinates_part_*.zarr"))
    if not os.path.exists(merged) or not coords:
        report[name] = {"error": "missing outputs"}
        continue
    cz = np.asarray(zarr.open(coords[0], mode="r")[:])
    o = cz.min(axis=0)
    e = cz.max(axis=0) + 192
    arr = zarr.open(merged, mode="r")
    lg = np.asarray(arr[..., o[0]:e[0], o[1]:e[1], o[2]:e[2]])
    while lg.ndim > 3:
        lg = lg[0]
    prob = 1.0 / (1.0 + np.exp(-lg.astype(np.float32)))

    ct = np.asarray(ct_z[o[0]:e[0], o[1]:e[1], o[2]:e[2]])
    # surface pred at band-L2: slice o/4 .. e/4 then upsample x4
    so, se = o // 4, (e + 3) // 4
    sp = np.asarray(surf_z[so[0]:se[0], so[1]:se[1], so[2]:se[2]]) > 127
    sp_up = np.repeat(np.repeat(np.repeat(sp, 4, 0), 4, 1), 4, 2)
    sp_up = sp_up[:prob.shape[0], :prob.shape[1], :prob.shape[2]]

    interior = ct > 5
    on_sheet = sp_up & interior
    off_sheet = (~sp_up) & interior
    d_on = float((prob[on_sheet] > THR).mean()) if on_sheet.any() else float("nan")
    d_off = float((prob[off_sheet] > THR).mean()) if off_sheet.any() else float("nan")
    d_void = float((prob[~interior] > THR).mean()) if (~interior).any() else float("nan")
    report[name] = {
        "origin": [int(v) for v in o], "shape": list(prob.shape),
        "prob_mean": round(float(prob.mean()), 4),
        "frac_gt_thr": round(float((prob > THR).mean()), 4),
        "density_on_sheet": round(d_on, 4), "density_off_sheet": round(d_off, 4),
        "density_in_void": round(d_void, 4),
        "on_off_ratio": round(d_on / d_off, 2) if d_off and np.isfinite(d_off) and d_off > 0 else None,
    }
    print(name, report[name], flush=True)

    zmid = prob.shape[0] // 2
    axes[0][i].imshow(ct[zmid], cmap="gray")
    axes[0][i].set_title(f"{name} CT z={o[0] + zmid}")
    axes[1][i].imshow(ct[zmid], cmap="gray")
    pm = np.ma.masked_less(prob[zmid], THR)
    axes[1][i].imshow(pm, cmap="autumn", alpha=0.65, vmin=THR, vmax=1)
    axes[1][i].set_title(f"on/off={report[name]['on_off_ratio']}")
    for r in range(2):
        axes[r][i].set_xticks([]); axes[r][i].set_yticks([])

with open(os.path.join(OUT, "sweep_report.json"), "w") as f:
    json.dump(report, f, indent=1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sweep_overlays.png"), dpi=95)
print("wrote sweep_overlays.png", flush=True)
