"""QC test 2: firing morphology of pulled prob tiles vs CT and surface prediction.

For each pulled 64^3 probL2 tile: stream matching CT at L2 and surface pred
(level 0 = band-L2 grid) from S3, compute prediction density on-sheet /
off-sheet-interior / background / near-edge, classify, render gallery.
"""
import glob
import json
import os

import numpy as np
import zarr
import fsspec
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RND = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\round_1"
OUTJ = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\qc_morph_result.json"
OUTP = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\qc_morph_gallery.png"
VOL = "vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
SURF = ("vesuvius-challenge-open-data/PHerc1203/representations/predictions/surfaces/"
        "20260319130212-surface-20260413222639-surface-m7-L2-th0.2.zarr")

fs = fsspec.filesystem("s3", anon=True)
ct2 = zarr.open(fs.get_mapper(VOL), mode="r")["2"]
surf_root = zarr.open(fs.get_mapper(SURF), mode="r")
try:
    surf = surf_root["0"]
except (KeyError, TypeError):
    surf = surf_root
print("ct L2 shape", ct2.shape, "surf shape", surf.shape, flush=True)

files = sorted(glob.glob(os.path.join(RND, "w*_prob_*.npy")))
rows = []
panels = []
for fp in files:
    base = os.path.basename(fp)[:-4]
    w = base.split("_")[0]
    oz, oy, ox = (int(v) for v in base.split("_")[2:5])
    cz, cy, cx = oz // 4, oy // 4, ox // 4
    prob = np.load(fp).astype(np.float32) / 255.0
    ct = np.asarray(ct2[cz:cz + 64, cy:cy + 64, cx:cx + 64]).astype(np.float32)
    sf = np.asarray(surf[cz:cz + 64, cy:cy + 64, cx:cx + 64])
    interior = ct > 5
    sheet = (sf > 127) & interior
    bg = ~interior
    offsheet = interior & ~sheet
    edge = binary_dilation(bg, iterations=2) & interior  # 2 L2-voxels (~8 L0) from mask edge

    def dens(m):
        if m.sum() < 50:
            return None
        return {"mean_p": round(float(prob[m].mean()), 4),
                "f05": round(float((prob[m] > 0.5).mean()), 4),
                "vox": int(m.sum())}

    r = {"file": base, "w": w, "tile": [oz, oy, ox],
         "frac_interior": round(float(interior.mean()), 3),
         "frac_sheet_of_interior": round(float(sheet.sum() / max(interior.sum(), 1)), 3),
         "on_sheet": dens(sheet), "off_sheet": dens(offsheet),
         "background": dens(bg), "edge_band": dens(edge)}

    # classification
    on = r["on_sheet"]["f05"] if r["on_sheet"] else 0.0
    off = r["off_sheet"]["f05"] if r["off_sheet"] else 0.0
    bgf = r["background"]["f05"] if r["background"] else 0.0
    ed = r["edge_band"]["f05"] if r["edge_band"] else 0.0
    if on > 2 * max(off, 1e-4) and on > 0.02:
        r["verdict"] = "sheet-following"
    elif ed > 2 * max(off, 1e-4) and ed > 0.02:
        r["verdict"] = "edge-halo"
    elif off > 1.5 * max(on, 1e-4) or bgf > 0.02:
        r["verdict"] = "cavity-blobbing"
    else:
        r["verdict"] = "diffuse-noise"
    rows.append(r)
    print(json.dumps(r), flush=True)

    # panel: slice with max prob mass
    zi = int(prob.sum(axis=(1, 2)).argmax())
    panels.append((base, r["verdict"], ct[zi], prob[zi], sf[zi]))

with open(OUTJ, "w") as f:
    json.dump(rows, f, indent=1)

# aggregate
agg = {}
for key in ("on_sheet", "off_sheet", "background", "edge_band"):
    vals = [r[key]["f05"] for r in rows if r[key]]
    mp = [r[key]["mean_p"] for r in rows if r[key]]
    agg[key] = {"f05_median": round(float(np.median(vals)), 4),
                "f05_mean": round(float(np.mean(vals)), 4),
                "mean_p_median": round(float(np.median(mp)), 4), "n": len(vals)}
verd = {}
for r in rows:
    verd[r["verdict"]] = verd.get(r["verdict"], 0) + 1
print("AGG", json.dumps(agg, indent=1))
print("VERDICTS", json.dumps(verd))
with open(OUTJ, "w") as f:
    json.dump({"tiles": rows, "aggregate": agg, "verdict_counts": verd}, f, indent=1)

# gallery: rows = tiles, cols = CT | prob | overlay(+sheet contour)
n = len(panels)
fig, axes = plt.subplots(n, 3, figsize=(9.6, 3.2 * n))
if n == 1:
    axes = axes[None, :]
for i, (base, v, ct_s, p_s, sf_s) in enumerate(panels):
    ax = axes[i, 0]
    ax.imshow(ct_s, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{base}\nCT L2", fontsize=7)
    ax = axes[i, 1]
    ax.imshow(p_s, cmap="inferno", vmin=0, vmax=1)
    ax.set_title(f"prob ({v})", fontsize=7)
    ax = axes[i, 2]
    ax.imshow(ct_s, cmap="gray", vmin=0, vmax=255)
    ax.imshow(np.ma.masked_less(p_s, 0.5), cmap="autumn", vmin=0, vmax=1, alpha=0.6)
    ax.contour(sf_s > 127, levels=[0.5], colors="cyan", linewidths=0.4)
    ax.set_title("overlay + sheet", fontsize=7)
    for a in axes[i]:
        a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
plt.savefig(OUTP, dpi=110)
print("WROTE", OUTP, flush=True)
