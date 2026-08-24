"""Positive control for the mesh-QC depth profile.

Runs the IDENTICAL profile code used on the new PHerc0813 meshes against
PHerc0139 w035 — the segment where letters are proven — to separate
"PHerc0813 laminae are unresolved" from "our normals/method are wrong".
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import tifffile
from scipy.ndimage import map_coordinates, uniform_filter

HUNT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt"
CT = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/"
      "20250728140407-9.362um-1.2m-113keV-masked.zarr")
SHAPE = (14376, 5820, 5240)   # corrected below from .zarray if reachable
CHUNK = 128
OFFSETS = np.arange(-10, 11)

try:
    import fsspec, zarr
    arr = zarr.open(fsspec.get_mapper(CT), mode="r")["0"]
    SHAPE = tuple(arr.shape)
except Exception as e:
    print("shape probe failed, using default:", str(e)[:80])
print("control CT shape:", SHAPE)


class Bbox:
    def __init__(self):
        self.s = requests.Session()

    def fetch(self, z0, z1, y0, y1, x0, x1):
        z0, y0, x0 = max(z0, 0), max(y0, 0), max(x0, 0)
        z1, y1, x1 = min(z1, SHAPE[0]), min(y1, SHAPE[1]), min(x1, SHAPE[2])
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        jobs = [(iz, iy, ix)
                for iz in range(z0 // CHUNK, (z1 - 1) // CHUNK + 1)
                for iy in range(y0 // CHUNK, (y1 - 1) // CHUNK + 1)
                for ix in range(x0 // CHUNK, (x1 - 1) // CHUNK + 1)]

        def get(j):
            iz, iy, ix = j
            for _ in range(3):
                try:
                    r = self.s.get(f"{CT}/0/{iz}/{iy}/{ix}", timeout=90)
                    if r.status_code == 404:
                        return j, None
                    r.raise_for_status()
                    return j, np.frombuffer(r.content, np.uint8).reshape(CHUNK, CHUNK, CHUNK)
                except Exception:
                    pass
            return j, None

        with ThreadPoolExecutor(8) as ex:
            for j, c in ex.map(get, jobs):
                if c is None:
                    continue
                iz, iy, ix = j
                zs, ys, xs = iz * CHUNK, iy * CHUNK, ix * CHUNK
                a0, b0, c0 = max(zs, z0), max(ys, y0), max(xs, x0)
                a1, b1, c1 = min(zs + CHUNK, z1), min(ys + CHUNK, y1), min(xs + CHUNK, x1)
                out[a0 - z0:a1 - z0, b0 - y0:b1 - y0, c0 - x0:c1 - x0] = \
                    c[a0 - zs:a1 - zs, b0 - ys:b1 - ys, c0 - xs:c1 - xs]
        return out, (z0, y0, x0)


def profile(mesh_dir, win_vertices, label, max_gb=1.5):
    x = tifffile.imread(os.path.join(mesh_dir, "x.tif")).astype(np.float64)
    y = tifffile.imread(os.path.join(mesh_dir, "y.tif")).astype(np.float64)
    z = tifffile.imread(os.path.join(mesh_dir, "z.tif")).astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0) & (y > 0) & (z > 0)
    print(f"{label}: grid {x.shape}, valid {valid.mean():.2f}")
    w = min(win_vertices, min(valid.shape))
    dens = uniform_filter(valid.astype(np.float32), size=w, mode="constant")
    r0, c0 = np.unravel_index(np.argmax(dens), dens.shape)
    r0 = int(np.clip(r0 - w // 2, 0, max(valid.shape[0] - w, 0)))
    c0 = int(np.clip(c0 - w // 2, 0, max(valid.shape[1] - w, 0)))
    sl = (slice(r0, r0 + w), slice(c0, c0 + w))
    xs, ys, zs, vs = x[sl], y[sl], z[sl], valid[sl]
    nx_, ny_, nz_ = np.gradient(xs), np.gradient(ys), np.gradient(zs)
    nx = ny_[0] * nz_[1] - nz_[0] * ny_[1]
    ny = nz_[0] * nx_[1] - nx_[0] * nz_[1]
    nz = nx_[0] * ny_[1] - ny_[0] * nx_[1]
    nrm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-9
    nx, ny, nz = nx / nrm, ny / nrm, nz / nrm
    pad = 14
    z0, z1 = int(zs[vs].min()) - pad, int(zs[vs].max()) + pad + 1
    y0, y1 = int(ys[vs].min()) - pad, int(ys[vs].max()) + pad + 1
    x0, x1 = int(xs[vs].min()) - pad, int(xs[vs].max()) + pad + 1
    gb = (z1 - z0) * (y1 - y0) * (x1 - x0) / 1e9
    print(f"  window {w} vertices -> bbox {gb:.2f} GB")
    if gb > max_gb:
        return None
    sub, (oz, oy, ox) = Bbox().fetch(z0, z1, y0, y1, x0, x1)
    prof = []
    for off in OFFSETS:
        vals = map_coordinates(sub.astype(np.float32),
                               [zs + nz * off - oz, ys + ny * off - oy, xs + nx * off - ox],
                               order=1, mode="constant", cval=0.0)
        prof.append(float(vals[vs].mean()))
    prof = np.array(prof)
    print(f"  profile min {prof.min():.1f} max {prof.max():.1f} contrast {prof.max()-prof.min():.1f} "
          f"peak@{OFFSETS[int(np.argmax(prof))]}")
    return prof


# w035 mesh grid is FULL-resolution (scale 1 vertex ~ 1 voxel), so a comparable
# physical window is ~480 voxels => 480 vertices, vs 24 on the 20-voxel-step
# PHerc0813 meshes. Try both to be sure the comparison is like-for-like.
ctrl_dir = None
for cand in glob.glob(os.path.join(HUNT, "meshcache", "w035", "**"), recursive=True):
    if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "x.tif")):
        ctrl_dir = cand
        break
if not ctrl_dir:
    print("control mesh not found under hunt/meshcache/w035")
    sys.exit(1)
print("control mesh:", ctrl_dir)
meta = {}
mp = os.path.join(ctrl_dir, "meta.json")
if os.path.exists(mp):
    meta = json.load(open(mp))
    print("control meta scale:", meta.get("scale"))

out = {}
for w in (24, 120, 480):
    p = profile(ctrl_dir, w, f"w035 (win {w})")
    if p is not None:
        out[f"w035_win{w}"] = [round(v, 1) for v in p]

json.dump(out, open(os.path.join(HUNT, "control_profile.json"), "w"), indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
qc = json.load(open(os.path.join(HUNT, "pherc0813_mesh_qc.json")))
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for k, v in out.items():
    axes[0].plot(OFFSETS, v, marker="o", ms=3, label=k)
axes[0].set_title("CONTROL PHerc0139 w035 (letters proven)")
for r in qc:
    if r.get("profile") and r.get("surface_zero_frac", 1) < 0.2:
        axes[1].plot(OFFSETS, r["profile"], marker="o", ms=3, label=f"...{r['name'][-6:]}")
axes[1].set_title("PHerc0813 newly grown (on-material only)")
for a in axes:
    a.axvline(0, color="k", lw=0.8, ls="--")
    a.set_xlabel("offset along surface normal (voxels)")
    a.set_ylabel("mean CT (DN)")
    a.legend(fontsize=7)
fig.suptitle("Does a known-good surface show lamella contrast under identical code?")
fig.tight_layout()
fig.savefig(os.path.join(HUNT, "control_vs_0813_profile.png"), dpi=110)
print("wrote control_vs_0813_profile.png")
