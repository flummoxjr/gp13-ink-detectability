"""QC the newly grown PHerc0813 meshes BEFORE spending render money.

The PHerc1447 lesson: published segments can be grown into empty volume (one had
91% of vertices on zero voxels). Same check here, plus a depth profile confirming
each mesh sits ON a lamella rather than beside one.

Access pattern matters: sampling voxels individually triggers one chunk fetch per
sample. Instead we take one contiguous mesh window per patch, fetch its enclosing
CT bbox ONCE with a threaded chunk reader, and sample inside that.

Outputs: hunt/pherc0813_mesh_qc.json + .png
"""
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import tifffile
from scipy.ndimage import map_coordinates

MESH_DIR = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\pherc0813_meshes"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt"
CT = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0813/volumes/"
      "20250821151723-9.362um-1.2m-113keV-masked.zarr")
SHAPE = (16993, 7947, 7947)
CHUNK = 128
# Mesh-grid window per patch, in VERTICES. The grow step_size was 20 voxels, so a
# 24-vertex window spans ~480 voxels per axis (~0.1 GB bbox) — 128 spanned 2,560
# voxels per axis and demanded 10+ GB.
WIN = 24
OFFSETS = np.arange(-10, 11)   # voxels along the surface normal


class Bbox:
    """Threaded chunk reader.

    CRITICAL: a failed fetch and a genuinely empty (masked-out) chunk both leave
    zeros in the output array — they are indistinguishable downstream, and a
    network fault would masquerade as 'this mesh was grown into empty space'.
    So failures are counted separately and surfaced; any window with failures is
    reported as UNKNOWN rather than empty.
    """

    def __init__(self):
        self.s = requests.Session()
        self.last_missing = 0   # chunks the server said do not exist (real emptiness)
        self.last_failed = 0    # chunks we could not retrieve (network/unknown)

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
            for attempt in range(3):
                try:
                    r = self.s.get(f"{CT}/0/{iz}/{iy}/{ix}", timeout=90)
                    if r.status_code == 404:
                        return j, None, "missing"      # server says no such chunk
                    r.raise_for_status()
                    return j, np.frombuffer(r.content, np.uint8).reshape(CHUNK, CHUNK, CHUNK), "ok"
                except Exception:
                    if attempt == 2:
                        return j, None, "failed"       # could not retrieve
            return j, None, "failed"

        self.last_missing = self.last_failed = 0
        with ThreadPoolExecutor(8) as ex:
            for j, c, status in ex.map(get, jobs):
                if c is None:
                    if status == "missing":
                        self.last_missing += 1
                    else:
                        self.last_failed += 1
                    continue
                iz, iy, ix = j
                zs, ys, xs = iz * CHUNK, iy * CHUNK, ix * CHUNK
                a0, b0, c0 = max(zs, z0), max(ys, y0), max(xs, x0)
                a1, b1, c1 = min(zs + CHUNK, z1), min(ys + CHUNK, y1), min(xs + CHUNK, x1)
                out[a0 - z0:a1 - z0, b0 - y0:b1 - y0, c0 - x0:c1 - x0] = \
                    c[a0 - zs:a1 - zs, b0 - ys:b1 - ys, c0 - xs:c1 - xs]
        return out, (z0, y0, x0)


vol = Bbox()
results = []
for d in sorted(glob.glob(os.path.join(MESH_DIR, "auto_grown_*"))):
    name = os.path.basename(d)
    meta = json.load(open(os.path.join(d, "meta.json")))
    area_cm2 = round(float(meta.get("area_vx2", 0)) * (9.362 ** 2) * 1e-8, 2)
    x = tifffile.imread(os.path.join(d, "x.tif")).astype(np.float64)
    y = tifffile.imread(os.path.join(d, "y.tif")).astype(np.float64)
    z = tifffile.imread(os.path.join(d, "z.tif")).astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x > 0) & (y > 0) & (z > 0)
    if valid.sum() < 500:
        results.append({"name": name, "area_cm2": area_cm2, "error": "too few valid vertices"})
        print(f"{name}: too few valid vertices"); continue

    # densest valid window of the mesh grid
    from scipy.ndimage import uniform_filter
    dens = uniform_filter(valid.astype(np.float32), size=min(WIN, min(valid.shape)), mode="constant")
    r0, c0 = np.unravel_index(np.argmax(dens), dens.shape)
    r0 = int(np.clip(r0 - WIN // 2, 0, max(valid.shape[0] - WIN, 0)))
    c0 = int(np.clip(c0 - WIN // 2, 0, max(valid.shape[1] - WIN, 0)))
    sl = (slice(r0, r0 + WIN), slice(c0, c0 + WIN))
    xs, ys, zs, vs = x[sl], y[sl], z[sl], valid[sl]
    if vs.sum() < 200:
        results.append({"name": name, "area_cm2": area_cm2, "error": "window empty"})
        continue

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
    if gb > 1.5:
        results.append({"name": name, "area_cm2": area_cm2, "error": f"window bbox {gb:.2f} GB"})
        print(f"{name}: window bbox too large ({gb:.2f} GB), skipped"); continue
    sub, (oz, oy, ox) = vol.fetch(z0, z1, y0, y1, x0, x1)
    if vol.last_failed:
        results.append({"name": name, "area_cm2": area_cm2,
                        "error": f"UNKNOWN — {vol.last_failed} chunk fetches failed "
                                 f"(network); cannot distinguish empty from unfetched"})
        print(f"{name}: SKIPPED — {vol.last_failed} chunk fetches failed", flush=True)
        continue

    prof = []
    for off in OFFSETS:
        vals = map_coordinates(sub.astype(np.float32),
                               [zs + nz * off - oz, ys + ny * off - oy, xs + nx * off - ox],
                               order=1, mode="constant", cval=0.0)
        prof.append(float(vals[vs].mean()))
    prof = np.array(prof)
    surf = map_coordinates(sub.astype(np.float32), [zs - oz, ys - oy, xs - ox],
                           order=1, mode="constant", cval=0.0)[vs]
    rec = {
        "name": name, "area_cm2": area_cm2,
        "valid_vertices": int(valid.sum()),
        "chunks_missing_404": int(vol.last_missing),
        "surface_mean_DN": round(float(surf.mean()), 1),
        "surface_zero_frac": round(float((surf < 5).mean()), 4),
        "profile": [round(v, 1) for v in prof],
        "peak_offset": int(OFFSETS[int(np.argmax(prof))]),
        "contrast": round(float(prof.max() - prof.min()), 1),
    }
    results.append(rec)
    print(f"{name}: {area_cm2} cm2 | surface {rec['surface_mean_DN']} DN | "
          f"empty {rec['surface_zero_frac']:.3f} | peak@{rec['peak_offset']} | "
          f"contrast {rec['contrast']}", flush=True)

json.dump(results, open(os.path.join(OUT, "pherc0813_mesh_qc.json"), "w"), indent=1)
good = [r for r in results if r.get("surface_zero_frac", 1) < 0.2]
print(f"\n=== {len(good)}/{len(results)} meshes on material (<20% empty) ===")
print(f"usable new surface: {sum(r['area_cm2'] for r in good):.1f} cm2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 5.5))
for r in results:
    if "profile" in r:
        ax.plot(OFFSETS, r["profile"], marker="o", ms=3,
                label=f"...{r['name'][-6:]} ({r['area_cm2']} cm²)")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("offset along surface normal (voxels)")
ax.set_ylabel("mean CT value (DN)")
ax.set_title("PHerc0813 newly grown meshes — depth profile through the surface")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pherc0813_mesh_qc.png"), dpi=110)
print("wrote pherc0813_mesh_qc.png")
