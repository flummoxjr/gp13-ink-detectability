"""Investigation D step 4b: how far off the sheet do the PHerc1203 auto-grown meshes sit?

Risk #1 for the 2.4 um path is mesh placement, not registration.  This measures the bias
distribution cheaply in the 9.362 um volume (4 MiB per patch): for many patches across the
in-band segments, sample the volume along the surface normal and record where the sheet peak
falls relative to the mesh.
"""
import io
import json
import os
import sys

import numpy as np
import requests
import tifffile
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zarr_http import Zarr3D  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
P9 = "PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr"
BAND = (7940.0, 11825.0)
W = 9              # stored cells per patch
NS = 48            # samples per side
NOFF = 41          # offsets -20..+20 voxels of 9.362 um = +-187 um


def fetch_tif(key):
    return tifffile.imread(io.BytesIO(requests.get(f"{B}/{key}", timeout=300).content))


def profile(vol, px, py, pz):
    du = np.stack([np.gradient(px, axis=0), np.gradient(py, axis=0), np.gradient(pz, axis=0)], -1)
    dv = np.stack([np.gradient(px, axis=1), np.gradient(py, axis=1), np.gradient(pz, axis=1)], -1)
    nrm = np.cross(du, dv)
    nrm /= np.maximum(1e-9, np.linalg.norm(nrm, axis=-1, keepdims=True))
    offs = np.arange(NOFF) - (NOFF - 1) // 2
    pts = np.stack([pz, py, px], -1)
    nzyx = np.stack([nrm[..., 2], nrm[..., 1], nrm[..., 0]], -1)
    allp = pts[None] + offs[:, None, None, None] * nzyx[None]
    flat = allp.reshape(-1, 3)
    lo = np.maximum(0, np.floor(flat.min(0)).astype(int) - 2)
    hi = np.ceil(flat.max(0)).astype(int) + 3
    if np.prod(hi - lo) > 40e6:
        return None
    blk = vol.read(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], workers=16).astype(np.float32)
    v = map_coordinates(blk, (flat - lo).T, order=1, mode="constant", cval=0)
    return offs, v.reshape(NOFF, -1).mean(1)


def main():
    cat = json.load(open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod\segment_catalog.json"))
    ext = json.load(open(os.path.join(OUT, "band_extents.json")))["segments"]
    order = sorted(ext, key=lambda r: -(r["area_cm2"] * (r.get("pt_in_bandA") or 0)))[:8]
    vol = Zarr3D(P9, 0)
    rows = []
    for r in order:
        d = r["seg_dir"]
        X = fetch_tif(d + "x.tif").astype(np.float64)
        Y = fetch_tif(d + "y.tif").astype(np.float64)
        Z = fetch_tif(d + "z.tif").astype(np.float64)
        ok = (X > 0) & (Y > 0) & (Z > 0) & (Z > BAND[0]) & (Z < BAND[1])
        H, Wd = X.shape
        cands = []
        for i in range(0, H - W, max(1, (H - W) // 12)):
            for j in range(0, Wd - W, max(1, (Wd - W) // 12)):
                if ok[i:i + W, j:j + W].all():
                    cands.append((i, j))
        rng = np.random.default_rng(0)
        if len(cands) > 6:
            cands = [cands[k] for k in rng.choice(len(cands), 6, replace=False)]
        for (i, j) in cands:
            u = np.linspace(0, W - 1, NS)
            gj, gi = np.meshgrid(u, u, indexing="xy")
            c = np.stack([gi.ravel(), gj.ravel()])
            px = map_coordinates(X[i:i + W, j:j + W], c, order=1).reshape(NS, NS)
            py = map_coordinates(Y[i:i + W, j:j + W], c, order=1).reshape(NS, NS)
            pz = map_coordinates(Z[i:i + W, j:j + W], c, order=1).reshape(NS, NS)
            res = profile(vol, px, py, pz)
            if res is None:
                continue
            offs, prof = res
            k = int(np.argmax(prof))
            rows.append(dict(seg=r["name"], ij=[i, j], z=float(pz.mean()),
                             peak_off_vox=int(offs[k]), peak_off_um=float(offs[k] * 9.362),
                             peak=float(prof.max()), at0=float(prof[(NOFF - 1) // 2]),
                             lo=float(prof.min()),
                             contrast=float((prof.max() - prof.min()) / max(1e-6, prof.max()))))
            print(f"{r['name'][-8:]} ij=({i:3d},{j:3d}) z={pz.mean():6.0f} "
                  f"peak {offs[k]:+3d} vox = {offs[k]*9.362:+7.1f} um   "
                  f"peak {prof.max():5.1f} at0 {prof[(NOFF-1)//2]:5.1f}", flush=True)
    if rows:
        off = np.array([r["peak_off_um"] for r in rows])
        print(f"\nn={len(off)} patches: |bias| median {np.median(np.abs(off)):.0f} um, "
              f"mean {off.mean():+.0f} um, sd {off.std():.0f} um, "
              f"p90 |bias| {np.percentile(np.abs(off), 90):.0f} um, max {np.abs(off).max():.0f} um")
        print(f"fraction with |bias| > 74 um (half a 62-layer 2.4um window): "
              f"{(np.abs(off) > 74).mean()*100:.0f}%")
    json.dump(rows, open(os.path.join(OUT, "mesh_bias_survey.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
