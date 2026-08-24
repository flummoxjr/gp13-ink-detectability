"""Reference for mesh_bias_survey.py: the same along-normal probe on the w035 control.

w035 (PHerc0139) is the segment where ink_9um reproduces human-verified Greek letters at pixel
AUC 0.9991, so its mesh is known to sit on the readable recto surface.  Running the identical
estimator here tells us whether the PHerc1203 numbers mean "meshes are off the sheet" or just
"the estimator is weak".
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
from mesh_bias_survey import profile, W, NS, NOFF, fetch_tif  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
VOL = "PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr"
MESH = "PHerc0139/segments/20260317000000-w035_2026031718/mesh/20260317000000-on-20250728140407-9.362um.tifxyz/"


def main():
    X = fetch_tif(MESH + "x.tif").astype(np.float64)
    Y = fetch_tif(MESH + "y.tif").astype(np.float64)
    Z = fetch_tif(MESH + "z.tif").astype(np.float64)
    ok = (X > 0) & (Y > 0) & (Z > 0)
    H, Wd = X.shape
    print("w035 stored grid", X.shape, "valid", ok.mean())
    vol = Zarr3D(VOL, 0)
    cands = []
    for i in range(0, H - W, max(1, (H - W) // 14)):
        for j in range(0, Wd - W, max(1, (Wd - W) // 14)):
            if ok[i:i + W, j:j + W].all():
                cands.append((i, j))
    rng = np.random.default_rng(0)
    if len(cands) > 24:
        cands = [cands[k] for k in rng.choice(len(cands), 24, replace=False)]
    rows = []
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
        rows.append(dict(ij=[i, j], z=float(pz.mean()), peak_off_um=float(offs[k] * 9.362),
                         peak=float(prof.max()), at0=float(prof[(NOFF - 1) // 2]),
                         lo=float(prof.min()),
                         contrast=float((prof.max() - prof.min()) / max(1e-6, prof.max()))))
        print(f"w035 ij=({i:3d},{j:3d}) z={pz.mean():6.0f} peak {offs[k]*9.362:+7.1f} um  "
              f"peak {prof.max():5.1f} at0 {prof[(NOFF-1)//2]:5.1f} contrast {rows[-1]['contrast']:.3f}",
              flush=True)
    o = np.array([r["peak_off_um"] for r in rows])
    c = np.array([r["contrast"] for r in rows])
    print(f"\nw035 n={len(o)}: |bias| median {np.median(np.abs(o)):.0f} um, sd {o.std():.0f} um, "
          f"contrast median {np.median(c):.3f}, "
          f"frac |bias|>74um {(np.abs(o)>74).mean()*100:.0f}%")
    json.dump(rows, open(os.path.join(OUT, "mesh_bias_w035.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
