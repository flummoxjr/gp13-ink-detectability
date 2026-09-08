"""Re-express a tifxyz mesh in another volume's coordinates (the open-data convention):
apply a 3x4 affine (rows x, y, z; p_out = M . [p_in, 1]) to every stored cell and resample the
stored grid by the transform's linear scale so one stored cell stays 1/scale = 20 output voxels
(PHerc0139 w035: the published 9.362 um mesh is 291 x 262 stored cells; the published
"on-20260102150214-2.399um" mesh is 1132 x 1020 = round(291 x 3.8925), scale still 0.05).

  python tifxyz_transform.py <in_dir> <out_dir> --matrix M.json [--inverse] [--crop r0 r1 c0 c1]
                             [--keep-tiles tiles.json --tile 9 --margin 1] [--factor F]

--matrix     JSON with "transformation_matrix" (3 rows of 4). With --inverse the matrix maps
             out -> in (the catalogue publishes 2.4um -> 9.362um; we go 9.362 -> 2.4).
--crop       keep only stored rows r0:r1, cols c0:c1 of the INPUT grid (others invalid).
--keep-tiles JSON list of {"seg":..., "ij":[i,j]} (c1_tiles.jsonl rows) -- keep the union of
             tile x tile cells at those input-grid origins (+margin cells); others invalid.
Validity follows villa (z > 0 and finite); invalid output cells are written as 0.
"""
import argparse
import json
import os

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates


def load(d):
    X = tifffile.imread(os.path.join(d, "x.tif")).astype(np.float64)
    Y = tifffile.imread(os.path.join(d, "y.tif")).astype(np.float64)
    Z = tifffile.imread(os.path.join(d, "z.tif")).astype(np.float64)
    meta = json.load(open(os.path.join(d, "meta.json")))
    valid = (Z > 0) & np.isfinite(Z) & np.isfinite(X) & np.isfinite(Y)
    return X, Y, Z, valid, meta


def affine_3x4(m, inverse):
    A = np.array(m, dtype=np.float64)
    assert A.shape == (3, 4), A.shape
    if inverse:
        R, t = A[:, :3], A[:, 3]
        Ri = np.linalg.inv(R)
        A = np.concatenate([Ri, (-Ri @ t)[:, None]], axis=1)
    return A


def resample_grid(F, valid, factor):
    """Bilinear resample of a coordinate raster by `factor` (area-aligned: out cell I sits at
    in position (I + 0.5) / factor - 0.5). Cells whose 2x2 support is not fully valid are invalid."""
    h, w = F.shape
    H, W = int(round(h * factor)), int(round(w * factor))
    ii = (np.arange(H) + 0.5) / factor - 0.5
    jj = (np.arange(W) + 0.5) / factor - 0.5
    gi, gj = np.meshgrid(ii, jj, indexing="ij")
    coords = np.stack([gi.ravel(), gj.ravel()])
    out = map_coordinates(F, coords, order=1, mode="nearest").reshape(H, W)
    vmin = map_coordinates(valid.astype(np.float64), coords, order=1, mode="constant", cval=0.0).reshape(H, W)
    return out, vmin > 0.999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--inverse", action="store_true")
    ap.add_argument("--crop", nargs=4, type=int, default=None, metavar=("R0", "R1", "C0", "C1"))
    ap.add_argument("--keep-tiles", default=None)
    ap.add_argument("--tile", type=int, default=9)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--factor", type=float, default=0.0, help="stored-grid resample factor (default: transform scale)")
    a = ap.parse_args()

    X, Y, Z, valid, meta = load(a.in_dir)
    h, w = X.shape
    M = affine_3x4(json.load(open(a.matrix))["transformation_matrix"], a.inverse)
    scale = float(np.cbrt(abs(np.linalg.det(M[:, :3]))))
    factor = a.factor or scale
    keep = np.ones_like(valid)
    if a.crop:
        r0, r1, c0, c1 = a.crop
        keep[:] = False
        keep[max(0, r0):min(h, r1), max(0, c0):min(w, c1)] = True
    if a.keep_tiles:
        rows = json.load(open(a.keep_tiles)) if a.keep_tiles.endswith(".json") else [json.loads(l) for l in open(a.keep_tiles) if l.strip()]
        keep[:] = False
        for r in rows:
            i, j = r["ij"]
            keep[max(0, i - a.margin):min(h, i + a.tile + a.margin), max(0, j - a.margin):min(w, j + a.tile + a.margin)] = True
    valid = valid & keep
    n_in = int(valid.sum())

    Xr, vX = resample_grid(X, valid, factor)
    Yr, vY = resample_grid(Y, valid, factor)
    Zr, vZ = resample_grid(Z, valid, factor)
    v = vX & vY & vZ
    P = np.stack([Xr.ravel(), Yr.ravel(), Zr.ravel(), np.ones(Xr.size)])
    Q = M @ P
    Xo, Yo, Zo = (Q[k].reshape(Xr.shape) for k in range(3))
    v &= (Zo > 0) & np.isfinite(Zo)
    for A_ in (Xo, Yo, Zo):
        A_[~v] = 0.0
    os.makedirs(a.out_dir, exist_ok=True)
    for name, A_ in (("x", Xo), ("y", Yo), ("z", Zo)):
        tifffile.imwrite(os.path.join(a.out_dir, f"{name}.tif"), A_.astype(np.float32))
    bbox = [[float(Xo[v].min()), float(Yo[v].min()), float(Zo[v].min())], [float(Xo[v].max()), float(Yo[v].max()), float(Zo[v].max())]] if v.any() else None
    m2 = dict(meta)
    m2.update(bbox=bbox, area_vx2=float(meta.get("area_vx2", 0.0)) * (float(v.sum()) / max(1, n_in)) * (1.0 / scale) ** 2 * factor ** 2 if meta.get("area_vx2") else None,
              transform_note=f"tifxyz_transform.py: {a.matrix} inverse={a.inverse} factor={factor:.5f} scale={scale:.5f} in_valid={n_in} out_valid={int(v.sum())}")
    if m2.get("area_vx2") is None:
        m2.pop("area_vx2", None)
    json.dump(m2, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=4)
    print(f"{a.in_dir} {h}x{w} ({n_in} kept valid cells) -> {a.out_dir} {Xo.shape[0]}x{Xo.shape[1]} ({int(v.sum())} valid); "
          f"factor {factor:.5f}, transform scale {scale:.5f}, bbox {bbox}")


if __name__ == "__main__":
    main()
