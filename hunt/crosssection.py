#!/usr/bin/env python
"""Cut a (depth x arclength) cross-section through each mesh: for one strip of
grid points, sample the volume at offsets -16..+16 along the normal.  A mesh that
sits ON a sheet shows a straight bright band at offset 0; a mis-placed mesh shows
the band drifting, doubled, or absent."""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\villa\vesuvius\src")
import numpy as np
from scipy.ndimage import map_coordinates
from vesuvius.tifxyz import read_tifxyz
from vesuvius.ink_detection.volume_io import open_volume, read_bbox_with_padding

CACHE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\meshcache"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\out"
S3 = "s3://vesuvius-challenge-open-data/"
OFF = np.arange(-16.0, 16.01, 0.5)
PAD = 18
LEN = 400          # full-res grid points along the strip
NROW = 5           # rows averaged to suppress fibre noise


def cross(seg):
    d = os.path.join(CACHE, seg["key"])
    surf = read_tifxyz(d, load_mask=False, validate=False)
    sv = surf._valid_mask.copy()
    scale = float(surf._scale[0])
    surf = surf.use_full_resolution()
    H, W = surf.shape
    # candidate fully-valid runs of stored columns, longest first
    cands = []
    for i in range(sv.shape[0]):
        row = sv[i]
        j = 0
        while j < len(row):
            if row[j]:
                k = j
                while k < len(row) and row[k]:
                    k += 1
                cands.append((i, j, k - j))
                j = k
            else:
                j += 1
    cands.sort(key=lambda t: -t[2])
    if not cands:
        return None
    vol = open_volume(S3 + seg["volume"], 0)
    pick = None
    for (i, j0, run) in cands[:12]:
        r0 = min(max(int(i / scale) - NROW // 2, 0), H - NROW - 1)
        c0 = int(j0 / scale)
        c1 = min(c0 + LEN, int((j0 + run) / scale), W)
        if c1 - c0 < 60:
            continue
        x, y, z, valid = surf[r0:r0 + NROW, c0:c1]
        nx, ny, nz = surf.get_normals(r0, r0 + NROW, c0, c1)
        ok = valid & np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)
        if ok.sum() >= 50:
            pick = (x, y, z, ok, nx, ny, nz)
            break
    if pick is None:
        return None
    x, y, z, ok, nx, ny, nz = pick
    z0 = int(np.floor(z[ok].min())) - PAD; z1 = int(np.ceil(z[ok].max())) + PAD + 1
    y0 = int(np.floor(y[ok].min())) - PAD; y1 = int(np.ceil(y[ok].max())) + PAD + 1
    x0 = int(np.floor(x[ok].min())) - PAD; x1 = int(np.ceil(x[ok].max())) + PAD + 1
    crop, _ = read_bbox_with_padding(vol, (z0, y0, x0, z1, y1, x1), fill_value=0)
    crop = crop.astype(np.float32, copy=False)
    img = np.full((OFF.size, x.shape[1]), np.nan, np.float32)
    for si, off in enumerate(OFF):
        acc = np.zeros(x.shape[1]); cnt = np.zeros(x.shape[1])
        for rr in range(x.shape[0]):
            m = ok[rr]
            if not m.any():
                continue
            coords = np.stack([z[rr, m] - z0 + off * nz[rr, m],
                               y[rr, m] - y0 + off * ny[rr, m],
                               x[rr, m] - x0 + off * nx[rr, m]])
            v = map_coordinates(crop, coords, order=1, mode="constant", cval=0.0)
            acc[m] += v; cnt[m] += 1
        img[si] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return img


if __name__ == "__main__":
    segs = json.load(open(os.path.join(CACHE, "segs.json")))
    store = {}
    p = os.path.join(OUT, "crosssections.npz")
    for s in segs:
        try:
            im = cross(s)
        except Exception as e:
            print("ERR", s["key"], type(e).__name__, e); im = None
        if im is not None:
            store[s["key"]] = im
            print("ok", s["key"], im.shape, float(np.nanmin(im)), float(np.nanmax(im)), flush=True)
    np.savez_compressed(p, **store)
    print("wrote", p)
