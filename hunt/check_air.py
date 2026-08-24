#!/usr/bin/env python
"""Whole-mesh check: does the mesh sit where there is actually scanned material?

Reads the mesh bounding box once from pyramid level 3 (8x downsample, ~75 um
voxels) and samples every valid grid vertex.  Level 3 is far too coarse to see a
lamella, but it answers the binary question 'is there anything here at all'
(masked/unscanned volume reads exactly 0)."""
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
LV = 3
F = 2 ** LV

if __name__ == "__main__":
    segs = json.load(open(os.path.join(CACHE, "segs.json")))
    rows = []
    for s in segs:
        su = read_tifxyz(os.path.join(CACHE, s["key"]), load_mask=False, validate=False)
        v = su._valid_mask
        x = su._x[v].astype(np.float64); y = su._y[v].astype(np.float64); z = su._z[v].astype(np.float64)
        v0 = open_volume(S3 + s["volume"], 0)
        Z0, Y0, X0 = v0.shape
        oob = (z < 1) | (z >= Z0 - 1) | (y < 1) | (y >= Y0 - 1) | (x < 1) | (x >= X0 - 1)
        vol = open_volume(S3 + s["volume"], LV)
        zl, yl, xl = z / F, y / F, x / F
        z0, z1 = max(int(zl.min()) - 2, 0), int(zl.max()) + 3
        y0, y1 = max(int(yl.min()) - 2, 0), int(yl.max()) + 3
        x0, x1 = max(int(xl.min()) - 2, 0), int(xl.max()) + 3
        crop, _ = read_bbox_with_padding(vol, (z0, y0, x0, z1, y1, x1), fill_value=0)
        val = map_coordinates(crop.astype(np.float32),
                              np.stack([zl - z0, yl - y0, xl - x0]),
                              order=1, mode="constant", cval=0.0)
        r = {"key": s["key"], "scroll": s["scroll"], "role": s["role"],
             "n_vertices": int(v.sum()),
             "frac_vertices_out_of_array": round(float(oob.mean()), 4),
             "level": LV, "crop_shape": list(crop.shape),
             "frac_vertex_lv3_zero": round(float((val < 1).mean()), 4),
             "frac_vertex_lv3_lt20": round(float((val < 20).mean()), 4),
             "vertex_lv3_med": round(float(np.median(val)), 1),
             "vertex_lv3_p10": round(float(np.percentile(val, 10)), 1),
             "vertex_lv3_p90": round(float(np.percentile(val, 90)), 1)}
        r["frac_no_material"] = round(float(np.maximum(val < 20, oob).mean()), 4)
        rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(OUT, "vertex_material.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "vertex_material.json"))
