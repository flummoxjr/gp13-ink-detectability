#!/usr/bin/env python
"""Direct level-3 material check on a targeted subset of the screened corpus,
to calibrate the cheap nonzero/valid proxy against measured ground truth."""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\villa\vesuvius\src")
import numpy as np
from scipy.ndimage import map_coordinates
from vesuvius.tifxyz import read_tifxyz
from vesuvius.ink_detection.volume_io import open_volume, read_bbox_with_padding

ROOT = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
CACHE = os.path.join(ROOT, "hunt", "corpuscache")
OUT = os.path.join(ROOT, "hunt", "out")
S3 = "s3://vesuvius-challenge-open-data/"
LV, F = 3, 8

TARGETS = [
    # worst material_ratio in PHerc1447
    "z_dbg_gen_00325_inp_hr", "z_dbg_gen_00070_inp_hr", "z_dbg_gen_00283_inp_hr",
    "z_dbg_gen_00316_inp_hr", "auto_grown_20250502161744358",
    "auto_grown_20250502163549332", "z_dbg_gen_00166_inp_hr",
    # mid / good PHerc1447 for contrast
    "auto_grown_20250703034159599", "z_dbg_gen_00320",
    # PHerc1203 + PHerc0800 calibration points
    "auto_grown_20250929222256117", "20251028220042-auto_grown_20251028220042762",
]

if __name__ == "__main__":
    cat = {c["name"]: c for c in json.load(open(os.path.join(ROOT, "runpod", "segment_catalog.json")))}
    outp = os.path.join(OUT, "corpus_material.json")
    rows = json.load(open(outp)) if os.path.exists(outp) else []
    done = {r["name"] for r in rows}
    for name in TARGETS:
        if name in done or name not in cat:
            continue
        c = cat[name]
        try:
            su = read_tifxyz(os.path.join(CACHE, name), load_mask=False, validate=False)
            v = su._valid_mask
            x = su._x[v].astype(np.float64); y = su._y[v].astype(np.float64); z = su._z[v].astype(np.float64)
            vpath = S3 + f"{c['scroll']}/volumes/{c['volume']}"
            v0 = open_volume(vpath, 0)
            Z0, Y0, X0 = v0.shape
            oob = (z < 1) | (z >= Z0 - 1) | (y < 1) | (y >= Y0 - 1) | (x < 1) | (x >= X0 - 1)
            vol = open_volume(vpath, LV)
            zl, yl, xl = z / F, y / F, x / F
            z0, z1 = max(int(zl.min()) - 2, 0), int(zl.max()) + 3
            y0, y1 = max(int(yl.min()) - 2, 0), int(yl.max()) + 3
            x0, x1 = max(int(xl.min()) - 2, 0), int(xl.max()) + 3
            crop, _ = read_bbox_with_padding(vol, (z0, y0, x0, z1, y1, x1), fill_value=0)
            val = map_coordinates(crop.astype(np.float32),
                                  np.stack([zl - z0, yl - y0, xl - x0]), order=1,
                                  mode="constant", cval=0.0)
            r = {"name": name, "scroll": c["scroll"], "n_vertices": int(v.sum()),
                 "frac_out_of_array": round(float(oob.mean()), 4),
                 "frac_no_material": round(float(np.maximum(val < 20, oob).mean()), 4),
                 "vertex_lv3_med": round(float(np.median(val)), 1)}
        except Exception as e:
            r = {"name": name, "scroll": c["scroll"], "error": f"{type(e).__name__}: {e}"[:200]}
        rows.append(r)
        json.dump(rows, open(outp, "w"), indent=1)
        print(json.dumps(r), flush=True)
    print("wrote", outp)
