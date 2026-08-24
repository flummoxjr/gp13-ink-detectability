"""Positive control for the mesh-vs-lamella angle measurement.

The PHerc0813 patches measure a median 67.3 deg between the mesh normal and the local
sheet normal, against a 60 deg random-orientation null. That is a strong claim about our
own geometry, so the measurement itself needs a control: run the identical code on
PUBLISHED Grand-Prize meshes, which were grown by the official pipeline WITH a
structure-tensor normal direction field. If those come back well aligned, the method is
sound and the PHerc0813 result is real. If they also come back near 60 deg, the method
is measuring nothing and the PHerc0813 result must be withdrawn.

Meshes are already cached in hunt/meshcache/; only the volume cubes at their centres
are fetched here.
"""
import json
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD")
from k2c_separability import open_level, vol_info, ROI  # noqa: E402
from k2c_analyze import coh_med  # noqa: E402
from hunt.mesh_lamella_alignment import mesh_normal, sheet_normal  # noqa: E402

T = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
MC = os.path.join(T, "hunt", "meshcache")
CACHE = r"D:\vesuvius-data\trackD\k2c_control"
VIDS = {"PHerc1203": "20250820131727", "PHerc1447": "20250521151220",
        "PHerc0800": "20250521135224", "PHerc0139": "20250728140407"}


def main():
    os.makedirs(CACHE, exist_ok=True)
    segs = {s["key"]: s for s in json.load(open(os.path.join(MC, "segs.json")))}
    keys = [k for k in sorted(os.listdir(MC)) if os.path.isdir(os.path.join(MC, k))]

    def one(key):
        d = os.path.join(MC, key)
        meta = json.load(open(os.path.join(d, "meta.json")))
        scroll = segs[key]["scroll"]
        bb = meta.get("bbox")
        if not bb:
            return None
        # bbox is [[x0,y0,z0],[x1,y1,z1]] in volume coords; centre it
        cx, cy, cz = [(bb[0][i] + bb[1][i]) / 2 for i in range(3)]
        long_id, shape0 = vol_info(scroll, VIDS[scroll])
        o = (int(np.clip(cz - ROI // 2, 0, shape0[0] - ROI)),
             int(np.clip(cy - ROI // 2, 0, shape0[1] - ROI)),
             int(np.clip(cx - ROI // 2, 0, shape0[2] - ROI)))
        cp = os.path.join(CACHE, f"{key}.npy")
        if os.path.exists(cp):
            a = np.load(cp)
        else:
            a = np.asarray(open_level(scroll, long_id, 0)[o[0]:o[0] + ROI,
                                                          o[1]:o[1] + ROI,
                                                          o[2]:o[2] + ROI])
            np.save(cp, a)
        return key, scroll, d, a, o

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(one, keys):
            if r is None:
                continue
            key, scroll, d, a, o = r
            fill = float((a > 0).mean())
            if fill < 0.5:
                print(f"{key:12} {scroll:10} fill={fill:.2f} — skipped (cube not in material)")
                continue
            mn, nv = mesh_normal(d)
            sn = sheet_normal(a)
            sep = coh_med(a)
            if mn is None or sn is None:
                print(f"{key:12} {scroll:10} insufficient data")
                continue
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(mn, sn)))))))
            rows.append(dict(key=key, scroll=scroll, angle_deg=ang, separability=sep,
                             fill=fill, origin=list(o)))
            print(f"{key:12} {scroll:10} sep={sep:.3f} fill={fill:.2f}  "
                  f"mesh-vs-sheet angle = {ang:5.1f} deg", flush=True)

    if rows:
        ang = [r["angle_deg"] for r in rows]
        print(f"\nPUBLISHED GP meshes (grown WITH a normal direction field):")
        print(f"  n={len(ang)}  median angle {np.median(ang):.1f} deg  "
              f"min {min(ang):.1f}  max {max(ang):.1f}")
        print(f"  within 30 deg of the sheets: {sum(1 for a in ang if a < 30)}/{len(ang)}")
        print(f"\nOUR PHerc0813 meshes (grown WITHOUT one): median 67.3 deg, 0/8 within 30 deg")
        print(f"random-orientation null median: 60.0 deg")
        json.dump({"meshes": rows, "median_angle_deg": float(np.median(ang)),
                   "n_within_30deg": int(sum(1 for a in ang if a < 30)),
                   "random_null_median_deg": 60.0},
                  open(os.path.join(T, "out", "k2c_separability",
                                    "published_mesh_alignment.json"), "w"), indent=1)
        print("wrote published_mesh_alignment.json")


if __name__ == "__main__":
    main()
