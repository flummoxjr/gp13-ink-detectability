"""Investigation D step 1: do PHerc1203's 22 segments fall inside the 2.4um band?

Fetches every segment's meta.json (top-level, i.e. the LARGEST grown version) plus the
tifxyz rasters, verifies the bbox coordinate order against the actual z.tif contents, and
computes overlap with the 2.4um band's physical z extent derived from beamline motor coords.
"""
import io
import json
import os
import sys

import numpy as np
import requests
import tifffile

B = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt"
CAT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod\segment_catalog.json"

# ---- geometry from trackD/meta/PHerc1203.json (ESRF BM18 acquisition metadata) ----
# 9.362um scan 20250720004030: helical, sz -358.3 -> -190.8626, det 1104px(bin2)*9.362um tall
# 2.403um scan 20250721080303: helical, sz -286.0552 -> -255.3276, det 2368px*2.403um tall
V9, V2 = 0.009362, 0.002403  # mm/voxel
N9, N2 = 18977, 15137        # z voxels
H9 = 1104 * V9               # detector vertical coverage, mm
H2 = 2368 * V2
Z9 = (-358.3 - H9 / 2, -190.8626 + H9 / 2)
Z2 = (-286.0552 - H2 / 2, -255.3276 + H2 / 2)


def report_geometry():
    print("=== physical z extents (BM18 stage coords, mm) ===")
    print(f"9.362um: motor span {Z9[0]:.4f} .. {Z9[1]:.4f} = {Z9[1]-Z9[0]:.4f} mm; "
          f"volume {N9}*{V9} = {N9*V9:.4f} mm; residual {Z9[1]-Z9[0]-N9*V9:+.4f} mm "
          f"({(Z9[1]-Z9[0]-N9*V9)/V9:+.1f} vx)")
    print(f"2.403um: motor span {Z2[0]:.4f} .. {Z2[1]:.4f} = {Z2[1]-Z2[0]:.4f} mm; "
          f"volume {N2}*{V2} = {N2*V2:.4f} mm; residual {Z2[1]-Z2[0]-N2*V2:+.4f} mm "
          f"({(Z2[1]-Z2[0]-N2*V2)/V2:+.1f} vx)")
    print(f"lateral FOV: 9um 6844*{V9} = {6844*V9:.4f} mm ; 2.4um 26493*{V2} = {26493*V2:.4f} mm")
    # hypothesis A: voxel z increases with motor z in BOTH volumes
    a0 = (Z2[0] - Z9[0]) / V9
    a1 = (Z2[1] - Z9[0]) / V9
    # hypothesis B: voxel z DEcreases with motor z in both volumes (both flipped)
    b0 = (Z9[1] - Z2[1]) / V9
    b1 = (Z9[1] - Z2[0]) / V9
    print(f"band in 9um voxel z -- hypothesis A (no flip): {a0:.0f} .. {a1:.0f}")
    print(f"band in 9um voxel z -- hypothesis B (both flipped): {b0:.0f} .. {b1:.0f}")
    return (a0, a1), (b0, b1)


def fetch(path, binary=False):
    r = requests.get(f"{B}/{path}", timeout=60)
    r.raise_for_status()
    return r.content if binary else r.text


def main():
    A, Bh = report_geometry()
    cat = json.load(open(CAT))
    segs = [e for e in cat if e["scroll"] == "PHerc1203"]
    rows = []
    for e in segs:
        d = e["seg_dir"]
        try:
            m = json.loads(fetch(d + "meta.json"))
        except Exception as ex:
            print("meta fail", d, ex)
            continue
        bbox = m.get("bbox")
        row = {"name": e["name"], "seg_dir": d, "surveyed_tifxyz": e["tifxyz"],
               "area_cm2": m.get("area_cm2"), "bbox": bbox, "seed": m.get("seed")}
        # verify order + get true extents from the rasters at top level
        try:
            zt = tifffile.imread(io.BytesIO(fetch(d + "z.tif", True)))
            xt = tifffile.imread(io.BytesIO(fetch(d + "x.tif", True)))
            yt = tifffile.imread(io.BytesIO(fetch(d + "y.tif", True)))
            # tifxyz uses -1 as the no-data sentinel
            msk = (zt > 0) & (xt > 0) & (yt > 0)
            row["raster_shape"] = list(zt.shape)
            row["n_valid"] = int(msk.sum())
            for nm, arr in (("x", xt), ("y", yt), ("z", zt)):
                v = arr[msk]
                row[f"{nm}_min"], row[f"{nm}_max"] = float(v.min()), float(v.max())
                row[f"{nm}_p05"], row[f"{nm}_p95"] = float(np.percentile(v, 5)), float(np.percentile(v, 95))
            zv = zt[msk]
            row["pt_in_bandA"] = float(((zv >= A[0]) & (zv <= A[1])).mean())
            row["pt_in_bandB"] = float(((zv >= Bh[0]) & (zv <= Bh[1])).mean())
        except Exception as ex:
            print("raster fail", d, ex)
        # also: what did the survey actually render?
        if e["tifxyz"] != d:
            try:
                mv = json.loads(fetch(e["tifxyz"] + "meta.json"))
                row["surveyed_area_cm2"] = mv.get("area_cm2")
                row["surveyed_bbox"] = mv.get("bbox")
            except Exception:
                pass
        rows.append(row)
        print(f"{row['name']}  area={row.get('area_cm2'):.2f}cm2  "
              f"z=[{row.get('z_min', -1):.0f},{row.get('z_max', -1):.0f}]  "
              f"x=[{row.get('x_min', -1):.0f},{row.get('x_max', -1):.0f}]  "
              f"y=[{row.get('y_min', -1):.0f},{row.get('y_max', -1):.0f}]  bbox={bbox}", flush=True)

    def frac_in(r, lo, hi, key="z"):
        # fraction of the segment's z span inside [lo,hi]
        a, b = r.get(f"{key}_min"), r.get(f"{key}_max")
        if a is None:
            return None
        ov = max(0.0, min(b, hi) - max(a, lo))
        return ov / max(1e-9, b - a)

    for r in rows:
        r["frac_in_bandA"] = frac_in(r, *A)
        r["frac_in_bandB"] = frac_in(r, *Bh)
    json.dump({"bandA_9umz": A, "bandB_9umz": Bh, "segments": rows},
              open(os.path.join(OUT, "band_extents.json"), "w"), indent=1)

    print("\n=== overlap summary ===")
    for r in sorted(rows, key=lambda r: -(r.get("pt_in_bandB") or 0)):
        print(f"{r['name']}  ptA={r.get('pt_in_bandA', -1):.3f}  ptB={r.get('pt_in_bandB', -1):.3f}  "
              f"z=[{r.get('z_min', -1):.0f},{r.get('z_max', -1):.0f}]  "
              f"area={r['area_cm2']:.2f}cm2  surveyed_area={r.get('surveyed_area_cm2', r['area_cm2']):.2f}cm2")
    tot = sum(r["area_cm2"] for r in rows)
    surv = sum(r.get("surveyed_area_cm2", r["area_cm2"]) for r in rows)
    print(f"\nTOTAL published 1203 segment area {tot:.1f} cm2; area actually screened in the "
          f"Aug survey {surv:.1f} cm2 ({100*surv/tot:.0f}%)")
    for tag, key in (("A", "pt_in_bandA"), ("B", "pt_in_bandB")):
        print(f"hypothesis {tag}: mean point-fraction in band = "
              f"{np.mean([r.get(key, 0) for r in rows]):.3f}; area-weighted = "
              f"{sum(r['area_cm2']*r.get(key, 0) for r in rows)/tot:.3f} "
              f"({sum(r['area_cm2']*r.get(key, 0) for r in rows):.1f} cm2 of surface inside the band)")
    for tag, key in (("A", "frac_in_bandA"), ("B", "frac_in_bandB")):
        full = [r for r in rows if (r[key] or 0) > 0.9]
        part = [r for r in rows if 0.1 < (r[key] or 0) <= 0.9]
        print(f"hypothesis {tag}: {len(full)} segments >90% inside, {len(part)} partial, "
              f"area fully-inside = {sum(r['area_cm2'] for r in full):.1f} cm2")


if __name__ == "__main__":
    sys.exit(main())
