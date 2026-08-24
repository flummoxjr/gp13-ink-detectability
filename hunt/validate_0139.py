"""Investigation D step 2c: validate the motor-coordinate transform derivation.

PHerc0139 publishes 2.403um -> 9.362um transforms in the open-data catalogue.  If the same
beamline-stage arithmetic that we used to bootstrap PHerc1203 reproduces the PUBLISHED 0139
matrices, the 1203 derivation stands on independent ground.  Also checks whether the w035
control segment (human-verified Greek letters) falls inside a 0139 2.4um band.
"""
import io
import json

import numpy as np
import requests
import tifffile

B = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
META = r"C:\Users\benbl\Desktop\Vsuvious\trackD\meta\PHerc0139.json"


def acq(scan):
    return scan["creation"]["metadata"]["metadata"]["tomo"]["acquisition"]


def main():
    d = json.load(open(META, encoding="utf-8"))
    scans = d["scans"]
    vols = d["volumes"]
    print("=== PHerc0139 scans ===")
    info = {}
    for sid, s in scans.items():
        a = acq(s)
        det = a["detector"]
        px = det["samplePixelSize"]              # mm
        roi = eval(det["roi_size"])              # [w, h] detector px
        binn = det.get("sensorBinning", 1)
        h = roi[1] * px                          # vertical coverage in mm (roi already binned?)
        info[sid] = dict(px=px, z0=a["z_start"], z1=a["z_end"], h=h, roi=roi, bin=binn,
                         e=a["energy"], radix=a["scanRadix"])
        print(f"  {sid} {px*1e3:.3f}um {a['energy']}keV  sz {a['z_start']:.4f}..{a['z_end']:.4f} "
              f"roi={roi} bin={binn} detH={h:.3f}mm")

    print("\n=== published transforms vs motor-derived prediction ===")
    v9 = None
    for vid, v in vols.items():
        if abs(v["properties"]["pixel_size_um"] - 9.362) < 0.01:
            v9 = vid
    p9 = vols[v9]["properties"]
    s9 = info[vols[v9]["scan_id"]]
    px9 = p9["pixel_size_um"] / 1000.0
    n9 = p9["shape"][0]
    # 9um volume z=0 in stage mm
    z9_lo = s9["z0"] - s9["h"] / 2
    print(f"  9.362um volume {v9} shape {p9['shape']}  z0_stage={z9_lo:.4f} mm "
          f"(span {n9*px9:.3f} mm vs motor {s9['z1']-s9['z0']+s9['h']:.3f} mm)")

    rows = []
    for vid, v in vols.items():
        pr = v["properties"]
        tr = pr.get("transforms")
        if not tr:
            continue
        t = [x for x in tr if x["to_volume_id"] == v9]
        if not t:
            continue
        M = np.array(t[0]["transformation_matrix"])
        sc = info[v["scan_id"]]
        pxv = pr["pixel_size_um"] / 1000.0
        nz = pr["shape"][0]
        zv_lo = sc["z0"] - sc["h"] / 2
        pred_scale = pxv / px9
        pred_tz = (zv_lo - z9_lo) / px9
        # published matrix rows are (x, y, z); z translation is M[2, 3]
        pub_scale = M[2, 2]
        pub_tz = M[2, 3]
        rows.append((vid, pr["pixel_size_um"], pr["shape"], pub_scale, pred_scale,
                     pub_tz, pred_tz, (pub_tz - pred_tz) * px9))
        print(f"  vol {vid} {pr['pixel_size_um']}um shape {pr['shape']}")
        print(f"      published  z9 = {pub_scale:.6f}*z2 + {pub_tz:9.2f}")
        print(f"      motor-pred z9 = {pred_scale:.6f}*z2 + {pred_tz:9.2f}   "
              f"-> dz = {pub_tz - pred_tz:+.1f} 9um vox = {(pub_tz-pred_tz)*px9*1e3:+.0f} um")
        print(f"      band covers 9um z {pub_tz:.0f} .. {pub_tz + pub_scale*nz:.0f} of {n9}")

    # --- does the w035 control segment sit inside one of those bands? ---
    print("\n=== w035 control segment vs the 0139 2.4um bands ===")
    seg = "PHerc0139/segments/20260317000000-w035_2026031718/mesh/20260317000000-on-20250728140407-9.362um.tifxyz/"
    try:
        zt = tifffile.imread(io.BytesIO(requests.get(f"{B}/{seg}z.tif", timeout=120).content))
        v = zt[zt > 0]
        print(f"  w035 mesh z in 9.362um volume: {v.min():.0f} .. {v.max():.0f} "
              f"(p05 {np.percentile(v,5):.0f}, p50 {np.percentile(v,50):.0f}, p95 {np.percentile(v,95):.0f})")
        for (vid, pxu, shape, sc, _, tz, _, _) in rows:
            lo, hi = tz, tz + sc * shape[0]
            frac = float(((v >= lo) & (v <= hi)).mean())
            print(f"  vs {vid} ({pxu}um) band 9um z [{lo:.0f},{hi:.0f}] -> {frac*100:.1f}% of w035 inside")
    except Exception as e:
        print("  mesh fetch failed:", e)


if __name__ == "__main__":
    main()
