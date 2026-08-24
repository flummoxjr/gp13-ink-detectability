"""m2b_deepdive.py — (a) local thickness of large 'blob' components (are they
solid masses or thin sprawling networks?); (b) patch subpopulation profile;
(c) per-class geometry summaries.
"""
import json, os
import numpy as np
from scipy import ndimage as ndi

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"
CACHE = os.path.join(OUT, "cache")
STRUCT = np.ones((3, 3, 3), bool)

inv = json.load(open(os.path.join(OUT, "inventory.json")))["samples"]
comps = json.load(open(os.path.join(OUT, "components.json")))

# ---------- (a) local thickness of big components ----------
# local half-thickness = EDT inside the binary mask; thickness ~ 2*EDT at the
# "core" (we report mean/max of 2*EDT over the component, and the p90).
big_rows = []
for rec in inv:
    z, y, x = rec["tile"]
    prob8 = np.load(rec["file"])
    binm = prob8 >= 128
    lab, n = ndi.label(binm, structure=STRUCT)
    sizes = np.bincount(lab.ravel())
    edt = ndi.distance_transform_edt(binm)
    for ci in range(1, n + 1):
        if sizes[ci] < 500:
            continue
        m = lab == ci
        th = 2.0 * edt[m]
        big_rows.append({
            "tile": [z, y, x], "size": int(sizes[ci]),
            "th_mean": float(th.mean()), "th_p90": float(np.percentile(th, 90)),
            "th_max": float(th.max()),
            "span_frac": float(max(np.ptp(np.argwhere(m), axis=0)) / 63.0),
        })

big_rows.sort(key=lambda r: -r["size"])
print("=== components >=500 vox: local thickness (2*EDT, L2 vox; 1 vox=9.6um) ===")
print(f"n={len(big_rows)}")
th_mean = np.array([r["th_mean"] for r in big_rows])
th_p90 = np.array([r["th_p90"] for r in big_rows])
spans = np.array([r["span_frac"] for r in big_rows])
sz = np.array([r["size"] for r in big_rows])
print(f"size: median {np.median(sz):.0f}, max {sz.max()}")
print(f"th_mean: median {np.median(th_mean):.1f}  p90-of-comp median {np.median(th_p90):.1f}")
print(f"span_frac (max extent / tile): median {np.median(spans):.2f}, frac spanning>0.9: {(spans>0.9).mean():.2f}")
for r in big_rows[:10]:
    print(f"  size={r['size']:6d} th_mean={r['th_mean']:.1f} th_p90={r['th_p90']:.1f} "
          f"th_max={r['th_max']:.1f} span={r['span_frac']:.2f} tile={r['tile']}")

# ---------- (b) patch subpopulation ----------
patches = [c for c in comps if c["class"] == "patch"]
print(f"\n=== patch components (flat, sheet-conformal): n={len(patches)} ===")
def arr(k): return np.array([p[k] for p in patches])
for k in ["size", "e1", "e2", "e3", "elong", "planar", "angle_deg",
          "onsheet_dil2", "p_mean", "p_max", "near_bg"]:
    a = arr(k)
    print(f"{k:12s} median {np.median(a):7.2f}  p25 {np.percentile(a,25):7.2f}  "
          f"p75 {np.percentile(a,75):7.2f}  max {a.max():7.2f}")
stroke_like = [p for p in patches if 2 <= p["e3"] <= 10 and p["e1"] >= 8
               and p["onsheet_dil2"] >= 0.5 and p["near_bg"] < 0.3]
print(f"stroke-like patches (2<=e3<=10, e1>=8, onsheet>=0.5, not near bg): {len(stroke_like)}")
per_tile = {}
for p in stroke_like:
    per_tile.setdefault(tuple(p["tile"]), []).append(p)
print("tiles with >=2 stroke-like patches:")
for t, ps in sorted(per_tile.items(), key=lambda kv: -len(kv[1])):
    if len(ps) >= 2:
        print(f"  {t}: n={len(ps)} sizes={[p['size'] for p in ps]} "
              f"e3={[round(p['e3'],1) for p in ps]}")

# in-plane aspect of stroke-like patches (stroke: elongated in-plane)
if stroke_like:
    ip = np.array([p["e1"] / p["e2"] for p in stroke_like])
    print(f"in-plane aspect e1/e2: median {np.median(ip):.2f} p25 {np.percentile(ip,25):.2f} p75 {np.percentile(ip,75):.2f}")
    szs = np.array([p["size"] for p in stroke_like])
    print(f"sizes: median {np.median(szs):.0f} max {szs.max()}")

# ---------- (c) per-class geometry ----------
print("\n=== per-class geometry (median [p25,p75]) ===")
for cl in ["patch", "ribbon", "oblique", "filament", "blob", "edge"]:
    cs = [c for c in comps if c["class"] == cl]
    if not cs:
        continue
    s = np.array([c["size"] for c in cs])
    e3 = np.array([c["e3"] for c in cs])
    pl = np.array([c["planar"] for c in cs])
    osf = np.array([c["onsheet_dil2"] for c in cs])
    ang = np.array([c["angle_deg"] for c in cs if c["angle_deg"] is not None])
    print(f"{cl:10s} n={len(cs):4d} size_med={np.median(s):6.0f} e3_med={np.median(e3):5.1f} "
          f"planar_med={np.median(pl):4.1f} onsheet_med={np.median(osf):.2f} "
          f"angle_med={'%5.1f' % np.median(ang) if len(ang) else '  n/a'}")

# fraction of all fired voxels in comps>=500 (the 'giant network' share)
tot = sum(c["size"] for c in comps)
gi = sum(c["size"] for c in comps if c["size"] >= 500)
gi2 = sum(c["size"] for c in comps if c["size"] >= 2000)
print(f"\nfired vox in comps>=20: {tot}; share in >=500: {gi/tot:.2f}; >=2000: {gi2/tot:.2f}")

json.dump({"big_components": big_rows,
           "n_patches": len(patches),
           "n_stroke_like": len(stroke_like),
           "stroke_like": stroke_like},
          open(os.path.join(OUT, "deepdive.json"), "w"), indent=1)
print("wrote deepdive.json")
