"""Boundary-halo quantification: response vs distance from the inference-patch edge.

The prediction canvas is a union of 128-px ink_9um patches on a 64-px stride;
patches with empty raw input are skipped, so the valid region has a long,
tile-quantised outline.  This measures the model's deterministic response
profile against that outline -> vf_rim.json, vf_rim2.json.
"""
import json, os, sys
import numpy as np
from scipy import ndimage as ndi

HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag"
sys.path.insert(0, HERE)
import vf_common as C

UM = C.PX_UM_DS4["PHerc1447"] / 1000.0      # mm per ds4 px
a = C.load(C.FLAG, "forward"); b = C.load(C.FLAG, "reverse"); m = a > 0
d = ndi.distance_transform_edt(m)           # ds4 px from the patch boundary

rows = []
print("mean DN vs distance from the inference-patch boundary:")
for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 10), (10, 15), (15, 20),
               (20, 30), (30, 50), (50, 10 ** 6)):
    s = (d > lo) & (d <= hi)
    if s.sum() < 50:
        continue
    v = a[s]
    rows.append({"dist_ds4px": [lo, hi], "dist_mm": [round(lo * UM, 3), round(hi * UM, 3)],
                 "n": int(s.sum()), "mean": round(float(v.mean()), 1),
                 "p99": round(float(np.percentile(v, 99)), 1),
                 "frac_gt195": round(float((v > 195).mean()), 5)})
    print(f"  {lo:3d}-{hi:3d} px ({lo*UM:5.2f}-{hi*UM:5.2f} mm): n={s.sum():7d} "
          f"mean={v.mean():6.1f} p99={np.percentile(v,99):5.1f} frac>195={(v>195).mean():.4f}")

rim, core = a[(d > 0) & (d <= 6)], a[d > 20]
json.dump({"rim_profile": rows, "rim_mean": float(rim.mean()),
           "interior_mean": float(core.mean()),
           "rim_frac_gt195": float((rim > 195).mean()),
           "interior_frac_gt195": float((core > 195).mean()),
           "rim_area_pct": float(100 * rim.size / m.sum())},
          open(os.path.join(HERE, "vf_rim.json"), "w"), indent=1)

hot = (a > 195) & m
halo = (d > 20) & (d <= 50) & m              # 0.69 - 1.73 mm from the boundary
near = (d <= 50) & m
print(f"\nhalo band 0.69-1.73 mm = {100*halo.sum()/m.sum():.1f}% of the valid area")
print(f"  but holds {100*(hot&halo).sum()/hot.sum():.1f}% of all pixels > 195 DN "
      f"({(hot&halo).sum()/halo.sum()/(hot.sum()/m.sum()):.1f}x enrichment)")
print(f"  {100*near.sum()/m.sum():.1f}% of the valid area is within 1.73 mm of a patch boundary")
print(f"  halo mean fwd {a[halo].mean():.1f} vs rev {b[halo].mean():.1f}  -> geometry, not ink")
json.dump({"halo_area_pct": float(100 * halo.sum() / m.sum()),
           "halo_share_of_hot_pct": float(100 * (hot & halo).sum() / hot.sum()),
           "halo_enrichment": float((hot & halo).sum() / halo.sum() / (hot.sum() / m.sum())),
           "pct_within_1.73mm_of_edge": float(100 * near.sum() / m.sum()),
           "halo_mean_fwd": float(a[halo].mean()), "halo_mean_rev": float(b[halo].mean())},
          open(os.path.join(HERE, "vf_rim2.json"), "w"), indent=1)
