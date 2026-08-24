"""Step 3: is the flagged spatial frequency a rendering / tiling pitch?

Measures the quantisation grid of the valid mask (the canvas is built from
rendered tiles, so mask edges snap to the tile lattice) and compares the
resulting scales to the flagged 7.24 mm.
"""
import json, sys, numpy as np
sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag")
import vf_common as C

px_ds4 = C.PX_UM_DS4["PHerc1447"]          # 34.56 um/px at ds4
res = {}
a = C.load(C.FLAG); mask = a > 0
H, W = mask.shape
print(f"ds4 canvas {mask.shape}; full-res canvas 4319x3719 (survey record)")

# --- edge positions in FULL-RES px ---------------------------------------
def edges(m, axis):
    d = np.diff(m.astype(np.int8), axis=axis)
    ys, xs = np.nonzero(d)
    return (xs if axis == 0 else ys), (ys if axis == 0 else xs)

pos = []
d = np.diff(mask.astype(np.int8), axis=0)          # horizontal edges -> row index
pos_h = np.unique(np.nonzero(d)[0]) + 1
d = np.diff(mask.astype(np.int8), axis=1)          # vertical edges -> col index
pos_v = np.unique(np.nonzero(d)[1]) + 1
print("distinct horizontal-edge rows (ds4):", pos_h)
print("distinct vertical-edge cols (ds4):", pos_v)
res["edge_rows_ds4"] = pos_h.tolist(); res["edge_cols_ds4"] = pos_v.tolist()

for tag, p in (("rows", pos_h), ("cols", pos_v)):
    f = p * 4                                     # full-res px
    print(f"\n{tag}: full-res edge coords {f}")
    # greatest common divisor of the edge coordinates and of their differences
    from math import gcd
    g = 0
    for v in f:
        g = gcd(g, int(v))
    dg = 0
    for v in np.diff(f):
        dg = gcd(dg, int(abs(v)))
    print(f"  gcd(coords)={g} px   gcd(diffs)={dg} px")
    res[f"{tag}_gcd_coords_fullres"] = g
    res[f"{tag}_gcd_diffs_fullres"] = dg
    # test candidate tile strides
    best = []
    for T in (32, 48, 64, 96, 128, 160, 192, 200, 208, 224, 256, 320, 384, 400, 416, 448, 512):
        r = np.abs(((f + T / 2) % T) - T / 2)
        best.append((T, float(r.max()), float(r.mean())))
    best.sort(key=lambda t: (t[1], -t[0]))
    print("  best-fitting strides (T, max residual px, mean residual px):")
    for t in best[:6]:
        print(f"    T={t[0]:4d}  maxres={t[1]:6.1f}  meanres={t[2]:6.1f}")
    res[f"{tag}_stride_fit"] = best

# --- what physical scales do plausible tile strides correspond to? -------
print("\nphysical scale of candidate strides (PHerc1447, 8.64 um/voxel):")
for T in (128, 192, 200, 208, 209, 256, 384, 512, 617, 838, 1024):
    print(f"  {T:5d} full-res px = {T*8.64/1000:6.3f} mm")
res["stride_mm"] = {str(T): T * 8.64 / 1000 for T in (128, 192, 256, 384, 512, 1024)}
print(f"\nflagged period 7.24 mm = {7.24*1000/8.64:.1f} full-res px "
      f"= {7.24*1000/px_ds4:.1f} ds4 px")
res["flag_period_fullres_px"] = 7.24 * 1000 / 8.64

# --- the flagged period vs the segment's own extent -----------------------
prof_len_mm = 838 * px_ds4 / 1000
print(f"projection profile length {prof_len_mm:.1f} mm -> the 7.24 mm 'period' is "
      f"Fourier bin k={prof_len_mm/7.24:.2f} (only ~4 cycles across the whole segment)")
res["profile_len_mm"] = prof_len_mm
res["cycles"] = prof_len_mm / 7.24

# --- connected components of the mask: the segment is a patchwork --------
from scipy.ndimage import label, find_objects
lab, n = label(mask)
sizes = np.bincount(lab.ravel())[1:]
order = np.argsort(-sizes)
print(f"\nmask has {n} connected components; sizes (ds4 px): {sizes[order][:8]}")
res["n_components"] = int(n)
res["component_sizes_ds4"] = sizes[order][:8].tolist()
sl = find_objects(lab)
comp = []
for i in order[:6]:
    s = sl[i]
    m = (lab[s] == i + 1)
    v = a[s][m]
    comp.append({"bbox_ds4": [int(s[0].start), int(s[0].stop), int(s[1].start), int(s[1].stop)],
                 "area_ds4": int(sizes[i]), "mean": float(v.mean()), "p50": float(np.median(v)),
                 "p99": float(np.percentile(v, 99))})
    print(f"  comp {i+1}: bbox rows {s[0].start}-{s[0].stop} cols {s[1].start}-{s[1].stop} "
          f"area={sizes[i]} mean={v.mean():.1f} p50={np.median(v):.0f} p99={np.percentile(v,99):.0f}")
res["components"] = comp
json.dump(res, open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag\vf_tiles.json", "w"),
          indent=1, default=float)
