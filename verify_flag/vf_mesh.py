"""Step 3b: mesh geometry of z_dbg_gen_00166_inp_hr (tifxyz from the open bucket)."""
import json, sys, numpy as np
sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag")
import vf_common as C
import tifffile

M = r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag\mesh"
res = {}
X = tifffile.imread(M + r"\x.tif").astype(np.float64)
Y = tifffile.imread(M + r"\y.tif").astype(np.float64)
Z = tifffile.imread(M + r"\z.tif").astype(np.float64)
print("tifxyz grid:", X.shape, X.dtype, "scale 0.05 -> render canvas",
      tuple(s * 20 for s in X.shape))
res["grid_shape"] = list(X.shape)
res["render_canvas_implied"] = [s * 20 for s in X.shape]

defined = (X != 0) | (Y != 0) | (Z != 0)
print(f"defined mesh fraction: {defined.mean():.4f}")
res["mesh_defined_frac"] = float(defined.mean())

a = C.load(C.FLAG); mask = a > 0
print(f"prediction ds4 canvas {mask.shape}, mask frac {mask.mean():.4f}")

# --- physical pitch of the mesh grid (voxel units, 8.64 um each) ---------
d_u = np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(Y, axis=1) ** 2 + np.diff(Z, axis=1) ** 2)
d_v = np.sqrt(np.diff(X, axis=0) ** 2 + np.diff(Y, axis=0) ** 2 + np.diff(Z, axis=0) ** 2)
ok_u = defined[:, :-1] & defined[:, 1:]
ok_v = defined[:-1] & defined[1:]
for tag, d, ok in (("u (cols)", d_u, ok_u), ("v (rows)", d_v, ok_v)):
    vv = d[ok]
    vv = vv[vv > 0]
    print(f"mesh step {tag}: median {np.median(vv):.3f} vox = {np.median(vv)*8.64/1000:.4f} mm "
          f"(p10 {np.percentile(vv,10):.2f}, p90 {np.percentile(vv,90):.2f})")
    res[f"mesh_step_{tag[0]}_vox_p50"] = float(np.median(vv))
    res[f"mesh_step_{tag[0]}_mm_p50"] = float(np.median(vv) * 8.64 / 1000)

# --- is the mesh grid itself periodic / patchy? --------------------------
# row-mean of the surface normal-ish quantity: use Z as a proxy for sheet depth
zz = np.where(defined, Z, np.nan)
rowm = np.nanmean(zz, axis=1)
print(f"\nZ range {np.nanmin(zz):.0f}-{np.nanmax(zz):.0f} vox "
      f"({(np.nanmax(zz)-np.nanmin(zz))*8.64/1000:.1f} mm)")
res["z_range_mm"] = float((np.nanmax(zz) - np.nanmin(zz)) * 8.64 / 1000)

# --- extent of the segment in mm ----------------------------------------
bb = json.load(open(M + r"\..\mesh_meta.json")) if False else None
print(f"prediction canvas 4319x3719 full-res px = "
      f"{4319*8.64/1000:.1f} x {3719*8.64/1000:.1f} mm")
res["canvas_mm"] = [4319 * 8.64 / 1000, 3719 * 8.64 / 1000]

# --- does the mesh 'defined' region explain the 64-px mask lattice? ------
from scipy.ndimage import zoom
dm = np.repeat(np.repeat(defined, 5, 0), 5, 1)      # grid -> ds4 (20/4 = 5)
h = min(dm.shape[0], mask.shape[0]); w = min(dm.shape[1], mask.shape[1])
dm = dm[:h, :w]; mk = mask[:h, :w]
inter = (dm & mk).sum(); union = (dm | mk).sum()
print(f"\nmesh-defined (upsampled) vs prediction mask: IoU={inter/union:.3f}; "
      f"mask covers {(mk & dm).sum()/max(dm.sum(),1):.3f} of mesh, "
      f"mesh covers {(mk & dm).sum()/max(mk.sum(),1):.3f} of mask")
res["mesh_mask_iou"] = float(inter / union)
res["mask_frac_of_mesh"] = float((mk & dm).sum() / max(dm.sum(), 1))

# lattice test on the mesh-defined region
d0 = np.diff(dm.astype(np.int8), axis=0); rows = np.unique(np.nonzero(d0)[0]) + 1
d1 = np.diff(dm.astype(np.int8), axis=1); cols = np.unique(np.nonzero(d1)[1]) + 1
from math import gcd
for tag, p in (("mesh rows", rows), ("mesh cols", cols)):
    f = p * 4
    g = 0
    for v in f[:200]:
        g = gcd(g, int(v))
    print(f"{tag}: n_edges={len(f)} gcd(full-res coords)={g}")
    res[f"{tag.replace(' ','_')}_gcd"] = g

json.dump(res, open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag\vf_mesh.json", "w"),
          indent=1, default=float)
