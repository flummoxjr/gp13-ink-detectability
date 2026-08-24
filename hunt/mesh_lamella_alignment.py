"""Do the PHerc0813 meshes actually follow the lamellae they sit in?

Three measurements now exist for the same 8 patches and they have to be reconciled:

  bulk material at the seed sites   separability 0.352-0.701 (median 0.613)
  the scroll's typical material     separability 0.665 (isotropic floor 0.105)
  the grown patches' depth profile  lamella modulation 0.037-0.073 (control 0.443)

The first two say the material at those sites has well-defined sheets. The third says
that along the mesh's own normal, density barely varies. Both can be true at once if
the mesh is not oriented along the sheet normal -- a surface lying oblique to the
lamellae samples across them and averages the modulation away.

This measures the angle directly: the mesh normal (from the tifxyz grid's tangent
cross-product, at the patch centre) against the local sheet normal (the leading
structure-tensor eigenvector of the cached cube at the same site). Near 0 deg means the
mesh follows the sheets and the flat profile is a material result. Near 90 deg means the
mesh cuts across them and the flat profile is a geometry result.
"""
import json
import os
import numpy as np
import tifffile
from scipy import ndimage as ndi

T = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
MESHES = os.path.join(T, "hunt", "pherc0813_meshes")
SEEDCACHE = r"D:\vesuvius-data\trackD\k2c_seeds"


def mesh_normal(d):
    """Median unit normal of a tifxyz patch, from grid tangent cross-products.

    NOTE: tifxyz marks invalid vertices with **-1**, not 0. An earlier version of
    this function masked on (x==0)&(y==0)&(z==0), which never matched anything, so
    -1 sentinels were treated as real coordinates. Their gradients at the
    valid/invalid boundary are enormous and dominate the orientation tensor. The
    published GP meshes are 47-52% invalid and ours are ~4%, so that bug biased the
    two populations by very different amounts. Mask on sign, and require a vertex's
    central-difference neighbours to be valid too (erode by one).
    """
    x = tifffile.imread(os.path.join(d, "x.tif")).astype(np.float64)
    y = tifffile.imread(os.path.join(d, "y.tif")).astype(np.float64)
    z = tifffile.imread(os.path.join(d, "z.tif")).astype(np.float64)
    valid = (x >= 0) & (y >= 0) & (z >= 0) & ~((x == 0) & (y == 0) & (z == 0))
    # a central difference at p needs p's 4-neighbours; erode so only interior
    # vertices of valid regions contribute
    core = ndi.binary_erosion(valid, structure=np.ones((3, 3), bool), border_value=0)
    if core.sum() < 50:
        return None, 0
    tu = np.stack([np.gradient(a, axis=0) for a in (z, y, x)], -1)
    tv = np.stack([np.gradient(a, axis=1) for a in (z, y, x)], -1)
    n = np.cross(tu, tv)
    ok = core & (np.linalg.norm(n, axis=-1) > 1e-6)
    if ok.sum() < 50:
        return None, int(core.sum())
    nn = n[ok]
    nn = nn / np.linalg.norm(nn, axis=-1, keepdims=True)
    # axial average (sign of a normal is arbitrary): leading eigenvector of the
    # orientation tensor, not the arithmetic mean
    Tn = np.einsum('ij,ik->jk', nn, nn) / len(nn)
    w, V = np.linalg.eigh(Tn)
    return V[:, 2], int(ok.sum())


def sheet_normal(a, block=32, sigma=1.0):
    """Coherence-weighted axial mean of local structure-tensor normals in a cube."""
    n = a.shape[0]
    N, C = [], []
    for z in range(0, n - block + 1, block):
        for y in range(0, n - block + 1, block):
            for x in range(0, n - block + 1, block):
                bl = a[z:z + block, y:y + block, x:x + block]
                if (bl > 0).mean() < 0.98:
                    continue
                v = ndi.gaussian_filter(bl.astype(np.float32), sigma)
                g = np.gradient(v)
                J = np.array([[float((g[i] * g[j]).mean()) for j in range(3)] for i in range(3)])
                w, V = np.linalg.eigh(J)
                C.append((w[2] - w[1]) / max(w[2] + w[1], 1e-9))
                N.append(V[:, 2])
    if len(N) < 8:
        return None
    N = np.array(N)
    C = np.array(C)
    Tn = np.einsum('i,ij,ik->jk', C / C.sum(), N, N)
    w, V = np.linalg.eigh(Tn)
    return V[:, 2]


def main():
    seeds = json.load(open(os.path.join(T, "out", "k2c_separability",
                                        "pherc0813_seed_separability.json")))
    dirs = sorted(d for d in os.listdir(MESHES) if os.path.isdir(os.path.join(MESHES, d)))
    rows = []
    print(f"{'mesh':30}{'sep@seed':>10}{'angle':>9}{'verts':>9}  interpretation")
    for i, dn in enumerate(dirs):
        cp = os.path.join(SEEDCACHE, f"PHerc0813_seed{i:02d}.npy")
        if not os.path.exists(cp):
            continue
        mn, nv = mesh_normal(os.path.join(MESHES, dn))
        sn = sheet_normal(np.load(cp))
        if mn is None or sn is None:
            print(f"{dn[-14:]:30}{'-':>10}{'-':>9}{nv:>9}  (insufficient data)")
            continue
        # axial angle: normals have no sign, so fold to [0, 90]
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(mn, sn)))))))
        sep = seeds["seeds"][i]["separability"] if i < len(seeds["seeds"]) else None
        note = ("mesh follows the sheets" if ang < 30 else
                "oblique" if ang < 60 else "mesh cuts ACROSS the sheets")
        rows.append(dict(mesh=dn, angle_deg=ang, sep_at_seed=sep, n_vertices=nv,
                         mesh_normal=list(map(float, mn)), sheet_normal=list(map(float, sn))))
        print(f"{dn[-14:]:30}{(sep if sep else float('nan')):>10.3f}{ang:>8.1f}°{nv:>9}  {note}")

    if rows:
        ang = [r["angle_deg"] for r in rows]
        # null: angle between two independent random axial directions in 3D has
        # median 60 deg (P(angle < a) = 1 - cos a)
        print(f"\nmedian mesh-vs-sheet angle: {np.median(ang):.1f}°  (n={len(ang)})")
        print(f"random-orientation null median: 60.0°   |   perfect alignment: 0°")
        print(f"meshes within 30° of the sheets: {sum(1 for a in ang if a < 30)}/{len(ang)}")
        json.dump({"scroll": "PHerc0813", "meshes": rows,
                   "median_angle_deg": float(np.median(ang)),
                   "random_null_median_deg": 60.0,
                   "n_within_30deg": int(sum(1 for a in ang if a < 30))},
                  open(os.path.join(T, "out", "k2c_separability",
                                    "pherc0813_mesh_alignment.json"), "w"), indent=1)
        print("wrote pherc0813_mesh_alignment.json")


if __name__ == "__main__":
    main()
