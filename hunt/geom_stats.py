#!/usr/bin/env python
"""Part A: whole-mesh geometry statistics from the stored tifxyz grid (no network)."""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\villa\vesuvius\src")
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from vesuvius.tifxyz import read_tifxyz

CACHE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\meshcache"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\out"
os.makedirs(OUT, exist_ok=True)


def ang(dot):
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def analyze(seg):
    d = os.path.join(CACHE, seg["key"])
    surf = read_tifxyz(d, load_mask=False, validate=False)
    x, y, z = surf._x.astype(np.float64), surf._y.astype(np.float64), surf._z.astype(np.float64)
    valid = surf._valid_mask
    h, w = x.shape
    vox = seg["vox_um"]
    r = {"key": seg["key"], "scroll": seg["scroll"], "role": seg["role"],
         "fwd_rev_r": seg["fwd_rev_r"], "vox_um": vox,
         "stored_shape": [int(h), int(w)], "scale": list(map(float, surf._scale)),
         "full_shape": list(surf.full_resolution_shape)}

    # --- coverage / holes -------------------------------------------------
    r["n_grid"] = int(h * w)
    r["valid_frac"] = round(float(valid.mean()), 4)
    lab, n = ndimage.label(valid)
    if n:
        sizes = ndimage.sum_labels(np.ones_like(lab, np.float32), lab, range(1, n + 1))
        r["n_valid_components"] = int(n)
        r["largest_component_frac_of_valid"] = round(float(sizes.max() / max(valid.sum(), 1)), 4)
    # interior holes: invalid pixels not connected to the border
    inv = ~valid
    labi, ni = ndimage.label(inv)
    border = set(np.unique(np.concatenate([labi[0], labi[-1], labi[:, 0], labi[:, -1]])))
    border.discard(0)
    hole_ids = [i for i in range(1, ni + 1) if i not in border]
    hole_px = int(sum((labi == i).sum() for i in hole_ids)) if hole_ids else 0
    r["n_interior_holes"] = len(hole_ids)
    r["interior_hole_frac"] = round(hole_px / max(int(valid.sum()) + hole_px, 1), 5)

    # --- grid regularity: edge lengths in voxels --------------------------
    def edges(axis):
        if axis == 0:
            m = valid[1:, :] & valid[:-1, :]
            dx = x[1:, :] - x[:-1, :]; dy = y[1:, :] - y[:-1, :]; dz = z[1:, :] - z[:-1, :]
        else:
            m = valid[:, 1:] & valid[:, :-1]
            dx = x[:, 1:] - x[:, :-1]; dy = y[:, 1:] - y[:, :-1]; dz = z[:, 1:] - z[:, :-1]
        L = np.sqrt(dx * dx + dy * dy + dz * dz)[m]
        return L[np.isfinite(L)]

    Lr, Lc = edges(0), edges(1)
    for nm, L in (("row", Lr), ("col", Lc)):
        if L.size:
            med = float(np.median(L))
            r[f"edge_{nm}_med_vox"] = round(med, 3)
            r[f"edge_{nm}_cv"] = round(float(L.std() / max(L.mean(), 1e-9)), 4)
            r[f"edge_{nm}_frac_gt2x"] = round(float((L > 2 * med).mean()), 5)
    if Lr.size and Lc.size:
        r["edge_anisotropy"] = round(float(np.median(Lr) / max(np.median(Lc), 1e-9)), 3)
        allL = np.concatenate([Lr, Lc])
        r["edge_all_cv"] = round(float(allL.std() / max(allL.mean(), 1e-9)), 4)
        r["step_um"] = round(float(np.median(allL)) * vox, 2)

    # --- area -------------------------------------------------------------
    try:
        area_vx2 = float(surf.quad_area())
    except Exception:
        area_vx2 = float(json.load(open(os.path.join(d, "meta.json"))).get("area_vx2", np.nan))
    r["area_mm2"] = round(area_vx2 * (vox / 1000.0) ** 2, 3)

    # --- normal coherence + curvature ------------------------------------
    nx, ny, nz = surf.compute_normals()
    fin = np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)
    r["normal_nan_frac"] = round(float(1 - fin.mean()), 4)
    N = np.stack([nx, ny, nz], -1)

    def pair_ang(axis):
        if axis == 0:
            m = fin[1:, :] & fin[:-1, :]
            dot = (N[1:, :] * N[:-1, :]).sum(-1)[m]
            L = np.sqrt(((np.stack([x, y, z], -1)[1:, :] - np.stack([x, y, z], -1)[:-1, :]) ** 2).sum(-1))[m]
        else:
            m = fin[:, 1:] & fin[:, :-1]
            dot = (N[:, 1:] * N[:, :-1]).sum(-1)[m]
            L = np.sqrt(((np.stack([x, y, z], -1)[:, 1:] - np.stack([x, y, z], -1)[:, :-1]) ** 2).sum(-1))[m]
        return ang(dot), L

    a0, L0 = pair_ang(0)
    a1, L1 = pair_ang(1)
    A = np.concatenate([a0, a1]); LL = np.concatenate([L0, L1])
    ok = np.isfinite(A) & np.isfinite(LL) & (LL > 1e-6)
    A, LL = A[ok], LL[ok]
    r["normal_dispersion_med_deg"] = round(float(np.median(A)), 3)
    r["normal_dispersion_p90_deg"] = round(float(np.percentile(A, 90)), 3)
    r["normal_dispersion_p99_deg"] = round(float(np.percentile(A, 99)), 3)
    r["normal_frac_gt30deg"] = round(float((A > 30).mean()), 5)
    r["normal_frac_antiparallel"] = round(float((A > 90).mean()), 6)
    # curvature: deg per mm along the surface
    kappa = A / (LL * vox / 1000.0)
    r["curv_med_deg_per_mm"] = round(float(np.median(kappa)), 2)
    r["curv_p90_deg_per_mm"] = round(float(np.percentile(kappa, 90)), 2)
    # radius of curvature in mm at the median
    r["radius_curv_med_mm"] = round(float(np.degrees(1.0) / max(np.median(kappa), 1e-9)), 3)

    # --- self-contact / doubling -----------------------------------------
    ii, jj = np.nonzero(valid)
    if ii.size > 4000:
        sel = np.random.default_rng(0).choice(ii.size, 4000, replace=False)
        ii, jj = ii[sel], jj[sel]
    P = np.stack([x[ii, jj], y[ii, jj], z[ii, jj]], -1)
    G = np.stack([ii, jj], -1).astype(np.float64)
    if P.shape[0] > 10:
        tree = cKDTree(P)
        # neighbours within 3 voxels in 3D
        pairs = tree.query_pairs(r=3.0, output_type="ndarray")
        if pairs.size:
            gd = np.abs(G[pairs[:, 0]] - G[pairs[:, 1]]).max(1)
            far = gd > 5  # >5 stored steps = >100 voxels apart along the sheet
            r["selfcontact_pairs_frac"] = round(float(far.sum()) / max(P.shape[0], 1), 5)
        else:
            r["selfcontact_pairs_frac"] = 0.0
    return r


if __name__ == "__main__":
    segs = json.load(open(os.path.join(CACHE, "segs.json")))
    res = []
    for s in segs:
        try:
            r = analyze(s)
        except Exception as e:
            r = {"key": s["key"], "error": f"{type(e).__name__}: {e}"}
        res.append(r)
        print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(OUT, "geom_stats.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "geom_stats.json"))
