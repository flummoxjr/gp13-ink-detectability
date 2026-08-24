"""m2_components.py — H2/H3 morphology analysis on sampled prob maps.

Per tile (64^3 probL2, CT L2, surface-pred L2-frame):
  - connected components of prob>0.5 (26-conn), >=20 vox
  - per component: size, PCA axes (elongation/planarity/thickness), plate normal
  - local sheet normal from surface-pred structure tensor (CT fallback)
  - angle(plate normal, sheet normal): 0 deg = sheet-conformal (ink-like),
    90 deg = through-sheet (crack-like)
  - classify: patch / ribbon / oblique / filament / blob / edge
Per-tile: class voxel fractions, flag ink-plausible tiles.
Global: coplanarity angle distribution + permutation nulls (within-tile and
across-tile shuffles, 200 perms).

Outputs: components.json, tile_scores.json, coplanarity_null.json
"""
import json, os, math
import numpy as np
from scipy import ndimage as ndi

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"
CACHE = os.path.join(OUT, "cache")
RNG = np.random.default_rng(1203)

MIN_VOX = 20
STRUCT = np.ones((3, 3, 3), bool)          # 26-connectivity
NBOX = 8                                    # half-width of normal-estimation box
SURF_COVER_MIN = 0.02                       # min surf>127 fraction in box to trust surf normal
COHER_MIN = 0.10

inv = json.load(open(os.path.join(OUT, "inventory.json")))["samples"]

def structure_tensor_field(vol, sigma):
    v = ndi.gaussian_filter(vol.astype(np.float32), sigma)
    gz, gy, gx = np.gradient(v)
    return {"zz": gz*gz, "yy": gy*gy, "xx": gx*gx,
            "zy": gz*gy, "zx": gz*gx, "yx": gy*gx}

def local_normal(J, cz, cy, cx):
    """Average structure tensor in a box around (cz,cy,cx); return
    (normal, coherence, energy)."""
    sl = tuple(slice(max(0, c - NBOX), min(64, c + NBOX + 1)) for c in (cz, cy, cx))
    S = np.array([[J["zz"][sl].mean(), J["zy"][sl].mean(), J["zx"][sl].mean()],
                  [J["zy"][sl].mean(), J["yy"][sl].mean(), J["yx"][sl].mean()],
                  [J["zx"][sl].mean(), J["yx"][sl].mean(), J["xx"][sl].mean()]])
    w, V = np.linalg.eigh(S)               # ascending
    tot = float(w.sum())
    if tot <= 1e-12:
        return None, 0.0, 0.0
    coher = float((w[2] - w[1]) / tot)     # plate-ness of the local intensity structure
    return V[:, 2].astype(float), coher, tot

def surf_cover(surf, cz, cy, cx):
    sl = tuple(slice(max(0, c - NBOX), min(64, c + NBOX + 1)) for c in (cz, cy, cx))
    return float((surf[sl] > 127).mean())

all_comps = []
tile_rows = []

for rec in inv:
    z, y, x = rec["tile"]
    prob8 = np.load(rec["file"])
    ct = np.load(os.path.join(CACHE, f"ct_{z}_{y}_{x}.npy"))
    surf = np.load(os.path.join(CACHE, f"surf_{z}_{y}_{x}.npy"))
    binm = prob8 >= 128                     # prob > 0.5

    interior = ct > 5
    bg = ~interior
    # distance to background (mask edge) in L2 voxels
    dist_bg = ndi.distance_transform_edt(interior)
    sheet = surf > 127
    sheet_dil = ndi.binary_dilation(sheet, STRUCT, iterations=2)

    J_surf = structure_tensor_field(surf, 2.0)
    J_ct = structure_tensor_field(ct, 1.5)

    lab, n = ndi.label(binm, structure=STRUCT)
    sizes = np.bincount(lab.ravel())
    keep = [i for i in range(1, n + 1) if sizes[i] >= MIN_VOX]

    comps_here = []
    for ci in keep:
        idx = np.argwhere(lab == ci)
        npx = len(idx)
        c = idx.mean(0)
        cz, cy, cx = [int(round(v)) for v in c]
        X = idx - c
        C = (X.T @ X) / npx
        w, V = np.linalg.eigh(C)            # ascending
        w = np.clip(w, 1e-6, None)
        a3, a2, a1 = np.sqrt(w)             # minor..major std
        v_major, v_mid, v_minor = V[:, 2], V[:, 1], V[:, 0]
        proj = X @ V                        # cols: minor, mid, major
        e3 = float(proj[:, 0].max() - proj[:, 0].min() + 1)   # thickness extent
        e2 = float(proj[:, 1].max() - proj[:, 1].min() + 1)
        e1 = float(proj[:, 2].max() - proj[:, 2].min() + 1)
        elong = float(a1 / a2)
        planar = float(a2 / a3)

        pv = prob8[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32) / 255.0
        onsheet = float(sheet[idx[:, 0], idx[:, 1], idx[:, 2]].mean())
        onsheet2 = float(sheet_dil[idx[:, 0], idx[:, 1], idx[:, 2]].mean())
        near_bg = float((dist_bg[idx[:, 0], idx[:, 1], idx[:, 2]] <= 3).mean())
        touches = bool((idx.min() == 0) or (idx.max(0) >= 63).any())

        cover = surf_cover(surf, cz, cy, cx)
        if cover >= SURF_COVER_MIN:
            nrm, coher, _ = local_normal(J_surf, cz, cy, cx)
            nsrc = "surf"
        else:
            nrm, coher, _ = local_normal(J_ct, cz, cy, cx)
            nsrc = "ct"
        if nrm is None or coher < COHER_MIN:
            angle = None
        else:
            angle = float(np.degrees(np.arccos(
                np.clip(abs(np.dot(v_minor, nrm)), 0, 1))))

        # ---- classification ----
        flat = planar >= 2.0 and e3 <= 10.0
        if near_bg > 0.5:
            cls = "edge"
        elif flat and angle is not None and angle <= 30:
            cls = "patch"
        elif flat and angle is not None and angle >= 60:
            cls = "ribbon"
        elif flat:
            cls = "oblique" if angle is not None else "flat_unoriented"
        elif elong >= 3.0:
            cls = "filament"
        else:
            cls = "blob"

        comp = {
            "tile": [z, y, x], "worker": rec["worker"], "size": int(npx),
            "a1": float(a1), "a2": float(a2), "a3": float(a3),
            "e1": e1, "e2": e2, "e3": e3,
            "elong": elong, "planar": planar,
            "plate_normal": [float(q) for q in v_minor],
            "sheet_normal": [float(q) for q in nrm] if nrm is not None else None,
            "normal_src": nsrc, "coherence": float(coher),
            "angle_deg": angle,
            "p_mean": float(pv.mean()), "p_max": float(pv.max()),
            "onsheet": onsheet, "onsheet_dil2": onsheet2,
            "near_bg": near_bg, "touches_border": touches,
            "class": cls,
        }
        comps_here.append(comp)
    all_comps.extend(comps_here)

    vox_tot = sum(cp["size"] for cp in comps_here)
    def frac(cl):
        return (sum(cp["size"] for cp in comps_here if cp["class"] == cl) / vox_tot
                if vox_tot else 0.0)
    patch_comps = [cp for cp in comps_here if cp["class"] == "patch"]
    patch_stroke = [cp for cp in patch_comps if 2 <= cp["e3"] <= 10]
    patch_onsheet_vox = sum(cp["size"] for cp in patch_comps if cp["onsheet_dil2"] >= 0.5)
    patch_vox = sum(cp["size"] for cp in patch_comps)
    row = {
        "tile": [z, y, x], "worker": rec["worker"], "fill": rec["fill"],
        "f05": rec["f05"], "pmax": rec["pmax"],
        "n_comps": len(comps_here), "vox_in_comps": vox_tot,
        "fired_frac": float(binm.mean()),
        "patch_frac": frac("patch"), "ribbon_frac": frac("ribbon"),
        "oblique_frac": frac("oblique"), "filament_frac": frac("filament"),
        "blob_frac": frac("blob"), "edge_frac": frac("edge"),
        "n_patch": len(patch_comps), "n_patch_stroke": len(patch_stroke),
        "patch_onsheet_share": (patch_onsheet_vox / patch_vox) if patch_vox else 0.0,
        "median_patch_e3": (float(np.median([cp["e3"] for cp in patch_comps]))
                            if patch_comps else None),
    }
    row["flag_ink_plausible"] = bool(
        row["patch_frac"] >= 0.5 and patch_vox > 0
        and row["patch_onsheet_share"] >= 0.5
        and (row["median_patch_e3"] is not None and 2 <= row["median_patch_e3"] <= 10))
    tile_rows.append(row)
    print(f"tile {z:5d} {y:5d} {x:5d} w={rec['worker']}: comps={len(comps_here):3d} "
          f"patch={row['patch_frac']:.2f} ribbon={row['ribbon_frac']:.2f} "
          f"oblique={row['oblique_frac']:.2f} blob={row['blob_frac']:.2f} "
          f"edge={row['edge_frac']:.2f} flag={row['flag_ink_plausible']}")

json.dump(all_comps, open(os.path.join(OUT, "components.json"), "w"))
json.dump(tile_rows, open(os.path.join(OUT, "tile_scores.json"), "w"), indent=1)

# ================= coplanarity: observed vs permutation nulls =================
valid = [cp for cp in all_comps if cp["angle_deg"] is not None and cp["class"] != "edge"]
angles = np.array([cp["angle_deg"] for cp in valid])
plate = np.array([cp["plate_normal"] for cp in valid])
sheetn = np.array([cp["sheet_normal"] for cp in valid])
tiles_of = np.array([tuple(cp["tile"]) for cp in valid])
tile_ids = np.array([hash(tuple(cp["tile"])) for cp in valid])

def med_frac(a):
    return float(np.median(a)), float((a <= 30).mean()), float((a >= 60).mean())

obs_med, obs_f30, obs_f60 = med_frac(angles)
NPERM = 200

def angles_for(perm_sheet):
    d = np.abs(np.einsum("ij,ij->i", plate, perm_sheet))
    return np.degrees(np.arccos(np.clip(d, 0, 1)))

# null A: shuffle sheet normals ACROSS all components (breaks tile pairing)
nullA = np.empty((NPERM, 3))
for k in range(NPERM):
    p = RNG.permutation(len(valid))
    nullA[k] = med_frac(angles_for(sheetn[p]))

# null B: shuffle WITHIN tile (conservative: local sheets are near-parallel)
nullB = np.empty((NPERM, 3))
order = np.arange(len(valid))
for k in range(NPERM):
    p = order.copy()
    for t in np.unique(tile_ids):
        m = np.where(tile_ids == t)[0]
        if len(m) > 1:
            p[m] = m[RNG.permutation(len(m))]
    nullB[k] = med_frac(angles_for(sheetn[p]))

# null C: isotropic random plate normals (geometry-free reference)
nullC = np.empty((NPERM, 3))
for k in range(NPERM):
    g = RNG.normal(size=(len(valid), 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    d = np.abs(np.einsum("ij,ij->i", g, sheetn))
    nullC[k] = med_frac(np.degrees(np.arccos(np.clip(d, 0, 1))))

def summarize(null, name):
    out = {}
    for j, stat in enumerate(["median_angle", "frac_le30", "frac_ge60"]):
        obs = [obs_med, obs_f30, obs_f60][j]
        mu, sd = float(null[:, j].mean()), float(null[:, j].std())
        # one-sided empirical p toward "alignment" (smaller median / larger f30 / smaller f60)
        if stat == "frac_le30":
            pemp = float((null[:, j] >= obs).mean())
        else:
            pemp = float((null[:, j] <= obs).mean())
        out[stat] = {"obs": obs, "null_mean": mu, "null_sd": sd,
                     "z": (obs - mu) / sd if sd > 0 else None, "p_emp_align": pemp}
    print(f"[{name}] median {obs_med:.1f} vs null {out['median_angle']['null_mean']:.1f}"
          f"+-{out['median_angle']['null_sd']:.1f}  f<=30 {obs_f30:.2f} vs "
          f"{out['frac_le30']['null_mean']:.2f}+-{out['frac_le30']['null_sd']:.2f}")
    return out

res = {
    "n_valid_comps": len(valid),
    "n_perm": NPERM,
    "observed": {"median_angle": obs_med, "frac_le30": obs_f30, "frac_ge60": obs_f60},
    "null_across_tiles": summarize(nullA, "across-tile shuffle"),
    "null_within_tile": summarize(nullB, "within-tile shuffle"),
    "null_isotropic": summarize(nullC, "isotropic plates"),
}

# split by class and on-sheet
for cl in ["patch", "ribbon", "oblique", "filament", "blob"]:
    a = np.array([cp["angle_deg"] for cp in valid if cp["class"] == cl])
    if len(a):
        res[f"angles_{cl}"] = {"n": int(len(a)), "median": float(np.median(a)),
                               "p25": float(np.percentile(a, 25)),
                               "p75": float(np.percentile(a, 75))}
on = np.array([cp["angle_deg"] for cp in valid if cp["onsheet_dil2"] >= 0.5])
off = np.array([cp["angle_deg"] for cp in valid if cp["onsheet_dil2"] < 0.5])
res["angles_onsheet"] = {"n": int(len(on)), "median": float(np.median(on)) if len(on) else None}
res["angles_offsheet"] = {"n": int(len(off)), "median": float(np.median(off)) if len(off) else None}

json.dump(res, open(os.path.join(OUT, "coplanarity_null.json"), "w"), indent=1)

# ---------------- global class summary ----------------
tot = sum(cp["size"] for cp in all_comps)
print("\n=== global voxel shares by class ===")
for cl in ["patch", "ribbon", "oblique", "filament", "blob", "edge", "flat_unoriented"]:
    v = sum(cp["size"] for cp in all_comps if cp["class"] == cl)
    nn = sum(1 for cp in all_comps if cp["class"] == cl)
    if nn:
        print(f"{cl:16s} n={nn:5d}  vox_share={v/tot:.3f}")
print(f"\ncomponents total: {len(all_comps)}, valid-normal non-edge: {len(valid)}")
print(f"flagged tiles: {sum(1 for r in tile_rows if r['flag_ink_plausible'])} / {len(tile_rows)}")
print("normal src counts:", {s: sum(1 for c in all_comps if c['normal_src'] == s) for s in ['surf', 'ct']})
