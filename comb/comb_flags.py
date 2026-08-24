"""HUNTER 2 / stage 2 — flag rule + letter-template cross-check.

Pre-registered rule (from comb_symmetry calibration):
  LOW-symmetry tile = tile r <= THR in BOTH seeds, where THR = pooled median
  tile-r of the control's LETTER-bearing tiles (the "at least as z-asymmetric
  as a median letter tile" regime). Patches = 8-connected components of
  flagged tiles (all sizes kept; contiguous >=2 emphasized).

For every patch, pooled pixel stats (both seeds, in-mask) + morphology at the
map's global in-mask p80 threshold, compared to the w035 letter template:
  letter tiles: frac195 p10 = 0.017, p99 >= 200, tile std p10 = 28
  letter comps: width p50 = 58 px, area p50 = 42k px
The SAME procedure is run on the control itself (both seeds) to calibrate the
procedure's yield on a surface that definitely bears letters.

Output: comb\comb_flags.json (+ prints)
"""
import sys
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_map, valid_mask, save_json

from pathlib import Path
COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")
TILE = 256


def load_tiles(key):
    return dict(np.load(COMB / f"sym_tiles_{key}.npz"))


def patch_pixel_stats(seg, tiles_yx, maps, masks):
    """Pooled in-mask pixel stats + p80-threshold morphology inside patch tiles."""
    out = {}
    for seed in ["s42", "s43"]:
        arr, m = maps[seed], masks[seed]
        sel = np.zeros(arr.shape, bool)
        for (iy, ix) in tiles_yx:
            sel[iy * TILE:(iy + 1) * TILE, ix * TILE:(ix + 1) * TILE] = True
        sel &= m
        v = arr[sel].astype(np.float32)
        thr80 = np.percentile(arr[m], 80)
        binary = (arr >= thr80) & sel
        lab, n = ndi.label(binary, structure=np.ones((3, 3), int))
        widths, areas = [], []
        if n:
            areas_all = np.bincount(lab.ravel())[1:]
            keep = np.flatnonzero(areas_all >= 20) + 1
            if keep.size:
                skel = skeletonize(binary)
                skl = np.bincount(lab[skel], minlength=n + 1)[1:]
                areas = areas_all[keep - 1].astype(float)
                widths = areas / np.maximum(skl[keep - 1], 1.0)
        out[seed] = {
            "n_px": int(sel.sum()),
            "p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max()), "std": float(v.std()),
            "frac195": float((v > 195).mean()),
            "n_comp20": int(len(areas)),
            "width_p50": float(np.median(widths)) if len(widths) else None,
            "width_max": float(np.max(widths)) if len(widths) else None,
            "area_max": float(np.max(areas)) if len(areas) else None,
        }
    return out


def run_segment(seg, thr, has_labels=False):
    t42, t43 = load_tiles(f"{seg}_s42"), load_tiles(f"{seg}_s43")
    ok = np.isfinite(t42["r"]) & np.isfinite(t43["r"])
    flagged = ok & (t42["r"] <= thr) & (t43["r"] <= thr)
    lab, n = ndi.label(flagged, structure=np.ones((3, 3), int))

    maps, masks = {}, {}
    for seed in ["s42", "s43"]:
        a = load_map(f"{seg}_{seed}")
        maps[seed] = a
        masks[seed] = valid_mask(a, erode=40)

    patches = []
    for pid in range(1, n + 1):
        ys, xs = np.where(lab == pid)
        tiles_yx = list(zip(ys.tolist(), xs.tolist()))
        p = {"id": pid, "n_tiles": len(tiles_yx), "tiles": tiles_yx,
             "r42": [float(t42["r"][iy, ix]) for iy, ix in tiles_yx],
             "r43": [float(t43["r"][iy, ix]) for iy, ix in tiles_yx],
             "tile_std42": [float(t42["stdf"][iy, ix]) for iy, ix in tiles_yx]}
        if has_labels:
            p["labfrac"] = [float(t42["labfrac"][iy, ix]) for iy, ix in tiles_yx]
        p.update(patch_pixel_stats(seg, tiles_yx, maps, masks))
        # letter-template verdict per patch (value channel)
        s = p["s42"]; s2 = p["s43"]
        p["letterlike_value"] = bool(
            min(s["frac195"], s2["frac195"]) >= 0.017 and
            min(s["p99"], s2["p99"]) >= 200)
        p["letterlike_morph"] = bool(
            (s["width_max"] or 0) >= 30 and (s["area_max"] or 0) >= 1e4)
        patches.append(p)
    patches.sort(key=lambda p: (-p["n_tiles"], -max(p["s42"]["frac195"], p["s43"]["frac195"])))
    return {"n_valid_tiles": int(ok.sum()), "n_flagged_tiles": int(flagged.sum()),
            "n_patches": n, "patches": patches}


def main():
    # THR from control letter tiles (pooled both seeds)
    c42, c43 = load_tiles("w035_s42"), load_tiles("w035_s43")
    ok = np.isfinite(c42["r"]) & np.isfinite(c43["r"])
    letter = ok & (c42["labfrac"] >= 0.02)
    pooled = np.concatenate([c42["r"][letter], c43["r"][letter]])
    thr = float(np.median(pooled))
    print(f"THR (pooled control letter-tile median r) = {thr:.4f}  "
          f"(n letter tiles per seed = {letter.sum()})", flush=True)

    out = {"thr": thr, "rule": "tile r <= THR in BOTH seeds; 8-connected patches",
           "letter_template": {"frac195_p10": 0.017, "p99_min": 200,
                               "width_p50_px": 58, "area_p50_px": 42000},
           "segments": {}}
    for seg, haslab in [("w035", True), ("1203A", False), ("1203B", False)]:
        res = run_segment(seg, thr, has_labels=haslab)
        out["segments"][seg] = res
        nl_v = sum(p["letterlike_value"] for p in res["patches"])
        nl_m = sum(p["letterlike_morph"] for p in res["patches"])
        big = sum(1 for p in res["patches"] if p["n_tiles"] >= 2)
        print(f"[{seg}] flagged {res['n_flagged_tiles']}/{res['n_valid_tiles']} tiles "
              f"-> {res['n_patches']} patches ({big} contiguous>=2); "
              f"letterlike value {nl_v}, morph {nl_m}", flush=True)
        for p in res["patches"][:8]:
            extras = ""
            if haslab:
                extras = f" labfrac_max={max(p['labfrac']):.3f}"
            print(f"    patch {p['id']}: {p['n_tiles']} tiles at {p['tiles'][:4]}"
                  f" r42={min(p['r42']):+.2f}..{max(p['r42']):+.2f}"
                  f" frac195={p['s42']['frac195']:.4f}/{p['s43']['frac195']:.4f}"
                  f" p99={p['s42']['p99']:.0f}/{p['s43']['p99']:.0f}"
                  f" wmax={p['s42']['width_max']}/{p['s43']['width_max']}"
                  f" LV={p['letterlike_value']} LM={p['letterlike_morph']}{extras}", flush=True)
    save_json(COMB / "comb_flags.json", out)
    print("saved comb_flags.json")


if __name__ == "__main__":
    main()
