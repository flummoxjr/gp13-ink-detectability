r"""Comb 1 (v2) — dense-ink scan of the K2b papyrus index cubes (14 scrolls).

Hunt for PHerc172-class dense-ink signatures: localized voxel populations far
above the papyrus mode. All 14 GP 9um volumes share uint8 window f32
[-0.03,0.145] (verified from meta) so DN values are cross-comparable.

v1 finding that forced this redesign: the histogram mode of ~half the cubes sits
AT the saturation spike (DN 254/255); saturated fraction ranges 0.02% (0139,
1447, 0800) to 33% (0191_pap0). The shared window CLIPS dense scrolls' papyrus
texture wholesale, so "above-mode outliers" is ill-posed there. Two regimes:
  - percolating clipping (sat_frac >~ 2%): dense-phase load is a scroll/window
    property, not a localized anomaly; largest component percolates.
  - quiet cubes (sat_frac < 2%): a >=100-vox bright component IS a localized
    object worth eyes-on.
Per cube: mode of sub-250 histogram, robust sigma (HWHM of smoothed histogram),
p99.9-vs-mode z, dense-phase mask DN>=250, 26-conn components >=100 vox with
PCA extents, mask-edge distance, dark-adjacency (unsharp-overshoot suspicion),
z-extent (ring suspicion). Render top-6 localized anomalies from quiet cubes.
Outputs: comb/dense_scan.json, comb/dense_anomalies.png
"""
import glob
import json
import os
import re

import numpy as np
from scipy import ndimage as ndi

CACHE = r"D:\vesuvius-data\trackD\k2b"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb"
THR_DENSE = 250       # near-saturation, shared window -> f32 ~0.142+
MIN_COMP = 100
SAT_PERCOLATE = 0.02  # sat_frac above this = clipping regime, not anomaly hunt

os.makedirs(OUT, exist_ok=True)


def cube_stats(a):
    m = a > 0
    v = a[m]
    h = np.bincount(v, minlength=256).astype(np.float64)
    h[0] = 0
    hs = ndi.uniform_filter1d(h, size=5)
    mode = int(np.argmax(hs[:250]))          # exclude saturation spike
    # robust sigma: half-width at half max on the LOW side of the mode
    half = hs[mode] / 2
    lo = mode
    while lo > 1 and hs[lo] > half:
        lo -= 1
    sig = max((mode - lo) / 1.1774, 3.0)     # HWHM -> Gaussian sigma
    p50, p99, p999 = np.percentile(v, [50, 99, 99.9])
    sat = float((v >= THR_DENSE).mean())
    return {
        "n_inmask": int(v.size), "fill": float(m.mean()), "mode": mode,
        "sigma": round(float(sig), 2), "p50": float(p50), "p99": float(p99),
        "p99_9": float(p999), "sat_frac": round(sat, 5),
        "tail_z_p999": round((float(p999) - mode) / sig, 2),
    }, m


def component_geometry(coords):
    c = coords - coords.mean(axis=0)
    cov = c.T @ c / len(c)
    w = np.linalg.eigvalsh(cov)[::-1]
    return np.sqrt(np.maximum(w, 0))


def scan_cube(path):
    a = np.load(path)
    st, mask = cube_stats(a)
    bright = (a >= THR_DENSE) & mask

    # context maps for mundane checks
    edt = ndi.distance_transform_edt(mask) if (~mask).any() else None
    dark = mask & (a < np.percentile(a[mask], 10))
    dark_near = ndi.binary_dilation(dark, iterations=3)

    lab, ncomp = ndi.label(bright, structure=np.ones((3, 3, 3), np.int8))
    comps = []
    largest = 0
    if ncomp:
        sizes = np.bincount(lab.ravel())
        largest = int(sizes[1:].max())
        order = np.argsort(sizes[1:])[::-1] + 1
        for ci in order[:40]:                # cap detailed geometry at top 40
            if sizes[ci] < MIN_COMP:
                break
            coords = np.argwhere(lab == ci)
            vals = a[tuple(coords.T)]
            s1, s2, s3 = component_geometry(coords)
            zmin, ymin, xmin = coords.min(axis=0)
            zmax, ymax, xmax = coords.max(axis=0)
            face_d = int(min(coords.min(), (255 - coords.max(axis=0)).min()))
            edge_d = float(edt[tuple(coords.T)].min()) if edt is not None else 256.0
            pk = coords[np.argmax(vals)]
            comps.append({
                "size": int(sizes[ci]), "peak_dn": int(vals.max()),
                "mean_dn": round(float(vals.mean()), 1),
                "centroid": [round(float(c), 1) for c in coords.mean(axis=0)],
                "peak_zyx": [int(p) for p in pk],
                "s1": round(float(s1), 2), "s2": round(float(s2), 2),
                "s3": round(float(s3), 2),
                "edge_dist": round(edge_d, 1), "face_dist": face_d,
                "z_extent": int(zmax - zmin + 1),
                "inplane_extent": int(max(ymax - ymin, xmax - xmin) + 1),
                "dark_adj_frac": round(float(dark_near[tuple(coords.T)].mean()), 3),
            })
    st["n_comp_ge100"] = sum(1 for c in comps if c["size"] >= MIN_COMP)
    st["largest_comp"] = largest
    st["largest_share"] = round(largest / max(int(bright.sum()), 1), 3)
    st["dense_vox"] = int(bright.sum())
    st["components"] = comps
    return st, a


def main():
    paths = sorted(glob.glob(os.path.join(CACHE, "*_pap*.npy")))
    per_cube = {}
    cubes = {}
    for p in paths:
        name = os.path.basename(p)[:-4]
        st, a = scan_cube(p)
        per_cube[name] = st
        cubes[name] = a
        regime = "PERCOLATING" if st["sat_frac"] >= SAT_PERCOLATE else "quiet"
        print(f"{name:22s} mode={st['mode']:3d} sig={st['sigma']:5.1f} "
              f"tailz={st['tail_z_p999']:6.1f} sat={st['sat_frac']:.4f} "
              f"[{regime}] comps>=100: {st['n_comp_ge100']} "
              f"largest={st['largest_comp']}", flush=True)

    # per-scroll aggregate: dense load, split by regime
    scrolls = {}
    for name, st in per_cube.items():
        s = re.match(r"(PHerc\d+)_", name).group(1)
        d = scrolls.setdefault(s, {"n": 0, "dense": 0, "sat": [], "modes": [],
                                   "quiet_comp_vox": 0, "quiet_ncomp": 0,
                                   "n_quiet": 0})
        d["n"] += st["n_inmask"]
        d["dense"] += st["dense_vox"]
        d["sat"].append(st["sat_frac"])
        d["modes"].append(st["mode"])
        if st["sat_frac"] < SAT_PERCOLATE:
            d["n_quiet"] += 1
            d["quiet_ncomp"] += st["n_comp_ge100"]
            d["quiet_comp_vox"] += sum(c["size"] for c in st["components"]
                                       if c["size"] >= MIN_COMP)
    rank = []
    for s, d in scrolls.items():
        rank.append({
            "scroll": s,
            "dense_ppm": round(1e6 * d["dense"] / d["n"], 1),
            "sat_frac_med": round(float(np.median(d["sat"])), 4),
            "mode_med": float(np.median(d["modes"])),
            "n_quiet_cubes": d["n_quiet"],
            "quiet_ncomp_ge100": d["quiet_ncomp"],
            "quiet_comp_vox": d["quiet_comp_vox"],
        })
    rank.sort(key=lambda r: -r["dense_ppm"])
    print("\n=== scroll ranking by dense-phase load (DN>=250) ===")
    for r in rank:
        print(f"  {r['scroll']}: {r['dense_ppm']:9.1f} ppm  sat_med "
              f"{r['sat_frac_med']:.4f}  mode {r['mode_med']:.0f}  quiet cubes "
              f"{r['n_quiet_cubes']}, their comps {r['quiet_ncomp_ge100']} "
              f"({r['quiet_comp_vox']} vox)")

    with open(os.path.join(OUT, "dense_scan.json"), "w") as f:
        json.dump({"per_cube": per_cube, "ranking": rank,
                   "params": {"thr_dense": THR_DENSE, "min_comp": MIN_COMP,
                              "sat_percolate": SAT_PERCOLATE}}, f, indent=1)

    # ---- top-6 LOCALIZED anomalies: components from quiet cubes only ----
    loc = []
    for name, st in per_cube.items():
        if st["sat_frac"] >= SAT_PERCOLATE:
            continue
        for c in st["components"]:
            if c["size"] >= MIN_COMP:
                loc.append((name, st, c))
    loc.sort(key=lambda t: -t[2]["size"])
    top = loc[:6]
    print(f"\nlocalized components in quiet cubes: {len(loc)}; rendering top {len(top)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = max(len(top), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 8.0), squeeze=False)
    H = 80
    for i, (name, st, c) in enumerate(top):
        a = cubes[name]
        z, y, x = c["peak_zyx"]
        y0, y1 = max(y - H, 0), min(y + H, 256)
        x0, x1 = max(x - H, 0), min(x + H, 256)
        sl = a[z, y0:y1, x0:x1]
        axes[0][i].imshow(sl, cmap="gray", vmin=0, vmax=255)
        axes[0][i].contour((sl >= THR_DENSE).astype(float), levels=[0.5],
                           colors="r", linewidths=0.7)
        axes[0][i].set_title(f"{name}\nz={z} n={c['size']} pk={c['peak_dn']} "
                             f"mode={st['mode']}", fontsize=8)
        z0_, z1_ = max(z - H, 0), min(z + H, 256)
        sl2 = a[z0_:z1_, y0:y1, x]
        axes[1][i].imshow(sl2, cmap="gray", vmin=0, vmax=255)
        axes[1][i].contour((sl2 >= THR_DENSE).astype(float), levels=[0.5],
                           colors="r", linewidths=0.7)
        axes[1][i].set_title(
            f"x={x} s=({c['s1']},{c['s2']},{c['s3']}) zext={c['z_extent']}\n"
            f"edge_d={c['edge_dist']} dark_adj={c['dark_adj_frac']}", fontsize=8)
        for rr in range(2):
            axes[rr][i].set_xticks([]); axes[rr][i].set_yticks([])
    if top:
        axes[0][0].set_ylabel("xy slice @ peak z", fontsize=9)
        axes[1][0].set_ylabel("zy slice @ peak x", fontsize=9)
    fig.suptitle("Top localized bright anomalies (DN>=250) in QUIET cubes "
                 "(sat<2%) — shared window [-0.03,0.145]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "dense_anomalies.png"), dpi=110)
    print("wrote dense_anomalies.png")


if __name__ == "__main__":
    main()
