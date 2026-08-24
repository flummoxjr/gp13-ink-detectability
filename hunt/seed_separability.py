"""Were the 8 PHerc0813 surfaces grown in laminated material, or not?

The grown patches show lamella modulation 0.037-0.073 against the control's 0.443,
which was read (in NEXT_SESSION.md) as "PHerc0813 has unresolvable sheets". But the
K2c random-frame measurement says PHerc0813's bulk material is well laminated
(separability 0.665, median of 12 uniformly-sampled ROIs, against an isotropic floor
of 0.105). Those two statements cannot both describe the same material.

This resolves it by measuring the separability statistic AT THE SEED COORDINATES the
meshes were actually grown from, and comparing against that scroll's own random-frame
distribution. If the seeds sit low in their own scroll's distribution, the flat depth
profiles are a mesh-placement result, not a property of the scroll.
"""
import json
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD")
from k2c_separability import open_level, vol_info, ROI  # noqa: E402
from k2c_analyze import coh_med  # noqa: E402

T = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
CACHE = r"D:\vesuvius-data\trackD\k2c_seeds"
SAMPLE, VID = "PHerc0813", "20250821151723"


def main():
    os.makedirs(CACHE, exist_ok=True)
    seeds = json.load(open(os.path.join(T, "hunt", "seeds_0813.json")))
    qc = json.load(open(os.path.join(T, "hunt", "pherc0813_mesh_qc.json")))
    ref = json.load(open(os.path.join(T, "out", "k2c_separability", f"{SAMPLE}.json")))
    rnd = sorted(r["coh_med"] for r in ref["rois"])

    long_id, shape0 = vol_info(SAMPLE, VID)
    z0 = open_level(SAMPLE, long_id, 0)

    def fetch(i_s):
        i, s = i_s
        # seeds are (x, y, z); the volume is indexed (z, y, x)
        o = (int(np.clip(s["z"] - ROI // 2, 0, shape0[0] - ROI)),
             int(np.clip(s["y"] - ROI // 2, 0, shape0[1] - ROI)),
             int(np.clip(s["x"] - ROI // 2, 0, shape0[2] - ROI)))
        cp = os.path.join(CACHE, f"{SAMPLE}_seed{i:02d}.npy")
        if os.path.exists(cp):
            return i, s, o, np.load(cp)
        a = np.asarray(z0[o[0]:o[0] + ROI, o[1]:o[1] + ROI, o[2]:o[2] + ROI])
        np.save(cp, a)
        return i, s, o, a

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, s, o, a in ex.map(fetch, list(enumerate(seeds))):
            c = coh_med(a)
            fill = float((a > 0).mean())
            pct = 100.0 * sum(1 for v in rnd if v < c) / len(rnd)
            rows.append(dict(seed=i, xyz=[s["x"], s["y"], s["z"]], origin=list(o),
                             sheet_frac=s.get("sheet_frac"), separability=None if not np.isfinite(c) else float(c),
                             fill=fill, mean_dn=float(a.astype(np.float32).mean()),
                             percentile_in_own_scroll=pct))
            print(f"seed {i}: xyz=({s['x']},{s['y']},{s['z']})  fill={fill:.3f}  "
                  f"DN={a.astype(np.float32).mean():6.1f}  sep={c:.3f}  "
                  f"= {pct:.0f}th pct of PHerc0813's random frame", flush=True)

    ok = [r for r in rows if r["separability"] is not None and r["fill"] > 0.5]
    out = dict(scroll=SAMPLE,
               random_frame_median=float(np.median(rnd)),
               random_frame_range=[float(min(rnd)), float(max(rnd))],
               isotropic_floor_air_median=0.105,
               seeds=rows,
               seed_median=float(np.median([r["separability"] for r in ok])) if ok else None,
               n_seeds_below_random_min=sum(1 for r in ok if r["separability"] < min(rnd)))
    # tie each mesh's measured depth-profile modulation to its seed, where the order matches
    for r, q in zip(rows, qc):
        r["mesh_name"] = q.get("name")
        r["mesh_surface_zero_frac"] = q.get("surface_zero_frac")
        r["mesh_contrast"] = q.get("contrast")
        r["mesh_surface_mean_DN"] = q.get("surface_mean_DN")
    p = os.path.join(T, "out", "k2c_separability", "pherc0813_seed_separability.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nrandom frame: median {np.median(rnd):.3f}, range {min(rnd):.3f}-{max(rnd):.3f}")
    if ok:
        print(f"seed sites:   median {np.median([r['separability'] for r in ok]):.3f} "
              f"over {len(ok)} in-material seeds")
    print("wrote", p)


if __name__ == "__main__":
    main()
