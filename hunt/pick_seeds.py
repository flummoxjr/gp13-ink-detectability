"""Pick growth seeds for any GP scroll from the separability index — the
pick_seeds_0358.py recipe, parametrised (2026-09-02).

  python hunt/pick_seeds.py --scroll PHerc0826 --n 24 [--keep-existing]

  1. rank the scroll's 24 uniformly-sampled ROIs by measured sheet separability
     (out/k2c_separability/<scroll>.json, key coh_med);
  2. for each cube, take the cube centre and search a ±SEARCH-voxel neighbourhood
     of the m7 surface PREDICTION (level 0, binary) for the nearest 255 voxel —
     the support gate against the 42–50 % phantom rate: a seed must sit on
     predicted sheet;
  3. emit up to N seeds (x, y, z, separability, sheet_frac, pred_value=255).

--keep-existing keeps the seeds already in hunt/seeds_<scroll>.json (e.g. the
8 PHerc0813 seeds that grew correctly on 2026-08-25) and adds new ones from
cubes not already used, so a rerun extends rather than replaces a bank.
"""
import argparse
import json
import os

import fsspec
import numpy as np
import zarr

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
PRED = {
    "PHerc0358": "20250821151737-surface-20260413222639-surface-m7-L0-th0.2.zarr",
    "PHerc0813": "20250821151723-surface-20260413222639-surface-m7-L0-th0.2.zarr",
    "PHerc0826": "20250821151701-surface-20260413222639-surface-m7-L0-th0.2.zarr",
}
SEARCH = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True, choices=sorted(PRED))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--keep-existing", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(T, "hunt", f"seeds_{a.scroll[-4:]}.json")
    rois = json.load(open(os.path.join(T, "out", "k2c_separability", f"{a.scroll}.json"), encoding="utf-8"))["rois"]
    rois = sorted(rois, key=lambda r: -r["coh_med"])
    url = f"{BUCKET}/{a.scroll}/representations/predictions/surfaces/{PRED[a.scroll]}"
    z0 = zarr.open(fsspec.get_mapper(url), mode="r")["0"]
    print(f"{a.scroll}: prediction volume {z0.shape} {z0.dtype}; {len(rois)} ROIs")
    seeds = []
    used = set()
    if a.keep_existing and os.path.exists(out):
        seeds = json.load(open(out))
        for s in seeds:
            used.add((s["x"] // 256, s["y"] // 256, s["z"] // 256))
        print(f"keeping {len(seeds)} existing seeds")
    for r in rois:
        if len(seeds) >= a.n:
            break
        oz, oy, ox = r["origin"]
        cz, cy, cx = oz + 128, oy + 128, ox + 128
        if (cx // 256, cy // 256, cz // 256) in used:
            continue
        sl = tuple(slice(max(0, c - SEARCH), c + SEARCH) for c in (cz, cy, cx))
        blk = np.asarray(z0[sl])
        hits = np.argwhere(blk == 255)
        if len(hits) == 0:
            print(f"  cube sep={r['coh_med']:.3f} at ({cx},{cy},{cz}): no sheet voxel in ±{SEARCH} — skipped")
            continue
        d = np.abs(hits - SEARCH).sum(axis=1)
        hz, hy, hx = hits[int(np.argmin(d))]
        seed = dict(x=int(sl[2].start + hx), y=int(sl[1].start + hy), z=int(sl[0].start + hz),
                    separability=float(r["coh_med"]), sheet_frac=float((blk == 255).mean()), pred_value=255)
        seeds.append(seed)
        used.add((cx // 256, cy // 256, cz // 256))
        print(f"  seed {len(seeds)}: ({seed['x']},{seed['y']},{seed['z']}) sep={r['coh_med']:.3f} sheet_frac={seed['sheet_frac']:.3f}")
    json.dump(seeds, open(out, "w"), indent=1)
    print(f"wrote {len(seeds)} verified on-sheet seeds -> {out}")


if __name__ == "__main__":
    main()
