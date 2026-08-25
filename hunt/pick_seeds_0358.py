"""Pick growth seeds for PHerc0358 from the separability index — measured, not guessed.

The 0813 lesson twice over: seeds must sit ON a predicted sheet (the tracer reports the
seed voxel's value; on the binary m7 prediction that must be 255), and the material
around them should be worth growing in. Here both are enforced:

  1. rank the scroll's 24 uniformly-sampled ROIs by measured sheet separability;
  2. for each of the top cubes, take the cube centre and search a 24-voxel
     neighbourhood of the m7 surface PREDICTION for the nearest 255 voxel;
  3. emit up to N seeds with their separability and the verified prediction value.

Output: hunt/seeds_0358.json in the same (x, y, z) shape as seeds_0813.json.
"""
import json
import os
import numpy as np
import zarr
import fsspec

T = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
PRED = (BUCKET + "/PHerc0358/representations/predictions/surfaces/"
        "20250821151737-surface-20260413222639-surface-m7-L0-th0.2.zarr")
N_SEEDS = 8
SEARCH = 12   # +/- voxels around the cube centre to find an on-sheet voxel


def main():
    rois = json.load(open(os.path.join(T, "out", "k2c_separability", "PHerc0358.json")))["rois"]
    rois = sorted(rois, key=lambda r: -r["coh_med"])
    z0 = zarr.open(fsspec.get_mapper(PRED), mode="r")["0"]
    print(f"prediction volume {z0.shape}, dtype {z0.dtype}")

    seeds = []
    for r in rois:
        if len(seeds) >= N_SEEDS:
            break
        oz, oy, ox = r["origin"]
        cz, cy, cx = oz + 128, oy + 128, ox + 128
        blk = np.asarray(z0[cz - SEARCH:cz + SEARCH, cy - SEARCH:cy + SEARCH, cx - SEARCH:cx + SEARCH])
        hits = np.argwhere(blk == 255)
        if len(hits) == 0:
            print(f"  cube sep={r['coh_med']:.3f} at ({cx},{cy},{cz}): no sheet voxel in ±{SEARCH} — skipped")
            continue
        # nearest on-sheet voxel to the centre
        d = np.abs(hits - SEARCH).sum(axis=1)
        hz, hy, hx = hits[int(np.argmin(d))]
        seed = dict(x=int(cx - SEARCH + hx), y=int(cy - SEARCH + hy), z=int(cz - SEARCH + hz),
                    separability=r["coh_med"],
                    sheet_frac=float((blk == 255).mean()))
        seeds.append(seed)
        print(f"  seed {len(seeds)}: ({seed['x']},{seed['y']},{seed['z']})  "
              f"sep={r['coh_med']:.3f}  sheet_frac={seed['sheet_frac']:.3f}")

    out = os.path.join(T, "hunt", "seeds_0358.json")
    json.dump(seeds, open(out, "w"), indent=1)
    print(f"\nwrote {len(seeds)} verified on-sheet seeds -> {out}")


if __name__ == "__main__":
    main()
