"""Build parts/selection.json for the C2a pod from the C1 ranking (out/bet_b/c1_rank.json).

Set H = every tile with contrast >= 0.24 (w035's 10th percentile; fixed in c1_rank_inband.py before
scoring), capped at 100 by contrast. Set L = the same number of tiles with the LOWEST contrast (the
merged-stack control). Both sets are rendered and scored by the identical pipeline.
The ctl entry is the w035 sub-crop (rows 1024:2432, cols 896:2560 of the 9.362 um canvas = the middle
1408 x 1664 of the harness's CTL_CROP 512:2944 / 384:3072) with the catalogue matrix.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKD = os.path.normpath(os.path.join(HERE, "..", ".."))
THR = 0.24
CAP = 100
CTL_SUB = (1024, 2432, 896, 2560)          # canvas px at 9.362 um (inside CTL_CROP 512:2944, 384:3072)
CTL_CROP = (512, 2944, 384, 3072)


def main():
    rank = json.load(open(os.path.join(TRACKD, "out", "bet_b", "c1_rank.json")))
    ext = {r["name"]: r for r in json.load(open(os.path.join(TRACKD, "hunt", "band_extents.json")))["segments"]}
    tiles = rank["ranked_tiles"]
    H = [t for t in tiles if t["contrast"] >= THR][:CAP]
    L = sorted(tiles, key=lambda t: t["contrast"])[:len(H)]
    segs = {}
    for t, s in [(t, "H") for t in H] + [(t, "L") for t in L]:
        d = segs.setdefault(t["seg"], dict(seg_dir=ext[t["seg"]]["seg_dir"], tiles=[]))
        d["tiles"].append(dict(ij=t["ij"], set=s, contrast=round(t["contrast"], 4), z9=round(t["z"], 1)))
    m0139 = json.load(open(os.path.join(TRACKD, "hunt", "pherc0139_2399um_to_9362um.json")))
    m1203 = json.load(open(os.path.join(TRACKD, "hunt", "pherc1203_2403um_to_9362um.json")))
    y0, y1, x0, x1 = CTL_SUB
    sel = dict(
        made_from="out/bet_b/c1_rank.json (hunt/c1_rank_inband.py, threshold 0.24 fixed before scoring)",
        threshold=THR, n_H=len(H), n_L=len(L), tile=9, margin=1, px_um_1203=2.403,
        ctl=dict(mesh="PHerc0139/segments/20260317000000-w035_2026031718/mesh/20260317000000-on-20250728140407-9.362um.tifxyz",
                 matrix=dict(transformation_matrix=m0139["transformation_matrix"]), inverse=True, px_um=2.399,
                 crop_cells=[y0 // 20 - 1, -(-y1 // 20) + 1, x0 // 20 - 1, -(-x1 // 20) + 1], crop_px_9362=list(CTL_SUB),
                 label_crop=[y0 - CTL_CROP[0], y1 - CTL_CROP[0], x0 - CTL_CROP[2], x1 - CTL_CROP[2]]),
        matrix_1203=dict(transformation_matrix=m1203["transformation_matrix"], _comment=m1203["_comment"][:200]),
        segments=segs,
        H_contrast=[round(t["contrast"], 3) for t in H], L_contrast=[round(t["contrast"], 3) for t in L])
    out = os.path.join(HERE, "parts", "selection.json")
    json.dump(sel, open(out, "w"), indent=1)
    print(f"H {len(H)} tiles (contrast {H[-1]['contrast']:.3f}-{H[0]['contrast']:.3f}), L {len(L)} ({L[0]['contrast']:.3f}-{L[-1]['contrast']:.3f}); "
          f"{len(segs)} segments; ctl crop cells {sel['ctl']['crop_cells']} label_crop {sel['ctl']['label_crop']}; wrote {out}")


if __name__ == "__main__":
    main()
