"""HUNTER 2 / stage 1 — tile-wise local z-symmetry maps.

For each map pair (forward, z-reversed render) compute per-tile Pearson r
on a fixed 256-px non-overlapping grid, restricted to the eroded valid mask
(erode=40, same as the verdict's stats mask — kills the 1203 rim artifact).

Maps processed:
  control  w035  seed42, seed43   (+ per-tile human-label fraction)
  1203A          seed42, seed43
  1203B          seed42, seed43

Per tile we also record intensity stats of the FORWARD map (mean, p99,
frac>195) so any low-symmetry flag can immediately be compared against the
letter template (letters: value p50=199, frac>195 = 0.69).

Output: comb\sym_tiles_<key>.npz  (one per pair) + comb\sym_summary.json
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_map, valid_mask, load_w035_label2d, save_json

from pathlib import Path
COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")

TILE = 256
MIN_VALID_FRAC = 0.60   # tile must be >=60% inside the eroded valid mask
MIN_STD = 1.0           # both maps need std > 1 DN in-tile for r to mean anything

PAIRS = {
    # key: (fwd map, rev map, has_labels)
    "w035_s42": ("w035_s42", "w035_s42r", True),
    "w035_s43": ("w035_s43", "w035_s43r", True),
    "1203A_s42": ("1203A_s42", "1203A_s42r", False),
    "1203A_s43": ("1203A_s43", "1203A_s43r", False),
    "1203B_s42": ("1203B_s42", "1203B_s42r", False),
    "1203B_s43": ("1203B_s43", "1203B_s43r", False),
}


def tile_stats(fwd, rev, mask, label2d=None):
    H, W = fwd.shape
    ny, nx = H // TILE, W // TILE
    r = np.full((ny, nx), np.nan, np.float32)
    validf = np.zeros((ny, nx), np.float32)
    stdf = np.full((ny, nx), np.nan, np.float32)
    stdr = np.full((ny, nx), np.nan, np.float32)
    meanf = np.full((ny, nx), np.nan, np.float32)
    p99f = np.full((ny, nx), np.nan, np.float32)
    frac195 = np.full((ny, nx), np.nan, np.float32)
    labfrac = np.full((ny, nx), np.nan, np.float32)
    for iy in range(ny):
        y0 = iy * TILE
        for ix in range(nx):
            x0 = ix * TILE
            m = mask[y0:y0 + TILE, x0:x0 + TILE]
            vf = m.mean()
            validf[iy, ix] = vf
            if vf < MIN_VALID_FRAC:
                continue
            a = fwd[y0:y0 + TILE, x0:x0 + TILE][m].astype(np.float32)
            b = rev[y0:y0 + TILE, x0:x0 + TILE][m].astype(np.float32)
            sa, sb = a.std(), b.std()
            stdf[iy, ix], stdr[iy, ix] = sa, sb
            meanf[iy, ix] = a.mean()
            p99f[iy, ix] = np.percentile(a, 99)
            frac195[iy, ix] = (a > 195).mean()
            if label2d is not None:
                lm = label2d[y0:y0 + TILE, x0:x0 + TILE][m]
                labfrac[iy, ix] = lm.mean()
            if sa < MIN_STD or sb < MIN_STD:
                continue
            a -= a.mean(); b -= b.mean()
            r[iy, ix] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    return dict(r=r, validf=validf, stdf=stdf, stdr=stdr, meanf=meanf,
                p99f=p99f, frac195=frac195, labfrac=labfrac)


def main():
    summary = {"tile": TILE, "min_valid_frac": MIN_VALID_FRAC, "min_std": MIN_STD,
               "mask_erode": 40, "pairs": {}}
    lab2d = None
    for key, (fk, rk, has_lab) in PAIRS.items():
        fwd = load_map(fk)
        rev = load_map(rk)
        assert fwd.shape == rev.shape, (key, fwd.shape, rev.shape)
        m = valid_mask(fwd, erode=40) & valid_mask(rev, erode=40)
        lab = None
        if has_lab:
            if lab2d is None:
                lab2d = load_w035_label2d(fwd.shape)
            lab = lab2d
        st = tile_stats(fwd, rev, m, lab)
        np.savez_compressed(COMB / f"sym_tiles_{key}.npz", **st)
        ok = np.isfinite(st["r"])
        rr = st["r"][ok]
        row = {"shape": list(fwd.shape), "n_tiles_grid": int(st["r"].size),
               "n_tiles_valid": int(ok.sum()),
               "r_p05": float(np.percentile(rr, 5)), "r_p25": float(np.percentile(rr, 25)),
               "r_p50": float(np.percentile(rr, 50)), "r_p75": float(np.percentile(rr, 75)),
               "r_p95": float(np.percentile(rr, 95)), "r_min": float(rr.min()),
               "r_max": float(rr.max())}
        if has_lab:
            lf = st["labfrac"]
            letter = ok & (lf >= 0.02)
            blank = ok & (lf == 0.0)
            row["n_letter_tiles"] = int(letter.sum())
            row["n_blank_tiles"] = int(blank.sum())
            if letter.sum():
                lv = st["r"][letter]
                row["letter_r"] = {"p10": float(np.percentile(lv, 10)),
                                   "p50": float(np.percentile(lv, 50)),
                                   "p90": float(np.percentile(lv, 90))}
            if blank.sum():
                bv = st["r"][blank]
                row["blank_r"] = {"p10": float(np.percentile(bv, 10)),
                                  "p50": float(np.percentile(bv, 50)),
                                  "p90": float(np.percentile(bv, 90))}
        summary["pairs"][key] = row
        print(key, row, flush=True)
    save_json(COMB / "sym_summary.json", summary)
    print("saved sym_summary.json")


if __name__ == "__main__":
    main()
