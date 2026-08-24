"""QC s1a-v2 attack 6 — measure actual registration between ink24_ds4 and the
v2 zoom-resampled ink11 (origin-aligned, no shift correction) over the analyzed
text region, and quantify the impact of any offset on the v2 letter/background
masks.
"""
import json
import os

import numpy as np
from scipy.ndimage import zoom, gaussian_filter, distance_transform_edt

CACHE = r"D:\vesuvius-data\trackD\w032"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc"
TILE = 512
TILES = [(12, 11), (11, 12), (8, 10), (9, 9)]


def ncc(a, b, valid):
    a = a[valid]; b = b[valid]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def main():
    ink24 = np.load(os.path.join(CACHE, "ink24_ds4.npy"))
    ink11 = np.load(os.path.join(CACHE, "ink11_ds4.npy"))
    zf = 2.258 / 2.399
    ink11r = zoom(ink11, zf, order=1)
    ink11a = np.zeros_like(ink24)
    h = min(ink11r.shape[0], ink24.shape[0]); w = min(ink11r.shape[1], ink24.shape[1])
    ink11a[:h, :w] = ink11r[:h, :w]

    # value floors
    r0, r1, c0, c1 = 8 * TILE, 16 * TILE, 8 * TILE, 15 * TILE
    a24 = ink24[r0:r1, c0:c1].astype(np.float32)
    a11 = ink11a[r0:r1, c0:c1].astype(np.float32)
    nz24 = a24[a24 > 0]; nz11 = a11[a11 > 0]
    res = {
        "ink24_nonzero_pct": {p: float(np.percentile(nz24, p)) for p in (0.1, 1, 5, 25, 50, 75)},
        "ink11a_nonzero_pct": {p: float(np.percentile(nz11, p)) for p in (0.1, 1, 5, 25, 50, 75)},
        "region_frac_ink11a_zero": float((a11 == 0).mean()),
        "region_frac_ink24_zero": float((a24 == 0).mean()),
    }
    print("value floors:", json.dumps(res, indent=1), flush=True)

    # --- registration by NCC over integer shifts of ink11a relative to ink24 ---
    # only where both have coverage
    valid_both = (a24 > 0) & (a11 > 0)
    print(f"region {a24.shape}, valid_both frac {valid_both.mean():.3f}", flush=True)

    # blur to blob scale (letters ~ 10-30 ds4 px strokes); z-score-ish
    b24 = gaussian_filter(a24, 4)
    b11 = gaussian_filter(a11, 4)

    def ncc_at(dy, dx):
        s11 = np.roll(np.roll(b11, dy, axis=0), dx, axis=1)
        v11 = np.roll(np.roll(a11 > 0, dy, axis=0), dx, axis=1)
        v = (a24 > 0) & v11
        return ncc(b24, s11, v)

    # coarse at ds4 with step 4 over +-48, then refine step 1
    best = (0, 0, ncc_at(0, 0))
    print(f"NCC at (0,0): {best[2]:.4f}", flush=True)
    coarse = {}
    for dy in range(-48, 49, 4):
        for dx in range(-48, 49, 4):
            c = ncc_at(dy, dx)
            coarse[(dy, dx)] = c
            if c > best[2]:
                best = (dy, dx, c)
    print(f"coarse best: dy={best[0]} dx={best[1]} ncc={best[2]:.4f}", flush=True)
    b0, b1 = best[0], best[1]
    for dy in range(b0 - 3, b0 + 4):
        for dx in range(b1 - 3, b1 + 4):
            c = ncc_at(dy, dx)
            if c > best[2]:
                best = (dy, dx, c)
    print(f"refined best: dy={best[0]} dx={best[1]} ncc={best[2]:.4f}", flush=True)
    res["ncc_at_00"] = round(ncc_at(0, 0), 4)
    res["best_shift_dy_dx"] = [int(best[0]), int(best[1])]
    res["ncc_at_best"] = round(best[2], 4)

    # --- per-tile impact on masks ---
    dy, dx = best[0], best[1]
    ink11c = np.roll(np.roll(ink11a, dy, axis=0), dx, axis=1)  # corrected
    dist_ok = distance_transform_edt(ink24 > 0) >= 8
    tiles_out = {}
    for (ty, tx) in TILES:
        y0, x0 = ty * TILE, tx * TILE
        s24 = ink24[y0:y0 + TILE, x0:x0 + TILE]
        s11 = ink11a[y0:y0 + TILE, x0:x0 + TILE]
        s11c = ink11c[y0:y0 + TILE, x0:x0 + TILE]
        dk = dist_ok[y0:y0 + TILE, x0:x0 + TILE]
        letters = (s24 >= 200) & (s11 >= 150) & dk
        letters_c = (s24 >= 200) & (s11c >= 150) & dk
        bg = (s24 >= 28) & (s24 <= 60) & (s11 <= 60) & dk
        bg_c = (s24 >= 28) & (s24 <= 60) & (s11c <= 60) & dk
        inter = (letters & letters_c).sum()
        dice = 2 * inter / max(letters.sum() + letters_c.sum(), 1)
        # how much of v2 letters loses dual confirmation under corrected reg
        lost = (letters & ~letters_c).sum() / max(letters.sum(), 1)
        gained = (letters_c & ~letters).sum() / max(letters_c.sum(), 1)
        bg_contam = (bg & (s11c > 60)).sum() / max(bg.sum(), 1)  # bg px actually high-ink11
        bg_nocov = (bg & (s11 == 0)).sum() / max(bg.sum(), 1)    # bg px with NO ink11 coverage
        let_nocov_c = (letters & (s11c == 0)).sum() / max(letters.sum(), 1)
        tiles_out[f"{ty}_{tx}"] = {
            "letters_px": int(letters.sum()), "letters_corr_px": int(letters_c.sum()),
            "dice_letters_orig_vs_corr": round(float(dice), 3),
            "frac_letters_lose_dual_conf": round(float(lost), 3),
            "frac_corr_letters_new": round(float(gained), 3),
            "bg_px": int(bg.sum()),
            "frac_bg_actually_ink11_gt60": round(float(bg_contam), 4),
            "frac_bg_no_ink11_coverage": round(float(bg_nocov), 4),
            "frac_letters_no_ink11_cov_after_corr": round(float(let_nocov_c), 4),
        }
        print(f"tile ({ty},{tx}): {tiles_out[f'{ty}_{tx}']}", flush=True)
    res["tiles"] = tiles_out

    # per-tile local registration (is the offset uniform?)
    local = {}
    for (ty, tx) in TILES:
        y0, x0 = ty * TILE - r0, tx * TILE - c0
        sa = slice(y0, y0 + TILE); sb = slice(x0, x0 + TILE)
        t24 = b24[sa, sb]; t11f = b11; ta24 = a24[sa, sb]

        def ncc_t(dyy, dxx):
            s = np.roll(np.roll(b11, dyy, axis=0), dxx, axis=1)[sa, sb]
            v = np.roll(np.roll(a11 > 0, dyy, axis=0), dxx, axis=1)[sa, sb] & (ta24 > 0)
            if v.mean() < 0.2:
                return -9
            return ncc(t24, s, v)

        tb = (0, 0, ncc_t(0, 0))
        for dyy in range(-48, 49, 4):
            for dxx in range(-48, 49, 4):
                c = ncc_t(dyy, dxx)
                if c > tb[2]:
                    tb = (dyy, dxx, c)
        for dyy in range(tb[0] - 3, tb[0] + 4):
            for dxx in range(tb[1] - 3, tb[1] + 4):
                c = ncc_t(dyy, dxx)
                if c > tb[2]:
                    tb = (dyy, dxx, c)
        local[f"{ty}_{tx}"] = {"dy_dx": [int(tb[0]), int(tb[1])], "ncc": round(tb[2], 4),
                               "ncc_00": round(ncc_t(0, 0), 4)}
        print(f"local reg tile ({ty},{tx}): {local[f'{ty}_{tx}']}", flush=True)
    res["local_registration"] = local

    with open(os.path.join(OUT, "qc_s1a_v2_register.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("wrote qc_s1a_v2_register.json", flush=True)


if __name__ == "__main__":
    main()
