"""QC confound analysis for s1a — runs offline on tiles extracted by
qc_s1a_extract.py. Prints a full report and writes qc_s1a_results.json.

Tests:
  T0 reproduction of original AUCs
  T1 no-data-corrected background (ink in [3,20), >=8px from 0-zones)
  T1b additionally exclude saturated-render zones from BOTH classes
  T2 placebo: rot180, +64 shift, 12 random shifts (null distribution)
  T3 independent ink11 mask + intersection mask
  T4 eroded letter masks (2,3 iterations)
  T5 per-tile letter-to-saturation proximity vs AUC
  T7 corrected depth profile
  plus: per-tile z-scored pooled AUC (removes between-tile offsets),
        Spearman rho ink-value vs brightness
"""
import glob
import json
import os
import re

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.stats import rankdata, spearmanr

QCD = r"D:\vesuvius-data\trackD\w032\qc"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc"
WIN = (-0.03, 0.145)
K_OFFSETS = list(range(-4, 5))
KI = {k: i for i, k in enumerate(K_OFFSETS)}
STRONG_INK = 170
WEAK_BG = 20
# The ink TIFs' nonzero values have a floor ~28 (P1 of nonzero = 30), so the
# original ink<20 background is ~pure ink==0 no-data. "Predicted-but-low"
# background must be defined as a low band of actual predictions:
BG_LO, BG_HI = 25, 60
SAT_RAW = 240          # uint8 threshold for "saturated white" render zones
SAT_RAW2 = 200         # softer saturation threshold variant
NEAR_PX = 8


def tie_auc(pos, neg, max_n=300_000):
    rng = np.random.default_rng(1)
    if len(pos) > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if len(neg) > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    allv = np.concatenate([pos, neg])
    ranks = rankdata(allv)
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def load_tiles():
    tiles = []
    for f in sorted(glob.glob(os.path.join(QCD, "tile_*.npz"))):
        m = re.search(r"tile_(\d+)_(\d+)\.npz", f)
        d = np.load(f)
        tiles.append({
            "id": (int(m.group(1)), int(m.group(2))),
            "sub_ink": d["sub_ink"], "ink11r": d["ink11r"],
            "ok_pre": d["ok_pre"], "raw0": d["raw0"], "vals": d["vals"],
            "X": d["X"], "Y": d["Y"], "Z": d["Z"],
        })
    return tiles


def pooled_auc_by_k(tiles, pos_key, neg_key, zscore=False):
    """pos_key/neg_key: functions tile -> bool mask (512,512)."""
    out = {}
    per_tile = {}
    for k in K_OFFSETS:
        pp, nn = [], []
        for t in tiles:
            v = t["vals"][KI[k]]
            good = t["ok_pre"] & (t["raw0"] > 5) & np.isfinite(v)
            pos = v[pos_key(t) & good]
            neg = v[neg_key(t) & good]
            if len(pos) > 100 and len(neg) > 100:
                a = tie_auc(pos, neg)
                per_tile.setdefault(t["id"], {})[k] = round(a, 4)
                if zscore:
                    med = np.median(neg)
                    iqr = np.subtract(*np.percentile(neg, [75, 25])) + 1e-9
                    pos = (pos - med) / iqr
                    neg = (neg - med) / iqr
                pp.append(pos)
                nn.append(neg)
        if pp:
            out[k] = round(tie_auc(np.concatenate(pp), np.concatenate(nn)), 4)
    return out, per_tile


def main():
    tiles = load_tiles()
    print(f"loaded {len(tiles)} tiles: {[t['id'] for t in tiles]}")
    lo, hi = WIN
    results = {}

    # --- derived masks per tile ---
    for t in tiles:
        ink = t["sub_ink"]
        t["letters"] = ink > STRONG_INK
        dil = binary_dilation(t["letters"], iterations=60)
        t["bg_orig"] = (ink < WEAK_BG) & dil
        nodata = ink == 0
        near_nodata = binary_dilation(nodata, iterations=NEAR_PX)
        t["bg_corr"] = (ink >= BG_LO) & (ink <= BG_HI) & dil & ~near_nodata
        raw_uint8_0 = (t["vals"][KI[0]] - lo) / (hi - lo) * 255.0
        raw_u8 = np.nan_to_num(raw_uint8_0, nan=255.0)
        sat = raw_u8 >= SAT_RAW
        t["sat"] = sat
        t["near_sat"] = binary_dilation(sat, iterations=NEAR_PX)
        t["near_sat2"] = binary_dilation(raw_u8 >= SAT_RAW2, iterations=NEAR_PX)
        t["letters_nosat"] = t["letters"] & ~t["near_sat"]
        t["bg_corr_nosat"] = t["bg_corr"] & ~t["near_sat"]
        good = t["ok_pre"] & (t["raw0"] > 5)
        bo = t["bg_orig"] & good
        frac0 = float((ink[bo] == 0).mean()) if bo.sum() else float("nan")
        pct = np.percentile(t["raw0"][good], [50, 90, 95, 98, 99]) if good.sum() else []
        print(f"tile {t['id']}: letters={t['letters'].sum()} "
              f"(nosat={t['letters_nosat'].sum()}) bg_orig={t['bg_orig'].sum()} "
              f"[frac ink==0 in used bg_orig: {frac0:.3f}] "
              f"bg_corr={t['bg_corr'].sum()} (nosat={t['bg_corr_nosat'].sum()}) "
              f"sat_frac={sat.mean():.3f} raw0 pct[50,90,95,98,99]={np.round(pct,1)}")
        t["frac0_bg_orig"] = frac0
    results["bg_orig_frac_ink0"] = {str(t["id"]): round(t["frac0_bg_orig"], 3)
                                    for t in tiles}

    # T0 reproduction
    a, pt = pooled_auc_by_k(tiles, lambda t: t["letters"], lambda t: t["bg_orig"])
    results["T0_repro_pooled"] = a
    results["T0_repro_per_tile"] = {str(k): v for k, v in pt.items()}
    print("\nT0 repro pooled:", a)

    # T1 corrected background
    a, pt = pooled_auc_by_k(tiles, lambda t: t["letters"], lambda t: t["bg_corr"])
    results["T1_bgcorr_pooled"] = a
    results["T1_bgcorr_per_tile"] = {str(k): v for k, v in pt.items()}
    print("T1 bg-corrected pooled:", a)

    # T1b saturation-excluded (both classes)
    a, pt = pooled_auc_by_k(tiles, lambda t: t["letters_nosat"],
                            lambda t: t["bg_corr_nosat"])
    results["T1b_nosat_pooled"] = a
    results["T1b_nosat_per_tile"] = {str(k): v for k, v in pt.items()}
    print("T1b sat-excluded pooled:", a)

    # T1b' softer saturation threshold (raw>=200)
    a, pt = pooled_auc_by_k(tiles, lambda t: t["letters"] & ~t["near_sat2"],
                            lambda t: t["bg_corr"] & ~t["near_sat2"])
    results["T1b2_nosat200_pooled"] = a
    results["T1b2_nosat200_per_tile"] = {str(k): v for k, v in pt.items()}
    print("T1b' sat200-excluded pooled:", a)

    # z-scored pooled (removes between-tile brightness offsets), corrected masks
    a, _ = pooled_auc_by_k(tiles, lambda t: t["letters_nosat"],
                           lambda t: t["bg_corr_nosat"], zscore=True)
    results["T1c_nosat_zscored_pooled"] = a
    print("T1c z-scored sat-excluded pooled:", a)

    # T2 placebo
    def shifted_mask(m, dy, dx):
        out = np.zeros_like(m)
        h, w = m.shape
        ys0, ys1 = max(dy, 0), min(h + dy, h)
        xs0, xs1 = max(dx, 0), min(w + dx, w)
        out[ys0:ys1, xs0:xs1] = m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        return out

    def placebo_auc(transform):
        pp, nn = [], []
        per = {}
        for t in tiles:
            fake = transform(t["letters"])
            fake = fake & (t["sub_ink"] >= BG_LO) & ~t["letters"] & ~t["near_sat"]
            neg_m = t["bg_corr_nosat"] & ~fake
            for k in (0, -1):
                v = t["vals"][KI[k]]
                good = t["ok_pre"] & (t["raw0"] > 5) & np.isfinite(v)
                pos = v[fake & good]
                neg = v[neg_m & good]
                if len(pos) > 500 and len(neg) > 500:
                    per.setdefault(k, {"pp": [], "nn": []})
                    per[k]["pp"].append(pos)
                    per[k]["nn"].append(neg)
        return {k: round(tie_auc(np.concatenate(d["pp"]), np.concatenate(d["nn"])), 4)
                for k, d in per.items()}

    results["T2_rot180"] = placebo_auc(lambda m: m[::-1, ::-1])
    results["T2_shift_u64"] = placebo_auc(lambda m: shifted_mask(m, 0, 64))
    results["T2_shift_v64"] = placebo_auc(lambda m: shifted_mask(m, 64, 0))
    rng = np.random.default_rng(7)
    null_k0 = []
    for i in range(12):
        dy, dx = [int(s * rng.choice([-1, 1])) for s in rng.integers(48, 192, 2)]
        r = placebo_auc(lambda m: shifted_mask(m, dy, dx))
        if 0 in r:
            null_k0.append(r[0])
    results["T2_random_shifts_k0"] = null_k0
    results["T2_null_mean_sd"] = [round(float(np.mean(null_k0)), 4),
                                  round(float(np.std(null_k0)), 4)] if null_k0 else None
    print("T2 rot180:", results["T2_rot180"], " shift_u64:", results["T2_shift_u64"],
          " shift_v64:", results["T2_shift_v64"])
    print("T2 random-shift null k0:", null_k0)

    # T3 independent mask from ink11
    # threshold: same 170; also report mask agreement
    for t in tiles:
        t["letters11"] = (t["ink11r"] > STRONG_INK) & ~t["near_sat"]
        t["both"] = t["letters11"] & t["letters_nosat"]
    dice = []
    for t in tiles:
        a_, b_ = t["letters"], t["ink11r"] > STRONG_INK
        d = 2 * (a_ & b_).sum() / max(a_.sum() + b_.sum(), 1)
        dice.append(round(float(d), 3))
    results["T3_dice_24_vs_11"] = dice
    bg11 = lambda t: t["bg_corr_nosat"] & (t["ink11r"] >= BG_LO) & (t["ink11r"] <= BG_HI)
    a, pt = pooled_auc_by_k(tiles, lambda t: t["letters11"], bg11)
    results["T3_ink11_pooled"] = a
    results["T3_ink11_per_tile"] = {str(k): v for k, v in pt.items()}
    a2, pt2 = pooled_auc_by_k(tiles, lambda t: t["both"], bg11)
    results["T3_intersect_pooled"] = a2
    results["T3_intersect_per_tile"] = {str(k): v for k, v in pt2.items()}
    print("T3 dice(letters24,letters11):", dice)
    print("T3 ink11-mask pooled:", a)
    print("T3 intersect-mask pooled:", a2)

    # T4 eroded letters
    for it in (2, 3):
        a, pt = pooled_auc_by_k(
            tiles,
            lambda t, it=it: binary_erosion(t["letters"], iterations=it) & ~t["near_sat"],
            lambda t: t["bg_corr_nosat"])
        results[f"T4_eroded{it}_pooled"] = a
        results[f"T4_eroded{it}_per_tile"] = {str(k): v for k, v in pt.items()}
        print(f"T4 eroded x{it} pooled:", a)

    # T5 letter proximity to saturated zones vs per-tile AUC
    t5 = []
    for t in tiles:
        if t["sat"].sum() == 0:
            dist = np.full_like(t["raw0"], 1e4)
        else:
            dist = distance_transform_edt(~t["sat"])
        good = t["ok_pre"] & (t["raw0"] > 5)
        ld = dist[t["letters"] & good]
        bd = dist[t["bg_corr"] & good]
        auc_corr = results["T1_bgcorr_per_tile"].get(str(t["id"]), {}).get(-1, np.nan)
        t5.append({"tile": t["id"], "mean_dist_letters": round(float(ld.mean()), 1),
                   "mean_dist_bg": round(float(bd.mean()), 1),
                   "frac_letters_within16": round(float((ld < 16).mean()), 3),
                   "auc_km1_corr": auc_corr})
    results["T5_proximity"] = [
        {**d, "tile": list(d["tile"])} for d in t5]
    fr = [d["frac_letters_within16"] for d in t5]
    au = [d["auc_km1_corr"] for d in t5]
    if len(set(fr)) > 1:
        rho, p = spearmanr(fr, au)
        results["T5_spearman_frac_vs_auc"] = [round(float(rho), 3), round(float(p), 3)]
    print("T5:", json.dumps(results["T5_proximity"], indent=1))
    print("T5 spearman(frac_letters_near_sat, auc):", results.get("T5_spearman_frac_vs_auc"))

    # supplementary: spearman rho of ink value vs brightness at k=0 in clean domain
    rhos = {}
    for t in tiles:
        v = t["vals"][KI[0]]
        dom = t["ok_pre"] & (t["raw0"] > 5) & np.isfinite(v) & \
            (t["sub_ink"] >= BG_LO) & ~t["near_sat"]
        if dom.sum() > 1000:
            rho, _ = spearmanr(t["sub_ink"][dom].astype(float), v[dom])
            rhos[str(t["id"])] = round(float(rho), 3)
    results["supp_spearman_ink_vs_brightness_k0"] = rhos
    print("supp spearman(ink value, brightness) per tile:", rhos)

    with open(os.path.join(OUT, "qc_s1a_results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print("\nwrote qc_s1a_results.json")


if __name__ == "__main__":
    main()
