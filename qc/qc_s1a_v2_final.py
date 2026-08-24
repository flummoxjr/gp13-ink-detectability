"""QC s1a-v2 final battery:
A) registration-corrected masks (shift ink11a by (-36,+39)) -> real AUC profiles
B) pooled excluding (9,9): real profile + 40-rep null at k=-1,0
C) on-sheet restriction (depth std>0.01): per-tile + pooled profiles + 40-rep null
   (both incl and excl (9,9))
Writes qc_s1a_v2_final.json
"""
import json
import os

import numpy as np
from scipy.ndimage import zoom, distance_transform_edt
from scipy.stats import rankdata

CACHE = r"D:\vesuvius-data\trackD\w032"
QCD = r"D:\vesuvius-data\trackD\w032\qc"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc"
K_OFFSETS = list(range(-4, 5))
KI = {k: i for i, k in enumerate(K_OFFSETS)}
TILE = 512
TILES = [(12, 11), (11, 12), (8, 10), (9, 9)]
SHIFT = (-36, 39)
N_NULL = 40


def tie_auc(pos, neg, max_n=300_000):
    rng = np.random.default_rng(1)
    if len(pos) > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if len(neg) > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    if len(pos) < 100 or len(neg) < 100:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = rankdata(allv)
    n1, n2 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def main():
    ink24 = np.load(os.path.join(CACHE, "ink24_ds4.npy"))
    ink11 = np.load(os.path.join(CACHE, "ink11_ds4.npy"))
    zf = 2.258 / 2.399
    ink11r = zoom(ink11, zf, order=1)
    ink11a = np.zeros_like(ink24)
    h = min(ink11r.shape[0], ink24.shape[0]); w = min(ink11r.shape[1], ink24.shape[1])
    ink11a[:h, :w] = ink11r[:h, :w]
    ink11c = np.roll(np.roll(ink11a, SHIFT[0], axis=0), SHIFT[1], axis=1)
    dist_ok_full = distance_transform_edt(ink24 > 0) >= 8

    tiles = []
    for (ty, tx) in TILES:
        d = np.load(os.path.join(QCD, f"tile_v2_{ty}_{tx}.npz"))
        y0, x0 = ty * TILE, tx * TILE
        t = {"id": (ty, tx)}
        for k in ("sub_ink", "s11", "dist_nd", "ok_pre", "raw0", "vals"):
            t[k] = d[k]
        t["s11c"] = ink11c[y0:y0 + TILE, x0:x0 + TILE]
        dk = t["dist_nd"] >= 8
        t["letters"] = (t["sub_ink"] >= 200) & (t["s11"] >= 150) & dk
        t["bg"] = (t["sub_ink"] >= 28) & (t["sub_ink"] <= 60) & (t["s11"] <= 60) & dk
        t["letters_c"] = (t["sub_ink"] >= 200) & (t["s11c"] >= 150) & dk
        t["bg_c"] = (t["sub_ink"] >= 28) & (t["sub_ink"] <= 60) & (t["s11c"] <= 60) & dk
        t["good0"] = t["ok_pre"] & (t["raw0"] > 5)
        t["struct"] = np.nanstd(t["vals"], axis=0) > 0.01
        tiles.append(t)

    results = {}

    # ---------- A) registration-corrected real profiles ----------
    regc = {}
    for t in tiles:
        prof = {}
        for k in K_OFFSETS:
            v = t["vals"][KI[k]]
            good = t["good0"] & np.isfinite(v)
            prof[k] = round(tie_auc(v[t["letters_c"] & good],
                                    v[t["bg_c"] & good & ~t["letters_c"]]), 4)
        regc[str(t["id"])] = prof
        print(f"reg-corrected {t['id']}: {prof}", flush=True)
    # pooled
    pl = {}
    for k in K_OFFSETS:
        pp, nn = [], []
        for t in tiles:
            v = t["vals"][KI[k]]
            good = t["good0"] & np.isfinite(v)
            pos = v[t["letters_c"] & good]; neg = v[t["bg_c"] & good & ~t["letters_c"]]
            if len(pos) > 100 and len(neg) > 100:
                pp.append(pos); nn.append(neg)
        pl[k] = round(tie_auc(np.concatenate(pp), np.concatenate(nn)), 4)
    regc["pooled"] = pl
    print("reg-corrected pooled:", pl, flush=True)
    results["reg_corrected"] = regc

    # ---------- generic pooled + null machinery ----------
    def pooled_profile(tt, use_struct, masks=None):
        out = {}
        for k in K_OFFSETS:
            pp, nn = [], []
            for i, t in enumerate(tt):
                v = t["vals"][KI[k]]
                good = t["good0"] & np.isfinite(v)
                if use_struct:
                    good = good & t["struct"]
                lm = t["letters"] if masks is None else masks[i]
                pos = v[lm & good]
                neg = v[t["bg"] & good & ~lm]
                if len(pos) > 100 and len(neg) > 100:
                    pp.append(pos); nn.append(neg)
            if pp:
                out[k] = tie_auc(np.concatenate(pp), np.concatenate(nn))
        return out

    def null_battery(tt, use_struct, label, seed=2027):
        rng = np.random.default_rng(seed)
        real = pooled_profile(tt, use_struct)
        reps = []
        for rep in range(N_NULL):
            masks = []
            for t in tt:
                for _ in range(20):
                    dy = int(rng.integers(32, 481)); dx = int(rng.integers(32, 481))
                    fake = np.roll(t["letters"], (dy, dx), axis=(0, 1))
                    fm = fake & (t["dist_nd"] >= 8) & (t["sub_ink"] > 0) & ~t["letters"]
                    if fm.sum() >= 500:
                        break
                masks.append(fm)
            reps.append(pooled_profile(tt, use_struct, masks))
        out = {"real": {k: round(v, 4) for k, v in real.items()}}
        for kq in (-1, 0):
            nv = np.array([r[kq] for r in reps if kq in r])
            rv = real.get(kq, float("nan"))
            out[f"k{kq}"] = {
                "real": round(rv, 4), "null_mean": round(float(nv.mean()), 4),
                "null_sd": round(float(nv.std()), 4),
                "null_min": round(float(nv.min()), 4),
                "null_max": round(float(nv.max()), 4),
                "z": round(float((rv - nv.mean()) / (nv.std() + 1e-9)), 2),
                "n_null_ge_real": int((nv >= rv).sum()), "n_null": len(nv)}
        # shape specificity of nulls
        n_peak = 0; n_ok = 0
        for r in reps:
            if len(r) < 9:
                continue
            ks = sorted(r); vv = np.array([r[k] for k in ks])
            n_ok += 1
            if abs(ks[int(np.argmax(vv))]) <= 1 and (vv.max() - vv.min()) > 0.08:
                n_peak += 1
        out["null_peak_le1_range08"] = [n_peak, n_ok]
        print(f"[{label}] {json.dumps(out)}", flush=True)
        return out

    # ---------- B) pooled excluding (9,9) ----------
    t3 = [t for t in tiles if t["id"] != (9, 9)]
    results["pooled_excl_99"] = null_battery(t3, False, "excl99")

    # ---------- C) on-sheet restricted ----------
    results["pooled_struct_all4"] = null_battery(tiles, True, "struct_all4")
    results["pooled_struct_excl99"] = null_battery(t3, True, "struct_excl99")

    # per-tile struct profiles (for the record)
    ps = {}
    for t in tiles:
        prof = {}
        for k in K_OFFSETS:
            v = t["vals"][KI[k]]
            good = t["good0"] & np.isfinite(v) & t["struct"]
            prof[k] = round(tie_auc(v[t["letters"] & good], v[t["bg"] & good]), 4)
        ps[str(t["id"])] = prof
    results["per_tile_struct"] = ps

    with open(os.path.join(OUT, "qc_s1a_v2_final.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print("wrote qc_s1a_v2_final.json", flush=True)


if __name__ == "__main__":
    main()
