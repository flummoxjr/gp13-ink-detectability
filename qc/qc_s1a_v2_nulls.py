"""QC s1a-v2 adversarial analysis — runs offline on tile_v2_*.npz.

N1  null distribution: 40 random cyclic shifts of the letter mask per tile,
    full k-profiles, two variants:
      raw     — exactly the v2 null construction (roll only)
      matched — roll & dist_ok & on-prediction & ~letters (symmetric w/ real)
N2  profile-shape specificity under the null (peak at |k|<=1 & range>0.08)
N3  confounds: dist-to-nodata, normal tilt, grid validity, ink11 coverage,
    trend/residual AUC decomposition (bg-only smooth trend)
N4  inverted tile (11,12) forensics
N5  block bootstrap (64px blocks) CI on pooled and per-tile AUC
N6  z-scored pooling
Writes qc_s1a_v2_nulls.json
"""
import glob
import json
import os
import re

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

QCD = r"D:\vesuvius-data\trackD\w032\qc"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc"
K_OFFSETS = list(range(-4, 5))
KI = {k: i for i, k in enumerate(K_OFFSETS)}
N_NULL = 40
TREND_SIGMA = 48.0


def tie_auc(pos, neg, max_n=300_000, seed=1):
    rng = np.random.default_rng(seed)
    if len(pos) > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if len(neg) > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = rankdata(allv)
    n1, n2 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def load_tiles():
    tiles = []
    for f in sorted(glob.glob(os.path.join(QCD, "tile_v2_*.npz"))):
        m = re.search(r"tile_v2_(\d+)_(\d+)\.npz", f)
        d = np.load(f)
        t = {"id": (int(m.group(1)), int(m.group(2)))}
        for k in ("sub_ink", "s11", "dist_nd", "ok_pre", "raw0", "vals",
                  "X", "Y", "Z", "normals"):
            t[k] = d[k]
        t["letters"] = (t["sub_ink"] >= 200) & (t["s11"] >= 150) & (t["dist_nd"] >= 8)
        t["bg"] = (t["sub_ink"] >= 28) & (t["sub_ink"] <= 60) & (t["s11"] <= 60) & \
                  (t["dist_nd"] >= 8)
        t["good0"] = t["ok_pre"] & (t["raw0"] > 5)
        tiles.append(t)
    return tiles


def profile(t, lm, exclude_from_neg=None):
    """AUC by k for letter-mask lm on tile t (v2 logic)."""
    out = {}
    ex = lm if exclude_from_neg is None else (lm | exclude_from_neg)
    for k in K_OFFSETS:
        v = t["vals"][KI[k]]
        good = t["good0"] & np.isfinite(v)
        pos = v[lm & good]
        neg = v[t["bg"] & good & ~ex]
        if len(pos) > 100 and len(neg) > 100:
            out[k] = tie_auc(pos, neg)
    return out


def pooled_profile(tiles, masks):
    out = {}
    for k in K_OFFSETS:
        pp, nn = [], []
        for t, lm in zip(tiles, masks):
            v = t["vals"][KI[k]]
            good = t["good0"] & np.isfinite(v)
            pos = v[lm & good]
            neg = v[t["bg"] & good & ~lm]
            if len(pos) > 100 and len(neg) > 100:
                pp.append(pos); nn.append(neg)
        if pp:
            out[k] = tie_auc(np.concatenate(pp), np.concatenate(nn))
    return out


def shape_stats(prof):
    if len(prof) < 9:
        return None
    ks = sorted(prof)
    vals = np.array([prof[k] for k in ks])
    pk = ks[int(np.argmax(vals))]
    tr = ks[int(np.argmin(vals))]
    rng_ = float(vals.max() - vals.min())
    return {"peak_k": pk, "trough_k": tr, "range": round(rng_, 4),
            "max": round(float(vals.max()), 4), "min": round(float(vals.min()), 4)}


def main():
    tiles = load_tiles()
    print(f"loaded {len(tiles)} tiles: {[t['id'] for t in tiles]}", flush=True)
    results = {}

    # ---------- real profiles ----------
    real = {}
    for t in tiles:
        p = profile(t, t["letters"])
        real[str(t["id"])] = {k: round(v, 4) for k, v in p.items()}
        t["real_prof"] = p
    results["real_per_tile"] = real
    rp = pooled_profile(tiles, [t["letters"] for t in tiles])
    results["real_pooled"] = {k: round(v, 4) for k, v in rp.items()}
    print("real pooled:", results["real_pooled"], flush=True)

    # ---------- N1/N2 null distributions ----------
    rng = np.random.default_rng(2026)
    null_reps = {"raw": [], "matched": []}          # list of dict tile-> profile
    pooled_reps = {"raw": [], "matched": []}
    for rep in range(N_NULL):
        masks_raw, masks_mat = [], []
        for t in tiles:
            for _ in range(20):
                dy = int(rng.integers(32, 481)); dx = int(rng.integers(32, 481))
                fake = np.roll(t["letters"], (dy, dx), axis=(0, 1))
                fm = fake & (t["dist_nd"] >= 8) & (t["sub_ink"] > 0) & ~t["letters"]
                if fm.sum() >= 500 and (fake & t["good0"]).sum() >= 500:
                    break
            masks_raw.append(fake)
            masks_mat.append(fm)
        rep_raw, rep_mat = {}, {}
        for t, fr, fm in zip(tiles, masks_raw, masks_mat):
            rep_raw[str(t["id"])] = profile(t, fr)
            rep_mat[str(t["id"])] = profile(t, fm)
        null_reps["raw"].append(rep_raw)
        null_reps["matched"].append(rep_mat)
        pooled_reps["raw"].append(pooled_profile(tiles, masks_raw))
        pooled_reps["matched"].append(pooled_profile(tiles, masks_mat))
        if (rep + 1) % 10 == 0:
            print(f"null rep {rep+1}/{N_NULL}", flush=True)

    # summarize
    summ = {}
    for variant in ("raw", "matched"):
        vs = {"per_tile": {}, "pooled": {}}
        # per tile at k=-1, 0 and at real peak
        for t in tiles:
            tid = str(t["id"])
            realp = t["real_prof"]
            rk = sorted(realp)
            rvals = np.array([realp[k] for k in rk])
            peak_k = rk[int(np.argmax(rvals))]
            ext_k = rk[int(np.argmax(np.abs(rvals - 0.5)))]  # strongest deviation
            entry = {}
            for kq, lab in [(-1, "km1"), (0, "k0"), (peak_k, "peak"), (ext_k, "ext")]:
                nv = [r[tid].get(kq) for r in null_reps[variant] if kq in r.get(tid, {})]
                nv = np.array([x for x in nv if x is not None and np.isfinite(x)])
                if len(nv) < 5:
                    continue
                rv = realp.get(kq, float("nan"))
                entry[lab] = {
                    "k": kq, "real": round(rv, 4),
                    "null_mean": round(float(nv.mean()), 4),
                    "null_sd": round(float(nv.std()), 4),
                    "null_min": round(float(nv.min()), 4),
                    "null_max": round(float(nv.max()), 4),
                    "z": round(float((rv - nv.mean()) / (nv.std() + 1e-9)), 2),
                    "pct_null_gt_real": round(float((nv >= rv).mean()), 3),
                    "pct_null_more_extreme": round(
                        float((np.abs(nv - 0.5) >= abs(rv - 0.5)).mean()), 3),
                    "n_null": int(len(nv)),
                }
            vs["per_tile"][tid] = entry
        for kq, lab in [(-1, "km1"), (0, "k0")]:
            nv = np.array([r.get(kq) for r in pooled_reps[variant]
                           if kq in r and np.isfinite(r[kq])])
            rv = rp.get(kq, float("nan"))
            vs["pooled"][lab] = {
                "k": kq, "real": round(rv, 4),
                "null_mean": round(float(nv.mean()), 4),
                "null_sd": round(float(nv.std()), 4),
                "null_min": round(float(nv.min()), 4),
                "null_max": round(float(nv.max()), 4),
                "z": round(float((rv - nv.mean()) / (nv.std() + 1e-9)), 2),
                "pct_null_gt_real": round(float((nv >= rv).mean()), 3),
                "n_null": int(len(nv)),
            }
        # N2 profile shape under null
        shp = {"per_tile_profiles": 0, "peak_le1_range08": 0,
               "peak_or_trough_le1_range08": 0, "range_ge_08": 0}
        per_tile_shape = {str(t["id"]): {"n": 0, "peak_le1_range08": 0,
                                         "extreme_le1_range08": 0} for t in tiles}
        for r in null_reps[variant]:
            for tid, prof in r.items():
                ss = shape_stats(prof)
                if ss is None:
                    continue
                shp["per_tile_profiles"] += 1
                per_tile_shape[tid]["n"] += 1
                if ss["range"] > 0.08:
                    shp["range_ge_08"] += 1
                if abs(ss["peak_k"]) <= 1 and ss["range"] > 0.08:
                    shp["peak_le1_range08"] += 1
                    per_tile_shape[tid]["peak_le1_range08"] += 1
                hit = (abs(ss["peak_k"]) <= 1 or abs(ss["trough_k"]) <= 1) and ss["range"] > 0.08
                if hit:
                    shp["peak_or_trough_le1_range08"] += 1
                    per_tile_shape[tid]["extreme_le1_range08"] += 1
        pooled_shape = [shape_stats(r) for r in pooled_reps[variant]]
        pooled_shape = [s for s in pooled_shape if s]
        shp["pooled_n"] = len(pooled_shape)
        shp["pooled_peak_le1_range08"] = sum(
            1 for s in pooled_shape if abs(s["peak_k"]) <= 1 and s["range"] > 0.08)
        shp["pooled_peak_or_trough_le1_range08"] = sum(
            1 for s in pooled_shape
            if (abs(s["peak_k"]) <= 1 or abs(s["trough_k"]) <= 1) and s["range"] > 0.08)
        vs["shape"] = shp
        vs["shape_per_tile"] = per_tile_shape
        summ[variant] = vs
        print(f"[{variant}] pooled km1: {vs['pooled']['km1']}", flush=True)
    results["null"] = summ
    results["real_shape_per_tile"] = {str(t["id"]): shape_stats(t["real_prof"])
                                      for t in tiles}
    results["real_shape_pooled"] = shape_stats(rp)

    # ---------- N3 confounds ----------
    conf = {}
    for t in tiles:
        tid = str(t["id"])
        g = t["good0"]
        L = t["letters"] & g
        B = t["bg"] & g
        nx, ny, nz_ = t["normals"]
        # tilt relative to tile-mean normal
        mn = np.array([nx[B].mean(), ny[B].mean(), nz_[B].mean()])
        mn /= np.linalg.norm(mn) + 1e-9
        cosang = np.clip(nx * mn[0] + ny * mn[1] + nz_ * mn[2], -1, 1)
        ang = np.degrees(np.arccos(cosang))
        entry = {
            "dist_nd_mean_letters": round(float(t["dist_nd"][L].mean()), 1),
            "dist_nd_mean_bg": round(float(t["dist_nd"][B].mean()), 1),
            "dist_nd_auc": round(tie_auc(t["dist_nd"][L], t["dist_nd"][B]), 3),
            "tilt_deg_mean_letters": round(float(ang[L].mean()), 2),
            "tilt_deg_mean_bg": round(float(ang[B].mean()), 2),
            "tilt_auc": round(tie_auc(ang[L], ang[B]), 3),
            "nz_mean_letters": round(float(nz_[L].mean()), 3),
            "nz_mean_bg": round(float(nz_[B].mean()), 3),
            "frac_bg_no_ink11_cov": round(float((t["s11"][B] == 0).mean()), 4),
            "letters_centroid_rc": [round(float(np.where(L)[0].mean()), 1),
                                    round(float(np.where(L)[1].mean()), 1)],
            "bg_centroid_rc": [round(float(np.where(B)[0].mean()), 1),
                               round(float(np.where(B)[1].mean()), 1)],
            "Z_median_letters": round(float(np.median(t["Z"][L])), 1),
            "Z_median_bg": round(float(np.median(t["Z"][B])), 1),
        }
        # trend / residual decomposition at k=-1 and k=0
        for kq in (-1, 0):
            v = t["vals"][KI[kq]]
            good = t["good0"] & np.isfinite(v)
            Bk = t["bg"] & good
            Lk = t["letters"] & good
            w = Bk.astype(np.float32)
            vv = np.where(Bk, v, 0).astype(np.float32)
            den = gaussian_filter(w, TREND_SIGMA)
            trend = gaussian_filter(vv, TREND_SIGMA) / (den + 1e-9)
            okd = den > 1e-3
            Lk2, Bk2 = Lk & okd, Bk & okd
            auc_trend = tie_auc(trend[Lk2], trend[Bk2])
            resid = v - trend
            auc_resid = tie_auc(resid[Lk2], resid[Bk2])
            auc_real = tie_auc(v[Lk2], v[Bk2])
            entry[f"k{kq}_auc_real"] = round(auc_real, 4)
            entry[f"k{kq}_auc_trend_only"] = round(auc_trend, 4)
            entry[f"k{kq}_auc_residual"] = round(auc_resid, 4)
        conf[tid] = entry
        print(f"confounds {tid}: {json.dumps(entry)}", flush=True)
    results["confounds"] = conf

    # residual profile by k for each tile (is the residual surface-peaked?)
    resid_prof = {}
    for t in tiles:
        pr = {}
        for kq in K_OFFSETS:
            v = t["vals"][KI[kq]]
            good = t["good0"] & np.isfinite(v)
            Bk = t["bg"] & good
            Lk = t["letters"] & good
            w = Bk.astype(np.float32)
            vv = np.where(Bk, v, 0).astype(np.float32)
            den = gaussian_filter(w, TREND_SIGMA)
            trend = gaussian_filter(vv, TREND_SIGMA) / (den + 1e-9)
            okd = den > 1e-3
            r_ = v - trend
            pr[kq] = round(tie_auc(r_[Lk & okd], r_[Bk & okd]), 4)
        resid_prof[str(t["id"])] = pr
        print(f"residual profile {t['id']}: {pr}", flush=True)
    results["residual_profile_per_tile"] = resid_prof

    # ---------- N4 inverted tile forensics ----------
    inv = {}
    for t in tiles:
        tid = str(t["id"])
        g = t["good0"]
        L = t["letters"] & g
        B = t["bg"] & g
        mprof_l = {k: round(float(np.nanmean(t["vals"][KI[k]][L])), 4) for k in K_OFFSETS}
        mprof_b = {k: round(float(np.nanmean(t["vals"][KI[k]][B])), 4) for k in K_OFFSETS}
        pk_l = max(mprof_l, key=mprof_l.get)
        pk_b = max(mprof_b, key=mprof_b.get)
        inv[tid] = {
            "mean_val_letters_by_k": mprof_l, "mean_val_bg_by_k": mprof_b,
            "letters_peak_k": pk_l, "bg_peak_k": pk_b,
            "raw0_median_letters": round(float(np.median(t["raw0"][L])), 1),
            "raw0_median_bg": round(float(np.median(t["raw0"][B])), 1),
        }
    results["class_brightness_profiles"] = inv

    # ---------- N5 block bootstrap ----------
    def block_boot(tiles_sub, kq, n_boot=300, bs=64):
        rngb = np.random.default_rng(7)
        # precompute per-tile block-indexed pos/neg values
        blocks = []
        for t in tiles_sub:
            v = t["vals"][KI[kq]]
            good = t["good0"] & np.isfinite(v)
            L = t["letters"] & good
            B = t["bg"] & good
            tb = []
            for by in range(0, 512, bs):
                for bx in range(0, 512, bs):
                    sl = (slice(by, by + bs), slice(bx, bx + bs))
                    tb.append((v[sl][L[sl]], v[sl][B[sl]]))
            blocks.append(tb)
        aucs = []
        for _ in range(n_boot):
            pp, nn = [], []
            for tb in blocks:
                idx = rngb.integers(0, len(tb), len(tb))
                pp.extend(tb[i][0] for i in idx)
                nn.extend(tb[i][1] for i in idx)
            pos = np.concatenate(pp); neg = np.concatenate(nn)
            if len(pos) > 50 and len(neg) > 50:
                aucs.append(tie_auc(pos, neg, max_n=150_000))
        a = np.array(aucs)
        return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std()), 4),
                "ci2.5": round(float(np.percentile(a, 2.5)), 4),
                "ci97.5": round(float(np.percentile(a, 97.5)), 4), "n_boot": len(a)}

    bb = {"pooled_km1": block_boot(tiles, -1)}
    for t in tiles:
        bb[f"tile_{t['id']}_km1"] = block_boot([t], -1, n_boot=200)
    # naive iid SD for comparison (pooled km1)
    npos = sum(int((t["letters"] & t["good0"] &
                    np.isfinite(t["vals"][KI[-1]])).sum()) for t in tiles)
    nneg = sum(int((t["bg"] & t["good0"] &
                    np.isfinite(t["vals"][KI[-1]])).sum()) for t in tiles)
    a0 = rp.get(-1, 0.6)
    naive_sd = float(np.sqrt(a0 * (1 - a0) * (1 / npos + 1 / nneg)))
    bb["pooled_naive_iid_sd"] = round(naive_sd, 5)
    bb["ess_deflation_factor"] = round((bb["pooled_km1"]["sd"] / naive_sd) ** 2, 1)
    bb["n_pos_pooled_km1"] = npos
    bb["n_neg_pooled_km1"] = nneg
    results["block_bootstrap"] = bb
    print("block bootstrap:", json.dumps(bb, indent=1), flush=True)

    # ---------- N6 z-scored pooling ----------
    zp = {}
    for kq in K_OFFSETS:
        pp, nn = [], []
        for t in tiles:
            v = t["vals"][KI[kq]]
            good = t["good0"] & np.isfinite(v)
            pos = v[t["letters"] & good]
            neg = v[t["bg"] & good]
            if len(pos) > 100 and len(neg) > 100:
                med = np.median(neg)
                iqr = np.subtract(*np.percentile(neg, [75, 25])) + 1e-9
                pp.append((pos - med) / iqr)
                nn.append((neg - med) / iqr)
        if pp:
            zp[kq] = round(tie_auc(np.concatenate(pp), np.concatenate(nn)), 4)
    results["zscored_pooled"] = zp
    print("z-scored pooled:", zp, flush=True)

    with open(os.path.join(OUT, "qc_s1a_v2_nulls.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print("wrote qc_s1a_v2_nulls.json", flush=True)


if __name__ == "__main__":
    main()
