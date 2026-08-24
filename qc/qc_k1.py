"""QC audit of K1 (intensity-AUC kill test on w035).

Checks:
  A. Tie handling of the argsort-of-argsort AUC vs average-rank (Mann-Whitney
     with tie correction); reproduce headline numbers both ways.
  B. Per-slice AUC sweep (all 28 z-slices, raw + detrended) vs the 2D labels
     (labels live only on slice 14 -> per-slice-label test == this sweep).
  C. Masked detrending (background from valid px only) vs the original
     unmasked gaussian detrend (which mixes in zeros outside the patches).
  D. Negative-set adequacy: AUC with negatives restricted to distance bands
     from the nearest labeled ink stroke.
  E. Texture statistics AUC (local std 3x3, gradient energy, z-std, per-slice
     local std averaged over z) -- computed on an eroded valid mask so patch
     borders don't leak.

Writes trackD/qc/qc_k1_results.json and prints a report.
"""
import json
import os
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata

CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc"
WIN = (-0.03, 0.145)


def auc_orig(pos, neg, max_n=2_000_000):
    """Verbatim copy of k1_intensity_auc.auc_rank (ordinal ranks, stable sort)."""
    rng = np.random.default_rng(0)
    if len(pos) > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if len(neg) > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv, kind="stable")).astype(np.float64) + 1
    rpos = ranks[: len(pos)].sum()
    n1, n2 = len(pos), len(neg)
    return float((rpos - n1 * (n1 + 1) / 2) / (n1 * n2))


def auc_avgrank(pos, neg):
    """Mann-Whitney AUC with average-rank tie correction (exact, no subsample)."""
    allv = np.concatenate([pos, neg])
    ranks = rankdata(allv, method="average")
    n1, n2 = len(pos), len(neg)
    rpos = ranks[:n1].sum()
    return float((rpos - n1 * (n1 + 1) / 2) / (n1 * n2))


def tie_prob(pos, neg, decimals=9):
    """P(pos value == neg value) for randomly drawn pair (via histogram overlap)."""
    pv, pc = np.unique(np.round(pos, decimals), return_counts=True)
    nv, nc = np.unique(np.round(neg, decimals), return_counts=True)
    common, ip, iq = np.intersect1d(pv, nv, return_indices=True)
    return float((pc[ip].astype(np.float64) * nc[iq]).sum() / (len(pos) * len(neg)))


def main():
    t0 = time.time()
    surf = np.load(os.path.join(CACHE, "w035_surf.npy"), mmap_mode="r")
    ink = np.load(os.path.join(CACHE, "w035_ink.npy"), mmap_mode="r")
    sup = np.load(os.path.join(CACHE, "w035_sup.npy"), mmap_mode="r")

    nz = surf.shape[0]
    z0, z1 = (nz - 17) // 2, (nz - 17) // 2 + 17
    lo, hi = WIN
    scale = (hi - lo) / 255.0

    # accumulate z-mean and z-var in float64 without holding full float volume
    acc = np.zeros(surf.shape[1:], np.float64)
    acc2 = np.zeros(surf.shape[1:], np.float64)
    nonzero_cnt = np.zeros(surf.shape[1:], np.int16)
    for z in range(z0, z1):
        s = surf[z].astype(np.float32) * scale + lo
        acc += s
        acc2 += s.astype(np.float64) ** 2
        nonzero_cnt += surf[z] > 0
    zmean = (acc / 17).astype(np.float32)
    zvar = (acc2 / 17 - (acc / 17) ** 2).clip(0)
    zstd = np.sqrt(zvar).astype(np.float32)
    del acc, acc2

    ink2d = ink[14] > 0
    sup2d = sup[14] > 0
    nonzero2d = (nonzero_cnt / 17.0) > 0.9
    valid = sup2d & nonzero2d
    pos_m = ink2d & valid
    neg_m = valid & ~ink2d
    res = {"valid_px": int(valid.sum()), "ink_px": int(pos_m.sum())}
    print(f"valid={res['valid_px']} ink={res['ink_px']}  (orig: 1071082 / 334002)")

    # ---------- A: reproduce + tie handling ----------
    print("\n[A] headline AUCs: original impl vs average-rank")
    res["A_headline"] = {}
    imgs = {"raw_zmean": zmean}
    for s in (25, 50, 100):
        imgs[f"detrend_s{s}"] = zmean - ndi.gaussian_filter(zmean, s)
    for name, img in imgs.items():
        pos, neg = img[pos_m], img[neg_m]
        a0 = auc_orig(pos, neg)
        a1 = auc_avgrank(pos, neg)
        tp = tie_prob(pos, neg)
        res["A_headline"][name] = {"auc_orig": round(a0, 5), "auc_avgrank": round(a1, 5),
                                   "tie_prob": round(tp, 6)}
        print(f"  {name:14s} orig={a0:.5f} avgrank={a1:.5f} tieP={tp:.2e}")

    # ---------- C: masked detrend ----------
    print("\n[C] masked detrend (background estimated from valid px only)")
    res["C_masked_detrend"] = {}
    vf = valid.astype(np.float32)
    for s in (25, 50):
        num = ndi.gaussian_filter(zmean * vf, s)
        den = ndi.gaussian_filter(vf, s)
        bg = np.where(den > 1e-4, num / np.maximum(den, 1e-4), 0.0)
        det = zmean - bg
        a = auc_avgrank(det[pos_m], det[neg_m])
        res["C_masked_detrend"][f"s{s}"] = round(a, 5)
        print(f"  masked detrend s={s}: AUC={a:.5f}")

    # ---------- B: per-slice sweep ----------
    print("\n[B] per-slice AUC (raw slice intensity vs 2D labels, within valid)")
    res["B_per_slice"] = {}
    for z in range(nz):
        s = surf[z].astype(np.float32)
        vz = valid & (surf[z] > 0)
        a = auc_avgrank(s[ink2d & vz], s[vz & ~ink2d])
        res["B_per_slice"][z] = round(a, 4)
    print("  z: " + " ".join(f"{z}:{a:.3f}" for z, a in res["B_per_slice"].items()))
    devs = {z: abs(a - 0.5) for z, a in res["B_per_slice"].items()}
    zmax = max(devs, key=devs.get)
    print(f"  max |AUC-0.5| at z={zmax}: AUC={res['B_per_slice'][zmax]}")

    # per-slice detrended (s50) for the 5 central slices around the label plane
    print("  per-slice detrended s50 (z=11..17):")
    res["B_per_slice_detrended"] = {}
    for z in range(11, 18):
        s = surf[z].astype(np.float32)
        d = s - ndi.gaussian_filter(s, 50)
        vz = valid & (surf[z] > 0)
        a = auc_avgrank(d[ink2d & vz], d[vz & ~ink2d])
        res["B_per_slice_detrended"][z] = round(a, 4)
        print(f"    z={z}: {a:.4f}")

    # ---------- D: negative distance bands ----------
    print("\n[D] AUC with negatives restricted by distance from labeled ink")
    dist = ndi.distance_transform_edt(~ink2d)
    res["D_neg_bands"] = {}
    pos = imgs["detrend_s50"][pos_m]
    pos_raw = zmean[pos_m]
    for name, band in [("0-5px", (0, 5)), ("5-15px", (5, 15)),
                       ("15-40px", (15, 40)), (">40px", (40, 1e9))]:
        bm = neg_m & (dist > band[0]) & (dist <= band[1])
        n = int(bm.sum())
        if n < 1000:
            continue
        a_raw = auc_avgrank(pos_raw, zmean[bm])
        a_det = auc_avgrank(pos, imgs["detrend_s50"][bm])
        res["D_neg_bands"][name] = {"n_neg": n, "auc_raw": round(a_raw, 4),
                                    "auc_det50": round(a_det, 4)}
        print(f"  neg {name:8s} n={n:8d}  raw={a_raw:.4f} det50={a_det:.4f}")

    # ---------- E: texture statistics ----------
    print("\n[E] texture-statistic AUCs (valid eroded by 3 px)")
    er = ndi.binary_erosion(valid, iterations=3)
    pos_e, neg_e = ink2d & er, er & ~ink2d
    print(f"  eroded valid={int(er.sum())} pos={int(pos_e.sum())}")

    def local_std(img, size=3):
        m = ndi.uniform_filter(img.astype(np.float64), size)
        m2 = ndi.uniform_filter(img.astype(np.float64) ** 2, size)
        return np.sqrt((m2 - m ** 2).clip(0)).astype(np.float32)

    tex = {}
    tex["std3_zmean"] = local_std(zmean)
    gx = ndi.sobel(zmean, 0)
    gy = ndi.sobel(zmean, 1)
    tex["grad_energy_zmean"] = np.sqrt(gx ** 2 + gy ** 2)
    del gx, gy
    tex["std3_det25"] = local_std(imgs["detrend_s25"])
    tex["zstd_17"] = zstd
    # mean over central 17 slices of per-slice 3x3 std ("texture the models see")
    accT = np.zeros(surf.shape[1:], np.float32)
    for z in range(z0, z1):
        accT += local_std(surf[z].astype(np.float32) * scale)
    tex["mean_std3_perslice"] = accT / 17

    res["E_texture"] = {}
    for name, img in tex.items():
        a = auc_avgrank(img[pos_e], img[neg_e])
        res["E_texture"][name] = round(a, 4)
        print(f"  {name:22s} AUC={a:.4f}")

    # smoothed-texture variants (aggregate texture over a stroke-width scale)
    print("  smoothed (sigma=8) texture maps:")
    res["E_texture_smoothed"] = {}
    for name in ("std3_zmean", "grad_energy_zmean", "zstd_17", "mean_std3_perslice"):
        sm = ndi.gaussian_filter(tex[name], 8)
        a = auc_avgrank(sm[pos_e], sm[neg_e])
        res["E_texture_smoothed"][name] = round(a, 4)
        print(f"  {name+'+s8':22s} AUC={a:.4f}")

    with open(os.path.join(OUT, "qc_k1_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\ndone in {time.time()-t0:.0f}s -> qc_k1_results.json")


if __name__ == "__main__":
    main()
