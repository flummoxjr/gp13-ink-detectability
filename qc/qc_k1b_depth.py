"""K1 follow-up: depth-contrast statistic + within-patch control.

The per-slice sweep showed AUC>0.5 at z=10-14 and AUC<0.5 at z=3-8 (sign
flip with depth). Test a simple depth-contrast feature, and verify the
signal exists WITHIN individual supervision patches (not just across
patches, which could be a patch-level confound).
"""
import json
import os

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata

CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc"


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    r = rankdata(allv, method="average")
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def main():
    surf = np.load(os.path.join(CACHE, "w035_surf.npy"), mmap_mode="r")
    ink = np.load(os.path.join(CACHE, "w035_ink.npy"), mmap_mode="r")
    sup = np.load(os.path.join(CACHE, "w035_sup.npy"), mmap_mode="r")
    ink2d = ink[14] > 0
    sup2d = sup[14] > 0

    def band_mean(a, b):
        acc = np.zeros(surf.shape[1:], np.float32)
        for z in range(a, b):
            acc += surf[z]
        return acc / (b - a)

    up = band_mean(10, 15)     # slices 10..14 (AUC>0.5 band)
    dn = band_mean(4, 9)       # slices 4..8   (AUC<0.5 band)
    nonzero = np.ones(surf.shape[1:], bool)
    for z in range(5, 22):
        nonzero &= surf[z] > 0
    valid = sup2d & nonzero
    pos_m = ink2d & valid
    neg_m = valid & ~ink2d

    res = {}
    contrast = up - dn
    a = auc(contrast[pos_m], contrast[neg_m])
    res["depth_contrast_auc"] = round(a, 4)
    print(f"depth contrast mean(z10-14)-mean(z4-8): AUC={a:.4f}")

    det = contrast - ndi.gaussian_filter(contrast, 50)
    a2 = auc(det[pos_m], det[neg_m])
    res["depth_contrast_det50_auc"] = round(a2, 4)
    print(f"  detrended s50: AUC={a2:.4f}")

    z12 = surf[12].astype(np.float32)
    a3 = auc(z12[pos_m], z12[neg_m])
    print(f"single slice z=12 raw: AUC={a3:.4f}")

    # within-patch control: connected components of the supervision mask
    lab, nlab = ndi.label(sup2d)
    print(f"\nwithin-patch AUCs ({nlab} supervision components):")
    res["per_patch"] = {}
    for i in range(1, nlab + 1):
        m = (lab == i) & valid
        p, ng = ink2d & m, m & ~ink2d
        if p.sum() < 2000 or ng.sum() < 2000:
            continue
        ap = auc(contrast[p], contrast[ng])
        az = auc(z12[p], z12[ng])
        res["per_patch"][i] = {"n_pos": int(p.sum()), "n_neg": int(ng.sum()),
                               "auc_contrast": round(ap, 4), "auc_z12": round(az, 4)}
        print(f"  patch {i:2d}: pos={int(p.sum()):7d} neg={int(ng.sum()):7d} "
              f"contrast={ap:.4f} z12={az:.4f}")

    with open(os.path.join(OUT, "qc_k1b_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote qc_k1b_results.json")


if __name__ == "__main__":
    main()
