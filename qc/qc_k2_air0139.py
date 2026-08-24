"""Retry a clean air ROI for PHerc0139: variance-aware selection.

The lowest-mean block picked first was structurally busy (L0 hp-std 13 DN,
PSD ~= papyrus PSD). Here: candidates = fill>0.999 & mean<70, ranked by L5
block std; evaluate the best few at both 256^3 and 128^3, stream, and report
PSD vs the papyrus ROI.
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD")
from k2_spectral_ceiling import radial_psd, vol_node, open_level  # noqa: E402

CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc"
SC = 32
sample, vid = "PHerc0139", "20250728140407"


def block_stats(arr, box):
    fill = uniform_filter((arr > 0).astype(np.float32), box, mode="constant")
    mean = uniform_filter(arr.astype(np.float32), box, mode="constant")
    m2 = uniform_filter(arr.astype(np.float32) ** 2, box, mode="constant")
    std = np.sqrt((m2 - mean ** 2).clip(0))
    return fill, mean, std


def main():
    meta = vol_node(sample, vid)
    shape0 = meta["shape"]
    arr = np.load(os.path.join(CACHE, f"qc_L5_{sample}.npy"))
    pap = np.load(os.path.join(CACHE, f"k2_roi_{sample}_{vid}.npy"))
    qp, pp = radial_psd(pap.astype(np.float32) / 255.0 * (meta["win"][1] - meta["win"][0]) + meta["win"][0])

    results = {}
    for n in (256, 128):
        box = n // SC
        fill, mean, std = block_stats(arr, box)
        zlo, zhi = arr.shape[0] // 8, 7 * arr.shape[0] // 8
        cand = (fill > 0.999) & (mean < 70)
        cand[:zlo] = False
        cand[zhi:] = False
        b = box // 2 + 1
        for ax in (1, 2):
            sl = [slice(None)] * 3
            sl[ax] = slice(0, b)
            cand[tuple(sl)] = False
            sl[ax] = slice(-b, None)
            cand[tuple(sl)] = False
        ncand = int(cand.sum())
        print(f"n={n}: {ncand} candidate blocks (fill>0.999, mean<70)")
        if ncand == 0:
            continue
        stds = np.where(cand, std, 1e9)
        idx = np.unravel_index(np.argmin(stds), stds.shape)
        origin = tuple(int(np.clip(i * SC - n // 2, 0, shape0[k] - n)) for k, i in enumerate(idx))
        print(f"  best-by-std: origin={origin} L5 mean={mean[idx]:.1f} std={std[idx]:.1f}")
        cache = os.path.join(CACHE, f"qc_roi_{sample}_air2_{n}.npy")
        if os.path.exists(cache):
            roi = np.load(cache)
        else:
            z0 = open_level(sample, meta["long_id"], 0)
            roi = z0[origin[0]:origin[0] + n, origin[1]:origin[1] + n, origin[2]:origin[2] + n]
            np.save(cache, roi)
        hp = float((roi.astype(np.float32) - gaussian_filter(roi.astype(np.float32), 2)).std())
        print(f"  L0: mean={roi.mean():.1f} std={roi.std():.2f} hp-std={hp:.2f} zero-frac={(roi==0).mean():.4f}")
        qa, pa = radial_psd(roi.astype(np.float32) / 255.0 * (meta["win"][1] - meta["win"][0]) + meta["win"][0])
        band = (qp > 0.005) & (qp <= 0.5)
        qb = qp[band]
        pa_i = np.interp(qb, qa[qa > 0.005], pa[qa > 0.005])
        ratio = pp[band] / pa_i
        at = {str(qq): float(np.interp(qq, qb, ratio)) for qq in (0.15, 0.25, 0.35, 0.45, 0.5)}
        w2 = np.where(ratio < 2)[0]
        c2 = float(qb[w2[0]]) if len(w2) else None
        print(f"  papyrus/air: {at}  2x-crossing: {c2}")
        results[f"n{n}"] = {"origin": origin, "l0_mean": float(roi.mean()), "l0_std": float(roi.std()),
                            "hp_std": hp, "pap_over_air": at, "q_below_2x": c2}

    with open(os.path.join(OUT, "qc_k2_air0139.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote qc_k2_air0139.json")


if __name__ == "__main__":
    main()
