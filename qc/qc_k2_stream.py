"""QC audit of K2 — streaming part.

  E. Air-ROI discriminator: for each scroll, find a low-density (air) ROI
     fully inside the release mask, stream it at L0, and compare its PSD to
     the papyrus ROI PSD.  Excess of papyrus over air = real structure; the
     frequency where papyrus sinks to the air level is the measured
     information ceiling (independent of the red-spectrum objection).
  F. ROI-to-ROI variance: two extra papyrus ROIs on PHerc0813 at different
     locations; report PSD spread.

Probes at level 5 (~30 MB per scroll), streams 256^3 (or 128^3 fallback)
L0 ROIs (~16 MB each), caches everything under D:/vesuvius-data/trackD/qc_.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD")
from k2_spectral_ceiling import radial_psd, vol_node, open_level  # noqa: E402

CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc"
TARGETS = [
    ("PHerc0813", "20250821151723"),
    ("PHerc1203", "20250820131727"),
    ("PHerc0139", "20250728140407"),
]
L = 5
SC = 2 ** L  # L5 voxel = 32 L0 px


def get_l5(sample, long_id):
    cache = os.path.join(CACHE, f"qc_L5_{sample}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    z = open_level(sample, long_id, L)
    arr = z[:]
    np.save(cache, arr)
    return arr


def stream_roi(sample, long_id, origin, n, tag):
    cache = os.path.join(CACHE, f"qc_roi_{sample}_{tag}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    z0 = open_level(sample, long_id, 0)
    roi = z0[origin[0]:origin[0] + n, origin[1]:origin[1] + n, origin[2]:origin[2] + n]
    np.save(cache, roi)
    return roi


def block_stats(arr, box):
    fill = uniform_filter((arr > 0).astype(np.float32), box, mode="constant")
    mean = uniform_filter(arr.astype(np.float32), box, mode="constant")
    m2 = uniform_filter(arr.astype(np.float32) ** 2, box, mode="constant")
    std = np.sqrt((m2 - mean ** 2).clip(0))
    return fill, mean, std


def to_origin(idx, box, shape0, n):
    o = [int(i * SC - n // 2) for i in idx]
    return tuple(int(np.clip(o[k], 0, shape0[k] - n)) for k in range(3))


def find_air(arr, shape0):
    """Lowest-mean fully-inside-mask block; try 256^3 then 128^3."""
    for n in (256, 128):
        box = n // SC
        fill, mean, std = block_stats(arr, box)
        zlo, zhi = arr.shape[0] // 8, 7 * arr.shape[0] // 8
        cand = (fill > 0.999)
        cand[:zlo] = False
        cand[zhi:] = False
        # keep away from array borders so the box is fully sampled
        b = box // 2 + 1
        for ax in (1, 2):
            sl = [slice(None)] * 3
            sl[ax] = slice(0, b)
            cand[tuple(sl)] = False
            sl[ax] = slice(-b, None)
            cand[tuple(sl)] = False
        if not cand.any():
            continue
        means = np.where(cand, mean, 1e9)
        idx = np.unravel_index(np.argmin(means), means.shape)
        return to_origin(idx, box, shape0, n), n, float(mean[idx]), float(std[idx])
    return None, None, None, None


def find_papyrus(arr, shape0, zrange, n=256):
    """Highest-mean fully-filled block within a z-range of the L5 grid."""
    box = n // SC
    fill, mean, std = block_stats(arr, box)
    cand = fill > 0.999
    cand[: zrange[0]] = False
    cand[zrange[1]:] = False
    b = box // 2 + 1
    for ax in (1, 2):
        sl = [slice(None)] * 3
        sl[ax] = slice(0, b)
        cand[tuple(sl)] = False
        sl[ax] = slice(-b, None)
        cand[tuple(sl)] = False
    means = np.where(cand, mean, -1)
    idx = np.unravel_index(np.argmax(means), means.shape)
    return to_origin(idx, box, shape0, n), float(mean[idx])


def psd_of(roi, win):
    lo, hi = win
    f = roi.astype(np.float32) / 255.0 * (hi - lo) + lo
    return radial_psd(f)


def main():
    t0 = time.time()
    res = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    for ax, (sample, vid) in zip(axes, TARGETS):
        meta = vol_node(sample, vid)
        shape0 = meta["shape"]
        print(f"\n=== {sample} ===")
        arr = get_l5(sample, meta["long_id"])
        print(f"  L5 {arr.shape}")

        # air ROI
        ao, an, am, astd = find_air(arr, shape0)
        print(f"  air ROI: origin={ao} n={an} L5mean={am:.1f} L5std={astd:.1f}")
        air = stream_roi(sample, meta["long_id"], ao, an, f"air{an}")
        noise_hp = float((air.astype(np.float32) - gaussian_filter(air.astype(np.float32), 2)).std())
        print(f"  air L0: mean={air.mean():.1f} std={air.std():.2f} DN, hp-std={noise_hp:.2f} DN, zero-frac={float((air==0).mean()):.4f}")
        qa, pa = psd_of(air, meta["win"])

        # papyrus (cached K2 ROI)
        pap = np.load(os.path.join(CACHE, f"k2_roi_{sample}_{vid}.npy"))
        qp, pp = psd_of(pap, meta["win"])

        # floor
        rng0 = np.random.default_rng(0)
        step = (meta["win"][1] - meta["win"][0]) / 255.0
        _, pfl = radial_psd(rng0.uniform(-step / 2, step / 2, pap.shape).astype(np.float32))

        band = (qp > 0.005) & (qp <= 0.5)
        qb = qp[band]
        pa_i = np.interp(qb, qa[(qa > 0.005)], pa[(qa > 0.005)])
        ratio_air = pp[band] / pa_i
        # ceiling metrics
        def first_below(thr):
            w = np.where(ratio_air < thr)[0]
            return float(qb[w[0]]) if len(w) else None
        c2, c1 = first_below(2.0), first_below(1.25)
        r = {
            "air_origin": ao, "air_n": an, "air_dn_mean": float(air.mean()),
            "air_dn_std": float(air.std()), "air_dn_hpstd": noise_hp,
            "pap_over_air_at": {str(qq): float(np.interp(qq, qb, ratio_air))
                                 for qq in (0.05, 0.15, 0.25, 0.35, 0.45, 0.5)},
            "q_pap_below_2x_air": c2, "q_pap_below_1.25x_air": c1,
            "air_over_floor_at_0.4": float(np.interp(0.4, qb, pa_i / pfl[band])),
        }
        res[sample] = r
        hp2 = 4.681 / c2 if c2 else None
        print(f"  papyrus/air: @0.15={r['pap_over_air_at']['0.15']:.1f} @0.25={r['pap_over_air_at']['0.25']:.1f} "
              f"@0.35={r['pap_over_air_at']['0.35']:.1f} @0.5={r['pap_over_air_at']['0.5']:.2f}")
        print(f"  papyrus sinks below 2x air at q={c2}, below 1.25x air at q={c1}"
              + (f"  (2x: half-period {hp2:.1f} um)" if hp2 else ""))
        print(f"  air PSD / quant floor at q=0.4: {r['air_over_floor_at_0.4']:.1f}")

        ax.loglog(qb, pp[band], label="papyrus ROI")
        ax.loglog(qa[(qa > 0.005) & (qa <= 0.5)], pa[(qa > 0.005) & (qa <= 0.5)], label=f"air ROI ({an}^3)")
        ax.loglog(qb, pfl[band], "--", label="uint8 rounding floor")
        ax.set_title(sample)
        ax.set_xlabel("cycles/px")
        ax.legend(fontsize=8)

        # F: extra papyrus ROIs (PHerc0813 only)
        if sample == "PHerc0813":
            zc = (arr.shape[0] // 4, 3 * arr.shape[0] // 4)
            third = (zc[1] - zc[0]) // 3
            spread = {}
            curves = [("cached", pp)]
            for k, zr in [("b", (zc[0], zc[0] + third)), ("c", (zc[1] - third, zc[1]))]:
                po, pm = find_papyrus(arr, shape0, zr)
                print(f"  extra papyrus ROI {k}: origin={po} L5mean={pm:.1f}")
                roi = stream_roi(sample, meta["long_id"], po, 256, f"pap_{k}")
                if np.array_equal(roi, pap):
                    print("    (identical to cached ROI — skip)")
                    continue
                _, pk = psd_of(roi, meta["win"])
                curves.append((k, pk))
                ax.loglog(qb, pk[band], lw=0.8, alpha=0.7, label=f"papyrus ROI {k}")
                print(f"    DN mean/std {roi.mean():.0f}/{roi.std():.0f}")
            for qq in (0.05, 0.15, 0.25, 0.35, 0.5):
                vals = [float(np.interp(qq, qb, c[band])) for _, c in curves]
                spread[str(qq)] = {"min": min(vals), "max": max(vals),
                                   "maxmin_ratio": max(vals) / min(vals)}
            res["PHerc0813_roi_spread"] = spread
            print("  PSD spread across ROIs (max/min): " +
                  " ".join(f"q={k}:{v['maxmin_ratio']:.2f}x" for k, v in spread.items()))
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qc_k2_air.png"), dpi=110)
    with open(os.path.join(OUT, "qc_k2_stream.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\ndone in {time.time()-t0:.0f}s -> qc_k2_stream.json / qc_k2_air.png")


if __name__ == "__main__":
    main()
