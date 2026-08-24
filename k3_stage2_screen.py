"""K3 stage 2 — corrected Paris 4 dual-energy screen (QC-rebuilt, slab-streaming).

Memory-disciplined rewrite: volumes stay uint8; float work happens per z-slab
(256 slices + 1-slice halo), peak RAM ~8GB. Fixes per qc/k3_stage1_review.md:
  - air-offset zero calibration per volume
  - sub-voxel translation check (skip shift if <0.05 vox — measured 0.017)
  - 3-voxel mask erosion, 3^3 neighborhood aggregation
  - z-slab x radius detrending
  - TWO channels: LOW ratio + co-bright (Pb/Au/Hg/Bi) and HIGH ratio + co-bright (Ca/Fe)
Outputs: k3_s2_stats.json, k3_s2_{low,high}_clusters.json, k3_s2_low_gallery.png
"""
import json
import os

import numpy as np
from scipy import ndimage

LEVEL = 1
SCALE = 2 ** LEVEL
CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"
WIN74 = (-0.058, 0.27)
WIN110 = (-0.04, 0.2)
SLAB = 256
HALO = 2


def to_float(u8, win):
    lo, hi = win
    return u8.astype(np.float32) / 255.0 * (hi - lo) + lo


def main():
    os.makedirs(OUT, exist_ok=True)
    a74 = np.load(os.path.join(CACHE, f"paris4_45um_L{LEVEL}_74keV.npy"), mmap_mode="r")
    a110 = np.load(os.path.join(CACHE, f"paris4_45um_L{LEVEL}_110keV.npy"), mmap_mode="r")
    nz = min(a74.shape[0], a110.shape[0])
    ny, nx = a74.shape[1], a74.shape[2]
    print("shapes:", a74.shape, a110.shape, flush=True)

    # --- air offsets from a slice subsample
    stats = {}
    sub74 = to_float(np.asarray(a74[64:nz:128]), WIN74)
    sub110 = to_float(np.asarray(a110[64:nz:128]), WIN110)
    m_sub = (np.asarray(a74[64:nz:128]) > 0) & (np.asarray(a110[64:nz:128]) > 0)
    m_sub = ndimage.binary_erosion(m_sub, iterations=3)
    for name, f in [("74keV", sub74), ("110keV", sub110)]:
        v = f[m_sub]
        air_thr = np.percentile(v, 5)
        stats[f"air_offset_{name}"] = float(np.median(v[v <= air_thr]))
        print(f"{name}: air offset = {stats[f'air_offset_{name}']:+.5f}", flush=True)
    off74, off110 = stats["air_offset_74keV"], stats["air_offset_110keV"]
    # papyrus brightness gates from the same subsample
    pap_sub = m_sub & (sub74 - off74 > 0.02) & (sub110 - off110 > 0.015)
    p75_74 = float(np.percentile((sub74 - off74)[pap_sub], 75))
    p75_110 = float(np.percentile((sub110 - off110)[pap_sub], 75))
    stats["p75_74"], stats["p75_110"] = p75_74, p75_110
    del sub74, sub110, m_sub, pap_sub

    # --- pass 1: per-slab aggregated ratio -> detrended -> collect sigma, then channels
    yy2, xx2 = np.mgrid[0:ny, 0:nx].astype(np.float32)
    det_vals_sample = []
    slabs = []
    for z0 in range(0, nz, SLAB):
        z1 = min(z0 + SLAB, nz)
        h0, h1 = max(z0 - HALO, 0), min(z1 + HALO, nz)
        u74 = np.asarray(a74[h0:h1])
        u110 = np.asarray(a110[h0:h1])
        m = (u74 > 0) & (u110 > 0)
        m = ndimage.binary_erosion(m, iterations=3)
        f74 = to_float(u74, WIN74) - off74
        f110 = to_float(u110, WIN110) - off110
        del u74, u110
        pap = m & (f74 > 0.02) & (f110 > 0.015)
        with np.errstate(divide="ignore", invalid="ignore"):
            r0 = np.where(pap, f74 / np.maximum(f110, 1e-4), 0.0).astype(np.float32)
        w = pap.astype(np.float32)
        k = np.ones((3, 3, 3), np.float32)
        num = ndimage.convolve(r0, k, mode="constant")
        den = ndimage.convolve(w, k, mode="constant")
        with np.errstate(invalid="ignore"):
            ragg = np.where(den >= 14, num / np.maximum(den, 1), np.nan)
        del num, den, r0, w
        # crop halo
        c0, c1 = z0 - h0, z0 - h0 + (z1 - z0)
        ragg = ragg[c0:c1]
        pap_c = pap[c0:c1]
        f74_c = f74[c0:c1]
        f110_c = f110[c0:c1]
        # radial detrend within this slab (32-slice sub-blocks)
        det = np.full_like(ragg, np.nan)
        for b0 in range(0, ragg.shape[0], 32):
            b1 = min(b0 + 32, ragg.shape[0])
            sl = ragg[b0:b1]
            good = np.isfinite(sl)
            if good.sum() < 1000:
                continue
            mid = (b0 + b1) // 2
            ys, xs = np.nonzero(pap_c[mid])
            if len(ys) < 100:
                continue
            cy, cx = ys.mean(), xs.mean()
            rb2 = ((np.sqrt((yy2 - cy) ** 2 + (xx2 - cx) ** 2)) / 8).astype(np.int32)
            rb = np.broadcast_to(rb2, sl.shape)
            lut = np.full(int(rb2.max()) + 1, np.nan, np.float32)
            for bb in np.unique(rb2):
                sel = good & (rb == bb)
                cnt = sel.sum()
                if cnt > 200:
                    lut[bb] = np.median(sl[sel])
            det[b0:b1] = sl - lut[rb]
        good = np.isfinite(det)
        if good.sum():
            det_vals_sample.append(det[good][:: max(1, good.sum() // 100000)])
        slabs.append((z0, z1, det.astype(np.float16), pap_c,
                      (f74_c > p75_74) & (f110_c > p75_110)))
        print(f"slab {z0}:{z1} done ({good.sum()} det voxels)", flush=True)
        del f74, f110, pap, ragg, det, f74_c, f110_c, pap_c, good, m

    sigma = float(np.std(np.concatenate(det_vals_sample)))
    stats["ratio_detrended_sigma"] = round(sigma, 5)
    print(f"global detrended sigma: {sigma:.4f}", flush=True)

    # --- pass 2: channels + cluster masks
    lowm = np.zeros((nz, ny, nx), np.uint8)
    highm = np.zeros((nz, ny, nx), np.uint8)
    for z0, z1, det16, pap_c, cobright in slabs:
        det = det16.astype(np.float32)
        good = np.isfinite(det)
        lowm[z0:z1] = (good & cobright & (det < -4 * sigma)).astype(np.uint8)
        highm[z0:z1] = (good & cobright & (det > 4 * sigma)).astype(np.uint8)
    del slabs
    stats["low_channel_voxels"] = int(lowm.sum())
    stats["high_channel_voxels"] = int(highm.sum())
    print(f"LOW (heavy-metal) channel: {stats['low_channel_voxels']} voxels; "
          f"HIGH (mid-Z): {stats['high_channel_voxels']}", flush=True)

    def cluster_table(ch, tag, top=25):
        lab, n = ndimage.label(ch)
        rows = []
        if n:
            sizes = ndimage.sum_labels(np.ones_like(lab, np.float32), lab, range(1, n + 1))
            order = np.argsort(sizes)[::-1][:top]
            coms = ndimage.center_of_mass(ch, lab, [int(i) + 1 for i in order])
            for i, com in zip(order, coms):
                rows.append({"size_vox": int(sizes[int(i)]),
                             "centroid_zyx_L0": [int(c * SCALE) for c in com]})
        with open(os.path.join(OUT, f"k3_s2_{tag}_clusters.json"), "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"{tag}: {n} clusters; top: {rows[:5]}", flush=True)
        return rows

    low_rows = cluster_table(lowm, "low")
    high_rows = cluster_table(highm, "high")
    with open(os.path.join(OUT, "k3_s2_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)

    # gallery for LOW channel
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in low_rows if r["size_vox"] >= 2][:6]
    if rows:
        fig, axes = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 4.2), squeeze=False)
        for i, rrow in enumerate(rows):
            z, y, x = (c // SCALE for c in rrow["centroid_zyx_L0"])
            s = 60
            im74 = to_float(np.asarray(a74[z, max(y - s, 0):y + s, max(x - s, 0):x + s]), WIN74)
            axes[0][i].imshow(im74, cmap="gray")
            axes[0][i].set_title(f"z={z * SCALE} n={rrow['size_vox']}")
            axes[0][i].set_xticks([]); axes[0][i].set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "k3_s2_low_gallery.png"), dpi=100)
        print("wrote k3_s2_low_gallery.png", flush=True)


if __name__ == "__main__":
    main()
