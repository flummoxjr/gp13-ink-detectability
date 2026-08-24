"""K1 — absolute-intensity ROC-AUC kill test on native-9.362um labeled segment w035.

Question: does calibrated absolute intensity (the channel per-crop MAD
normalization deletes) carry ANY linearly-recoverable ink signal at 9um, on the
segment with the strongest known 9um ink (PHerc0139 w035, letters found by
ink_9um)?

Method (pre-registered kill rules):
  - z-mean the central 17 of 28 surface-volume slices -> 2D intensity map
  - detrend with a large-sigma Gaussian (removes illumination/thickness drift)
  - ROC-AUC of (+/-) detrended intensity against max-projected ink labels,
    within the supervision mask
  - |AUC - 0.5| < 0.02 on this best-case segment => channel dead at 9um.

Outputs: trackD/out/k1_w035_auc.json, k1_w035_maps.png
Caches:  D:/vesuvius-data/trackD/w035_{surf,ink,sup}.npy
"""
import json
import os

import numpy as np

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
HF = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices"
SURF_URL = f"{S3}/PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr"
INK_URL = f"{HF}/w035/w035_inklabels.zarr"
SUP_URL = f"{HF}/w035/w035_supervision_mask.zarr"

CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"

# Recorded window of the 9.362um 113keV volume (same for all 2025 batch)
WIN = (-0.03, 0.145)


def load_zarr(url, cache_name, level="0"):
    cache = os.path.join(CACHE, cache_name)
    if os.path.exists(cache):
        return np.load(cache)
    import zarr, fsspec
    print("fetching", url)
    g = zarr.open(fsspec.get_mapper(url), mode="r")
    arr = g[level][:] if level in getattr(g, "keys", lambda: [])() or hasattr(g, "keys") else g[:]
    arr = np.asarray(arr)
    np.save(cache, arr)
    return arr


def auc_rank(pos, neg, max_n=2_000_000):
    """ROC-AUC via rank statistic (Mann-Whitney), subsampled for speed."""
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


def main():
    os.makedirs(OUT, exist_ok=True)
    surf = load_zarr(SURF_URL, "w035_surf.npy")
    print("surface volume:", surf.shape, surf.dtype)
    ink = load_zarr(INK_URL, "w035_ink.npy")
    print("ink labels:", ink.shape, ink.dtype, "positives:", int((ink > 0).sum()))
    sup = load_zarr(SUP_URL, "w035_sup.npy")
    print("supervision mask:", sup.shape, sup.dtype, "valid:", int((sup > 0).sum()))

    nz = surf.shape[0]
    z0, z1 = (nz - 17) // 2, (nz - 17) // 2 + 17
    lo, hi = WIN
    f = surf[z0:z1].astype(np.float32) / 255.0 * (hi - lo) + lo
    zmean = f.mean(axis=0)

    ink2d = (ink > 0).max(axis=0)
    sup2d = (sup > 0).max(axis=0)
    # valid papyrus: supervised AND surface volume non-empty there
    nonzero2d = (surf[z0:z1] > 0).mean(axis=0) > 0.9
    valid = sup2d & nonzero2d
    print(f"valid px: {int(valid.sum())}, ink px in valid: {int((ink2d & valid).sum())}")

    from scipy.ndimage import gaussian_filter
    results = {"segment": "w035", "z_slices": [int(z0), int(z1)], "valid_px": int(valid.sum()),
               "ink_px": int((ink2d & valid).sum()), "aucs": {}}

    for name, img in [
        ("raw_zmean", zmean),
        ("detrend_s25", zmean - gaussian_filter(zmean, 25)),
        ("detrend_s50", zmean - gaussian_filter(zmean, 50)),
        ("detrend_s100", zmean - gaussian_filter(zmean, 100)),
    ]:
        pos = img[ink2d & valid]
        neg = img[valid & ~ink2d]
        a = auc_rank(pos, neg)
        results["aucs"][name] = round(a, 4)
        print(f"AUC[{name}] = {a:.4f}  (ink-{'brighter' if a > 0.5 else 'darker'})")

    with open(os.path.join(OUT, "k1_w035_auc.json"), "w") as fh:
        json.dump(results, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    det = zmean - gaussian_filter(zmean, 50)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    v0, v1 = np.percentile(zmean[valid], [2, 98])
    axes[0].imshow(np.where(valid, zmean, np.nan), cmap="gray", vmin=v0, vmax=v1)
    axes[0].set_title("w035 z-mean calibrated intensity")
    d0, d1 = np.percentile(det[valid], [2, 98])
    axes[1].imshow(np.where(valid, det, np.nan), cmap="gray", vmin=d0, vmax=d1)
    axes[1].set_title("detrended (sigma=50)")
    axes[2].imshow(ink2d, cmap="gray")
    axes[2].set_title("ink labels (max-z)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "k1_w035_maps.png"), dpi=90)
    print("wrote k1_w035_maps.png")


if __name__ == "__main__":
    main()
