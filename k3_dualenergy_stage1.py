"""K3 stage 1 — Paris 4 45.5um dual-energy (74/110 keV) alignment + ratio prototype.

Downloads level-3 pyramids of both masked volumes (~40MB each, cached locally),
estimates the inter-volume translation by phase correlation, inverts the recorded
uint8 conversion windows back to reconstruction-float values, and computes a
first papyrus-masked energy-ratio map with dense-material candidates flagged.

Outputs (in trackD/out/):
  k3_stage1_stats.json      — alignment shift, mask overlap, ratio stats
  k3_stage1_overview.png    — mid-slices, ratio map, histograms
Cache (on D:): D:/vesuvius-data/trackD/paris4_45um_L{level}_{energy}.npy
"""
import json
import os

import numpy as np

LEVEL = 3
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
VOLS = {
    "74keV": {
        "path": "PHercParis4/volumes/20260310170716-45.532um-11.0m-74keV-masked.zarr",
        "win": (-0.058, 0.27),
        "shape0": (4071, 2264, 2264),
    },
    "110keV": {
        "path": "PHercParis4/volumes/20260310173927-45.532um-11.0m-110keV-masked.zarr",
        "win": (-0.04, 0.2),
        "shape0": (4066, 2264, 2264),
    },
}
CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def load_level(name, spec, level=LEVEL):
    cache = os.path.join(CACHE, f"paris4_45um_L{level}_{name}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import zarr
    import fsspec

    url = f"{BUCKET}/{spec['path']}"
    store = fsspec.get_mapper(url)
    g = zarr.open(store, mode="r")
    arr = g[str(level)][:]
    np.save(cache, arr)
    return arr


def to_float(u8, win):
    """Invert the recorded float32->uint8 conversion window. 0 is the mask fill."""
    lo, hi = win
    f = u8.astype(np.float32) / 255.0 * (hi - lo) + lo
    return f


def phase_corr_shift(a, b):
    """Integer-voxel translation of b relative to a via 3D phase correlation."""
    A = np.fft.rfftn(a)
    B = np.fft.rfftn(b, s=a.shape)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-9
    c = np.fft.irfftn(R, s=a.shape)
    peak = np.unravel_index(np.argmax(c), c.shape)
    shift = []
    for p, n in zip(peak, a.shape):
        shift.append(int(p) if p <= n // 2 else int(p) - n)
    return tuple(shift), float(c.max())


def main():
    print("Loading level", LEVEL, "volumes (cached after first run)...")
    a74 = load_level("74keV", VOLS["74keV"])
    a110 = load_level("110keV", VOLS["110keV"])
    print("shapes:", a74.shape, a110.shape)

    # Crop both to the common shape before any comparison
    nz = min(a74.shape[0], a110.shape[0])
    a74c, a110c = a74[:nz], a110[:nz]

    # Masked-region agreement before registration (fill value is 0)
    m74 = a74c > 0
    m110 = a110c > 0
    inter = np.logical_and(m74, m110).sum()
    union = np.logical_or(m74, m110).sum()
    iou_pre = inter / union if union else 0.0
    print(f"mask IoU (unshifted): {iou_pre:.4f}")

    # Phase correlation on the central subvolume of the mask (binary → robust to
    # brightness differences between energies)
    zc, yc, xc = (s // 2 for s in a74c.shape)
    hz, hy, hx = min(zc, 192), min(yc, 96), min(xc, 96)
    sub74 = m74[zc - hz:zc + hz, yc - hy:yc + hy, xc - hx:xc + hx].astype(np.float32)
    sub110 = m110[zc - hz:zc + hz, yc - hy:yc + hy, xc - hx:xc + hx].astype(np.float32)
    shift, peak = phase_corr_shift(sub74, sub110)
    print(f"phase-correlation shift (110 rel. 74), level-{LEVEL} voxels: {shift}, peak={peak:.4f}")

    # Apply integer shift to 110keV and recompute IoU
    a110s = np.roll(a110c, shift, axis=(0, 1, 2))
    m110s = a110s > 0
    iou_post = (np.logical_and(m74, m110s).sum() / np.logical_or(m74, m110s).sum())
    print(f"mask IoU (shifted): {iou_post:.4f}")

    # Convert to reconstruction floats
    f74 = to_float(a74c, VOLS["74keV"]["win"])
    f110 = to_float(a110s, VOLS["110keV"]["win"])

    both = np.logical_and(m74, m110s)
    # Papyrus voxels: positive attenuation-ish values in both volumes
    pap = both & (f74 > 0.02) & (f110 > 0.02)
    ratio = np.zeros_like(f74)
    ratio[pap] = f74[pap] / f110[pap]

    r = ratio[pap]
    stats = {
        "level": LEVEL,
        "shift_110_rel_74_voxels": list(shift),
        "phase_corr_peak": peak,
        "mask_iou_pre": float(iou_pre),
        "mask_iou_post": float(iou_post),
        "papyrus_voxels": int(pap.sum()),
        "ratio_median": float(np.median(r)),
        "ratio_p05": float(np.percentile(r, 5)),
        "ratio_p95": float(np.percentile(r, 95)),
        "ratio_p999": float(np.percentile(r, 99.9)),
        "f74_p999": float(np.percentile(f74[m74], 99.9)),
        "f110_p999": float(np.percentile(f110[m110s], 99.9)),
    }

    # Dense-material candidates: brightest voxels in the 74keV frame within mask
    thr = np.percentile(f74[m74], 99.95)
    dense = m74 & (f74 >= thr)
    stats["dense_thr_f74"] = float(thr)
    stats["dense_voxels"] = int(dense.sum())
    if dense.sum() > 50:
        dr = f74[dense & m110s] / np.maximum(f110[dense & m110s], 1e-4)
        stats["dense_ratio_median"] = float(np.median(dr))
        stats["dense_ratio_p90"] = float(np.percentile(dr, 90))

    with open(os.path.join(OUT, "k3_stage1_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats, indent=1))

    # Overview figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zmid = nz // 2
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(f74[zmid], cmap="gray")
    axes[0, 0].set_title(f"74 keV, z={zmid} (L{LEVEL})")
    axes[0, 1].imshow(f110[zmid], cmap="gray")
    axes[0, 1].set_title(f"110 keV shifted {shift}")
    rp = np.where(pap[zmid], ratio[zmid], np.nan)
    im = axes[0, 2].imshow(rp, cmap="coolwarm", vmin=stats["ratio_p05"], vmax=stats["ratio_p95"])
    axes[0, 2].set_title("74/110 ratio (papyrus)")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.8)
    axes[1, 0].hist(f74[pap].ravel(), bins=120)
    axes[1, 0].set_title("f74 papyrus values")
    axes[1, 1].hist(f110[pap].ravel(), bins=120)
    axes[1, 1].set_title("f110 papyrus values")
    axes[1, 2].hist(r, bins=150, range=(stats["ratio_p05"] * 0.8, stats["ratio_p95"] * 1.2))
    axes[1, 2].axvline(stats["ratio_median"], color="k", ls="--")
    axes[1, 2].set_title("ratio histogram")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "k3_stage1_overview.png"), dpi=110)
    print("wrote", os.path.join(OUT, "k3_stage1_overview.png"))


if __name__ == "__main__":
    main()
