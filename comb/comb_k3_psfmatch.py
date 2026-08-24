r"""Comb 3 — blur-equalization re-check of K3 HIGH-channel clusters #3 and #5.

Context: k3 stage-2 flagged mid-Z (Ca/Fe-like) HIGH-ratio clusters; verification
imagery showed #1/#2/#4 = nodular incrustation (real), but #3/#5 (the z~3900-4020
group) show STREAKY fiber-scale ratio patterns suspected to be PSF mismatch:
the 74keV Paganin recon is ~22% blurrier (width ~ sqrt(lambda)), so at fiber
edges f74 is smoothed while f110 keeps contrast -> spurious high f74/f110 on
thin bright fibers' flanks and dark-gap rims.

Test: fetch L0 windows (21 x 160 x 160) at both energies around each cluster
centroid; blur the SHARPER 110keV volume with a 3D Gaussian (sigma sweep
0-1.5 px, mask-aware normalized convolution), recompute the offset-corrected
ratio, and track the high-ratio anomaly load vs sigma. Independently estimate
the true blur mismatch sigma_eq from the in-plane radial PSD ratio
(PSD74/PSD110 ~ exp(-4 pi^2 sigma^2 q^2) where both are structure-dominated).

If the streaks collapse at sigma ~ sigma_eq -> artifact confirmed.
If they survive -> genuinely mineralized fibers (mildly interesting).

Outputs: comb/k3_psfmatch.png, comb/k3_psfmatch.json
"""
import json
import os

import numpy as np
from scipy import ndimage as ndi

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
V74 = f"{BUCKET}/PHercParis4/volumes/20260310170716-45.532um-11.0m-74keV-masked.zarr"
V110 = f"{BUCKET}/PHercParis4/volumes/20260310173927-45.532um-11.0m-110keV-masked.zarr"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb"
CACHE = os.path.join(OUT, "cache_k3")
WIN74 = (-0.058, 0.27)
WIN110 = (-0.04, 0.2)
OFF74, OFF110 = -0.03099, -0.02682  # identical to k3_verify_clusters.py
CLUSTERS = {  # 1-indexed rank in k3_s2_high_clusters.json
    "#3": (3925, 1058, 989),
    "#5": (4022, 1237, 914),
}
HZ, HY, HX = 10, 80, 80
SIGMAS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
RATIO_THR = 1.5     # stage-1 erosion-corrected dense-phase level
ERODE = 2           # L0 voxels off the papyrus mask rim

os.makedirs(CACHE, exist_ok=True)


def fetch(tag, vol_url, c):
    p = os.path.join(CACHE, f"{tag}_{c[0]}_{c[1]}_{c[2]}.npy")
    if os.path.exists(p):
        return np.load(p)
    import zarr, fsspec
    zl = zarr.open(fsspec.get_mapper(vol_url), mode="r")["0"]
    z, y, x = c
    a = np.asarray(zl[z - HZ:z + HZ + 1, y - HY:y + HY + 1, x - HX:x + HX + 1])
    np.save(p, a)
    return a


def to_f(a, win, off):
    return a.astype(np.float32) / 255 * (win[1] - win[0]) + win[0] - off


def masked_blur(f, m, sigma):
    if sigma <= 0:
        return f
    mf = m.astype(np.float32)
    num = ndi.gaussian_filter(f * mf, sigma)
    den = ndi.gaussian_filter(mf, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-3, num / np.maximum(den, 1e-3), f)


def radial_psd_2d(sl):
    n0, n1 = sl.shape
    w = np.hanning(n0)[:, None] * np.hanning(n1)[None, :]
    v = (sl - sl.mean()) * w
    P = np.abs(np.fft.fftn(v)) ** 2
    f0 = np.fft.fftfreq(n0)
    f1 = np.fft.fftfreq(n1)
    q = np.sqrt(f0[:, None] ** 2 + f1[None, :] ** 2)
    bins = np.linspace(0, 0.5, 40)
    idx = np.digitize(q.ravel(), bins)
    num = np.bincount(idx, weights=P.ravel(), minlength=len(bins) + 1)
    cnt = np.bincount(idx, minlength=len(bins) + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = num / cnt
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, prof[1:len(bins)]


def estimate_sigma_eq(f74, f110, m):
    """ln(PSD74/PSD110) = -4 pi^2 sigma^2 q^2 + const, fit over structure band."""
    p74 = []
    p110 = []
    for iz in range(HZ - 3, HZ + 4):
        q, a = radial_psd_2d(np.where(m[iz], f74[iz], f74[iz].mean()))
        _, b = radial_psd_2d(np.where(m[iz], f110[iz], f110[iz].mean()))
        p74.append(a)
        p110.append(b)
    p74 = np.nanmedian(np.stack(p74), axis=0)
    p110 = np.nanmedian(np.stack(p110), axis=0)
    band = (q >= 0.06) & (q <= 0.30)   # structure-dominated, away from noise floor
    ln_r = np.log(p74[band] / p110[band])
    q2 = q[band] ** 2
    A = np.column_stack([q2, np.ones_like(q2)])
    coef, *_ = np.linalg.lstsq(A, ln_r, rcond=None)
    slope = coef[0]
    sig2 = -slope / (4 * np.pi ** 2)
    return float(np.sqrt(max(sig2, 0.0))), q, p74, p110


def main():
    results = {}
    panels = {}
    for tag, c in CLUSTERS.items():
        a74 = fetch("v74", V74, c)
        a110 = fetch("v110", V110, c)
        f74 = to_f(a74, WIN74, OFF74)
        f110 = to_f(a110, WIN110, OFF110)
        pap = (a74 > 0) & (a110 > 0) & (f74 > 0.02) & (f110 > 0.015)
        core = ndi.binary_erosion(pap, iterations=ERODE)
        print(f"{tag} {c}: pap {pap.mean():.3f}, core {core.mean():.3f} "
              f"({int(core.sum())} vox)", flush=True)

        sig_eq, q, p74, p110 = estimate_sigma_eq(f74, f110, pap)
        print(f"{tag}: spectral blur-mismatch sigma_eq = {sig_eq:.2f} px", flush=True)

        rows = []
        ratio_maps = {}
        for s in SIGMAS:
            f110b = masked_blur(f110, pap, s)
            f74b = f74  # 74 keV left untouched (it is the blurrier one)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(core, f74b / np.maximum(f110b, 1e-4), np.nan)
            rv = ratio[core]
            load = float((rv > RATIO_THR).mean())
            # streak-shape metric: high-ratio voxels remaining
            rows.append({"sigma": s,
                         "ratio_p50": round(float(np.nanmedian(rv)), 4),
                         "ratio_p99": round(float(np.nanpercentile(rv, 99)), 4),
                         "ratio_p999": round(float(np.nanpercentile(rv, 99.9)), 4),
                         "frac_gt_thr": round(load, 6),
                         "n_gt_thr": int((rv > RATIO_THR).sum())})
            ratio_maps[s] = ratio
            print(f"  sigma={s:4.2f}: p50={rows[-1]['ratio_p50']:.3f} "
                  f"p99={rows[-1]['ratio_p99']:.3f} "
                  f"frac>{RATIO_THR}={load:.5f}", flush=True)
        results[tag] = {"centroid_zyx_L0": list(c), "sigma_eq_px": round(sig_eq, 3),
                        "core_vox": int(core.sum()), "sweep": rows}
        panels[tag] = (f74, f110, ratio_maps, core)

    with open(os.path.join(OUT, "k3_psfmatch.json"), "w") as f:
        json.dump({"params": {"ratio_thr": RATIO_THR, "erode": ERODE,
                              "window_hzyx": [HZ, HY, HX], "sigmas": SIGMAS},
                   "clusters": results}, f, indent=1)

    # ---- figure: per cluster, one row of images + load-vs-sigma curves ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 6, figsize=(21, 7.4), squeeze=False)
    for r, (tag, c) in enumerate(CLUSTERS.items()):
        f74, f110, ratio_maps, core = panels[tag]
        zc = HZ
        axes[r][0].imshow(f74[zc], cmap="gray")
        axes[r][0].set_title(f"{tag} {c}\n74keV (blurrier)", fontsize=9)
        axes[r][1].imshow(f110[zc], cmap="gray")
        axes[r][1].set_title("110keV (sharper)", fontsize=9)
        for k, s in enumerate([0.0, 0.75, 1.0, 1.5]):
            im = axes[r][2 + k].imshow(ratio_maps[s][zc], cmap="coolwarm",
                                       vmin=0.9, vmax=1.7)
            n = results[tag]["sweep"][SIGMAS.index(s)]["n_gt_thr"]
            axes[r][2 + k].set_title(
                f"ratio, sigma={s}\nn>{RATIO_THR}: {n}"
                + ("  (~sigma_eq)" if abs(s - results[tag]["sigma_eq_px"]) < 0.2
                   else ""), fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[:, -1], shrink=0.6)
    fig.suptitle("K3 HIGH clusters #3/#5 — ratio after blurring 110keV to match "
                 "74keV PSF (central z slice; core mask, rim eroded "
                 f"{ERODE} vox)", fontsize=12)

    fig.savefig(os.path.join(OUT, "k3_psfmatch.png"), dpi=110,
                bbox_inches="tight")

    # load-vs-sigma curve
    fig2, ax = plt.subplots(figsize=(6, 4))
    for tag in CLUSTERS:
        sw = results[tag]["sweep"]
        ax.plot([r["sigma"] for r in sw], [r["n_gt_thr"] for r in sw],
                marker="o", label=f"{tag} (sigma_eq={results[tag]['sigma_eq_px']})")
        ax.axvline(results[tag]["sigma_eq_px"], ls="--", alpha=0.4)
    ax.set_xlabel("Gaussian sigma applied to 110keV (px)")
    ax.set_ylabel(f"core voxels with ratio > {RATIO_THR}")
    ax.set_yscale("symlog")
    ax.legend()
    ax.set_title("High-ratio anomaly load vs blur equalization")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "k3_psfmatch_curve.png"), dpi=110)
    print("wrote k3_psfmatch.png / k3_psfmatch_curve.png")


if __name__ == "__main__":
    main()
