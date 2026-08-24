r"""Comb 3b — K3 #3/#5 PSF-match, corrected statistic.

Stage 2 flagged HIGH voxels as detrended ratio > 4*sigma_det (0.126) above the
LOCAL slab/radius baseline. The fixed ratio>1.5 count in comb_k3_psfmatch.py is
baseline-shift-dominated (blurring 110keV raises the in-fiber median), so here
the anomaly statistic matches stage 2: excess = ratio - median(ratio in core
window, same sigma); count excess > 0.504 and its connected-component shape.

Verdict logic: if the excess load collapses at sigma ~ sigma_eq (0.56-0.62 px
from the PSD fit) the streaks are PSF-mismatch artifacts; survival = real.
Outputs: comb/k3_psfmatch2.png, appends verdict numbers -> k3_psfmatch2.json
"""
import json
import os

import numpy as np
from scipy import ndimage as ndi

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb"
CACHE = os.path.join(OUT, "cache_k3")
WIN74 = (-0.058, 0.27)
WIN110 = (-0.04, 0.2)
OFF74, OFF110 = -0.03099, -0.02682
CLUSTERS = {"#3": (3925, 1058, 989), "#5": (4022, 1237, 914)}
HZ = 10
SIGMAS = [0.0, 0.25, 0.5, 0.62, 0.75, 1.0, 1.25, 1.5]
EXCESS_THR = 4 * 0.126   # stage-2 HIGH channel: 4 sigma_det
ERODE = 2


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


def comp_shapes(mask3d):
    lab, n = ndi.label(mask3d, structure=np.ones((3, 3, 3), np.int8))
    if not n:
        return 0, 0, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    big = sizes.max()
    # elongation of the largest component (streakiness)
    coords = np.argwhere(lab == (np.argmax(sizes) + 1))
    if len(coords) < 8:
        return int(n), int(big), 0.0
    c = coords - coords.mean(0)
    w = np.linalg.eigvalsh(c.T @ c / len(c))[::-1]
    s = np.sqrt(np.maximum(w, 0))
    return int(n), int(big), round(float(s[0] / max(s[2], 0.5)), 2)


def main():
    results = {}
    figs = {}
    for tag, c in CLUSTERS.items():
        a74 = np.load(os.path.join(CACHE, f"v74_{c[0]}_{c[1]}_{c[2]}.npy"))
        a110 = np.load(os.path.join(CACHE, f"v110_{c[0]}_{c[1]}_{c[2]}.npy"))
        f74 = to_f(a74, WIN74, OFF74)
        f110 = to_f(a110, WIN110, OFF110)
        pap = (a74 > 0) & (a110 > 0) & (f74 > 0.02) & (f110 > 0.015)
        core = ndi.binary_erosion(pap, iterations=ERODE)
        rows = []
        maps = {}
        for s in SIGMAS:
            f110b = masked_blur(f110, pap, s)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(core, f74 / np.maximum(f110b, 1e-4), np.nan)
            med = float(np.nanmedian(ratio))
            excess = ratio - med
            hot = np.isfinite(excess) & (excess > EXCESS_THR)
            ncomp, big, elong = comp_shapes(hot)
            rows.append({"sigma": s, "median": round(med, 4),
                         "n_hot": int(hot.sum()),
                         "hot_frac": round(float(hot.sum() / core.sum()), 5),
                         "n_comp": ncomp, "largest": big,
                         "largest_elong": elong})
            maps[s] = excess
            print(f"{tag} sigma={s:4.2f}: med={med:.3f} hot={hot.sum():6d} "
                  f"comps={ncomp:4d} largest={big:5d} elong={elong}", flush=True)
        results[tag] = rows
        figs[tag] = (f74, f110, maps)

    with open(os.path.join(OUT, "k3_psfmatch2.json"), "w") as f:
        json.dump({"excess_thr": EXCESS_THR, "clusters": results}, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [0.0, 0.5, 0.62, 0.75, 1.0]
    fig, axes = plt.subplots(2, len(show) + 2, figsize=(3.1 * (len(show) + 2), 6.8),
                             squeeze=False)
    for r, (tag, c) in enumerate(CLUSTERS.items()):
        f74, f110, maps = figs[tag]
        axes[r][0].imshow(f74[HZ], cmap="gray")
        axes[r][0].set_title(f"{tag} {c}\n74keV", fontsize=9)
        axes[r][1].imshow(f110[HZ], cmap="gray")
        axes[r][1].set_title("110keV", fontsize=9)
        for k, s in enumerate(show):
            im = axes[r][2 + k].imshow(maps[s][HZ], cmap="coolwarm",
                                       vmin=-0.5, vmax=0.5)
            row = [x for x in results[tag] if x["sigma"] == s][0]
            axes[r][2 + k].set_title(
                f"excess, s={s}\nhot={row['n_hot']}"
                + ("  <-sigma_eq" if s == 0.62 else ""), fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[:, -1], shrink=0.6)
    fig.suptitle("K3 #3/#5 — LOCAL ratio excess (ratio - window median) vs 110keV "
                 f"blur; hot = excess > {EXCESS_THR:.3f} (stage-2 4-sigma)",
                 fontsize=12)
    fig.savefig(os.path.join(OUT, "k3_psfmatch2.png"), dpi=110,
                bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(6.5, 4))
    for tag in CLUSTERS:
        sw = results[tag]
        ax.plot([x["sigma"] for x in sw], [x["n_hot"] for x in sw], marker="o",
                label=f"{tag}")
    ax.axvline(0.62, ls="--", alpha=0.5, label="sigma_eq (PSD fit)")
    ax.set_xlabel("Gaussian sigma applied to 110keV (px)")
    ax.set_ylabel("core voxels with local excess > 0.504")
    ax.legend()
    ax.set_title("K3 #3/#5: 4-sigma hot-voxel load vs blur equalization")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "k3_psfmatch2_curve.png"), dpi=110)
    print("wrote k3_psfmatch2.png / k3_psfmatch2_curve.png")


if __name__ == "__main__":
    main()
