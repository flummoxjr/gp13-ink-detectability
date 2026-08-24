"""K3 stage 2a — spatially localize dense-material candidates in the Paris 4 45.5um pair.

Connected-component analysis of the brightest 74keV voxels: where are they
(mount vs interior), how big, and what is their 74/110 ratio? Writes a cluster
table (JSON) and a gallery of the top clusters in context.
"""
import json
import os

import numpy as np
from scipy import ndimage

CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"

WIN74 = (-0.058, 0.27)
WIN110 = (-0.04, 0.2)


def to_float(u8, win):
    lo, hi = win
    return u8.astype(np.float32) / 255.0 * (hi - lo) + lo


def main():
    a74 = np.load(os.path.join(CACHE, "paris4_45um_L3_74keV.npy"))
    a110 = np.load(os.path.join(CACHE, "paris4_45um_L3_110keV.npy"))
    nz = min(a74.shape[0], a110.shape[0])
    a74, a110 = a74[:nz], a110[:nz]
    m74, m110 = a74 > 0, a110 > 0

    f74 = to_float(a74, WIN74)
    f110 = to_float(a110, WIN110)

    # Erode the mask so partial-volume rim voxels can't masquerade as dense material
    m_core = ndimage.binary_erosion(m74 & m110, iterations=2)

    thr = np.percentile(f74[m_core], 99.95)
    dense = m_core & (f74 >= thr)
    lab, n = ndimage.label(dense)
    print(f"dense threshold f74={thr:.4f}; {int(dense.sum())} voxels in {n} clusters (mask-eroded)")

    # Scroll axis proxy: per-slice centroid of the papyrus mask
    clusters = []
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    com_all = ndimage.center_of_mass(dense, lab, [i + 1 for i in order[:40]])
    for rank, (idx, com) in enumerate(zip(order[:40], com_all)):
        cid = int(idx) + 1
        sel = lab == cid
        size = int(sizes[idx])
        zc, yc, xc = (float(c) for c in com)
        zi = int(round(zc))
        mask_slice = m74[zi]
        if mask_slice.sum() > 0:
            ys, xs = np.nonzero(mask_slice)
            cy, cx = ys.mean(), xs.mean()
            r_norm = float(np.hypot(yc - cy, xc - cx) / max(np.hypot(ys - cy, xs - cx).max(), 1))
        else:
            r_norm = float("nan")
        rvals = f74[sel] / np.maximum(f110[sel], 1e-4)
        clusters.append({
            "id": cid,
            "size_vox": size,
            "centroid_zyx_L3": [round(zc, 1), round(yc, 1), round(xc, 1)],
            "centroid_zyx_L0": [round(zc * 8, 0), round(yc * 8, 0), round(xc * 8, 0)],
            "radial_frac": round(r_norm, 3),
            "f74_mean": round(float(f74[sel].mean()), 4),
            "ratio_median": round(float(np.median(rvals)), 3),
        })

    with open(os.path.join(OUT, "k3_dense_clusters.json"), "w") as fh:
        json.dump({"threshold_f74": float(thr), "n_clusters": int(n), "top": clusters}, fh, indent=1)

    for c in clusters[:15]:
        print(c)

    # Gallery: biggest 6 clusters in slice context
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, c in zip(axes.ravel(), clusters[:6]):
        z = int(round(c["centroid_zyx_L3"][0]))
        ax.imshow(f74[z], cmap="gray", vmax=np.percentile(f74[z][m74[z]], 99.5) if m74[z].any() else None)
        y, x = c["centroid_zyx_L3"][1], c["centroid_zyx_L3"][2]
        ax.add_patch(plt.Circle((x, y), 8, fill=False, color="red", lw=1.5))
        ax.set_title(f"#{c['id']} z={z} size={c['size_vox']} r={c['radial_frac']} ratio={c['ratio_median']}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "k3_dense_gallery.png"), dpi=110)
    print("wrote gallery")


if __name__ == "__main__":
    main()
