"""Investigation D: visual QC of the derived PHerc1203 9.362um <-> 2.403um transform."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zarr_http import Zarr3D  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
P9 = "PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr"
P2 = "PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
V9, V2 = 0.009362, 0.002403
RATIO = V9 / V2
T = np.array([7940.1, 19.2, -18.8])   # level-0 9um voxels (z, y, x)


def main():
    print(json.loads(requests.get(
        "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1203/representations/"
        "predictions/surfaces/20260319130212-surface-20260413222639-surface-m7-L2-th0.2."
        "normal-grids/metadata.json", timeout=30).text))

    L9, L2 = 2, 4
    z9 = Zarr3D(P9, L9)
    z2 = Zarr3D(P2, L2)
    rescale = (V2 * 2 ** L2) / (V9 * 2 ** L9)

    # a z-plane in the middle of the band, lateral centre near the segment cloud
    z2c, y2c, x2c = 8500, (3600 - T[1]) * RATIO, (3300 - T[2]) * RATIO
    n = 192
    a2 = (np.array([z2c, y2c, x2c]) / 2 ** L2 - n / 2).astype(int)
    Braw = z2.read(a2[0], a2[0] + n, a2[1], a2[1] + n, a2[2], a2[2] + n).astype(np.float32)
    Bz = zoom(Braw, rescale, order=1)
    m = min(Bz.shape)
    Bz = Bz[:m, :m, :m]
    p9 = np.array([z2c, y2c, x2c]) / RATIO + T
    b9 = (p9 / 2 ** L9 - m / 2).astype(int)
    A = z9.read(b9[0], b9[0] + m, b9[1], b9[1] + m, b9[2], b9[2] + m).astype(np.float32)

    k = m // 2
    fig, ax = plt.subplots(2, 3, figsize=(13, 9))
    for j, (img, ttl) in enumerate([(A[k], "9.362um (lvl2, 37.4um/vox)"),
                                    (Bz[k], "2.403um (lvl4, resampled to 9um grid)"),
                                    (None, "overlay (R=9um, G=2.4um)")]):
        if img is None:
            rgb = np.zeros(A[k].shape + (3,), np.float32)
            rgb[..., 0] = (A[k] - A[k].min()) / max(1e-6, float(np.ptp(A[k])))
            rgb[..., 1] = (Bz[k] - Bz[k].min()) / max(1e-6, float(np.ptp(Bz[k])))
            ax[0, j].imshow(rgb)
        else:
            ax[0, j].imshow(img, cmap="gray")
        ax[0, j].set_title(ttl, fontsize=9)
        ax[0, j].axis("off")
    # xz sections
    for j, (img, ttl) in enumerate([(A[:, k], "9.362um xz"), (Bz[:, k], "2.403um xz"),
                                    (None, "overlay xz")]):
        if img is None:
            rgb = np.zeros(A[:, k].shape + (3,), np.float32)
            rgb[..., 0] = (A[:, k] - A[:, k].min()) / max(1e-6, float(np.ptp(A[:, k])))
            rgb[..., 1] = (Bz[:, k] - Bz[:, k].min()) / max(1e-6, float(np.ptp(Bz[:, k])))
            ax[1, j].imshow(rgb)
        else:
            ax[1, j].imshow(img, cmap="gray")
        ax[1, j].set_title(ttl, fontsize=9)
        ax[1, j].axis("off")
    aa = A - A.mean(); bb = Bz - Bz.mean()
    r = float((aa * bb).sum() / np.sqrt((aa * aa).sum() * (bb * bb).sum()))
    fig.suptitle(f"PHerc1203 derived transform QC  |  2.4um block centred at z2={z2c} "
                 f"-> 9um z={p9[0]:.0f}   volumetric NCC={r:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qc_transform.png"), dpi=110)
    print("NCC at derived transform:", round(r, 4), "-> trackD/hunt/qc_transform.png")

    # also render the native 2.4um detail that the 9um scan cannot show
    Draw = Zarr3D(P2, 0).read(int(z2c) - 2, int(z2c) + 2,
                              int(y2c) - 400, int(y2c) + 400,
                              int(x2c) - 400, int(x2c) + 400)
    E = Zarr3D(P9, 0).read(int(p9[0]) - 1, int(p9[0]) + 1,
                           int(p9[1]) - 103, int(p9[1]) + 103,
                           int(p9[2]) - 103, int(p9[2]) + 103)
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
    ax2[0].imshow(E[0], cmap="gray"); ax2[0].set_title("9.362um level 0 (1.93 mm FOV)", fontsize=9)
    ax2[1].imshow(Draw[0], cmap="gray"); ax2[1].set_title("2.403um level 0, same place (1.92 mm FOV)", fontsize=9)
    for a in ax2:
        a.axis("off")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "qc_native_detail.png"), dpi=120)
    print("-> trackD/hunt/qc_native_detail.png")


if __name__ == "__main__":
    main()
