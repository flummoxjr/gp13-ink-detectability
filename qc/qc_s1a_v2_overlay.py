"""QC s1a-v2 attack 5 — visual overlay. For tiles (9,9) and (8,10):
  col 1: render at tile's peak k with letter contours (red)
  col 2: same render with rot180-null contours (cyan)
  col 3: bg-trend-removed residual image with letter contours
  col 4: zoom on largest letter cluster (render, contours)
Writes qc_s1a_v2_overlay.png
"""
import os

import numpy as np
from scipy.ndimage import gaussian_filter, label

QCD = r"D:\vesuvius-data\trackD\w032\qc"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc"
K_OFFSETS = list(range(-4, 5))
KI = {k: i for i, k in enumerate(K_OFFSETS)}
TREND_SIGMA = 48.0

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(ty, tx):
    d = np.load(os.path.join(QCD, f"tile_v2_{ty}_{tx}.npz"))
    t = {k: d[k] for k in ("sub_ink", "s11", "dist_nd", "ok_pre", "raw0", "vals")}
    t["letters"] = (t["sub_ink"] >= 200) & (t["s11"] >= 150) & (t["dist_nd"] >= 8)
    t["bg"] = (t["sub_ink"] >= 28) & (t["sub_ink"] <= 60) & (t["s11"] <= 60) & \
              (t["dist_nd"] >= 8)
    t["good0"] = t["ok_pre"] & (t["raw0"] > 5)
    return t


def main():
    picks = [((9, 9), -1), ((8, 10), 0)]
    fig, axes = plt.subplots(len(picks), 4, figsize=(21, 5.5 * len(picks)),
                             squeeze=False)
    for r, ((ty, tx), kq) in enumerate(picks):
        t = load(ty, tx)
        v = t["vals"][KI[kq]]
        good = t["good0"] & np.isfinite(v)
        img = np.where(good, v, np.nan)
        fin = img[np.isfinite(img)]
        v0, v1 = np.percentile(fin, [2, 98])
        rot = t["letters"][::-1, ::-1]

        ax = axes[r][0]
        ax.imshow(img, cmap="gray", vmin=v0, vmax=v1)
        ax.contour(t["letters"], levels=[0.5], colors="red", linewidths=0.7)
        ax.set_title(f"({ty},{tx}) k={kq} + letters (red)")

        ax = axes[r][1]
        ax.imshow(img, cmap="gray", vmin=v0, vmax=v1)
        ax.contour(rot, levels=[0.5], colors="cyan", linewidths=0.7)
        ax.set_title(f"({ty},{tx}) k={kq} + rot180 null (cyan)")

        # residual
        Bk = t["bg"] & good
        w = Bk.astype(np.float32)
        vv = np.where(Bk, v, 0).astype(np.float32)
        den = gaussian_filter(w, TREND_SIGMA)
        trend = gaussian_filter(vv, TREND_SIGMA) / (den + 1e-9)
        resid = np.where(good & (den > 1e-3), v - trend, np.nan)
        rfin = resid[np.isfinite(resid)]
        r0, r1_ = np.percentile(rfin, [2, 98])
        ax = axes[r][2]
        ax.imshow(resid, cmap="gray", vmin=r0, vmax=r1_)
        ax.contour(t["letters"], levels=[0.5], colors="red", linewidths=0.7)
        ax.set_title("bg-trend removed residual + letters")

        # zoom on largest letter cluster
        lab, nl = label(t["letters"])
        if nl:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            big = sizes.argmax()
            ys, xs = np.where(lab == big)
            cy, cx = int(ys.mean()), int(xs.mean())
            hw = 90
            y0z, y1z = max(cy - hw, 0), min(cy + hw, 512)
            x0z, x1z = max(cx - hw, 0), min(cx + hw, 512)
            ax = axes[r][3]
            ax.imshow(img[y0z:y1z, x0z:x1z], cmap="gray", vmin=v0, vmax=v1)
            ax.contour(t["letters"][y0z:y1z, x0z:x1z], levels=[0.5],
                       colors="red", linewidths=0.9)
            ax.set_title(f"zoom largest cluster ({sizes[big]} px)")
        for c in range(4):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qc_s1a_v2_overlay.png"), dpi=110)
    print("wrote qc_s1a_v2_overlay.png")


if __name__ == "__main__":
    main()
