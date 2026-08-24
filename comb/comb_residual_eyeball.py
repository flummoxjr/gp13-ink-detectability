r"""Comb 2 — human-review gallery of the top unexplained-RESIDUAL tiles.

From salvage/tiles.parquet: recompute the M1 nuisance residual for f05
(f05 ~ fill + meanct_mat + z + rnorm, standardized OLS — matches
salvage/analysis.py; stored R2 = 0.6207, reproduced here as a gate).
The ~36%-unexplained residual carries real spatial structure (Moran z=+31);
nobody has LOOKED at what sits in the high-residual tiles. Take the top-20
positive-residual tiles (model fires above covariate prediction), spatially
deduplicated (Chebyshev >= 2 tiles), stream each tile's central CT slice at L1
from the PHerc1203 2.4um band volume, and render a pure look-and-see gallery.

Output: comb/residual_eyeball.png, comb/residual_picks.json
"""
import json
import os

import numpy as np
import pandas as pd

SAL = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb"
CACHE = os.path.join(OUT, "cache_residual")
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
VOL = f"{BUCKET}/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
N_PICK = 20
MIN_SEP = 2   # tiles, Chebyshev, in (ti,tj,tk)
CTX = 128     # L1 px on each side of tile center -> 256^2 field (1.23 mm)

os.makedirs(CACHE, exist_ok=True)


def main():
    df = pd.read_parquet(os.path.join(SAL, "tiles.parquet"))
    sc = df[~df.skipped].copy()
    sc["ti"], sc["tj"], sc["tk"] = sc.z // 256, sc.y // 256, sc.x // 256
    sc = sc.dropna(subset=["meanct_mat", "rnorm", "f05", "f08", "fill"]).reset_index(drop=True)

    def zs(a):
        return (a - a.mean()) / a.std()

    X = np.column_stack([zs(sc.fill.values), zs(sc.meanct_mat.values),
                         zs(sc.z.values.astype(float)), zs(sc.rnorm.values)])
    X1 = np.column_stack([np.ones(len(X)), X])
    y = sc.f05.values
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    fit = X1 @ beta
    r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"M1 R2 = {r2:.4f} (stored: 0.6207)")
    assert abs(r2 - 0.6207) < 0.005, "regression does not reproduce stored M1"
    sc["resid"] = y - fit

    # top positive residuals, tile-space NMS
    order = np.argsort(sc.resid.values)[::-1]
    picks = []
    for i in order:
        t = sc.iloc[i]
        if all(max(abs(t.ti - p.ti), abs(t.tj - p.tj), abs(t.tk - p.tk)) >= MIN_SEP
               for p in picks):
            picks.append(t)
        if len(picks) >= N_PICK:
            break
    print(f"picked {len(picks)} tiles; resid range "
          f"{picks[-1].resid:.4f}..{picks[0].resid:.4f}")

    meta = [{"z": int(t.z), "y": int(t.y), "x": int(t.x),
             "resid": round(float(t.resid), 4), "f05": round(float(t.f05), 4),
             "pmax": float(t.pmax), "meanct": round(float(t.meanct), 1),
             "fill": float(t.fill), "rnorm": round(float(t.rnorm), 3),
             "theta": round(float(t.theta), 2)} for t in picks]
    with open(os.path.join(OUT, "residual_picks.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # ---- stream L1 central slices ----
    import zarr
    import fsspec
    z1 = zarr.open(fsspec.get_mapper(VOL), mode="r")["1"]
    print("L1 shape:", z1.shape)

    slices = []
    for k, t in enumerate(picks):
        cpath = os.path.join(CACHE, f"t{int(t.z)}_{int(t.y)}_{int(t.x)}.npy")
        if os.path.exists(cpath):
            slices.append(np.load(cpath))
            continue
        zc = (int(t.z) + 128) // 2
        yc = (int(t.y) + 128) // 2
        xc = (int(t.x) + 128) // 2
        y0, x0 = max(yc - CTX, 0), max(xc - CTX, 0)
        a = np.asarray(z1[zc, y0:y0 + 2 * CTX, x0:x0 + 2 * CTX])
        np.save(cpath, a)
        slices.append(a)
        print(f"  fetched {k + 1}/{len(picks)} L1 z={zc}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.7 * rows),
                             squeeze=False)
    for k, (t, a) in enumerate(zip(picks, slices)):
        ax = axes[k // cols][k % cols]
        ax.imshow(a, cmap="gray", vmin=0, vmax=255)
        # tile footprint: central 128x128 L1 px of the 256^2 context
        ax.add_patch(Rectangle((CTX - 64, CTX - 64), 128, 128, fill=False,
                               edgecolor="orange", linewidth=1.0))
        ax.set_title(f"L0({int(t.z)},{int(t.y)},{int(t.x)})\n"
                     f"resid={t.resid:+.3f} f05={t.f05:.3f} "
                     f"ct={t.meanct:.0f} rn={t.rnorm:.2f}", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Top-20 unexplained-residual tiles — PHerc1203 2.4um band, "
                 "central CT slice at L1 (field 1.23 mm, orange = scored tile)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "residual_eyeball.png"), dpi=110)
    print("wrote residual_eyeball.png")


if __name__ == "__main__":
    main()
