"""Investigation D step 5: does the 2.403 um band resolve laminae the 9.362 um scan merges?

mesh_bias_survey.py found that PHerc1203's in-band mesh patches have a median along-normal
intensity contrast of 0.138 at 9.362 um, against 0.397 for the w035 control (identical export
window, p = 2.6e-11).  If that is a *resolution* limit, the same patches should show much more
structure at 2.403 um.  If it is the *material*, they will look just as flat.  That distinction
decides whether the 2.4 um path is worth running.
"""
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import tifffile
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zarr_http import Zarr3D  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
P9 = "PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr"
P2 = "PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
RATIO = 9.362 / 2.403
T = np.array([7940.1, 19.2, -18.8])
W = 9
NS = 300              # 2.4 um sampling across the patch
SPAN_UM = 190.0       # half-span of the along-normal sweep


def fetch_tif(key):
    return tifffile.imread(io.BytesIO(requests.get(f"{B}/{key}", timeout=300).content))


def sample(vol, px, py, pz, nrm, step_um, n_off, um_per_vox):
    offs = (np.arange(n_off) - (n_off - 1) // 2) * (step_um / um_per_vox)
    pts = np.stack([pz, py, px], -1)
    nzyx = np.stack([nrm[..., 2], nrm[..., 1], nrm[..., 0]], -1)
    allp = pts[None] + offs[:, None, None, None] * nzyx[None]
    flat = allp.reshape(-1, 3)
    lo = np.maximum(0, np.floor(flat.min(0)).astype(int) - 2)
    hi = np.ceil(flat.max(0)).astype(int) + 3
    mb = np.prod(hi - lo) / 2**20
    blk = vol.read(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], workers=24).astype(np.float32)
    v = map_coordinates(blk, (flat - lo).T, order=1, mode="constant", cval=0)
    return offs * um_per_vox, v.reshape(n_off, px.shape[0], px.shape[1]), mb


def main():
    surv = json.load(open(os.path.join(OUT, "mesh_bias_survey.json")))
    surv.sort(key=lambda r: r["contrast"])
    picks = surv[:3] + surv[len(surv) // 2:len(surv) // 2 + 1]   # 3 flattest + 1 median
    v9, v2 = Zarr3D(P9, 0), Zarr3D(P2, 0)
    cat = {e["name"]: e for e in json.load(
        open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod\segment_catalog.json"))}
    cache = {}
    rows = []
    fig, axes = plt.subplots(len(picks), 3, figsize=(13, 3.4 * len(picks)))
    for row, p in enumerate(picks):
        seg = p["seg"]
        if seg not in cache:
            d = cat[seg]["seg_dir"]
            cache[seg] = tuple(fetch_tif(d + f"{a}.tif").astype(np.float64) for a in "xyz")
        X, Y, Z = cache[seg]
        i, j = p["ij"]
        u = np.linspace(0, W - 1, NS)
        gj, gi = np.meshgrid(u, u, indexing="xy")
        c = np.stack([gi.ravel(), gj.ravel()])
        px = map_coordinates(X[i:i + W, j:j + W], c, order=1).reshape(NS, NS)
        py = map_coordinates(Y[i:i + W, j:j + W], c, order=1).reshape(NS, NS)
        pz = map_coordinates(Z[i:i + W, j:j + W], c, order=1).reshape(NS, NS)
        du = np.stack([np.gradient(px, axis=0), np.gradient(py, axis=0), np.gradient(pz, axis=0)], -1)
        dv = np.stack([np.gradient(px, axis=1), np.gradient(py, axis=1), np.gradient(pz, axis=1)], -1)
        nrm = np.cross(du, dv)
        nrm /= np.maximum(1e-9, np.linalg.norm(nrm, axis=-1, keepdims=True))

        n9 = int(2 * SPAN_UM / 9.362) | 1
        o9, s9, mb9 = sample(v9, px, py, pz, nrm, 9.362, n9, 9.362)
        p2z, p2y, p2x = (pz - T[0]) * RATIO, (py - T[1]) * RATIO, (px - T[2]) * RATIO
        n2 = int(2 * SPAN_UM / 4.806) | 1        # sample every 2 voxels to keep the block small
        o2, s2, mb2 = sample(v2, p2x, p2y, p2z, nrm, 4.806, n2, 2.403)

        pr9 = s9.reshape(n9, -1).mean(1)
        pr2 = s2.reshape(n2, -1).mean(1)
        # contrast of the mean profile, and per-pixel along-normal std (structure, not just mean)
        c9 = (pr9.max() - pr9.min()) / max(1e-6, pr9.max())
        c2 = (pr2.max() - pr2.min()) / max(1e-6, pr2.max())
        sd9 = float(s9.std(0).mean() / max(1e-6, s9.mean()))
        sd2 = float(s2.std(0).mean() / max(1e-6, s2.mean()))
        rows.append(dict(seg=seg, ij=[i, j], z=float(pz.mean()), mb=[mb9, mb2],
                         contrast_9=float(c9), contrast_2=float(c2),
                         along_normal_relstd_9=sd9, along_normal_relstd_2=sd2,
                         survey_contrast_9=p["contrast"]))
        print(f"{seg[-8:]} ij=({i},{j}) z={pz.mean():.0f}: mean-profile contrast "
              f"9um {c9:.3f} -> 2.4um {c2:.3f} | per-pixel along-normal rel-std "
              f"9um {sd9:.3f} -> 2.4um {sd2:.3f}  ({mb9:.0f}+{mb2:.0f} MiB)", flush=True)

        ax = axes[row] if len(picks) > 1 else axes
        ax[0].plot(o9, pr9, "o-", ms=3, label=f"9.362 um (c={c9:.2f})")
        ax[0].plot(o2, pr2, "s-", ms=2, label=f"2.403 um (c={c2:.2f})")
        ax[0].axvline(0, color="k", lw=0.8, ls="--")
        ax[0].set_xlabel("offset along normal (um)"); ax[0].legend(fontsize=7)
        ax[0].set_title(f"{seg[-8:]} z={pz.mean():.0f}", fontsize=8)
        ax[1].imshow(s9[n9 // 2], cmap="gray"); ax[1].set_title("surface @ 9.362 um", fontsize=8); ax[1].axis("off")
        ax[2].imshow(s2[n2 // 2], cmap="gray"); ax[2].set_title("surface @ 2.403 um", fontsize=8); ax[2].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "contrast_2um_vs_9um.png"), dpi=110)
    json.dump(rows, open(os.path.join(OUT, "contrast_2um_vs_9um.json"), "w"), indent=1)
    a = np.array([[r["contrast_9"], r["contrast_2"]] for r in rows])
    b = np.array([[r["along_normal_relstd_9"], r["along_normal_relstd_2"]] for r in rows])
    print(f"\nmean-profile contrast   9um {a[:,0].mean():.3f} -> 2.4um {a[:,1].mean():.3f}")
    print(f"along-normal rel-std    9um {b[:,0].mean():.3f} -> 2.4um {b[:,1].mean():.3f}")
    print("-> trackD/hunt/contrast_2um_vs_9um.png")


if __name__ == "__main__":
    main()
