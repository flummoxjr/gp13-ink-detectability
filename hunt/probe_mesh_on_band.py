"""Investigation D step 4: does a transformed PHerc1203 mesh land ON the sheet at 2.4 um?

Takes a small patch of one in-band segment, maps it through the derived transform, samples the
2.403 um volume along the surface normal, and reports the along-normal intensity profile.  A
mesh that sits on the recto surface gives a profile peaked at offset 0 with a width of roughly
one sheet; a mesh that straddles or misses gives a flat or double-peaked profile.  The same
patch is sampled in the 9.362 um volume as a reference.
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
T = np.array([7940.1, 19.2, -18.8])          # (z, y, x) level-0 9 um voxels
SEG = "PHerc1203/segments/raw/auto_grown_20250925223153537/"   # 78.4% in band
NOFF = 41                                    # offsets -20..+20


def fetch_tif(key):
    return tifffile.imread(io.BytesIO(requests.get(f"{B}/{key}", timeout=180).content))


def main():
    meta = json.loads(requests.get(f"{B}/{SEG}meta.json", timeout=60).text)
    sc = meta["scale"][0]
    X = fetch_tif(SEG + "x.tif").astype(np.float64)
    Y = fetch_tif(SEG + "y.tif").astype(np.float64)
    Z = fetch_tif(SEG + "z.tif").astype(np.float64)
    ok = (X > 0) & (Y > 0) & (Z > 0)
    print("stored grid", X.shape, "scale", sc, "-> one cell =", 1 / sc, "volume voxels at 9 um")

    # pick a compact valid window whose z sits inside the band 7940..11825
    band = ok & (Z > 8200) & (Z < 11500)
    W = 9                                       # 9x9 stored cells ~ 1.7 mm
    best, bz = None, None
    for i in range(0, X.shape[0] - W):
        for j in range(0, X.shape[1] - W):
            blk = band[i:i + W, j:j + W]
            if blk.all():
                zc = Z[i:i + W, j:j + W].mean()
                if best is None or abs(zc - 9800) < abs(bz - 9800):
                    best, bz = (i, j), zc
    if best is None:
        print("no fully valid in-band window found")
        return
    i, j = best
    print(f"window at stored ({i},{j}) size {W}x{W}, mean 9um z = {bz:.0f}")

    # upsample the patch to 2.403 um sampling density
    n = int(round((W - 1) / sc * RATIO))        # samples across the patch at 2.4 um pitch
    n = min(n, 420)
    u = np.linspace(0, W - 1, n)
    gj, gi = np.meshgrid(u, u, indexing="xy")
    coords = np.stack([gi.ravel(), gj.ravel()])
    px = map_coordinates(X[i:i + W, j:j + W], coords, order=1).reshape(n, n)
    py = map_coordinates(Y[i:i + W, j:j + W], coords, order=1).reshape(n, n)
    pz = map_coordinates(Z[i:i + W, j:j + W], coords, order=1).reshape(n, n)
    print(f"patch {n}x{n} samples, 9um extent x {px.min():.0f}-{px.max():.0f} "
          f"y {py.min():.0f}-{py.max():.0f} z {pz.min():.0f}-{pz.max():.0f}")

    # surface normal from the parameterisation (points are (x, y, z) in 9 um voxels)
    du = np.stack(np.gradient(np.stack([px, py, pz], -1), axis=0), -1)[..., 0]
    dv = np.stack(np.gradient(np.stack([px, py, pz], -1), axis=1), -1)[..., 0]
    du = np.stack(np.gradient(px, axis=0) * 0 + [np.gradient(px, axis=0), np.gradient(py, axis=0),
                                                 np.gradient(pz, axis=0)], -1)
    dv = np.stack([np.gradient(px, axis=1), np.gradient(py, axis=1), np.gradient(pz, axis=1)], -1)
    nrm = np.cross(du, dv)
    nrm /= np.maximum(1e-9, np.linalg.norm(nrm, axis=-1, keepdims=True))

    offs = np.arange(NOFF) - (NOFF - 1) // 2     # in voxels of the target volume

    results = {}
    for tag, path, vox in (("9.362um", P9, 1.0), ("2.403um", P2, 1.0)):
        # sample points in the target volume's voxel coords
        if tag == "9.362um":
            pxx, pyy, pzz = px, py, pz
            nn = nrm
        else:
            pzz = (pz - T[0]) * RATIO
            pyy = (py - T[1]) * RATIO
            pxx = (px - T[2]) * RATIO
            nn = nrm                              # isotropic scale -> normal unchanged
        pts = np.stack([pzz, pyy, pxx], -1)       # (n, n, 3) in (z, y, x)
        nzyx = np.stack([nn[..., 2], nn[..., 1], nn[..., 0]], -1)
        all_pts = pts[None] + offs[:, None, None, None] * nzyx[None]
        lo = np.floor(all_pts.reshape(-1, 3).min(0)).astype(int) - 2
        hi = np.ceil(all_pts.reshape(-1, 3).max(0)).astype(int) + 3
        z = Zarr3D(path, 0)
        vol_mb = np.prod(hi - lo) / 2**20
        print(f"{tag}: reading block {tuple(hi-lo)} = {vol_mb:.0f} MiB of voxels")
        blk = z.read(max(0, lo[0]), hi[0], max(0, lo[1]), hi[1], max(0, lo[2]), hi[2],
                     workers=16).astype(np.float32)
        c = all_pts - np.array([max(0, lo[0]), max(0, lo[1]), max(0, lo[2])])
        vals = map_coordinates(blk, c.reshape(-1, 3).T, order=1, mode="constant", cval=0)
        vals = vals.reshape(NOFF, n, n)
        prof = vals.reshape(NOFF, -1).mean(1)
        results[tag] = dict(profile=prof.tolist(), offsets=offs.tolist(),
                            peak_offset=int(offs[int(np.argmax(prof))]),
                            peak=float(prof.max()), at_zero=float(prof[(NOFF - 1) // 2]),
                            um_per_offset=9.362 if tag == "9.362um" else 2.403)
        print(f"{tag}: profile peak at offset {offs[int(np.argmax(prof))]:+d} "
              f"({offs[int(np.argmax(prof))] * (9.362 if tag=='9.362um' else 2.403):+.0f} um), "
              f"peak {prof.max():.1f}, value at 0 {prof[(NOFF-1)//2]:.1f}, min {prof.min():.1f}")
        results[tag]["face"] = vals[(NOFF - 1) // 2].tolist() if n <= 64 else None
        np.save(os.path.join(OUT, f"probe_face_{tag}.npy"), vals)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for tag, style in (("9.362um", "o-"), ("2.403um", "s-")):
        r = results[tag]
        ax[0].plot(np.array(r["offsets"]) * r["um_per_offset"], r["profile"], style,
                   ms=3, label=f"{tag} (peak {r['peak_offset']*r['um_per_offset']:+.0f} um)")
    ax[0].axvline(0, color="k", lw=0.8, ls="--")
    ax[0].set_xlabel("offset along surface normal (um)")
    ax[0].set_ylabel("mean intensity")
    ax[0].set_title("along-normal profile of the transformed mesh")
    ax[0].legend(fontsize=8)
    v9 = np.load(os.path.join(OUT, "probe_face_9.362um.npy"))
    v2 = np.load(os.path.join(OUT, "probe_face_2.403um.npy"))
    ax[1].imshow(v9[(NOFF - 1) // 2], cmap="gray"); ax[1].set_title("surface at 9.362 um"); ax[1].axis("off")
    ax[2].imshow(v2[(NOFF - 1) // 2], cmap="gray"); ax[2].set_title("same surface at 2.403 um"); ax[2].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "probe_mesh_on_band.png"), dpi=110)
    json.dump(results, open(os.path.join(OUT, "probe_mesh_on_band.json"), "w"), indent=1)
    print("-> trackD/hunt/probe_mesh_on_band.png")


if __name__ == "__main__":
    main()
