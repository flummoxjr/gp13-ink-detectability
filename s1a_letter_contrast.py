"""Stage 1a — model-free letter-contrast test on PHerc1667 w032.

At coordinates where the June-2026 2.4um read PROVES letters exist, does the
legacy 7.91um scan (the resolution class of all 13 GP scrolls) show ANY
statistical intensity/texture signal? Pure measurement, no ML.

Method:
  - letter mask from the released 2.4um-model ink TIF (ds4 ~9.6um/px),
    strong-ink threshold, high-confidence = also confirmed by the 1.129um-model TIF
  - control mask: near-letter background on the same papyrus (valid grid, low ink)
  - map uv -> 7.91um voxels via the released on-7.91um tifxyz grid (bilinear),
    surface normals from grid derivatives
  - sample the 7.91um volume (trilinear, bbox-fetched per tile) at offsets
    k = -4..+4 voxels along the normal
  - per-tile + pooled AUC (tie-corrected) letters vs control, per depth offset;
    plus uv-rendered images for eyeballing

Outputs: trackD/out/s1a_w032_stats.json, s1a_w032_tiles.png
"""
import json
import os

import numpy as np
import requests
import tifffile
from scipy.ndimage import map_coordinates, binary_dilation, gaussian_filter

CACHE = r"D:\vesuvius-data\trackD\w032"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"
VOL_URL = "https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr"
VOL_SHAPE = (11173, 3340, 3440)  # z,y,x
WIN = (-0.03, 0.145)

INK_SCALE = 4 * 2.399 / 7.91          # ds4 ink px -> uv(7.91um px)
GRID_SCALE = 0.05                      # uv(7.91um px) -> grid px
TILE = 512                             # ds4 px (~4.9mm)
N_TILES = 6
K_OFFSETS = list(range(-4, 5))
STRONG_INK = 170
WEAK_BG = 20


def tie_auc(pos, neg, max_n=300_000):
    rng = np.random.default_rng(1)
    if len(pos) > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if len(neg) > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    from scipy.stats import rankdata
    allv = np.concatenate([pos, neg])
    ranks = rankdata(allv)  # average ranks on ties
    n1, n2 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


class BboxVolume:
    """Fetch a zarr bbox once per tile over HTTPS (chunk-level, threaded)."""

    def __init__(self, url, shape, chunks=(128, 128, 128)):
        self.url, self.shape, self.chunks = url, shape, chunks
        self.sess = requests.Session()

    def fetch_bbox(self, z0, z1, y0, y1, x0, x1):
        cz, cy, cx = self.chunks
        z0, y0, x0 = max(z0, 0), max(y0, 0), max(x0, 0)
        z1, y1, x1 = min(z1, self.shape[0]), min(y1, self.shape[1]), min(x1, self.shape[2])
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        jobs = []
        for iz in range(z0 // cz, (z1 - 1) // cz + 1):
            for iy in range(y0 // cy, (y1 - 1) // cy + 1):
                for ix in range(x0 // cx, (x1 - 1) // cx + 1):
                    jobs.append((iz, iy, ix))

        def get(job):
            iz, iy, ix = job
            r = self.sess.get(f"{self.url}/0/{iz}/{iy}/{ix}", timeout=120)
            if r.status_code == 404:
                return job, None
            r.raise_for_status()
            return job, np.frombuffer(r.content, np.uint8).reshape(self.chunks)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(16) as ex:
            for job, chunk in ex.map(get, jobs):
                if chunk is None:
                    continue
                iz, iy, ix = job
                zs, ys, xs = iz * 128, iy * 128, ix * 128
                sz0, sy0, sx0 = max(zs, z0), max(ys, y0), max(xs, x0)
                sz1, sy1, sx1 = min(zs + 128, z1), min(ys + 128, y1), min(xs + 128, x1)
                out[sz0 - z0:sz1 - z0, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                    chunk[sz0 - zs:sz1 - zs, sy0 - ys:sy1 - ys, sx0 - xs:sx1 - xs]
        return out, (z0, y0, x0)


def main():
    os.makedirs(OUT, exist_ok=True)
    ink = np.load(os.path.join(CACHE, "ink24_ds4.npy"))
    print("ink ds4:", ink.shape)
    gx = tifffile.imread(os.path.join(CACHE, "grid_x.tif"))
    gy = tifffile.imread(os.path.join(CACHE, "grid_y.tif"))
    gz = tifffile.imread(os.path.join(CACHE, "grid_z.tif"))
    gvalid = (gx > 0) & (gy > 0) & (gz > 0)

    # grid coords for every ds4 ink px (vectorized per tile later)
    H, W = ink.shape

    # tile ranking: want LETTER-LIKE tiles (moderate ink fraction — saturated
    # blobs are model garbage over the segment's empty margin), with both
    # strong-ink and clean-background pixels present, on a valid grid patch.
    n_ty, n_tx = H // TILE, W // TILE
    scores = np.zeros((n_ty, n_tx))
    for ty in range(n_ty):
        for tx in range(n_tx):
            sub = ink[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE]
            ink_frac = (sub > STRONG_INK).mean()
            bg_frac = (sub < WEAK_BG).mean()
            gy0 = int(ty * TILE * INK_SCALE * GRID_SCALE)
            gy1 = int((ty + 1) * TILE * INK_SCALE * GRID_SCALE) + 1
            gx0 = int(tx * TILE * INK_SCALE * GRID_SCALE)
            gx1 = int((tx + 1) * TILE * INK_SCALE * GRID_SCALE) + 1
            gpatch = gvalid[gy0:gy1, gx0:gx1]
            if gpatch.size == 0 or gpatch.mean() < 0.98:
                continue
            if 0.02 <= ink_frac <= 0.30 and bg_frac >= 0.30:
                scores[ty, tx] = ink_frac
    order = np.dstack(np.unravel_index(np.argsort(scores.ravel())[::-1], scores.shape))[0]
    tiles = [tuple(t) for t in order[:N_TILES * 3] if scores[tuple(t)] > 0.002]
    print("candidate tiles:", tiles[:12], "scores:", [round(float(scores[t]), 4) for t in tiles[:12]])

    vol = BboxVolume(VOL_URL, VOL_SHAPE)
    lo, hi = WIN
    results = {"tiles": [], "pooled": {}}
    pooled = {k: {"pos": [], "neg": []} for k in K_OFFSETS}
    renders = []

    for (ty, tx) in tiles:
        if len(results["tiles"]) >= N_TILES:
            break
        y0d, x0d = ty * TILE, tx * TILE
        sub_ink = ink[y0d:y0d + TILE, x0d:x0d + TILE]
        letters = sub_ink > STRONG_INK
        background = (sub_ink < WEAK_BG) & binary_dilation(letters, iterations=60)

        # uv(ds4) -> grid px (float)
        vv, uu = np.mgrid[0:TILE, 0:TILE]
        gpy = (vv + y0d) * INK_SCALE * GRID_SCALE
        gpx = (uu + x0d) * INK_SCALE * GRID_SCALE
        X = map_coordinates(gx, [gpy, gpx], order=1)
        Y = map_coordinates(gy, [gpy, gpx], order=1)
        Z = map_coordinates(gz, [gpy, gpx], order=1)
        vmask = map_coordinates(gvalid.astype(np.float32), [gpy, gpx], order=1) > 0.99
        ok = vmask & (X > 1) & (Y > 1) & (Z > 1)

        # normals from grid derivatives (in 7.91um voxel units)
        dXu = map_coordinates(np.gradient(gx, axis=1), [gpy, gpx], order=1)
        dYu = map_coordinates(np.gradient(gy, axis=1), [gpy, gpx], order=1)
        dZu = map_coordinates(np.gradient(gz, axis=1), [gpy, gpx], order=1)
        dXv = map_coordinates(np.gradient(gx, axis=0), [gpy, gpx], order=1)
        dYv = map_coordinates(np.gradient(gy, axis=0), [gpy, gpx], order=1)
        dZv = map_coordinates(np.gradient(gz, axis=0), [gpy, gpx], order=1)
        nx = dYu * dZv - dZu * dYv
        ny = dZu * dXv - dXu * dZv
        nz = dXu * dYv - dYu * dXv
        norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-9
        nx, ny, nz = nx / norm, ny / norm, nz / norm

        zmin, zmax = int(Z[ok].min()) - 8, int(Z[ok].max()) + 9
        ymin, ymax = int(Y[ok].min()) - 8, int(Y[ok].max()) + 9
        xmin, xmax = int(X[ok].min()) - 8, int(X[ok].max()) + 9
        gb = (zmax - zmin) * (ymax - ymin) * (xmax - xmin) / 1e9
        print(f"tile ({ty},{tx}): letters={int(letters.sum())} bg={int(background.sum())} "
              f"bbox {zmax - zmin}x{ymax - ymin}x{xmax - xmin} = {gb:.2f} GB")
        if gb > 1.2:
            print("  bbox too large, skipping tile (likely seam)")
            continue
        sub, (oz, oy, ox) = vol.fetch_bbox(zmin, zmax, ymin, ymax, xmin, xmax)
        subf = sub.astype(np.float32) / 255.0 * (hi - lo) + lo

        # on-papyrus check: sample the surface (k=0) and require mostly nonzero voxels
        pz0, py0, px0 = Z - oz, Y - oy, X - ox
        raw0 = map_coordinates(sub.astype(np.float32), [pz0, py0, px0], order=1,
                               mode="constant", cval=0.0)
        onpap = float((raw0[ok] > 5).mean()) if ok.sum() else 0.0
        print(f"  on-papyrus fraction of mapped surface: {onpap:.2f}")
        if onpap < 0.7:
            print("  tile maps mostly off-papyrus, skipping")
            continue
        ok = ok & (raw0 > 5)

        tile_res = {"tile": [int(ty), int(tx)], "letters_px": int(letters.sum()),
                    "bg_px": int(background.sum()), "onpap_frac": round(onpap, 3),
                    "auc_by_k": {}}
        render_k0 = np.full((TILE, TILE), np.nan, np.float32)
        for k in K_OFFSETS:
            pz, py, px = Z + nz * k - oz, Y + ny * k - oy, X + nx * k - ox
            vals = map_coordinates(subf, [pz, py, px], order=1, mode="constant", cval=np.nan)
            if k == 0:
                render_k0 = np.where(ok, vals, np.nan)
            good = ok & np.isfinite(vals)
            pos = vals[letters & good]
            neg = vals[background & good]
            if len(pos) > 100 and len(neg) > 100:
                a = tie_auc(pos, neg)
                tile_res["auc_by_k"][k] = round(a, 4)
                pooled[k]["pos"].append(pos)
                pooled[k]["neg"].append(neg)
        results["tiles"].append(tile_res)
        print("  ", tile_res["auc_by_k"])
        renders.append(((ty, tx), sub_ink.copy(), render_k0))

    for k in K_OFFSETS:
        if pooled[k]["pos"]:
            a = tie_auc(np.concatenate(pooled[k]["pos"]), np.concatenate(pooled[k]["neg"]))
            results["pooled"][k] = round(a, 4)
    print("POOLED AUC by depth offset:", results["pooled"])

    with open(os.path.join(OUT, "s1a_w032_stats.json"), "w") as fh:
        json.dump(results, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(renders)
    if n:
        fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 9), squeeze=False)
        for i, ((ty, tx), tink, rk0) in enumerate(renders):
            axes[0][i].imshow(tink, cmap="inferno")
            axes[0][i].set_title(f"tile({ty},{tx}) ink@2.4um")
            v = rk0[np.isfinite(rk0)]
            if len(v):
                v0, v1 = np.percentile(v, [2, 98])
                axes[1][i].imshow(rk0, cmap="gray", vmin=v0, vmax=v1)
            axes[1][i].set_title("7.91um render k=0")
            for ax in (axes[0][i], axes[1][i]):
                ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "s1a_w032_tiles.png"), dpi=95)
        print("wrote s1a_w032_tiles.png")


if __name__ == "__main__":
    main()
