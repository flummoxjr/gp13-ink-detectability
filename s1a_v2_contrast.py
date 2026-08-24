"""S1a-v2 — letter-contrast test on w032 with QC-verified masks and built-in nulls.

Fixes vs v1 (per qc/s1a_verification.md — v1 REFUTED):
  - letters: dual-model confirmed (ink24 >= 200 AND resampled ink11 >= 150)
  - background: ON-PREDICTION band ink24 in [28,60] AND ink11 <= 60,
    both >= 8 ds4-px from any no-data (ink24==0) pixel
  - tiles: dual-coverage candidates (rows 8-15, cols 8-14), gallery-verified text region
  - placebo nulls reported alongside: rot180 mask + (+64,+64) shifted mask
  - per-tile AUC + pooled + null distribution; depth offsets k=-4..+4

Outputs: trackD/out/s1a_v2_stats.json, s1a_v2_tiles.png
"""
import json
import os

import numpy as np
import requests
import tifffile
from scipy.ndimage import map_coordinates, distance_transform_edt, zoom

CACHE = r"D:\vesuvius-data\trackD\w032"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"
VOL_URL = "https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr"
VOL_SHAPE = (11173, 3340, 3440)
WIN = (-0.03, 0.145)
INK_SCALE = 4 * 2.399 / 7.91
GRID_SCALE = 0.05
TILE = 512
K_OFFSETS = list(range(-4, 5))

TILES = [(10, 11), (12, 11), (10, 10), (10, 12), (11, 12), (8, 10), (15, 13), (9, 9)]


def tie_auc(pos, neg, max_n=300_000):
    from scipy.stats import rankdata
    rng = np.random.default_rng(1)
    if len(pos) > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if len(neg) > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    allv = np.concatenate([pos, neg])
    ranks = rankdata(allv)
    n1, n2 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


class BboxVolume:
    def __init__(self, url, shape, chunks=(128, 128, 128)):
        self.url, self.shape, self.chunks = url, shape, chunks
        self.sess = requests.Session()

    def fetch_bbox(self, z0, z1, y0, y1, x0, x1):
        cz, cy, cx = self.chunks
        z0, y0, x0 = max(z0, 0), max(y0, 0), max(x0, 0)
        z1, y1, x1 = min(z1, self.shape[0]), min(y1, self.shape[1]), min(x1, self.shape[2])
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        jobs = [(iz, iy, ix)
                for iz in range(z0 // cz, (z1 - 1) // cz + 1)
                for iy in range(y0 // cy, (y1 - 1) // cy + 1)
                for ix in range(x0 // cx, (x1 - 1) // cx + 1)]

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
                a0, b0, c0 = max(zs, z0), max(ys, y0), max(xs, x0)
                a1, b1, c1 = min(zs + 128, z1), min(ys + 128, y1), min(xs + 128, x1)
                out[a0 - z0:a1 - z0, b0 - y0:b1 - y0, c0 - x0:c1 - x0] = \
                    chunk[a0 - zs:a1 - zs, b0 - ys:b1 - ys, c0 - xs:c1 - xs]
        return out, (z0, y0, x0)


def main():
    os.makedirs(OUT, exist_ok=True)
    ink24 = np.load(os.path.join(CACHE, "ink24_ds4.npy"))
    ink11 = np.load(os.path.join(CACHE, "ink11_ds4.npy"))
    zf = 2.258 / 2.399
    ink11r = zoom(ink11, zf, order=1)
    ink11a = np.zeros_like(ink24)
    h = min(ink11r.shape[0], ink24.shape[0]); w = min(ink11r.shape[1], ink24.shape[1])
    ink11a[:h, :w] = ink11r[:h, :w]

    gx = tifffile.imread(os.path.join(CACHE, "grid_x.tif"))
    gy = tifffile.imread(os.path.join(CACHE, "grid_y.tif"))
    gz = tifffile.imread(os.path.join(CACHE, "grid_z.tif"))
    gvalid = (gx > 0) & (gy > 0) & (gz > 0)

    dist_ok = distance_transform_edt(ink24 > 0) >= 8

    vol = BboxVolume(VOL_URL, VOL_SHAPE)
    lo, hi = WIN
    results = {"tiles": [], "pooled": {}, "null": {}}
    pooled = {k: {"pos": [], "neg": []} for k in K_OFFSETS}
    null_pool = {"rot": {k: {"pos": [], "neg": []} for k in K_OFFSETS},
                 "shift": {k: {"pos": [], "neg": []} for k in K_OFFSETS}}
    renders = []

    for (ty, tx) in TILES:
        y0d, x0d = ty * TILE, tx * TILE
        s24 = ink24[y0d:y0d + TILE, x0d:x0d + TILE]
        s11 = ink11a[y0d:y0d + TILE, x0d:x0d + TILE]
        dk = dist_ok[y0d:y0d + TILE, x0d:x0d + TILE]
        letters = (s24 >= 200) & (s11 >= 150) & dk
        background = (s24 >= 28) & (s24 <= 60) & (s11 <= 60) & dk
        if letters.sum() < 800 or background.sum() < 3000:
            print(f"tile ({ty},{tx}): too few px (let {letters.sum()}, bg {background.sum()}), skip")
            continue

        vv, uu = np.mgrid[0:TILE, 0:TILE]
        gpy = (vv + y0d) * INK_SCALE * GRID_SCALE
        gpx = (uu + x0d) * INK_SCALE * GRID_SCALE
        X = map_coordinates(gx, [gpy, gpx], order=1)
        Y = map_coordinates(gy, [gpy, gpx], order=1)
        Z = map_coordinates(gz, [gpy, gpx], order=1)
        vmask = map_coordinates(gvalid.astype(np.float32), [gpy, gpx], order=1) > 0.99
        ok = vmask & (X > 1) & (Y > 1) & (Z > 1)
        if ok.mean() < 0.5:
            print(f"tile ({ty},{tx}): grid mostly invalid, skip")
            continue

        dXu = map_coordinates(np.gradient(gx, axis=1), [gpy, gpx], order=1)
        dYu = map_coordinates(np.gradient(gy, axis=1), [gpy, gpx], order=1)
        dZu = map_coordinates(np.gradient(gz, axis=1), [gpy, gpx], order=1)
        dXv = map_coordinates(np.gradient(gx, axis=0), [gpy, gpx], order=1)
        dYv = map_coordinates(np.gradient(gy, axis=0), [gpy, gpx], order=1)
        dZv = map_coordinates(np.gradient(gz, axis=0), [gpy, gpx], order=1)
        nx = dYu * dZv - dZu * dYv
        ny = dZu * dXv - dXu * dZv
        nz_ = dXu * dYv - dYu * dXv
        norm = np.sqrt(nx ** 2 + ny ** 2 + nz_ ** 2) + 1e-9
        nx, ny, nz_ = nx / norm, ny / norm, nz_ / norm

        sel = ok
        zmin, zmax = int(Z[sel].min()) - 8, int(Z[sel].max()) + 9
        ymin, ymax = int(Y[sel].min()) - 8, int(Y[sel].max()) + 9
        xmin, xmax = int(X[sel].min()) - 8, int(X[sel].max()) + 9
        gb = (zmax - zmin) * (ymax - ymin) * (xmax - xmin) / 1e9
        print(f"tile ({ty},{tx}): let={int(letters.sum())} bg={int(background.sum())} bbox {gb:.2f} GB")
        if gb > 1.2:
            print("  bbox too large, skip")
            continue
        sub, (oz, oy, ox) = vol.fetch_bbox(zmin, zmax, ymin, ymax, xmin, xmax)
        subf = sub.astype(np.float32) / 255.0 * (hi - lo) + lo

        raw0 = map_coordinates(sub.astype(np.float32), [Z - oz, Y - oy, X - ox], order=1,
                               mode="constant", cval=0.0)
        onpap = float((raw0[ok] > 5).mean())
        print(f"  on-papyrus fraction: {onpap:.2f}")
        if onpap < 0.7:
            print("  skip (off-papyrus)")
            continue
        ok = ok & (raw0 > 5)

        # null masks
        letters_rot = letters[::-1, ::-1]
        letters_shift = np.roll(letters, (64, 64), axis=(0, 1))

        tile_res = {"tile": [int(ty), int(tx)], "letters_px": int(letters.sum()),
                    "bg_px": int(background.sum()), "onpap": round(onpap, 3),
                    "auc_by_k": {}, "auc_rot_by_k": {}, "auc_shift_by_k": {}}
        render_k0 = None
        for k in K_OFFSETS:
            vals = map_coordinates(subf, [Z + nz_ * k - oz, Y + ny * k - oy, X + nx * k - ox],
                                   order=1, mode="constant", cval=np.nan)
            good = ok & np.isfinite(vals)
            if k == 0:
                render_k0 = np.where(good, vals, np.nan)
            for tag, lm, store, pool_ in [
                ("auc_by_k", letters, pooled, pooled),
                ("auc_rot_by_k", letters_rot, null_pool["rot"], null_pool["rot"]),
                ("auc_shift_by_k", letters_shift, null_pool["shift"], null_pool["shift"]),
            ]:
                pos = vals[lm & good]
                neg = vals[background & good & ~lm]
                if len(pos) > 100 and len(neg) > 100:
                    a = tie_auc(pos, neg)
                    tile_res[tag][k] = round(a, 4)
                    pool_[k]["pos"].append(pos)
                    pool_[k]["neg"].append(neg)
        results["tiles"].append(tile_res)
        print(f"  real: {tile_res['auc_by_k']}")
        print(f"  rot null: {tile_res['auc_rot_by_k'].get(0)}  shift null: {tile_res['auc_shift_by_k'].get(0)}")
        renders.append(((ty, tx), s24.copy(), letters.copy(), render_k0))

    for k in K_OFFSETS:
        if pooled[k]["pos"]:
            results["pooled"][k] = round(tie_auc(np.concatenate(pooled[k]["pos"]),
                                                 np.concatenate(pooled[k]["neg"])), 4)
        for nm in ("rot", "shift"):
            if null_pool[nm][k]["pos"]:
                results["null"].setdefault(nm, {})[k] = round(
                    tie_auc(np.concatenate(null_pool[nm][k]["pos"]),
                            np.concatenate(null_pool[nm][k]["neg"])), 4)
    print("\nPOOLED:", results["pooled"])
    print("NULL rot:", results["null"].get("rot"))
    print("NULL shift:", results["null"].get("shift"))

    with open(os.path.join(OUT, "s1a_v2_stats.json"), "w") as fh:
        json.dump(results, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(renders)
    if n:
        fig, axes = plt.subplots(3, n, figsize=(3.8 * n, 12), squeeze=False)
        for i, ((ty, tx), tink, lmask, rk0) in enumerate(renders):
            axes[0][i].imshow(tink, cmap="inferno", vmin=0, vmax=255)
            axes[0][i].set_title(f"({ty},{tx}) ink24")
            if rk0 is not None:
                v = rk0[np.isfinite(rk0)]
                v0, v1 = np.percentile(v, [2, 98]) if len(v) else (0, 1)
                axes[1][i].imshow(rk0, cmap="gray", vmin=v0, vmax=v1)
                axes[2][i].imshow(rk0, cmap="gray", vmin=v0, vmax=v1)
                axes[2][i].contour(lmask, levels=[0.5], colors="red", linewidths=0.5)
            axes[1][i].set_title("7.91um k=0")
            axes[2][i].set_title("k=0 + letters")
            for r_ in range(3):
                axes[r_][i].set_xticks([]); axes[r_][i].set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "s1a_v2_tiles.png"), dpi=95)
        print("wrote s1a_v2_tiles.png")


if __name__ == "__main__":
    main()
