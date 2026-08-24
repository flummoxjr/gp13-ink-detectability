"""QC extraction for s1a — replicate original sampling exactly, save per-tile
value stacks + masks + coords so all confound tests run offline.

Saves per tile: D:\\vesuvius-data\\trackD\\w032\\qc\\tile_{ty}_{tx}.npz
  sub_ink  (512,512) uint8   ink24_ds4 tile
  ink11r   (512,512) float32 ink11_ds4 resampled into this tile frame
  ok_pre   (512,512) bool    valid-grid mask BEFORE raw0 gate
  raw0     (512,512) float32 trilinear uint8-volume sample at k=0
  vals     (9,512,512) float32 windowed trilinear samples k=-4..+4 (NaN outside)
  X,Y,Z    (512,512) float32 volume coords of surface
Also caches the fetched bbox as .npy for possible re-extraction.
"""
import json
import os
import sys

import numpy as np
import requests
import tifffile
from scipy.ndimage import map_coordinates, binary_dilation

CACHE = r"D:\vesuvius-data\trackD\w032"
QCD = r"D:\vesuvius-data\trackD\w032\qc"
VOL_URL = "https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr"
VOL_SHAPE = (11173, 3340, 3440)
WIN = (-0.03, 0.145)

INK_SCALE = 4 * 2.399 / 7.91
GRID_SCALE = 0.05
TILE = 512
N_TILES = 6
K_OFFSETS = list(range(-4, 5))
STRONG_INK = 170
WEAK_BG = 20
INK11_SCALE = 9.596 / 9.032  # ink24_ds4 px -> ink11_ds4 px (2.399/2.258)


class BboxVolume:
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
            for attempt in range(3):
                try:
                    r = self.sess.get(f"{self.url}/0/{iz}/{iy}/{ix}", timeout=120)
                    if r.status_code == 404:
                        return job, None
                    r.raise_for_status()
                    return job, np.frombuffer(r.content, np.uint8).reshape(self.chunks)
                except Exception as e:
                    if attempt == 2:
                        raise
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
    os.makedirs(QCD, exist_ok=True)
    ink = np.load(os.path.join(CACHE, "ink24_ds4.npy"))
    ink11 = np.load(os.path.join(CACHE, "ink11_ds4.npy"))
    print("ink24 ds4:", ink.shape, ink.dtype, "ink11 ds4:", ink11.shape, ink11.dtype, flush=True)
    gx = tifffile.imread(os.path.join(CACHE, "grid_x.tif"))
    gy = tifffile.imread(os.path.join(CACHE, "grid_y.tif"))
    gz = tifffile.imread(os.path.join(CACHE, "grid_z.tif"))
    gvalid = (gx > 0) & (gy > 0) & (gz > 0)
    H, W = ink.shape

    # --- identical tile selection to original ---
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
    print("candidates:", tiles[:12], flush=True)

    vol = BboxVolume(VOL_URL, VOL_SHAPE)
    lo, hi = WIN
    n_done = 0
    for (ty, tx) in tiles:
        if n_done >= N_TILES:
            break
        y0d, x0d = ty * TILE, tx * TILE
        sub_ink = ink[y0d:y0d + TILE, x0d:x0d + TILE]
        letters = sub_ink > STRONG_INK

        vv, uu = np.mgrid[0:TILE, 0:TILE]
        gpy = (vv + y0d) * INK_SCALE * GRID_SCALE
        gpx = (uu + x0d) * INK_SCALE * GRID_SCALE
        X = map_coordinates(gx, [gpy, gpx], order=1)
        Y = map_coordinates(gy, [gpy, gpx], order=1)
        Z = map_coordinates(gz, [gpy, gpx], order=1)
        vmask = map_coordinates(gvalid.astype(np.float32), [gpy, gpx], order=1) > 0.99
        ok = vmask & (X > 1) & (Y > 1) & (Z > 1)

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
        print(f"tile ({ty},{tx}): bbox {gb:.2f} GB", flush=True)
        if gb > 1.2:
            print("  skip: bbox too large", flush=True)
            continue

        bboxf = os.path.join(QCD, f"bbox_{ty}_{tx}.npz")
        if os.path.exists(bboxf):
            d = np.load(bboxf)
            sub, (oz, oy, ox) = d["sub"], d["origin"]
        else:
            sub, (oz, oy, ox) = vol.fetch_bbox(zmin, zmax, ymin, ymax, xmin, xmax)
            np.savez(bboxf, sub=sub, origin=np.array([oz, oy, ox]))
        print(f"  bbox loaded {sub.shape}", flush=True)
        subf = sub.astype(np.float32) / 255.0 * (hi - lo) + lo

        pz0, py0, px0 = Z - oz, Y - oy, X - ox
        raw0 = map_coordinates(sub.astype(np.float32), [pz0, py0, px0], order=1,
                               mode="constant", cval=0.0)
        onpap = float((raw0[ok] > 5).mean()) if ok.sum() else 0.0
        print(f"  onpap={onpap:.3f}", flush=True)
        if onpap < 0.7:
            print("  skip: off-papyrus", flush=True)
            continue

        vals = np.full((len(K_OFFSETS), TILE, TILE), np.nan, np.float32)
        for i, k in enumerate(K_OFFSETS):
            pz, py, px = Z + nz * k - oz, Y + ny * k - oy, X + nx * k - ox
            vals[i] = map_coordinates(subf, [pz, py, px], order=1,
                                      mode="constant", cval=np.nan)

        # ink11 resampled into this tile frame (bilinear on ds4 grid)
        i11y = (vv + y0d) * INK11_SCALE
        i11x = (uu + x0d) * INK11_SCALE
        ink11r = map_coordinates(ink11.astype(np.float32), [i11y, i11x], order=1)

        np.savez_compressed(os.path.join(QCD, f"tile_{ty}_{tx}.npz"),
                            sub_ink=sub_ink, ink11r=ink11r.astype(np.float32),
                            ok_pre=ok, raw0=raw0.astype(np.float32),
                            vals=vals, X=X.astype(np.float32),
                            Y=Y.astype(np.float32), Z=Z.astype(np.float32),
                            nz_comp=np.stack([nx, ny, nz]).astype(np.float32),
                            onpap=np.float32(onpap))
        print(f"  saved tile_{ty}_{tx}.npz", flush=True)
        n_done += 1
    print("DONE", n_done, "tiles", flush=True)


if __name__ == "__main__":
    main()
