"""QC extraction for s1a-v2 — replicate the v2 pipeline exactly for the 4
usable tiles, verify reproduction of reported AUCs, and save per-tile stacks
for offline adversarial analysis.

Saves per tile: D:\\vesuvius-data\\trackD\\w032\\qc\\tile_v2_{ty}_{tx}.npz
  sub_ink   (512,512) uint8    ink24_ds4 tile
  s11       (512,512) uint8    v2 zoom-resampled ink11 tile (exact v2 method)
  dist_nd   (512,512) float32  distance (ds4 px) to nearest ink24==0 pixel
  ok_pre    (512,512) bool     valid-grid mask BEFORE raw0 gate
  raw0      (512,512) float32  trilinear uint8-volume sample at k=0
  vals      (9,512,512) float32 windowed trilinear samples k=-4..+4
  X,Y,Z     (512,512) float32  volume coords of surface
  normals   (3,512,512) float32 nx,ny,nz
Bbox cache: bbox_v2_{ty}_{tx}.npz
Also prints reproduction check of the v2 real/rot/shift AUCs.
"""
import json
import os

import numpy as np
import requests
import tifffile
from scipy.ndimage import map_coordinates, distance_transform_edt, zoom
from scipy.stats import rankdata

CACHE = r"D:\vesuvius-data\trackD\w032"
QCD = r"D:\vesuvius-data\trackD\w032\qc"
VOL_URL = "https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr"
VOL_SHAPE = (11173, 3340, 3440)
WIN = (-0.03, 0.145)
INK_SCALE = 4 * 2.399 / 7.91
GRID_SCALE = 0.05
TILE = 512
K_OFFSETS = list(range(-4, 5))

TILES = [(12, 11), (11, 12), (8, 10), (9, 9)]


def tie_auc(pos, neg, max_n=300_000):
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
            for attempt in range(4):
                try:
                    r = self.sess.get(f"{self.url}/0/{iz}/{iy}/{ix}", timeout=120)
                    if r.status_code == 404:
                        return job, None
                    r.raise_for_status()
                    return job, np.frombuffer(r.content, np.uint8).reshape(self.chunks)
                except Exception:
                    if attempt == 3:
                        raise
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
    os.makedirs(QCD, exist_ok=True)
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

    dist_nd_full = distance_transform_edt(ink24 > 0)
    dist_ok = dist_nd_full >= 8

    vol = BboxVolume(VOL_URL, VOL_SHAPE)
    lo, hi = WIN
    repro = {}

    for (ty, tx) in TILES:
        y0d, x0d = ty * TILE, tx * TILE
        s24 = ink24[y0d:y0d + TILE, x0d:x0d + TILE]
        s11 = ink11a[y0d:y0d + TILE, x0d:x0d + TILE]
        dk = dist_ok[y0d:y0d + TILE, x0d:x0d + TILE]
        letters = (s24 >= 200) & (s11 >= 150) & dk
        background = (s24 >= 28) & (s24 <= 60) & (s11 <= 60) & dk
        print(f"tile ({ty},{tx}): let={int(letters.sum())} bg={int(background.sum())}", flush=True)

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
        nz_ = dXu * dYv - dYu * dXv
        norm = np.sqrt(nx ** 2 + ny ** 2 + nz_ ** 2) + 1e-9
        nx, ny, nz_ = nx / norm, ny / norm, nz_ / norm

        sel = ok
        zmin, zmax = int(Z[sel].min()) - 8, int(Z[sel].max()) + 9
        ymin, ymax = int(Y[sel].min()) - 8, int(Y[sel].max()) + 9
        xmin, xmax = int(X[sel].min()) - 8, int(X[sel].max()) + 9
        gb = (zmax - zmin) * (ymax - ymin) * (xmax - xmin) / 1e9
        print(f"  bbox {gb:.2f} GB", flush=True)

        bboxf = os.path.join(QCD, f"bbox_v2_{ty}_{tx}.npz")
        if os.path.exists(bboxf):
            d = np.load(bboxf)
            sub, (oz, oy, ox) = d["sub"], d["origin"]
        else:
            sub, (oz, oy, ox) = vol.fetch_bbox(zmin, zmax, ymin, ymax, xmin, xmax)
            np.savez(bboxf, sub=sub, origin=np.array([oz, oy, ox]))
        print(f"  bbox loaded {sub.shape}", flush=True)
        subf = sub.astype(np.float32) / 255.0 * (hi - lo) + lo

        raw0 = map_coordinates(sub.astype(np.float32), [Z - oz, Y - oy, X - ox], order=1,
                               mode="constant", cval=0.0)
        onpap = float((raw0[ok] > 5).mean())
        print(f"  onpap={onpap:.3f}", flush=True)
        ok2 = ok & (raw0 > 5)

        vals = np.full((len(K_OFFSETS), TILE, TILE), np.nan, np.float32)
        for i, k in enumerate(K_OFFSETS):
            vals[i] = map_coordinates(subf, [Z + nz_ * k - oz, Y + ny * k - oy, X + nx * k - ox],
                                      order=1, mode="constant", cval=np.nan)

        # reproduction check (identical to v2 script logic)
        letters_rot = letters[::-1, ::-1]
        letters_shift = np.roll(letters, (64, 64), axis=(0, 1))
        rep = {"auc_by_k": {}, "auc_rot_by_k": {}, "auc_shift_by_k": {}}
        for i, k in enumerate(K_OFFSETS):
            v = vals[i]
            good = ok2 & np.isfinite(v)
            for tag, lm in [("auc_by_k", letters), ("auc_rot_by_k", letters_rot),
                            ("auc_shift_by_k", letters_shift)]:
                pos = v[lm & good]
                neg = v[background & good & ~lm]
                if len(pos) > 100 and len(neg) > 100:
                    rep[tag][k] = round(tie_auc(pos, neg), 4)
        repro[f"{ty}_{tx}"] = rep
        print(f"  REPRO real: {rep['auc_by_k']}", flush=True)

        np.savez_compressed(os.path.join(QCD, f"tile_v2_{ty}_{tx}.npz"),
                            sub_ink=s24, s11=s11,
                            dist_nd=dist_nd_full[y0d:y0d + TILE, x0d:x0d + TILE].astype(np.float32),
                            ok_pre=ok, raw0=raw0.astype(np.float32),
                            vals=vals, X=X.astype(np.float32),
                            Y=Y.astype(np.float32), Z=Z.astype(np.float32),
                            normals=np.stack([nx, ny, nz_]).astype(np.float32),
                            onpap=np.float32(onpap))
        print(f"  saved tile_v2_{ty}_{tx}.npz", flush=True)

    with open(os.path.join(QCD, "qc_s1a_v2_repro.json"), "w") as fh:
        json.dump(repro, fh, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
