"""Bet B, step C1 ($0, local): rank the in-band PHerc1203 surface by the 9.362 um along-normal contrast.

The estimator is mesh_bias_survey.py's, verbatim (W = 9 stored cells per tile, 48 x 48 samples,
41 normal offsets of one 9.362 um voxel = +-187 um, contrast = (max - min) / max of the mean
along-normal profile). mesh_bias_survey.py sampled 6 random tiles on each of the 8 largest in-band
segments (48 tiles); this runs the SAME estimator densely -- every non-overlapping 9 x 9-cell tile
whose cells all lie inside band A of the 2.403 um volume -- on all 22 catalogued PHerc1203 segments,
so C2a can render the w035-like subset (contrast >= 0.24; w035's own 24 tiles read 0.22-0.58,
median 0.397) instead of the whole 24.1 cm^2.

Tile geometry: one stored cell = 20 voxels = 187.24 um, so a 9-cell tile is 1.685 mm on a side
= 2.84 mm^2 = 0.0284 cm^2.

Outputs (out/bet_b/):
  c1_tiles.jsonl   one line per scored tile (appended; the run resumes from it)
  c1_rank.json     per-segment and global summary + the ranked tile list
  c1_rank.md       the human-readable table

Usage:  python hunt/c1_rank_inband.py [--segments name,name] [--max-tiles N] [--threshold 0.24]
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import requests
import tifffile
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zarr_http  # noqa: E402
from zarr_http import Zarr3D  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out", "bet_b")
MESHCACHE = os.path.join(HERE, "meshcache")
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
P9 = "PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr"
W = 9              # stored cells per tile (mesh_bias_survey.py)
NS = 48            # samples per side
NOFF = 41          # offsets -20..+20 voxels of 9.362 um = +-187 um
CELL_UM = 20 * 9.362
TILE_CM2 = (W * CELL_UM / 1e4) ** 2


def fetch_tif(key, cache_path):
    if os.path.exists(cache_path):
        return tifffile.imread(cache_path)
    r = requests.get(f"{B}/{key}", timeout=300)
    r.raise_for_status()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(r.content)
    return tifffile.imread(io.BytesIO(r.content))


def profile(vol, px, py, pz):
    """Verbatim from mesh_bias_survey.py: mean intensity along the local normal, 41 offsets."""
    du = np.stack([np.gradient(px, axis=0), np.gradient(py, axis=0), np.gradient(pz, axis=0)], -1)
    dv = np.stack([np.gradient(px, axis=1), np.gradient(py, axis=1), np.gradient(pz, axis=1)], -1)
    nrm = np.cross(du, dv)
    nrm /= np.maximum(1e-9, np.linalg.norm(nrm, axis=-1, keepdims=True))
    offs = np.arange(NOFF) - (NOFF - 1) // 2
    pts = np.stack([pz, py, px], -1)
    nzyx = np.stack([nrm[..., 2], nrm[..., 1], nrm[..., 0]], -1)
    allp = pts[None] + offs[:, None, None, None] * nzyx[None]
    flat = allp.reshape(-1, 3)
    lo = np.maximum(0, np.floor(flat.min(0)).astype(int) - 2)
    hi = np.ceil(flat.max(0)).astype(int) + 3
    if np.prod(hi - lo) > 40e6:
        return None
    blk = vol.read(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], workers=16).astype(np.float32)
    v = map_coordinates(blk, (flat - lo).T, order=1, mode="constant", cval=0)
    return offs, v.reshape(NOFF, -1).mean(1)


def tile_stats(vol, X, Y, Z, i, j):
    u = np.linspace(0, W - 1, NS)
    gj, gi = np.meshgrid(u, u, indexing="xy")
    c = np.stack([gi.ravel(), gj.ravel()])
    px = map_coordinates(X[i:i + W, j:j + W], c, order=1).reshape(NS, NS)
    py = map_coordinates(Y[i:i + W, j:j + W], c, order=1).reshape(NS, NS)
    pz = map_coordinates(Z[i:i + W, j:j + W], c, order=1).reshape(NS, NS)
    res = profile(vol, px, py, pz)
    if res is None:
        return None
    offs, prof = res
    k = int(np.argmax(prof))
    return dict(ij=[int(i), int(j)], z=float(pz.mean()), y=float(py.mean()), x=float(px.mean()),
                peak_off_vox=int(offs[k]), peak_off_um=float(offs[k] * 9.362),
                peak=float(prof.max()), at0=float(prof[(NOFF - 1) // 2]), lo=float(prof.min()),
                contrast=float((prof.max() - prof.min()) / max(1e-6, prof.max())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="", help="comma list of segment names (default: all in-band)")
    ap.add_argument("--max-tiles", type=int, default=0, help="stop after N new tiles (smoke)")
    ap.add_argument("--threshold", type=float, default=0.24, help="w035-like contrast threshold")
    ap.add_argument("--chunk-cache", default=os.path.join(os.environ.get("C1_CACHE", os.path.join(ROOT, "out", "bet_b")), "zcache"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(a.chunk_cache, exist_ok=True)
    zarr_http.CACHE = a.chunk_cache
    ext = json.load(open(os.path.join(HERE, "band_extents.json")))
    band = tuple(ext["bandA_9umz"])
    segs = [r for r in ext["segments"] if (r.get("pt_in_bandA") or 0) > 0]
    segs.sort(key=lambda r: -(r["area_cm2"] * (r.get("pt_in_bandA") or 0)))
    if a.segments:
        keep = set(a.segments.split(","))
        segs = [r for r in segs if r["name"] in keep]
    jl = os.path.join(OUT, "c1_tiles.jsonl")
    done = set()
    rows = []
    if os.path.exists(jl):
        for line in open(jl):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(r)
            done.add((r["seg"], r["ij"][0], r["ij"][1]))
    print(f"band A z-range (9.362 um voxels): {band[0]:.0f}-{band[1]:.0f}; {len(segs)} in-band segments; "
          f"{len(rows)} tiles already scored", flush=True)
    vol = Zarr3D(P9, 0)
    t0 = time.time()
    n_new = 0
    for r in segs:
        d = r["seg_dir"]
        cdir = os.path.join(MESHCACHE, "1203_" + r["name"])
        X = fetch_tif(d + "x.tif", os.path.join(cdir, "x.tif")).astype(np.float64)
        Y = fetch_tif(d + "y.tif", os.path.join(cdir, "y.tif")).astype(np.float64)
        Z = fetch_tif(d + "z.tif", os.path.join(cdir, "z.tif")).astype(np.float64)
        ok = (X > 0) & (Y > 0) & (Z > 0) & (Z > band[0]) & (Z < band[1])
        H, Wd = X.shape
        tiles = [(i, j) for i in range(0, H - W + 1, W) for j in range(0, Wd - W + 1, W) if ok[i:i + W, j:j + W].all()]
        todo = [(i, j) for (i, j) in tiles if (r["name"], i, j) not in done]
        print(f"{r['name']}: raster {H}x{Wd}, {len(tiles)} in-band tiles ({len(tiles) * TILE_CM2:.2f} cm2), {len(todo)} to score",
              flush=True)
        with open(jl, "a") as f:
            for (i, j) in todo:
                st = tile_stats(vol, X, Y, Z, i, j)
                if st is None:
                    continue
                st["seg"] = r["name"]
                rows.append(st)
                f.write(json.dumps(st) + "\n")
                f.flush()
                n_new += 1
                if n_new % 10 == 0:
                    el = time.time() - t0
                    print(f"  {n_new} new tiles, {el / n_new:.1f} s/tile, last {r['name'][-8:]} ij=({i},{j}) contrast {st['contrast']:.3f}",
                          flush=True)
                if a.max_tiles and n_new >= a.max_tiles:
                    break
        if a.max_tiles and n_new >= a.max_tiles:
            break
    # ---- summary
    thr = a.threshold
    per = {}
    for r in rows:
        p = per.setdefault(r["seg"], dict(n=0, n_like=0, n_022=0, contrasts=[]))
        p["n"] += 1
        p["contrasts"].append(r["contrast"])
        p["n_like"] += r["contrast"] >= thr
        p["n_022"] += r["contrast"] >= 0.22
    summary = {}
    for s, p in per.items():
        c = np.array(p["contrasts"])
        summary[s] = dict(tiles=p["n"], cm2=round(p["n"] * TILE_CM2, 3), like_tiles=int(p["n_like"]),
                          like_cm2=round(p["n_like"] * TILE_CM2, 3), ge022_tiles=int(p["n_022"]),
                          contrast_median=round(float(np.median(c)), 4), contrast_p90=round(float(np.percentile(c, 90)), 4),
                          contrast_max=round(float(c.max()), 4))
    allc = np.array([r["contrast"] for r in rows]) if rows else np.zeros(0)
    ranked = sorted(rows, key=lambda r: -r["contrast"])
    out = dict(estimator="mesh_bias_survey.py profile(): W=9 cells, NS=48, NOFF=41 (+-187 um), contrast=(max-min)/max",
               volume=P9, band_A_9umz=list(band), tile_cm2=TILE_CM2, threshold=thr,
               w035_reference=dict(n=24, contrast_median=0.397, contrast_min=0.22, contrast_max=0.58, source="hunt/mesh_bias_w035.json"),
               pilot_reference=dict(n=48, contrast_median=0.138, range=[0.05, 0.30], source="hunt/mesh_bias_survey.json"),
               totals=dict(tiles=len(rows), cm2=round(len(rows) * TILE_CM2, 3),
                           like_tiles=int((allc >= thr).sum()), like_cm2=round(float((allc >= thr).sum() * TILE_CM2), 3),
                           ge022_tiles=int((allc >= 0.22).sum()), ge022_cm2=round(float((allc >= 0.22).sum() * TILE_CM2), 3),
                           contrast_median=round(float(np.median(allc)), 4) if len(allc) else None,
                           contrast_p90=round(float(np.percentile(allc, 90)), 4) if len(allc) else None),
               per_segment=summary, ranked_tiles=ranked)
    json.dump(out, open(os.path.join(OUT, "c1_rank.json"), "w"), indent=1)
    md = ["# Bet B C1 — 9.362 µm along-normal contrast ranking of the in-band PHerc1203 surface", "",
          f"Estimator: `mesh_bias_survey.py` profile (W = 9 cells, 48² samples, ±20 voxels = ±187 µm; contrast = (max − min)/max). "
          f"Dense non-overlapping tiles of {TILE_CM2:.4f} cm² inside band A (z {band[0]:.0f}–{band[1]:.0f} at 9.362 µm). "
          f"Reference: w035 (readable at 9 µm) 24 tiles median 0.397, range 0.22–0.58; pilot 48 PHerc1203 tiles median 0.138.", "",
          f"**Totals:** {len(rows)} tiles = {len(rows) * TILE_CM2:.2f} cm²; contrast median {out['totals']['contrast_median']}, "
          f"p90 {out['totals']['contrast_p90']}; **≥ {thr}: {out['totals']['like_tiles']} tiles = {out['totals']['like_cm2']} cm²**; "
          f"≥ 0.22: {out['totals']['ge022_tiles']} tiles = {out['totals']['ge022_cm2']} cm².", "",
          "| segment | tiles | cm² | median | p90 | max | ≥ thr tiles | ≥ thr cm² |", "|---|---|---|---|---|---|---|---|"]
    for s, v in sorted(summary.items(), key=lambda kv: -kv[1]["like_tiles"]):
        md.append(f"| {s} | {v['tiles']} | {v['cm2']:.2f} | {v['contrast_median']:.3f} | {v['contrast_p90']:.3f} | {v['contrast_max']:.3f} | {v['like_tiles']} | {v['like_cm2']:.2f} |")
    md += ["", "Top 20 tiles:", "", "| segment | ij | z | contrast | peak offset (µm) |", "|---|---|---|---|---|"]
    for r in ranked[:20]:
        md.append(f"| {r['seg']} | {r['ij']} | {r['z']:.0f} | {r['contrast']:.3f} | {r['peak_off_um']:+.0f} |")
    open(os.path.join(OUT, "c1_rank.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"\nDONE: {len(rows)} tiles, {out['totals']['like_tiles']} at >= {thr} ({out['totals']['like_cm2']} cm2); "
          f"{time.time() - t0:.0f} s; wrote {os.path.join(OUT, 'c1_rank.json')}", flush=True)


if __name__ == "__main__":
    main()
