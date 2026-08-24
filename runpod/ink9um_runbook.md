# ink_9um inference runbook (RunPod, RTX 5090)

Date: 2026-08-17. Derived from villa source at `villa/vesuvius` (argparse of
`src/vesuvius/ink_detection/inference/infer.py` and `infer_full3d_tifxyz.py`,
`docs/ink_detection.md`, `configs/aligned21_hybrid_3d2d.json`), the HF model card
`scrollprize/ink_9um`, and anonymous S3 XML listings. Every flag below was checked
against the argparse in source. Nothing here has been executed yet.

---

## CRITICAL FINDING — read first

**All 14 released ink_9um checkpoints are `mode: "flat"`, not `full_3d`.**
The HF repo contains exactly `hybrid_3d2d-seed42/step-{010000,020000,030000,040000,050000,060000,075000}.pth`
and the same seven for `seed43` (138 MB each, no config JSON in the model repo — the
config is embedded in each `.pth`; the shipped copy is
`src/vesuvius/ink_detection/configs/aligned21_hybrid_3d2d.json`, which says
`"mode": "flat"`, `model_type: vesuvius_unet_3d_stem_2d`, `patch_size [17,128,128]`,
`robust_mad` 1/99 normalization).

`infer_full3d_tifxyz.py` hard-refuses flat checkpoints (`_plan_from_args`:
`requires mode 'full_3d' or 'full_3d_single_wrap'` → ValueError). **The native-tifxyz
path is therefore not runnable with any released checkpoint.** Additionally, the S3
tifxyz dirs have no `volume_source.txt` (required file), so it would fail even on mode.

Consequence:
- **JOB 1 primary** = flat `infer.py` on the existing w035 9.362 µm surface-volume zarr.
- **JOB 2** = render a 21-slice surface volume from each PHerc1203 tifxyz mesh
  (Python render script below, using `vesuvius.tifxyz`), then flat `infer.py`.
- The w035 surface volume has **28 slices** (shape `[28, 5820, 5240]` uint8,
  uncompressed, chunks `[28,128,128]`); `infer.py` center-crops to the model's
  17-slice input automatically (`select_layer_indices`), so no preprocessing is needed.

Checkpoint choice: the card designates no single best ("worth trying a few").
Plan: **seed42/step-075000** (final) as primary, **seed43/step-075000** as replicate,
**seed42/step-050000** as alternate if 075000 looks over/under-confident.

---

## Verified data inventory (anonymous S3, bucket `vesuvius-challenge-open-data`)

JOB 1 — PHerc0139 w035 (`PHerc0139/segments/20260317000000-w035_2026031718/`):
- `surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/` — Zarr-v2 OME group,
  levels 0..N; level 0 `[28, 5820, 5240]` uint8, `compressor: null` (≈855 MB level 0, ≈1.1 GB total).
- `mesh/20260317000000-on-20250728140407-9.362um.tifxyz/` — `x.tif,y.tif,z.tif,meta.json`
  only (stored grid ≈291×262, `scale [0.05,0.05]` → full-res 5820×5240 = exactly the
  surface-volume canvas). No `volume_source.txt`, no mask.
- `ink-detection/` — existing predictions from the 1 µm and 2.4 µm pipelines (good
  visual references for letter locations/orientation):
  `PHerc0139-20260317000000-2.399um-...-new_canon_autoresearch_recipe-tile256-stride128.tif` etc.
- Matching scroll volume for the render-based control:
  `PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr`.

JOB 2 — PHerc1203:
- Volume: `PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr` —
  group levels `0..5`; level 0 `[18977, 6844, 6844]` uint8, chunks 128³, uncompressed.
- 22 `auto_grown_*` dirs under `PHerc1203/segments/raw/`. Layout per dir:
  `x.tif y.tif z.tif meta.json overlapping.json generations.tif` (+ `versions/`
  history). **tifxyz mesh only — no surface volume, no volume_source.txt.**
  Coordinates are in 9.362 µm voxel units of volume `20250820131727` (grown on it
  by `vc_grow_seg_from_seed`).
- Chosen 3 (largest area, all central in z 5.4k–10.2k of 19k):

  | segment | area_cm2 | bbox x | bbox y | bbox z |
  |---|---|---|---|---|
  | `auto_grown_20251005230830031` | 16.10 | 1064–5323 | 4381–5120 | 5819–10150 |
  | `auto_grown_20251005231446965` | 12.53 | 908–4634  | 3318–4758 | 5405–9443  |
  | `auto_grown_20251005221856743` | 11.79 | 2066–4378 | 3309–4233 | 5403–9077  |

---

## Step 0 — environment assumptions

- villa at `/workspace/villa`, vesuvius env synced: run everything from
  `/workspace/villa/vesuvius` with `uv run --no-sync --extra models` (drop
  `--no-sync` only if the env was never synced; per AGENTS.md do not resync casually).
- Python 3.14, torch ≥2.12 CUDA, zarr 2.18.7–3.x, s3fs, huggingface-hub all in the env.
- Anonymous S3 works out of the box: any `s3://` path containing
  `vesuvius-challenge-open-data` is opened with `anon=True` by
  `vesuvius.ink_detection.volume_io.open_volume_root` — no AWS credentials needed.

```bash
mkdir -p /workspace/ckpts/ink_9um /workspace/data/w035 /workspace/data/pherc1203 \
         /workspace/preds/w035 /workspace/preds/pherc1203 /workspace/scripts /workspace/cache
cd /workspace/villa/vesuvius
```

## Step 1 — download checkpoints and verify their embedded config

```bash
cd /workspace/villa/vesuvius
uv run --no-sync --extra models python - <<'PY'
from huggingface_hub import hf_hub_download
for seed in (42, 43):
    for step in ("050000", "075000"):
        p = hf_hub_download(
            "scrollprize/ink_9um",
            f"hybrid_3d2d-seed{seed}/step-{step}.pth",
            local_dir="/workspace/ckpts/ink_9um",
        )
        print(p)
PY
```

Sanity-check the embedded config (must print `flat`):

```bash
uv run --no-sync --extra models python - <<'PY'
import torch
cfg = torch.load(
    "/workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth",
    map_location="cpu", weights_only=False,
)["config"]
print(cfg["mode"], cfg["model_type"], cfg["patch_size"], cfg["image_normalization"])
PY
```

If this ever prints `full_3d`, the native path below becomes viable — but as of the
current HF listing it will print `flat`.

---

## JOB 1 — positive control on PHerc0139 w035

### 1a. Download the surface volume locally (recommended; it is uncompressed, ≈1.1 GB)

```bash
uv run --no-sync --extra models python - <<'PY'
import s3fs
fs = s3fs.S3FileSystem(anon=True)
fs.get(
    "vesuvius-challenge-open-data/PHerc0139/segments/20260317000000-w035_2026031718/"
    "surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr",
    "/workspace/data/w035/surface-volume.zarr",
    recursive=True,
)
PY
```

(`aws s3 sync --no-sign-request s3://vesuvius-challenge-open-data/... dest` works too
if the AWS CLI is installed.)

### 1b. PRIMARY — flat inference on the existing surface volume

```bash
cd /workspace/villa/vesuvius
uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
  /workspace/data/w035/surface-volume.zarr \
  /workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth \
  /workspace/preds/w035/w035_seed42-075000.tif \
  --direction both --batch-size 16 --num-workers 8 --gpus 0
```

- Positional form is `input_zarr checkpoint output_tiff` (single mode).
- `--direction both` runs forward and z-reversed; the second output gets a
  `_reverse` suffix → `w035_seed42-075000.tif` and `w035_seed42-075000_reverse.tif`.
  This matters because the rendered stack's slice order vs training orientation is
  not guaranteed; one of the two should show the letters clearly.
- Defaults already correct: `--resolution 0`, `--overlap 0.5` (stride 64 → Hann blend),
  `--amp-dtype auto` (reads fp16 from the checkpoint), torch.compile on
  (`reduce-overhead`; first batch adds ~1–2 min — add `--no-compile` if compile
  misbehaves on the 5090/torch combo).
- Depth handling is automatic: 28 source slices → centered 17 (`indices 6..22`).

Then repeat with the replicate/alternate checkpoints (same command, swap paths):

```bash
for CK in hybrid_3d2d-seed43/step-075000 hybrid_3d2d-seed42/step-050000; do
  TAG=$(echo "$CK" | tr '/' '_')
  uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
    /workspace/data/w035/surface-volume.zarr \
    /workspace/ckpts/ink_9um/$CK.pth \
    /workspace/preds/w035/w035_${TAG}.tif \
    --direction both --batch-size 16 --num-workers 8 --gpus 0
done
```

Optional quality bump (~4x slower): add `--tta-mirror` (with depth 17 ≠ 128, only the
two 128-px Y/X axes mirror → 4 variants).

Streaming variant (no download; slower, S3 read per patch):

```bash
uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
  "s3://vesuvius-challenge-open-data/PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr" \
  /workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth \
  /workspace/preds/w035/w035_seed42-075000_stream.tif \
  --direction both --batch-size 16 --num-workers 8 --gpus 0
```

### 1c. Control reference

Pull an existing (different-model) prediction to compare letter positions/orientation:

```bash
curl -s -o /workspace/preds/w035/reference_2p4um_pipeline.tif \
  "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260317000000-w035_2026031718/ink-detection/PHerc0139-20260317000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"
```

Note it is from the 2.4 µm pipeline at a different canvas scale — compare shapes of
letterforms, not pixels.

### 1d. BACKUP A — render-based control (validates the JOB 2 pipeline end-to-end)

Render our own 21-slice surface volume from the w035 9.362 µm tifxyz mesh + PHerc0139
9 µm volume, then infer. The tifxyz full-res grid (5820×5240) equals the published
surface-volume canvas, so the result is directly comparable to 1b. Uses the render
script from JOB 2 step 2a (write it first):

```bash
uv run --no-sync --extra models python - <<'PY'
import s3fs
fs = s3fs.S3FileSystem(anon=True)
fs.get(
    "vesuvius-challenge-open-data/PHerc0139/segments/20260317000000-w035_2026031718/"
    "mesh/20260317000000-on-20250728140407-9.362um.tifxyz",
    "/workspace/data/w035/mesh-9.362um.tifxyz", recursive=True)
PY

cd /workspace/villa/vesuvius
uv run --no-sync --extra models python /workspace/scripts/render_tifxyz_sv.py \
  /workspace/data/w035/mesh-9.362um.tifxyz \
  "s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr" \
  /workspace/data/w035/rendered-sv.zarr \
  --num-slices 21 --cache-dir /workspace/cache

uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
  /workspace/data/w035/rendered-sv.zarr \
  /workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth \
  /workspace/preds/w035/w035_rendered_seed42-075000.tif \
  --direction both --batch-size 16 --num-workers 8 --gpus 0
```

If 1b shows letters and 1d does too, the JOB 2 render→infer pipeline is proven.

### 1e. BACKUP B — native tifxyz path: DO NOT RUN (documented for completeness)

`python -m vesuvius.ink_detection.inference.infer_full3d_tifxyz <tifxyz> <ckpt> <out.ome.zarr> --plan-only`
will fail twice over with released checkpoints: (1) `Missing volume_source.txt`
(none of the S3 tifxyz dirs have one; you'd have to create it pointing at the scroll
volume URL), and then (2) `infer_full3d_tifxyz requires mode 'full_3d' or
'full_3d_single_wrap', got 'flat'`. Only revisit if scrollprize publishes a
`full_3d` checkpoint. Its verified argparse (for that day): positionals
`tifxyz_dir checkpoint output_zarr`; options `--resolution --overwrite --batch-size
--num-workers --prefetch-factor --downsample-workers --overlap --chunk-halo
--write-region {expanded,occupied} --blend-mode {gaussian,constant} --tta
--tta-batch-size --amp-dtype --compile-mode --no-compile --gpus --plan-only
--max-target-chunks --cache-dir --cache-max-gb --log-level`.

---

## JOB 2 — PHerc1203 auto-grown segments

Segments: `auto_grown_20251005230830031`, `auto_grown_20251005231446965`,
`auto_grown_20251005221856743` (largest three, all in the central z band).

### 2a. Write the render script (once)

Samples the scroll volume trilinearly at `surface + t·normal` for t = −10..+10 voxels
(21 slices, matching `vc_render_tifxyz --num-slices 21 --slice-step 1` conventions:
normals from grid cross products, coordinates interpolated to full resolution).
Writes a Zarr-v2 group with level `"0"` `(21, H, W)` uint8 plus a `"3"` max-pooled
level so `infer.py`'s occupancy scan can skip empty tiles. Normal sign is not
resolved globally — that is exactly what `--direction both` at inference covers.

```bash
cat > /workspace/scripts/render_tifxyz_sv.py <<'PY'
#!/usr/bin/env python
"""Render a centered N-slice uint8 surface volume from a tifxyz mesh + volume zarr."""
import argparse
import math
import numpy as np
from numcodecs import Blosc
from scipy.ndimage import map_coordinates

from vesuvius.tifxyz import read_tifxyz
from vesuvius.ink_detection.volume_io import open_volume, read_bbox_with_padding
from vesuvius.label_zarr import open_v2_group, create_v2_array


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("tifxyz_dir")
    ap.add_argument("volume", help="scroll volume (local path or s3:// OME-Zarr); level 0 is read")
    ap.add_argument("output_zarr")
    ap.add_argument("--num-slices", type=int, default=21)
    ap.add_argument("--slice-step", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--max-crop-voxels", type=float, default=1.5e9)
    ap.add_argument("--cache-dir", default=None, help="zarr-3 chunk cache dir (optional)")
    ap.add_argument("--cache-max-gb", type=float, default=100.0)
    return ap.parse_args()


def main():
    args = parse_args()
    surf = read_tifxyz(args.tifxyz_dir, load_mask=False, validate=False)
    surf.use_full_resolution()
    H, W = surf.shape
    n = int(args.num_slices)
    offsets = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * float(args.slice_step)
    pad = int(math.ceil(np.abs(offsets).max())) + int(args.margin)
    print(f"full-res grid: {H} x {W}, {n} slices, offsets {offsets[0]}..{offsets[-1]}")

    kwargs = {}
    if args.cache_dir:
        kwargs.update(cache_dir=args.cache_dir, cache_max_gb=args.cache_max_gb)
    try:
        vol = open_volume(args.volume, 0, **kwargs)
    except NotImplementedError:  # disk cache requires zarr 3
        print("zarr<3: continuing without chunk cache")
        vol = open_volume(args.volume, 0)

    group = open_v2_group(args.output_zarr)
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    out = create_v2_array(group, "0", shape=(n, H, W), chunks=(n, 256, 256),
                          dtype=np.uint8, compressor=comp, fill_value=0)

    def render_tile(r0, r1, c0, c1):
        x, y, z, valid = surf[r0:r1, c0:c1]
        if not np.any(valid):
            return
        nx, ny, nz = surf.get_normals(r0, r1, c0, c1)
        ok = valid & np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)
        if not np.any(ok):
            return
        z0 = int(np.floor(z[ok].min())) - pad; z1 = int(np.ceil(z[ok].max())) + pad + 1
        y0 = int(np.floor(y[ok].min())) - pad; y1 = int(np.ceil(y[ok].max())) + pad + 1
        x0 = int(np.floor(x[ok].min())) - pad; x1 = int(np.ceil(x[ok].max())) + pad + 1
        nvox = float(z1 - z0) * (y1 - y0) * (x1 - x0)
        if nvox > args.max_crop_voxels and (r1 - r0 > 64 or c1 - c0 > 64):
            rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
            for (a, b, c, d) in ((r0, rm, c0, cm), (r0, rm, cm, c1),
                                 (rm, r1, c0, cm), (rm, r1, cm, c1)):
                if a < b and c < d:
                    render_tile(a, b, c, d)
            return
        crop, _ = read_bbox_with_padding(vol, (z0, y0, x0, z1, y1, x1), fill_value=0)
        crop = crop.astype(np.float32, copy=False)
        th, tw = r1 - r0, c1 - c0
        tile_out = np.zeros((n, th, tw), dtype=np.uint8)
        zi, yi, xi = z[ok] - z0, y[ok] - y0, x[ok] - x0
        nzo, nyo, nxo = nz[ok], ny[ok], nx[ok]
        for si, off in enumerate(offsets):
            coords = np.stack([zi + off * nzo, yi + off * nyo, xi + off * nxo])
            vals = map_coordinates(crop, coords, order=1, mode="constant", cval=0.0)
            plane = np.zeros((th, tw), dtype=np.float32)
            plane[ok] = vals
            tile_out[si] = np.clip(np.rint(plane), 0, 255).astype(np.uint8)
        out[:, r0:r1, c0:c1] = tile_out

    t = int(args.tile)
    rows = list(range(0, H, t))
    for i, r0 in enumerate(rows):
        for c0 in range(0, W, t):
            render_tile(r0, min(H, r0 + t), c0, min(W, c0 + t))
        print(f"row band {i + 1}/{len(rows)} done")

    # occupancy level "3" (YX max-pool by 8) so infer.py can skip empty tiles
    p = 8
    occ = create_v2_array(group, "3", shape=(n, (H + p - 1) // p, (W + p - 1) // p),
                          chunks=(n, 256, 256), dtype=np.uint8, compressor=comp,
                          fill_value=0)
    band = 4096  # multiple of p
    for r0 in range(0, H, band):
        r1 = min(H, r0 + band)
        block = np.asarray(out[:, r0:r1, :])
        h = block.shape[1]
        ph, pw = (-h) % p, (-W) % p
        if ph or pw:
            block = np.pad(block, ((0, 0), (0, ph), (0, pw)))
        pooled = block.reshape(n, (h + ph) // p, p, (W + pw) // p, p).max(axis=(2, 4))
        occ[:, r0 // p : r0 // p + pooled.shape[1], :] = pooled
    print("done:", args.output_zarr)


if __name__ == "__main__":
    main()
PY
```

### 2b. Per segment: fetch tifxyz → render → infer

```bash
cd /workspace/villa/vesuvius
VOL="s3://vesuvius-challenge-open-data/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr"

for SEG in auto_grown_20251005230830031 auto_grown_20251005231446965 auto_grown_20251005221856743; do
  # 1) fetch the tifxyz mesh (a few hundred KB)
  uv run --no-sync --extra models python - <<PY
import os, s3fs
fs = s3fs.S3FileSystem(anon=True)
os.makedirs("/workspace/data/pherc1203/${SEG}.tifxyz", exist_ok=True)
for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
    fs.get(f"vesuvius-challenge-open-data/PHerc1203/segments/raw/${SEG}/{f}",
           f"/workspace/data/pherc1203/${SEG}.tifxyz/{f}")
PY

  # 2) render 21-slice surface volume at 9.362 um (level 0)
  uv run --no-sync --extra models python /workspace/scripts/render_tifxyz_sv.py \
    /workspace/data/pherc1203/${SEG}.tifxyz \
    "$VOL" \
    /workspace/data/pherc1203/${SEG}.sv.zarr \
    --num-slices 21 --cache-dir /workspace/cache

  # 3) flat inference, both z directions, two checkpoints
  for CK in hybrid_3d2d-seed42/step-075000 hybrid_3d2d-seed43/step-075000; do
    TAG=$(echo "$CK" | tr '/' '_')
    uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
      /workspace/data/pherc1203/${SEG}.sv.zarr \
      /workspace/ckpts/ink_9um/$CK.pth \
      /workspace/preds/pherc1203/${SEG}_${TAG}.tif \
      --direction both --batch-size 16 --num-workers 8 --gpus 0
  done
done
```

### 2c. BACKUP renderer — `vc_render_tifxyz` (volume-cartographer, C++)

Only if the Python renderer misbehaves. Requires building volume-cartographer
(`villa/volume-cartographer`, CMake + `scripts/build_dependencies.sh`; ~1 h, big
dependency chain) — that is why it is the backup. Verified flags from
`apps/src/vc_render_tifxyz.cpp`:

```bash
vc_render_tifxyz \
  --volume /workspace/data/pherc1203/volume-9um.zarr \  # local OME-Zarr (or a cache init'd with --remote-url)
  -s /workspace/data/pherc1203/auto_grown_20251005230830031.tifxyz \
  --scale 1.0 --group-idx 0 \
  --num-slices 21 --slice-step 1.0 \
  --zarr-output /workspace/data/pherc1203/auto_grown_20251005230830031.sv.zarr \
  --cache-gb 16
```

(`--volume`, `--scale`, `--group-idx` are required; `--crop-*` default 0 = full.)

---

## Expected outputs

| step | file | format |
|---|---|---|
| JOB 1 primary | `/workspace/preds/w035/w035_seed42-075000.tif` + `..._reverse.tif` | 5820×5240 uint8 tiled LZW BigTIFF, prob×255 (truncated) |
| JOB 1 backup A | `/workspace/preds/w035/w035_rendered_seed42-075000.tif` + reverse | same canvas as primary |
| JOB 2 | `/workspace/preds/pherc1203/<seg>_<ckpt>.tif` + reverse per checkpoint | canvas = tifxyz full-res grid (stored grid ÷ 0.05) |
| renders | `/workspace/data/**/*.sv.zarr` | Zarr-v2 group, `0` = (21,H,W) uint8, `3` = occupancy |

Success criterion for JOB 1: legible letterforms in at least one direction, in the
same locations as the reference tif from step 1c (allowing for scale/mirror between
pipelines). If both directions are noise on all three checkpoints, suspect the
input path (wrong level/canvas) before suspecting the model.

## Runtime / VRAM estimates (RTX 5090 32 GB)

- Model is small (≈34 M params, fp16 autocast): **< 6 GB VRAM at batch 16**; batch 32
  is safe if you want.
- JOB 1 flat inference: ~7.3 k patches/direction (5820×5240, patch 128, stride 64)
  → **a few minutes per direction** including compile warmup; whole JOB 1 with 3
  checkpoints × both directions ≈ 30–45 min plus the ~1.1 GB download.
- JOB 2 render: dominated by S3 chunk reads of the uncompressed 9 µm volume
  (uncompressed chunks = every touched 128³ chunk is a 2 MB GET). Expect
  **~15–60 min per segment** depending on pod bandwidth; the `--cache-dir` chunk
  cache makes reruns cheap. Renders are ~0.5–1.5 GB per segment.
- JOB 2 inference: grids of order 4–10 k × 10–20 k → **~5–15 min per
  segment × checkpoint × direction** (occupancy level skips empty canvas).

## Gotchas

1. **Do not use `infer_full3d_tifxyz` with released checkpoints** — flat-mode refusal
   (see 1e). This is the single biggest trap in the task framing.
2. **S3 anonymity is path-triggered**: only `s3://` paths containing
   `vesuvius-challenge-open-data` get `anon=True` automatically. The regional URL
   spelling with `?list-type=2` is only for `curl` listings; give Python the plain
   `s3://vesuvius-challenge-open-data/...` form.
3. **uint8 contract**: surface volumes must be raw uint8 intensities. `robust_mad`
   normalization happens inside `infer.py`; do not pre-normalize. The render script
   rounds trilinear samples back to uint8 (matching how the published surface volumes
   are stored).
4. **Depth**: model input is 17 slices; any ≥17-slice stack works (center crop, upper
   center for even excess). 28-slice w035 and our 21-slice renders are both fine.
   `--layer-start/--layer-end` exist but are not needed.
5. **Flat inference accepts `robust_mad` only with percentiles 1/99 and `divide` only
   with 255** — the released checkpoints embed exactly `robust_mad` 1/99, so no action.
6. **Direction ambiguity is real**: recto/verso and normal sign differ per segment;
   always run `--direction both` and keep both outputs.
7. **torch.compile**: on by default (`reduce-overhead`). If torch 2.12 + Blackwell
   (5090) trips over compile, add `--no-compile` — correctness is unchanged.
8. **Zarr-v2 vs v3**: everything written here is explicit Zarr v2 via
   `vesuvius.label_zarr` helpers, readable under either installed zarr major.
9. **Folder mode** of `infer.py` expects `<segdir>/<segdir>.zarr`-style layouts; we
   bypass it and use single mode with explicit paths everywhere.
10. **Output encoding**: flat TIFFs truncate `prob×255` to uint8 (not rounded) — a
    faint-signal pixel at p=0.499 reads 127; don't threshold at exactly 128 in QC.
11. `hf_hub_download` needs no token (public repo). Checkpoints are ~138 MB each.
12. The PHerc1203 `versions/N/` subdirs are growth-history snapshots — always use the
    top-level `x/y/z.tif` (latest generation).
