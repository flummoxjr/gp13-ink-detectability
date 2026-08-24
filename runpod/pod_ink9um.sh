#!/bin/bash
# ink_9um probe ladder: checkpoints -> w035 control (prebuilt SV) -> render-control -> PHerc1203 segments.
set -x
exec > /workspace/ink9um.log 2>&1
cd /workspace/villa/vesuvius
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace/scripts /workspace/preds/w035 /workspace/preds/pherc1203 /workspace/data /workspace/cache

echo "=== STEP1 checkpoints ==="
uv run --no-sync --extra models python - <<'PY'
from huggingface_hub import hf_hub_download
for seed in (42, 43):
    for step in ("050000", "075000"):
        print(hf_hub_download("scrollprize/ink_9um", f"hybrid_3d2d-seed{seed}/step-{step}.pth",
                              local_dir="/workspace/ckpts/ink_9um"))
PY
uv run --no-sync --extra models python - <<'PY'
import torch
cfg = torch.load("/workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth",
                 map_location="cpu", weights_only=False)["config"]
print("CKPT_CFG", cfg["mode"], cfg["model_type"], cfg["patch_size"], cfg["image_normalization"])
PY

echo "=== STEP1a w035 surface volume download ==="
uv run --no-sync --extra models python - <<'PY'
import s3fs
fs = s3fs.S3FileSystem(anon=True)
fs.get("vesuvius-challenge-open-data/PHerc0139/segments/20260317000000-w035_2026031718/"
       "surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr",
       "/workspace/data/w035/surface-volume.zarr", recursive=True)
print("w035 SV downloaded")
PY

echo "=== STEP1b w035 control inference (seed42-075000) ==="
uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
  /workspace/data/w035/surface-volume.zarr \
  /workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth \
  /workspace/preds/w035/w035_seed42-075000.tif \
  --direction both --batch-size 16 --num-workers 8 --gpus 0

echo "=== STEP1b-replicate (seed43-075000) ==="
uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
  /workspace/data/w035/surface-volume.zarr \
  /workspace/ckpts/ink_9um/hybrid_3d2d-seed43/step-075000.pth \
  /workspace/preds/w035/w035_seed43-075000.tif \
  --direction both --batch-size 16 --num-workers 8 --gpus 0

echo "=== STEP1c reference TIF ==="
curl -s -o /workspace/preds/w035/reference_2p4um_pipeline.tif \
  "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260317000000-w035_2026031718/ink-detection/PHerc0139-20260317000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"
ls -la /workspace/preds/w035/

echo "=== STEP1d render-control: fetch mesh, render, infer ==="
uv run --no-sync --extra models python - <<'PY'
import s3fs
fs = s3fs.S3FileSystem(anon=True)
fs.get("vesuvius-challenge-open-data/PHerc0139/segments/20260317000000-w035_2026031718/"
       "mesh/20260317000000-on-20250728140407-9.362um.tifxyz",
       "/workspace/data/w035/mesh-9.362um.tifxyz", recursive=True)
print("w035 mesh downloaded")
PY
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

echo "=== JOB2 PHerc1203 segments ==="
VOL="s3://vesuvius-challenge-open-data/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr"
for SEG in auto_grown_20251005230830031 auto_grown_20251005231446965 auto_grown_20251005221856743; do
  echo "=== JOB2 $SEG fetch ==="
  uv run --no-sync --extra models python - <<PY
import os, s3fs
fs = s3fs.S3FileSystem(anon=True)
os.makedirs("/workspace/data/pherc1203/${SEG}.tifxyz", exist_ok=True)
for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
    fs.get(f"vesuvius-challenge-open-data/PHerc1203/segments/raw/${SEG}/{f}",
           f"/workspace/data/pherc1203/${SEG}.tifxyz/{f}")
print("fetched ${SEG}")
PY
  echo "=== JOB2 $SEG render ==="
  uv run --no-sync --extra models python /workspace/scripts/render_tifxyz_sv.py \
    /workspace/data/pherc1203/${SEG}.tifxyz \
    "$VOL" \
    /workspace/data/pherc1203/${SEG}.sv.zarr \
    --num-slices 21 --cache-dir /workspace/cache
  for CK in hybrid_3d2d-seed42/step-075000 hybrid_3d2d-seed43/step-075000; do
    TAG=$(echo "$CK" | tr '/' '_')
    echo "=== JOB2 $SEG infer $TAG ==="
    uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
      /workspace/data/pherc1203/${SEG}.sv.zarr \
      /workspace/ckpts/ink_9um/$CK.pth \
      /workspace/preds/pherc1203/${SEG}_${TAG}.tif \
      --direction both --batch-size 16 --num-workers 8 --gpus 0
  done
done
echo "=== INK9UM LADDER DONE ==="
