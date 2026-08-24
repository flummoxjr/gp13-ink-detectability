#!/bin/bash
# JOB2 retry: no disk cache (zarr-3 ContextVar bug suspect); fallback smaller tiles.
set -x
exec > /workspace/ink9um_part3.log 2>&1
cd /workspace/villa/vesuvius
export PATH="$HOME/.local/bin:$PATH"
VOL="s3://vesuvius-challenge-open-data/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr"

for SEG in auto_grown_20251005230830031 auto_grown_20251005231446965 auto_grown_20251005221856743; do
  echo "=== $SEG render (no cache) ==="
  if ! uv run --no-sync --extra models python /workspace/scripts/render_tifxyz_sv.py \
      /workspace/data/pherc1203/${SEG}.tifxyz "$VOL" \
      /workspace/data/pherc1203/${SEG}.sv.zarr --num-slices 21; then
    echo "=== $SEG render retry (no cache, tile 256) ==="
    rm -rf /workspace/data/pherc1203/${SEG}.sv.zarr
    uv run --no-sync --extra models python /workspace/scripts/render_tifxyz_sv.py \
      /workspace/data/pherc1203/${SEG}.tifxyz "$VOL" \
      /workspace/data/pherc1203/${SEG}.sv.zarr --num-slices 21 --tile 256 \
      || { echo "RENDER_FAILED_FINAL $SEG"; rm -rf /workspace/data/pherc1203/${SEG}.sv.zarr; continue; }
  fi
  for CK in hybrid_3d2d-seed42/step-075000 hybrid_3d2d-seed43/step-075000; do
    TAG=$(echo "$CK" | tr '/' '_')
    echo "=== $SEG infer $TAG ==="
    uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
      /workspace/data/pherc1203/${SEG}.sv.zarr \
      /workspace/ckpts/ink_9um/$CK.pth \
      /workspace/preds/pherc1203/${SEG}_${TAG}.tif \
      --direction both --batch-size 16 --num-workers 8 --gpus 0 || echo "INFER_FAILED $SEG $TAG"
  done
  rm -rf /workspace/data/pherc1203/${SEG}.sv.zarr
done
echo "=== PART3 DONE ==="
ls -la /workspace/preds/pherc1203/
