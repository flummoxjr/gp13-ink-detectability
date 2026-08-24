#!/bin/bash
# Rerun the disk-failed rungs: STEP1d render-control + JOB2, with bounded cache + space gates.
set -x
exec > /workspace/ink9um_part2.log 2>&1
cd /workspace/villa/vesuvius
export PATH="$HOME/.local/bin:$PATH"

echo "=== CLEANUP ==="
rm -rf /workspace/cache /workspace/data/w035/surface-volume.zarr /workspace/data/w035/rendered-sv.zarr
rm -f /workspace/preds/w035/reference_2p4um_pipeline.tif /workspace/preds/w035/w035_rendered_seed42-075000.tif
df -h /workspace | tail -1

echo "=== STEP1d render-control (bounded cache) ==="
uv run --no-sync --extra models python /workspace/scripts/render_tifxyz_sv.py \
  /workspace/data/w035/mesh-9.362um.tifxyz \
  "s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr" \
  /workspace/data/w035/rendered-sv.zarr \
  --num-slices 21 --cache-dir /workspace/cache --cache-max-gb 8 || { echo "RENDER1D_FAILED"; exit 1; }
df -h /workspace | tail -1
uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
  /workspace/data/w035/rendered-sv.zarr \
  /workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth \
  /workspace/preds/w035/w035_rendered_seed42-075000.tif \
  --direction both --batch-size 16 --num-workers 8 --gpus 0 || { echo "INFER1D_FAILED"; exit 1; }
rm -rf /workspace/data/w035/rendered-sv.zarr

echo "=== JOB2 segments ==="
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
    --num-slices 21 --cache-dir /workspace/cache --cache-max-gb 8 || { echo "RENDER_FAILED $SEG"; continue; }
  df -h /workspace | tail -1
  for CK in hybrid_3d2d-seed42/step-075000 hybrid_3d2d-seed43/step-075000; do
    TAG=$(echo "$CK" | tr '/' '_')
    echo "=== JOB2 $SEG infer $TAG ==="
    uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer \
      /workspace/data/pherc1203/${SEG}.sv.zarr \
      /workspace/ckpts/ink_9um/$CK.pth \
      /workspace/preds/pherc1203/${SEG}_${TAG}.tif \
      --direction both --batch-size 16 --num-workers 8 --gpus 0 || echo "INFER_FAILED $SEG $TAG"
  done
  rm -rf /workspace/data/pherc1203/${SEG}.sv.zarr
done
echo "=== PART2 DONE ==="
ls -la /workspace/preds/pherc1203/
