#!/bin/bash
set -x
exec > /workspace/smoke2.log 2>&1
cd /workspace/villa/vesuvius
export PATH="$HOME/.local/bin:$PATH"

echo "=== install torch ==="
uv pip install torch 2>&1 | tail -3
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== smoke 192 ==="
/usr/bin/time -f "ELAPSED %E MAXRSS %MkB" uv run vesuvius.predict \
  --model_path /workspace/models/ink3d/ckpt_78k_fullsup.pth \
  --input_dir "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr" \
  --output_dir /workspace/out/smoke \
  --bbox "7392:7776,7168:7552,11776:12160" \
  --device cuda --disable_tta --batch_size 1 --patch_size 192,192,192 --input_anon 2>&1 | tail -12

echo "=== smoke 256 ==="
/usr/bin/time -f "ELAPSED %E MAXRSS %MkB" uv run vesuvius.predict \
  --model_path /workspace/models/ink3d/ckpt_78k_fullsup.pth \
  --input_dir "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr" \
  --output_dir /workspace/out/smoke256 \
  --bbox "7392:7776,7168:7552,11776:12160" \
  --device cuda --disable_tta --batch_size 1 --patch_size 256,256,256 --input_anon 2>&1 | tail -12

echo "=== SMOKE2 DONE ==="
