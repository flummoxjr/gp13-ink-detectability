#!/bin/bash
set -x
exec > /workspace/smoke3.log 2>&1
cd /workspace/villa/vesuvius
export PATH="$HOME/.local/bin:$PATH"

echo "=== torch cu128 (driver-compatible) ==="
uv pip install "torch==2.12.*" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2 \
  || uv pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2 \
  || uv pip install torch --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2

echo "=== runtime deps ==="
uv pip install tqdm scipy scikit-image pandas einops opencv-python-headless tifffile aiohttp \
  numba monai timm accelerate pytorch-lightning pytorch-optimizer huggingface-hub \
  dynamic-network-architectures nnunetv2 batchgenerators fft-conv-pytorch fvcore \
  connected-components-3d tensorstore typed-argument-parser psutil nest-asyncio \
  blosc2 lxml imagecodecs pynrrd cachetools edt wandb 2>&1 | tail -2

echo "=== torch check ==="
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== smoke 192 ==="
/usr/bin/time -f "ELAPSED %E MAXRSS %MkB" uv run vesuvius.predict \
  --model_path /workspace/models/ink3d/ckpt_78k_fullsup.pth \
  --input_dir "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr" \
  --output_dir /workspace/out/smoke \
  --bbox "7392:7776,7168:7552,11776:12160" \
  --device cuda --disable_tta --batch_size 1 --patch_size 192,192,192 --input_anon 2>&1 | tail -14

echo "=== smoke 256 ==="
/usr/bin/time -f "ELAPSED %E MAXRSS %MkB" uv run vesuvius.predict \
  --model_path /workspace/models/ink3d/ckpt_78k_fullsup.pth \
  --input_dir "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr" \
  --output_dir /workspace/out/smoke256 \
  --bbox "7392:7776,7168:7552,11776:12160" \
  --device cuda --disable_tta --batch_size 1 --patch_size 256,256,256 --input_anon 2>&1 | tail -14

echo "=== SMOKE3 DONE ==="
