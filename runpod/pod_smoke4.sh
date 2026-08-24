#!/bin/bash
set -x
exec > /workspace/smoke4.log 2>&1
cd /workspace/villa/vesuvius
export PATH="$HOME/.local/bin:$PATH"

echo "=== pin torch to cu128 (driver max 12.8) ==="
if ! uv pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu128; then
  echo "2.12.0+cu128 unavailable, trying 2.11.0"
  uv pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
fi
uv run python -c "import torch; print('TORCH_OK', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "=== checkpoint (re)download with size check ==="
ls -la /workspace/models/ink3d/ || true
uv run python - <<'EOF'
from huggingface_hub import hf_hub_download
import os
d = "/workspace/models/ink3d"
os.makedirs(d, exist_ok=True)
for f in ["ckpt_78k_fullsup.pth", "config.json", "avg_ref_embedding.npy"]:
    p = hf_hub_download("scrollprize/ink_3d_dino_guided", f, local_dir=d)
    print("GOT", p, os.path.getsize(p))
EOF
ls -la /workspace/models/ink3d/

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

echo "=== SMOKE4 DONE ==="
