#!/bin/bash
# Track D pod setup: villa env + model checkpoint + smoke inference timing.
set -x
cd /workspace
exec > /workspace/setup.log 2>&1

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python --version

echo "=== apt deps ==="
apt-get update -qq && apt-get install -y -qq cmake ninja-build git-lfs time > /dev/null

echo "=== clone villa ==="
git clone --depth 1 https://github.com/ScrollPrize/villa.git
cd /workspace/villa/vesuvius

echo "=== install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== uv sync --extra models (documented path) ==="
/usr/bin/time -v uv sync --extra models 2>&1 | tail -40
SYNC_RC=$?
echo "uv sync rc=$SYNC_RC"

echo "=== accept terms ==="
uv run vesuvius.accept_terms --yes || true

echo "=== download checkpoint ==="
uv run python - <<'EOF'
from huggingface_hub import hf_hub_download
import os
d = "/workspace/models/ink3d"
os.makedirs(d, exist_ok=True)
for f in ["ckpt_78k_fullsup.pth", "config.json", "avg_ref_embedding.npy"]:
    print(hf_hub_download("scrollprize/ink_3d_dino_guided", f, local_dir=d))
EOF

echo "=== smoke inference (same bbox as laptop baseline) ==="
cd /workspace/villa/vesuvius
/usr/bin/time -v uv run vesuvius.predict \
  --model_path /workspace/models/ink3d/ckpt_78k_fullsup.pth \
  --input_dir "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr" \
  --output_dir /workspace/out/smoke \
  --bbox "7392:7776,7168:7552,11776:12160" \
  --device cuda --disable_tta --batch_size 1 --patch_size 192,192,192 --input_anon 2>&1 | tail -30

echo "=== smoke 256^3 patch test (5090 32GB) ==="
/usr/bin/time -v uv run vesuvius.predict \
  --model_path /workspace/models/ink3d/ckpt_78k_fullsup.pth \
  --input_dir "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr" \
  --output_dir /workspace/out/smoke256 \
  --bbox "7392:7776,7168:7552,11776:12160" \
  --device cuda --disable_tta --batch_size 1 --patch_size 256,256,256 --input_anon 2>&1 | tail -20

echo "=== DONE ==="
