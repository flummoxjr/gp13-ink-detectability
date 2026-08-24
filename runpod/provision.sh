#!/bin/bash
# Provision a fresh pod and start a screening worker.
# Usage: bash provision.sh <worker_id> <n_workers>
set -x
exec > /workspace/provision.log 2>&1
WID=$1
NW=$2

cd /workspace
apt-get update -qq && apt-get install -y -qq git > /dev/null
[ -d villa ] || git clone --depth 1 https://github.com/ScrollPrize/villa.git
cd /workspace/villa/vesuvius
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra models 2>&1 | tail -2
uv pip install "torch==2.11.0" torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
uv pip install tqdm scipy scikit-image pandas einops opencv-python-headless tifffile aiohttp \
  numba monai timm accelerate pytorch-lightning pytorch-optimizer huggingface-hub \
  dynamic-network-architectures nnunetv2 batchgenerators fft-conv-pytorch fvcore \
  connected-components-3d tensorstore typed-argument-parser psutil nest-asyncio \
  blosc2 lxml imagecodecs pynrrd cachetools edt wandb 2>&1 | tail -2
uv run python -c "import torch; print('TORCH_OK', torch.__version__, torch.cuda.is_available())"
uv run vesuvius.accept_terms --yes || true

uv run python - <<'EOF'
from huggingface_hub import hf_hub_download
import os
d = "/workspace/models/ink3d"
os.makedirs(d, exist_ok=True)
for f in ["ckpt_78k_fullsup.pth", "config.json", "avg_ref_embedding.npy"]:
    print(hf_hub_download("scrollprize/ink_3d_dino_guided", f, local_dir=d))
EOF

echo "=== starting worker $WID/$NW ==="
cd /workspace/villa/vesuvius
nohup uv run python /workspace/screen_band.py worker $WID $NW /workspace/out/screen_full \
  > /workspace/worker.log 2>&1 &
echo "PROVISION_DONE worker=$WID"
