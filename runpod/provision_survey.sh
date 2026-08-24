#!/bin/bash
# Provision + start a survey shard worker. Usage: bash provision_survey.sh <shard> <nshards>
set -x
exec > /workspace/provision.log 2>&1
SHARD=$1
NSHARDS=$2
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
os.makedirs("/workspace/ckpts/ink_9um", exist_ok=True)
print(hf_hub_download("scrollprize/ink_9um", "hybrid_3d2d-seed42/step-075000.pth",
                      local_dir="/workspace/ckpts/ink_9um"))
EOF
mkdir -p /workspace/data /workspace/survey
echo "=== starting survey shard $SHARD/$NSHARDS ==="
cd /workspace/villa/vesuvius
nohup uv run --no-sync --extra models python /workspace/survey_segments.py $SHARD $NSHARDS \
  > /workspace/survey.log 2>&1 &
echo "SURVEY_STARTED shard=$SHARD"
