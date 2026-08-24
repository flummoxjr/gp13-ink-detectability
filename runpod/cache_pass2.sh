#!/bin/bash
# Second-pass cache-budget test on real scroll data: does a fresh process
# re-fill an already-populated cache dir past its own budget?
exec > /workspace/cache_pass2.log 2>&1
set -x
cd /workspace/villa/vesuvius
export PATH="$HOME/.local/bin:$PATH"
BUDGET_GB=4
echo "BUDGET_BYTES=$((BUDGET_GB * 1000000000))"
echo "BEFORE_BYTES=$(du -sb /workspace/ctxcache | cut -f1)"
echo "BEFORE_FILES=$(find /workspace/ctxcache -type f | wc -l)"
rm -rf /workspace/data/ctx.sv.zarr
timeout 1200 uv run --no-sync --extra models python /workspace/scripts/render_tifxyz_sv.py \
  /workspace/data/ctx.tifxyz \
  "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr" \
  /workspace/data/ctx.sv.zarr --num-slices 21 \
  --cache-dir /workspace/ctxcache --cache-max-gb $BUDGET_GB 2>&1 | tail -3
echo "EXIT=$?"
echo "AFTER_BYTES=$(du -sb /workspace/ctxcache | cut -f1)"
echo "AFTER_FILES=$(find /workspace/ctxcache -type f | wc -l)"
df -h /workspace | tail -1
echo "PASS2_DONE"
