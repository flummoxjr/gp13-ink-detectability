#!/bin/bash
# Pass 3: same real PHerc1203 render, same 4 GB budget, same already-oversized
# cache dir — but with the fix applied. Expect: cache bounded at <= budget.
exec > /workspace/cache_pass3.log 2>&1
set -x
cd /workspace/villa
export PATH="$HOME/.local/bin:$PATH"

echo "BUDGET_BYTES=4000000000"
echo "BEFORE_BYTES=$(du -sb /workspace/ctxcache | cut -f1)"
echo "BEFORE_FILES=$(find /workspace/ctxcache -type f | wc -l)"

echo "=== APPLY FIX ==="
git apply --verbose /workspace/cachefix.patch && echo "PATCH_APPLIED=yes" || echo "PATCH_APPLIED=no"
git diff --stat

cd /workspace/villa/vesuvius
rm -rf /workspace/data/ctx.sv.zarr
timeout 1500 uv run --no-sync --extra models python /workspace/scripts/render_tifxyz_sv.py \
  /workspace/data/ctx.tifxyz \
  "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr" \
  /workspace/data/ctx.sv.zarr --num-slices 21 \
  --cache-dir /workspace/ctxcache --cache-max-gb 4 2>&1 | tail -4
echo "RENDER_EXIT=$?"
echo "AFTER_BYTES=$(du -sb /workspace/ctxcache | cut -f1)"
echo "AFTER_FILES=$(find /workspace/ctxcache -type f | wc -l)"
ls -la /workspace/data/ctx.sv.zarr/0 2>/dev/null | head -3
df -h /workspace | tail -1
echo "PASS3_DONE"
