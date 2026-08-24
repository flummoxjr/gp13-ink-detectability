#!/bin/bash
# Pass 4: swap in the fixed volume_io.py (patch application failed; file copy is
# robust) and re-open the same 5.98 GB cache dir with the same 4 GB budget.
# Expected with the fix: the directory is swept AND seeded at open, so it ends
# at or below budget instead of growing.
exec > /workspace/cache_pass4.log 2>&1
set -x
export PATH="$HOME/.local/bin:$PATH"
TARGET=/workspace/villa/vesuvius/src/vesuvius/ink_detection/volume_io.py

echo "BUDGET_BYTES=4000000000"
echo "BEFORE_BYTES=$(du -sb /workspace/ctxcache | cut -f1)"
echo "BEFORE_FILES=$(find /workspace/ctxcache -type f | wc -l)"

cp "$TARGET" /workspace/volume_io_orig.py
cp /workspace/volume_io_fixed.py "$TARGET"
grep -c "_seed_cache_store_lru" "$TARGET" && echo "FIX_INSTALLED=yes" || echo "FIX_INSTALLED=no"

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
df -h /workspace | tail -1

echo "=== output sanity (voxel stats of the rendered surface volume) ==="
uv run --no-sync --extra models python - <<'PY'
import numpy as np, zarr
a = zarr.open("/workspace/data/ctx.sv.zarr", mode="r")["0"]
mid = np.asarray(a[10, ::8, ::8])
print("RENDER_STATS shape", a.shape, "mid-slice mean", round(float(mid.mean()), 3),
      "nonzero", round(float((mid > 0).mean()), 4))
PY

echo "=== restore pristine file ==="
cp /workspace/volume_io_orig.py "$TARGET"
echo "PASS4_DONE"
