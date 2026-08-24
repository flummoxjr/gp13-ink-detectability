#!/bin/bash
# Reproduce the villa cache-path ContextVar crash and capture the verbatim traceback.
set -x
exec > /workspace/ctxvar_repro.log 2>&1
cd /workspace/villa/vesuvius
export PATH="$HOME/.local/bin:$PATH"

echo "=== ENVIRONMENT ==="
git -C /workspace/villa rev-parse --short HEAD
uv run --no-sync --extra models python - <<'PY'
import importlib.metadata as md
for p in ("zarr", "s3fs", "aiobotocore", "botocore", "fsspec", "numcodecs", "torch"):
    try:
        print(p, md.version(p))
    except Exception as e:
        print(p, "ERR", e)
import sys; print("python", sys.version)
PY

echo "=== FETCH MESH ==="
rm -rf /workspace/data/ctx.tifxyz /workspace/data/ctx.sv.zarr /workspace/ctxcache
mkdir -p /workspace/data/ctx.tifxyz
uv run --no-sync --extra models python - <<'PY'
import s3fs
fs = s3fs.S3FileSystem(anon=True)
base = "vesuvius-challenge-open-data/PHerc1203/segments/raw/auto_grown_20251005230830031/"
for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
    try:
        fs.get(base + f, "/workspace/data/ctx.tifxyz/" + f)
        print("got", f)
    except Exception as e:
        print("skip", f, type(e).__name__)
PY

echo "=== RENDER WITH CACHE (expected: ContextVar ValueError) ==="
timeout 900 uv run --no-sync --extra models python /workspace/scripts/render_tifxyz_sv.py \
  /workspace/data/ctx.tifxyz \
  "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr" \
  /workspace/data/ctx.sv.zarr --num-slices 21 --cache-dir /workspace/ctxcache --cache-max-gb 4
echo "CACHED_RUN_EXIT=$?"

echo "=== CONTROL: SAME RENDER, NO CACHE (expected: proceeds) ==="
rm -rf /workspace/data/ctx.sv.zarr
timeout 420 uv run --no-sync --extra models python /workspace/scripts/render_tifxyz_sv.py \
  /workspace/data/ctx.tifxyz \
  "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr" \
  /workspace/data/ctx.sv.zarr --num-slices 21 2>&1 | tail -5
echo "UNCACHED_RUN_EXIT=$?"

echo "=== CACHE SIZE VS BUDGET (4 GB requested) ==="
du -sb /workspace/ctxcache 2>/dev/null
echo "=== DONE ==="
