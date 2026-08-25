#!/bin/bash
# THE DECISIVE TEST: current-main tracer vs the stale May-13 image.
# Build vc_grow_seg_from_seed from villa main, re-grow the same 8 seeds, so |n_z|
# can be compared against: stale build 0.876-1.000, published meshes 0.004-0.307.
set -x
SEEDS="$1"
S=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0813/representations/predictions/surfaces
B=20250821151723-surface-20260413222639-surface-m7-L0-th0.2

git clone --depth 1 https://github.com/ScrollPrize/villa /villa 2>&1 | tail -1
cd /villa && git rev-parse HEAD
cmake -S /villa/volume-cartographer -B /build -DCMAKE_BUILD_TYPE=Release > /build_cfg.log 2>&1 || { tail -30 /build_cfg.log; echo "CMAKE_CONFIG_FAILED"; exit 1; }
cmake --build /build --target vc_grow_seg_from_seed -j"$(nproc)" > /build.log 2>&1 || { tail -30 /build.log; echo "BUILD_FAILED"; exit 1; }
BIN=$(find /build -name vc_grow_seg_from_seed -type f | head -1); echo "BIN=$BIN"
"$BIN" --help > /dev/null 2>&1; echo "HELP_RC=$?"

mkdir -p /work/paths_M /work/cache /work/ngrid
printf '{"url":"%s/%s.normal-grids"}' "$S" "$B" > /work/ngrid/normal-grids-remote.json
# NOTE: no metadata.json fetch — main should fetch it itself (re-verifies the #1588 fix end-to-end)

cat > /work/seed_M.json <<'EOF'
{
  "mode": "seed",
  "generations": 75,
  "min_area_cm": 0.0,
  "thread_limit": 1,
  "use_cuda": false,
  "voxelsize": 9.362,
  "cache_size": 4000000000,
  "cache_root": "/work/cache",
  "normal_grid_path": "/work/ngrid"
}
EOF
i=0
for s in $SEEDS; do
  X=$(echo "$s" | cut -d, -f1); Y=$(echo "$s" | cut -d, -f2); Z=$(echo "$s" | cut -d, -f3)
  ( timeout 1800 "$BIN" --volume "$S/$B.zarr" --target-dir /work/paths_M \
      --params /work/seed_M.json --seed "$X" "$Y" "$Z" > /work/M_$i.log 2>&1; \
    echo "M_${i}_EXIT=$?" >> /work/M_$i.log ) &
  i=$((i+1)); sleep 2
done
wait
grep -h "seed location" /work/M_*.log | head -8
echo "M patches: $(ls /work/paths_M | wc -l)"
echo "=== ALL DONE ==="
