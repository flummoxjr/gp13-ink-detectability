#!/bin/bash
# THE DECISIVE TEST: current-main tracer vs the stale May-13 image.
# Build vc_grow_seg_from_seed from villa main, re-grow the same 8 seeds, so |n_z|
# can be compared against: stale build 0.876-1.000, published meshes 0.004-0.307.
set -x
SEEDS="$1"
S=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0358/representations/predictions/surfaces
B=20250821151737-surface-20260413222639-surface-m7-L0-th0.2

git clone --depth 1 https://github.com/ScrollPrize/villa /villa 2>&1 | tail -1
cd /villa && git rev-parse HEAD
cmake -S /villa/volume-cartographer -B /build -DCMAKE_BUILD_TYPE=Release > /build_cfg.log 2>&1 || { tail -30 /build_cfg.log; echo "CMAKE_CONFIG_FAILED"; exit 1; }
cmake --build /build --target vc_grow_seg_from_seed -j"$(nproc)" > /build.log 2>&1 || { tail -30 /build.log; echo "BUILD_FAILED"; exit 1; }
BIN=$(find /build -name vc_grow_seg_from_seed -type f | head -1); echo "BIN=$BIN"
mkdir -p /binpack/lib && cp "$BIN" /binpack/ && ldd "$BIN" | awk '/=> \//{print $3}' | xargs -r -I{} cp -n {} /binpack/lib/ 2>/dev/null
git -C /villa rev-parse HEAD > /binpack/BUILT_FROM.txt; mkdir -p /work; tar czf /work/vc_bin.tgz -C /binpack .
"$BIN" --help > /dev/null 2>&1; echo "HELP_RC=$?"

mkdir -p /work/paths_0358 /work/cache /work/ngrid
printf '{"url":"%s/%s.normal-grids"}' "$S" "$B" > /work/ngrid/normal-grids-remote.json
# NOTE: no metadata.json fetch — main should fetch it itself (re-verifies the #1588 fix end-to-end)

cat > /work/seed_0358.json <<'EOF'
{
  "mode": "seed",
  "generations": 75,
  "min_area_cm": 0.3,
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
  ( timeout 1800 "$BIN" --volume "$S/$B.zarr" --target-dir /work/paths_0358 \
      --params /work/seed_0358.json --seed "$X" "$Y" "$Z" > /work/G_$i.log 2>&1; \
    echo "G_${i}_EXIT=$?" >> /work/G_$i.log ) &
  i=$((i+1)); sleep 2
done
wait
grep -h "seed location" /work/G_*.log | head -8
echo "M patches: $(ls /work/paths_0358 | wc -l)"
echo "=== ALL DONE ==="
