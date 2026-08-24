#!/bin/bash
# CONTROLLED RE-GROW: identical to grow_0813_v3.sh except ONE variable —
# normal_grid_path is supplied, streaming the released m7 normal grids.
# Tests whether the missing grid is why our 8 patches came out with |n_z| ~ 0.97
# (flat across the wraps) instead of ~0.2 (following them).
set -x
SEEDS="$1"
# NOTE: the tracer runs on the m7 SURFACE-PREDICTION zarr, not the raw CT. An earlier
# version of this script pointed --volume at the masked CT, which silently changed a
# second variable and invalidated the comparison against grow_0813_v3.sh.
S=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0813/representations/predictions/surfaces
B=20250821151723-surface-20260413222639-surface-m7-L0-th0.2
VOL=$S/$B.zarr
NG=$S/$B.normal-grids

mkdir -p /work/paths_ng /work/cache_ng /work/ngrid
printf '{"url":"%s"}' "$NG" > /work/ngrid/normal-grids-remote.json
# the tracer opens <normal_grid_path>/metadata.json LOCALLY even when the store is
# remote; without it: terminate called ... Cannot open: /work/ngrid/metadata.json
curl -sf "$NG/metadata.json" -o /work/ngrid/metadata.json
cat /work/ngrid/normal-grids-remote.json

cat > /work/seed_ng.json <<'EOF'
{
  "mode": "seed",
  "generations": 100,
  "step_size": 20,
  "min_area_cm": 0.0,
  "thread_limit": 1,
  "use_cuda": false,
  "voxelsize": 9.362,
  "cache_size": 4000000000,
  "cache_root": "/work/cache_ng",
  "normal_grid_path": "/work/ngrid"
}
EOF

i=0
for s in $SEEDS; do
  X=$(echo "$s" | cut -d, -f1); Y=$(echo "$s" | cut -d, -f2); Z=$(echo "$s" | cut -d, -f3)
  ( timeout 1500 vc_grow_seg_from_seed --volume "$VOL" --target-dir /work/paths_ng \
      --params /work/seed_ng.json --seed "$X" "$Y" "$Z" > /work/ng_$i.log 2>&1; \
    echo "SEED_${i}_EXIT=$?" >> /work/ng_$i.log ) &
  i=$((i+1)); sleep 2
done
wait
grep -h "Loaded normal grid" /work/ng_*.log | head -3 > /work/NGRID_PROOF.txt
ls /work/paths_ng > /work/RESULT_LIST.txt
echo "=== ALL DONE ===" 
