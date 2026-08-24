#!/bin/bash
# A/B test: does mode:"random_seed" (the documented recipe) orient surfaces to the
# sheets where our explicit --seed runs came out flat across the wraps?
# Arm R = documented recipe (random_seed, gens 75, default step) with min_area_cm 0.0 —
#   0.3 discards every patch on a bare prediction zarr because computed area is 0.000000
#   (no volpkg meta -> no voxelsize; re-confirmed 2026-08-25, 16/16 surfaces grown then dropped).
# Arm E = identical params but mode:"seed" + our 8 explicit seeds -> isolates seed handling.
set -x
SEEDS="$1"
S=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0813/representations/predictions/surfaces
B=20250821151723-surface-20260413222639-surface-m7-L0-th0.2
VOL=$S/$B.zarr

mkdir -p /work/paths_R /work/paths_E /work/cache /work/ngrid
printf '{"url":"%s/%s.normal-grids"}' "$S" "$B" > /work/ngrid/normal-grids-remote.json
curl -sf "$S/$B.normal-grids/metadata.json" -o /work/ngrid/metadata.json  # villa #1588

cat > /work/seed_R.json <<'EOF'
{
  "mode": "random_seed",
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
sed 's/"random_seed"/"seed"/' /work/seed_R.json > /work/seed_E.json

echo "=== ARM R: documented random_seed recipe, 8 workers ==="
for i in $(seq 8); do
  ( timeout 1800 vc_grow_seg_from_seed --volume "$VOL" --target-dir /work/paths_R \
      --params /work/seed_R.json --skip-overlap-check > /work/R_$i.log 2>&1; \
    echo "R_${i}_EXIT=$?" >> /work/R_$i.log ) &
  sleep 2
done
wait
echo "=== ARM E: identical params, explicit seeds ==="
i=0
for s in $SEEDS; do
  X=$(echo "$s" | cut -d, -f1); Y=$(echo "$s" | cut -d, -f2); Z=$(echo "$s" | cut -d, -f3)
  ( timeout 1800 vc_grow_seg_from_seed --volume "$VOL" --target-dir /work/paths_E \
      --params /work/seed_E.json --seed "$X" "$Y" "$Z" > /work/E_$i.log 2>&1; \
    echo "E_${i}_EXIT=$?" >> /work/E_$i.log ) &
  i=$((i+1)); sleep 2
done
wait
echo "R patches: $(ls /work/paths_R | wc -l)  E patches: $(ls /work/paths_E | wc -l)"
echo "=== ALL DONE ==="
