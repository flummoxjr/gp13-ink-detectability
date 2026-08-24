#!/bin/bash
# Stage A retry: same growth, WITHOUT normal_grid_path (documented-optional; the
# remote-URL stub is not a valid grid dir). Args: "x,y,z x,y,z ..."
exec > /grow2.log 2>&1
set -x
SEEDS="$1"
S=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0813/representations/predictions/surfaces
B=20250821151723-surface-20260413222639-surface-m7-L0-th0.2

rm -rf /work/paths2 /work/seed2_*.log
mkdir -p /work/paths2 /work/cache

cat > /work/seed2.json <<'EOF'
{
  "mode": "seed",
  "generations": 75,
  "step_size": 20,
  "min_area_cm": 0.3,
  "thread_limit": 1,
  "use_cuda": false,
  "voxelsize": 9.362,
  "cache_size": 4000000000,
  "cache_root": "/work/cache"
}
EOF
cat /work/seed2.json

echo "=== SMOKE: one seed, foreground, 120s cap, to surface the real error early ==="
FIRST=$(echo $SEEDS | cut -d' ' -f1)
FX=$(echo "$FIRST" | cut -d, -f1); FY=$(echo "$FIRST" | cut -d, -f2); FZ=$(echo "$FIRST" | cut -d, -f3)
timeout 150 vc_grow_seg_from_seed --volume "$S/$B.zarr" --target-dir /work/paths2 \
  --params /work/seed2.json --seed "$FX" "$FY" "$FZ" 2>&1 | tail -25
echo "SMOKE_EXIT=$?"

if ls /work/paths2/*/meta.json >/dev/null 2>&1; then
  echo "=== SMOKE PRODUCED A PATCH — fanning out the rest ==="
  i=1
  for s in $(echo $SEEDS | cut -d' ' -f2-); do
    X=$(echo "$s" | cut -d, -f1); Y=$(echo "$s" | cut -d, -f2); Z=$(echo "$s" | cut -d, -f3)
    ( timeout 900 vc_grow_seg_from_seed --volume "$S/$B.zarr" --target-dir /work/paths2 \
        --params /work/seed2.json --seed "$X" "$Y" "$Z" > /work/seed2_$i.log 2>&1; \
      echo "SEED_${i}_EXIT=$?" >> /work/seed2_$i.log ) &
    i=$((i+1)); sleep 2
  done
  wait
else
  echo "=== SMOKE PRODUCED NOTHING — not fanning out; diagnostics follow ==="
fi

echo "=== AREA SUMMARY ==="
python3 - <<'PY'
import glob, json
n = ok = 0; tot = 0.0
for m in sorted(glob.glob("/work/paths2/*/meta.json")):
    try: d = json.load(open(m))
    except Exception: continue
    a = float(d.get("area_cm2") or d.get("area_cm") or 0)
    n += 1; tot += a; ok += (a >= 2.0)
    print(f"{m.split('/')[-2]}: area_cm2={a}")
print(f"PATCHES={n} TOTAL_AREA_CM2={tot:.2f} PATCHES_GE_2CM2={ok}")
print("VERDICT=GO" if ok >= 3 else "VERDICT=NO-GO")
PY
echo "GROW2_DONE"
