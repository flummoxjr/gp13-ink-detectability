#!/bin/bash
# Stage A — grow the first surface patches on PHerc0813 (highest-SNR GP volume, no
# published segments). Per trackD/hunt/HUNT_PLAN.md §3. Args: "x,y,z x,y,z ..."
exec > /grow.log 2>&1
set -x
SEEDS="$1"
S=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0813/representations/predictions/surfaces
B=20250821151723-surface-20260413222639-surface-m7-L0-th0.2

mkdir -p /work/ngrid /work/paths /work/cache
which vc_grow_seg_from_seed || ls /usr/local/bin | head -40

cat > /work/ngrid/normal-grids-remote.json <<EOF
{"url":"$S/$B.normal-grids"}
EOF

cat > /work/seed.json <<'EOF'
{
  "mode": "seed",
  "generations": 75,
  "step_size": 20,
  "min_area_cm": 0.3,
  "thread_limit": 1,
  "use_cuda": false,
  "voxelsize": 9.362,
  "cache_size": 4000000000,
  "cache_root": "/work/cache",
  "normal_grid_path": "/work/ngrid"
}
EOF
cat /work/seed.json

i=0
for s in $SEEDS; do
  X=$(echo "$s" | cut -d, -f1); Y=$(echo "$s" | cut -d, -f2); Z=$(echo "$s" | cut -d, -f3)
  echo "=== SEED $i: $X $Y $Z ==="
  ( vc_grow_seg_from_seed --volume "$S/$B.zarr" --target-dir /work/paths \
      --params /work/seed.json --seed "$X" "$Y" "$Z" > /work/seed_$i.log 2>&1 ; \
    echo "SEED_${i}_EXIT=$?" >> /work/seed_$i.log ) &
  i=$((i+1))
  sleep 3
done
wait
echo "=== ALL SEEDS DONE ==="

echo "=== GATE 1: volume opened (expect shape 16993 x 7947 x 7947) ==="
grep -h -m2 -i "zarr dataset size\|scale group" /work/seed_*.log | head -4
echo "=== GATE 2: normal grid streaming ==="
grep -h -m2 -i "normal grid" /work/seed_*.log | head -4
echo "=== GATE 3: seed on a sheet (expect value: 255) ==="
grep -h -m1 -i "found seed location" /work/seed_*.log | head -8
echo "=== EXITS ==="
grep -h "SEED_._EXIT" /work/seed_*.log
echo "=== ERRORS (first lines) ==="
grep -h -i -m1 "error\|exception\|failed\|terminate" /work/seed_*.log | head -8

echo "=== PATCHES GROWN ==="
ls -la /work/paths/ 2>/dev/null | head -20
for m in /work/paths/*/meta.json; do
  [ -f "$m" ] && echo "$m: $(cat $m | tr -d '\n' | cut -c1-300)"
done
echo "=== AREA SUMMARY ==="
python3 - <<'PY'
import glob, json
tot = 0.0; n = 0; ok = 0
for m in sorted(glob.glob("/work/paths/*/meta.json")):
    try:
        d = json.load(open(m))
    except Exception as e:
        print("unreadable", m, e); continue
    a = d.get("area_cm2") or d.get("area_cm") or 0
    n += 1; tot += float(a)
    if float(a) >= 2.0: ok += 1
    print(f"{m.split('/')[-2]}: area_cm2={a}")
print(f"PATCHES={n} TOTAL_AREA_CM2={tot:.2f} PATCHES_GE_2CM2={ok}")
print("VERDICT=GO" if ok >= 3 else "VERDICT=NO-GO")
PY
echo "GROW_DONE"
