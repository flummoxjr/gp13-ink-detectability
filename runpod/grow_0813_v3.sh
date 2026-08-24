#!/bin/bash
# Stage A v3: v2 proved the tracer grows ~8.7M vx^2 (~7.6 cm^2 at 9.362 um) in 75
# generations — but reported 0.000000 cm^2, so min_area_cm=0.3 discarded it. The
# volume is a bare prediction zarr with no volpkg meta, so voxelsize never reaches
# the area calc. Fix: min_area_cm = 0 so patches are always written; convert area
# ourselves from the vx^2 the tracer reports. Args: "x,y,z x,y,z ..."
exec > /grow3.log 2>&1
set -x
SEEDS="$1"
S=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0813/representations/predictions/surfaces
B=20250821151723-surface-20260413222639-surface-m7-L0-th0.2

rm -rf /work/paths3 /work/seed3_*.log
mkdir -p /work/paths3 /work/cache

cat > /work/seed3.json <<'EOF'
{
  "mode": "seed",
  "generations": 100,
  "step_size": 20,
  "min_area_cm": 0.0,
  "thread_limit": 1,
  "use_cuda": false,
  "voxelsize": 9.362,
  "cache_size": 4000000000,
  "cache_root": "/work/cache"
}
EOF

i=0
for s in $SEEDS; do
  X=$(echo "$s" | cut -d, -f1); Y=$(echo "$s" | cut -d, -f2); Z=$(echo "$s" | cut -d, -f3)
  ( timeout 1200 vc_grow_seg_from_seed --volume "$S/$B.zarr" --target-dir /work/paths3 \
      --params /work/seed3.json --seed "$X" "$Y" "$Z" > /work/seed3_$i.log 2>&1; \
    echo "SEED_${i}_EXIT=$?" >> /work/seed3_$i.log ) &
  i=$((i+1)); sleep 2
done
wait
echo "=== ALL DONE ==="

echo "=== REPORTED SURFACE AREAS (vx^2 -> cm^2 at 9.362 um) ==="
python3 - <<'PY'
import glob, re, json, os
vx_um = 9.362
tot_cm2 = 0.0; rows = []
for lg in sorted(glob.glob("/work/seed3_*.log")):
    txt = open(lg, errors="ignore").read()
    m = re.findall(r"generated surface ([\d.]+) vx\^2", txt)
    ex = re.search(r"SEED_(\d+)_EXIT=(\d+)", txt)
    vx2 = float(m[-1]) if m else 0.0
    cm2 = vx2 * (vx_um ** 2) * 1e-8   # um^2 -> cm^2
    tot_cm2 += cm2
    rows.append((os.path.basename(lg), vx2, cm2, ex.group(2) if ex else "?"))
    print(f"{os.path.basename(lg)}: {vx2:.0f} vx^2 = {cm2:.2f} cm^2 (exit {ex.group(2) if ex else '?'})")
print(f"TOTAL_AREA_CM2={tot_cm2:.2f}")
n_ge2 = sum(1 for r in rows if r[2] >= 2.0)
print(f"PATCHES_GE_2CM2={n_ge2}")
print("VERDICT=GO" if n_ge2 >= 3 else "VERDICT=NO-GO")
PY

echo "=== PATCH DIRS WRITTEN ==="
ls -la /work/paths3/ 2>/dev/null | head -20
for m in /work/paths3/*/meta.json; do
  [ -f "$m" ] && echo "$(dirname $m | xargs basename): $(tr -d '\n' < $m | cut -c1-200)"
done
du -sh /work/paths3 2>/dev/null
echo "GROW3_DONE"
