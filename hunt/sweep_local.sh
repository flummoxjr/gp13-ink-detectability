#!/bin/bash
# Real ink_9um depth-offset sweep on the w035 control crop (laptop 4090).
# Window = 17 slices centred at c; the model's native (aligned) window is c=14.
PY="C:/Users/benbl/Desktop/Vsuvious/.venv314/Scripts/python.exe"
Z="D:/vesuvius-data/trackD/w035_crop.sv.zarr"
CK="D:/vesuvius-data/models/ink9um/step-075000_seed42.pth"
O="C:/Users/benbl/Desktop/Vsuvious/trackD/out/hunt/preds"
mkdir -p "$O"
for c in 8 9 10 11 12 13 14 15 16 17 18 19; do
  lo=$((c-8)); hi=$((c+9))
  for d in forward reverse; do
    f="$O/c${c}_${d}.tif"
    [ -f "$f" ] && { echo "skip $f"; continue; }
    echo "=== centre $c layers $lo..$hi $d ==="
    t0=$(date +%s)
    "$PY" -m vesuvius.ink_detection.inference.infer "$Z" "$CK" "$f" \
      --direction $d --layer-start $lo --layer-end $hi \
      --batch-size 8 --num-workers 0 --gpus 0 --no-compile > /dev/null 2>&1
    echo "  done in $(( $(date +%s) - t0 ))s -> $(ls -la "$f" 2>/dev/null | awk '{print $5}')"
  done
done
echo "SWEEP COMPLETE"
