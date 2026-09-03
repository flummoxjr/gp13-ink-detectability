
S3="s3://vesuvius-challenge-open-data"
HTTPB="https://vesuvius-challenge-open-data.s3.amazonaws.com"
W035="PHerc0139/segments/20260317000000-w035_2026031718"
W059="PHerc0139/segments/20250223000000-w059_2025022312"
SV_W035_A="$W035/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr"
SV_W035_B="$W035/surface-volumes/1.129um-0.22m-59keV-volume-20260413113053-L1.zarr"
SV_W059_B="$W059/surface-volumes/1.129um-0.22m-59keV-volume-20260413113053-L1.zarr"
REF_W035_A="$W035/ink-detection/PHerc0139-20260317000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"
MODEL_REPO="scrollprize/ink_canonical_2um"
mkdir -p "$OUT/maps" "$OUT/logs" "$PREDS" "$DATA"

if [ "$DRY" = 1 ]; then
  for st in provision c0 c2 w059 finalize; do stage_open "$st"; sleep 0.2; stage_close "$st"; done
  say "ALL DONE (DRY)"; echo IDLE > "$VAR/stage"
  if [ "$LINGER_EXIT" = 1 ]; then exit 0; fi
  while :; do sleep 300; say "IDLE (DRY)"; done
fi

# ============================================================================
# STAGE provision -- villa main optimized_inference (sparse checkout) in its own
# venv; anon S3 reads; the model is pulled by the entrypoint from HF.
# ============================================================================
if stage_done provision; then
  say "=== STAGE provision already done, skipping ==="
else
  stage_open provision
  cd /workspace
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq git python3-venv >/dev/null 2>&1 || true
  if [ ! -d "$OI/.git" ]; then
    retry 3 timeout 600 git clone --depth 1 --filter=blob:none --sparse https://github.com/ScrollPrize/villa "$OI" >> "$OUT/logs/provision.log" 2>&1 || die "villa sparse clone failed"
    (cd "$OI" && git sparse-checkout set ink-detection/optimized_inference >> "$OUT/logs/provision.log" 2>&1) || die "sparse checkout failed"
  fi
  VSHA=$(cd "$OI" && git rev-parse HEAD)
  say "provision: villa @ $VSHA (optimized_inference)"
  echo "$VSHA" > "$VAR/villa_sha.txt"
  OID="$OI/ink-detection/optimized_inference"
  [ -f "$OID/entrypoint.py" ] || die "entrypoint.py missing after sparse checkout"
  N=$(grep -c "anon=False" "$OID/processing.py" || true)
  sed -i 's/anon=False/anon=True/g' "$OID/processing.py"
  say "provision: patched anon=False -> anon=True in processing.py ($N sites) for the public bucket"
  if [ ! -x "$OIVENV/bin/python" ]; then
    python3 -m venv --system-site-packages "$OIVENV" >> "$OUT/logs/provision.log" 2>&1 || die "venv failed"
  fi
  # 2026-09-03: a SECURE 5090 host read-timed-out on files.pythonhosted.org (pod wluhqd2196l2o6); measure first, fail fast, then pip with real retries
  PF_URL="https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.2.0.tar.gz"
  PF_SPEED=$(curl -s -L --max-time 40 -r 0-8388607 -o /dev/null -w "%{speed_download}" "$PF_URL" || echo 0)
  PF_MBS=$(awk -v s="$PF_SPEED" 'BEGIN{printf "%.2f", s/1048576}')
  say "PREFLIGHT files.pythonhosted.org: ${PF_MBS} MB/s on an 8 MB range"
  for U in https://huggingface.co/api/models/scrollprize/ink_canonical_2um https://vesuvius-challenge-open-data.s3.amazonaws.com/; do
    C=$(curl -s -o /dev/null --max-time 20 -w "%{http_code}" "$U" || echo 000); say "PREFLIGHT $U -> http $C"
  done
  if awk -v s="$PF_MBS" 'BEGIN{exit !(s < 1.0)}'; then die "PREFLIGHT: files.pythonhosted.org ${PF_MBS} MB/s (< 1 MB/s) from this host; relaunch on another host/cloud"; fi
  # 2026-09-03 pod 0q61zt4fkn0nki: `torch>=2.0.0` in requirements.txt pulled torch 2.14.0 over the image's
  # 2.8 build (torchvision/torchaudio then mismatched; the import check died). Constrain torch* to the image's versions.
  "$OIVENV/bin/python" -c 'import importlib
for m in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{m}=={importlib.import_module(m).__version__}")
    except Exception:
        pass' > "$VAR/constraints.txt"
  say "provision: pip constraints: $(tr '\n' ' ' < "$VAR/constraints.txt")"
  retry 3 timeout 2400 "$OIVENV/bin/pip" install -q --timeout 180 --retries 8 -c "$VAR/constraints.txt" -r "$OID/requirements.txt" >> "$OUT/logs/provision.log" 2>&1 || die "pip install -r requirements.txt failed (logs/provision.log)"
  "$OIVENV/bin/pip" install -q numpy scipy tifffile opencv-python-headless >> "$OUT/logs/provision.log" 2>&1 || true
  "$OIVENV/bin/python" - <<'PY' >> "$OUT/logs/provision.log" 2>&1 || { tail -14 "$OUT/logs/provision.log" | while read -r L; do say "import: $L"; done; die "optimized_inference import check failed"; }
import sys, os
sys.path.insert(0, os.path.join(os.environ["OI"], "ink-detection", "optimized_inference"))
import torch, zarr, s3fs, pytorch_lightning
import processing, inference
print("OI_IMPORT_OK torch", torch.__version__, "cuda", torch.cuda.is_available(), "zarr", zarr.__version__, "lightning", pytorch_lightning.__version__)
PY
  say "provision: $(grep OI_IMPORT_OK "$OUT/logs/provision.log" | tail -1)"
  stage_close provision
fi
OID="$OI/ink-detection/optimized_inference"

run_oi() { # run_oi <tag> <sv s3 path> <start> <end> <reverse true|false>
  # villa optimized_inference is map/reduce: STEP=inference writes zarr partitions to
  # ZARR_OUTPUT_DIR and STEP=reduce blends them into a tiled TIFF at OUTPUT_PATH.
  # (Pod 4f690f8ejklc7i, 2026-09-03, ran inference alone and found no PNG: 25 min lost.)
  local TAG=$1 SV=$2 S=$3 E=$4 REV=$5
  local OUTP="$PREDS/$TAG.tif" PARTS="$ROOT/parts_$TAG"
  if [ -s "$OUTP" ] && [ "$FORCE" != 1 ]; then say "oi skip (exists): $TAG"; return 0; fi
  say "OI OPEN $TAG: $SV layers [$S,$E) reverse=$REV tile 256 stride 128 batch $BATCH_OI"
  local t0=$SECONDS
  # Two villa caches are keyed WITHOUT the volume identity and poisoned every pass after the
  # first on pod hyj20dywx5vg8a (2026-09-03: the arm-B "control" came out identical to arm A):
  #   1. processing.get_cached_zarr_store: ZARR_CACHE_DIR defaults to ./zarr_cache (cwd) and the
  #      LocalStore is keyed by zarr-internal chunk names, so a second S3 volume reads the first
  #      volume's metadata and chunks  -> give every run its own ZARR_CACHE_DIR and delete it after.
  #   2. reduce_partitions copies partitions to /tmp/partition_cache/mask_pred_part_000.zarr only
  #      "if not already cached"                -> wipe /tmp/partition_cache before every reduce.
  local ZC="$ROOT/zcache_$TAG"
  rm -rf "$PARTS" "$ZC" /tmp/partition_cache "$OID/zarr_cache" 2>/dev/null || true
  mkdir -p "$PARTS" "$ZC"
  ( cd "$OID" && MODEL="$MODEL_REPO" MODEL_TYPE=resnet3d-152-3d-decoder STEP=inference NUM_PARTS=1 PART_ID=0 ZARR_OUTPUT_DIR="$PARTS" ZARR_CACHE_DIR="$ZC" \
      SURFACE_VOLUME_ZARR="$S3/$SV" START_LAYER="$S" END_LAYER="$E" TILE_SIZE=256 STRIDE=128 BATCH_SIZE="$BATCH_OI" FORCE_REVERSE="$REV" \
      OUTPUT_PATH="$OUTP" COMPILE=0 PROFILING_LEVEL=basic "$OIVENV/bin/python" entrypoint.py > "$OUT/logs/oi_$TAG.log" 2>&1 ) || {
    tail -15 "$OUT/logs/oi_$TAG.log" | while read -r L; do say "oi_$TAG: $L"; done
    return 1; }
  say "OI inference $TAG done ($((SECONDS - t0))s); partitions $(du -sh "$PARTS" 2>/dev/null | cut -f1); zarr cache $(du -sh "$ZC" 2>/dev/null | cut -f1); reduce next"
  rm -rf /tmp/partition_cache 2>/dev/null || true
  ( cd "$OID" && MODEL="$MODEL_REPO" MODEL_TYPE=resnet3d-152-3d-decoder STEP=reduce NUM_PARTS=1 ZARR_OUTPUT_DIR="$PARTS" ZARR_CACHE_DIR="$ZC" \
      SURFACE_VOLUME_ZARR="$S3/$SV" START_LAYER="$S" END_LAYER="$E" TILE_SIZE=256 STRIDE=128 FORCE_REVERSE="$REV" \
      OUTPUT_PATH="$OUTP" PROFILING_LEVEL=basic "$OIVENV/bin/python" entrypoint.py > "$OUT/logs/oi_${TAG}_reduce.log" 2>&1 ) || {
    tail -15 "$OUT/logs/oi_${TAG}_reduce.log" | while read -r L; do say "oi_${TAG}_reduce: $L"; done
    return 1; }
  [ -s "$OUTP" ] || { say "oi_$TAG: no output written after reduce; preds/: $(ls "$PREDS" | tr '\n' ' ')"; tail -8 "$OUT/logs/oi_${TAG}_reduce.log" | while read -r L; do say "oi_${TAG}_reduce: $L"; done; return 1; }
  rm -rf "$PARTS" "$ZC" /tmp/partition_cache /tmp/prediction_*.tif 2>/dev/null || true
  say "OI DONE $TAG ($((SECONDS - t0))s): $(du -h "$OUTP" | cut -f1) $(pyrun -c "import tifffile;print(tifffile.imread('$OUTP').shape)" 2>/dev/null)"
}

# ============================================================================
# STAGE c0 -- reproduce the published w035 A-arm prediction (gate r >= 0.90).
# ============================================================================
if stage_done c0; then
  say "=== STAGE c0 already done, skipping ==="
else
  stage_open c0
  [ -s "$DATA/ref_w035_A.tif" ] || retry 3 curl -fsSL --max-time 900 -o "$DATA/ref_w035_A.tif" "$HTTPB/$REF_W035_A" || die "reference prediction download failed"
  say "c0: reference prediction $(du -h "$DATA/ref_w035_A.tif" | cut -f1)"
  BEST=""; BESTR=0
  for WIN in "23 85" "12 74" "34 96"; do
    set -- $WIN
    TAG="c0_w035A_${1}_${2}"
    run_oi "$TAG" "$SV_W035_A" "$1" "$2" false || die "C0 inference failed for window [$1,$2)"
    RC=0; pyrun "$SCRIPTS/oi_score.py" c0 "$PREDS/$TAG.tif" "$DATA/ref_w035_A.tif" "$TAG" || RC=$?
    R=$(pyrun -c "import json;print(json.load(open('$RESULTS/c0_$TAG.json'))['r_ds4_joint'])")
    if pyrun -c "import sys; sys.exit(0 if float('$R') > float('$BESTR') else 1)"; then BEST="$1 $2"; BESTR=$R; fi
    if [ $RC = 0 ]; then break; fi
    [ "$SWEEP" = 1 ] || break
  done
  echo "$BEST" > "$VAR/window.txt"; echo "$BESTR" > "$VAR/c0_r.txt"
  if pyrun -c "import sys; sys.exit(0 if float('$BESTR') >= 0.90 else 1)"; then
    say "C0 PASSED: window [$BEST) r=$BESTR -- contract reproduced; this window is used for every other run"
  else
    touch "$VAR/contract_unverified"
    say "C0 NOT REPRODUCED: best r=$BESTR at window [$BEST) -- everything downstream is flagged CONTRACT_UNVERIFIED"
  fi
  stage_close c0
fi
read -r WS WE < "$VAR/window.txt"
# the L1 volumes have 116 planes (vs 109): shift the same 62-layer window to their centre
LS=$(( WS + 3 )); LE=$(( WE + 3 ))

# ============================================================================
# STAGE c2 -- w035 B-arm (1.129um-L1) forward: the modality control for the battery.
# ============================================================================
if stage_done c2; then
  say "=== STAGE c2 already done, skipping ==="
else
  stage_open c2
  run_oi "c2_w035B" "$SV_W035_B" "$LS" "$LE" false || die "c2 inference failed"
  pyrun "$SCRIPTS/oi_score.py" ds c2_w035B "$PREDS/c2_w035B.tif" 2.258 || die "ds failed c2"
  stage_close c2
fi

# ============================================================================
# STAGE w059 -- arm B at full coverage, forward + reverse.
# ============================================================================
if stage_done w059; then
  say "=== STAGE w059 already done, skipping ==="
else
  stage_open w059
  run_oi "w059B_fwd" "$SV_W059_B" "$LS" "$LE" false || die "w059 forward failed"
  pyrun "$SCRIPTS/oi_score.py" ds w059B_fwd "$PREDS/w059B_fwd.tif" 2.258 || die "ds failed w059 fwd"
  run_oi "w059B_rev" "$SV_W059_B" "$LS" "$LE" true || die "w059 reverse failed"
  pyrun "$SCRIPTS/oi_score.py" ds w059B_rev "$PREDS/w059B_rev.tif" 2.258 || die "ds failed w059 rev"
  pyrun "$SCRIPTS/oi_score.py" fwdrev w059B "$PREDS/w059B_fwd.tif" "$PREDS/w059B_rev.tif" || die "fwdrev failed"
  stage_close w059
fi

# ============================================================================
# STAGE finalize
# ============================================================================
stage_open finalize
pyrun - <<'PY' || die "finalize failed"
import json, os, time
R = os.environ["RESULTS"]; V = os.environ["VAR"]; O = os.environ["OUT"]
c0 = {f[3:-5]: json.load(open(os.path.join(R, f))) for f in os.listdir(R) if f.startswith("c0_")}
maps = json.load(open(os.path.join(R, "maps.json"))) if os.path.exists(os.path.join(R, "maps.json")) else {}
agg = dict(run="pod_w059_c0 v1", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           prereg=json.load(open(os.path.join(O, "prereg.json"))), villa_sha=open(os.path.join(V, "villa_sha.txt")).read().strip(),
           window=open(os.path.join(V, "window.txt")).read().strip(), c0=c0, c0_best_r=float(open(os.path.join(V, "c0_r.txt")).read()),
           contract_unverified=os.path.exists(os.path.join(V, "contract_unverified")), maps=maps,
           battery="runs LOCALLY on out/maps/*_ds16.npy (36.1 um/px) with PROTOCOL_V2; no verdict on the pod")
for p in (os.path.join(R, "results.json"), os.path.join(O, "results.json")):
    json.dump(agg, open(p, "w"), indent=1)
print("SUMMARY c0 best r", agg["c0_best_r"], "window", agg["window"], "unverified", agg["contract_unverified"])
for k, v in maps.items():
    print("SUMMARY", k, json.dumps(v)[:160])
PY
cp -f "$STATUS" "$OUT/status_at_done.txt" 2>/dev/null || true
( cd "$ROOT" && tar czf "$OUT/bundle.tgz.part" out/results out/maps out/logs out/prereg.json out/status_at_done.txt ) && mv -f "$OUT/bundle.tgz.part" "$OUT/bundle.tgz" || say "bundle FAILED (non-fatal)"
say "bundle: $(du -h "$OUT/bundle.tgz" | cut -f1) (results + ds4/ds16 maps + logs); full-res PNGs stay in preds/ (fetch-files if wanted)"
stage_close finalize
say "ALL DONE -- results.json + bundle.tgz served on :$PORT; the laptop guard harvests and TERMINATES."
echo IDLE > "$VAR/stage"
if [ "$LINGER_EXIT" = 1 ]; then exit 0; fi
while :; do sleep 300; say "IDLE -- ALL DONE; terminate the pod"; done
