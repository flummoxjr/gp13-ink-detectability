
S3="s3://vesuvius-challenge-open-data"
HTTPB="https://vesuvius-challenge-open-data.s3.amazonaws.com"
VOL_0139_2399="PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr"
VOL_1203_2403="PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
W035="PHerc0139/segments/20260317000000-w035_2026031718"
MESH_W035_9="$W035/mesh/20260317000000-on-20250728140407-9.362um.tifxyz"
REF_W035_A="$W035/ink-detection/PHerc0139-20260317000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif"
MODEL_REPO="scrollprize/ink_canonical_2um"
mkdir -p "$OUT/maps" "$OUT/logs" "$PREDS" "$DATA" "$MESHES" "$SV" "$OUT/previews"

if [ "$DRY" = 1 ]; then
  for st in provision oiprov meshes render_ctl ctl render_c2a c2a finalize; do stage_open "$st"; sleep 0.2; stage_close "$st"; done
  say "ALL DONE (DRY)"; echo IDLE > "$VAR/stage"
  if [ "$LINGER_EXIT" = 1 ]; then exit 0; fi
  while :; do sleep 300; say "IDLE (DRY)"; done
fi

# ============================================================================
# STAGE provision -- villa-pin + uv env (the renderer and the scorer live here;
# verbatim the p2a_v3 / screen0358_v2 recipe).
# ============================================================================
export PATH="$HOME/.local/bin:$PATH"
if stage_done provision; then
  say "=== STAGE provision already done, skipping ==="
else
  stage_open provision
  cd /workspace
  apt-get update -qq && apt-get install -y -qq git python3-venv >/dev/null 2>&1 || true
  command -v git >/dev/null || die "git unavailable after apt"
  PF_URL="https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.2.0.tar.gz"
  PF_SPEED=$(curl -s -L --max-time 40 -r 0-8388607 -o /dev/null -w "%{speed_download}" "$PF_URL" || echo 0)
  PF_MBS=$(awk -v s="$PF_SPEED" 'BEGIN{printf "%.2f", s/1048576}')
  say "PREFLIGHT files.pythonhosted.org: ${PF_MBS} MB/s on an 8 MB range"
  for U in https://huggingface.co/api/models/scrollprize/ink_canonical_2um https://vesuvius-challenge-open-data.s3.amazonaws.com/ https://github.com/; do
    C=$(curl -s -o /dev/null --max-time 20 -w "%{http_code}" "$U" || echo 000); say "PREFLIGHT $U -> http $C"
  done
  if awk -v s="$PF_MBS" 'BEGIN{exit !(s < 1.0)}'; then die "PREFLIGHT: files.pythonhosted.org ${PF_MBS} MB/s (< 1 MB/s) from this host; relaunch on another host/cloud"; fi
  if [ ! -d villa ]; then
    retry 3 timeout 300 git clone --depth 1 https://github.com/flummoxjr/villa-pin-37e300d3.git villa >> "$OUT/provision.log" 2>&1 || die "villa-pin clone failed - see provision.log"
  fi
  VSHA=$(cd villa && git rev-parse --short HEAD)
  say "provision: villa-pin @ $VSHA"
  echo "$VSHA" > "$VAR/villa_pin_sha.txt"
  cd /workspace/villa/vesuvius
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  export UV_HTTP_TIMEOUT=900
  say "provision: uv sync starting (full log at /provision.log on :$PORT)"
  retry 3 timeout 1200 uv sync --extra models >> "$OUT/provision.log" 2>&1 || die "uv sync failed - see provision.log"
  retry 2 timeout 1800 uv pip install "torch==2.11.0" torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/provision.log" 2>&1 || die "torch pin install failed - see provision.log"
  timeout 1200 uv pip install tqdm scipy scikit-image pandas einops opencv-python-headless \
    tifffile aiohttp numba monai timm accelerate pytorch-lightning \
    pytorch-optimizer huggingface-hub dynamic-network-architectures nnunetv2 \
    batchgenerators fft-conv-pytorch fvcore connected-components-3d tensorstore \
    typed-argument-parser psutil nest-asyncio blosc2 lxml imagecodecs pynrrd \
    cachetools edt wandb s3fs pillow >> "$OUT/provision.log" 2>&1 || die "dep install failed - see provision.log"
  pyrun -c "import torch,scipy,tifffile,numcodecs,PIL,zarr; \
print('ENV_OK torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || die "environment import check failed"
  pyrun -c "import torch; assert torch.cuda.is_available(); \
print('GPU', torch.cuda.get_device_name(0))" || die "no CUDA device"
  uv run vesuvius.accept_terms --yes >/dev/null 2>&1 || true
  export TORCH_COMPILE_DISABLE=1
  stage_close provision
fi
cd /workspace/villa/vesuvius
export TORCH_COMPILE_DISABLE=1
say "programs: $(cd "$SCRIPTS" && sha256sum render_tifxyz_sv.py tifxyz_transform.py c2a_meshes.py c2a_score.py selection.json | awk '{print substr($1,1,12), $2}' | tr '\n' ' ')"

# ============================================================================
# STAGE oiprov -- villa main optimized_inference (sparse checkout) in its own
# venv; anon S3 reads; the model is pulled by the entrypoint from HF.
# ============================================================================
if stage_done oiprov; then
  say "=== STAGE oiprov already done, skipping ==="
else
  stage_open oiprov
  cd /workspace
  if [ ! -d "$OI/.git" ]; then
    retry 3 timeout 600 git clone --depth 1 --filter=blob:none --sparse https://github.com/ScrollPrize/villa "$OI" >> "$OUT/logs/oiprov.log" 2>&1 || die "villa sparse clone failed"
    (cd "$OI" && git sparse-checkout set ink-detection/optimized_inference >> "$OUT/logs/oiprov.log" 2>&1) || die "sparse checkout failed"
  fi
  VSHA=$(cd "$OI" && git rev-parse HEAD)
  say "oiprov: villa @ $VSHA (optimized_inference)"
  echo "$VSHA" > "$VAR/villa_sha.txt"
  OID="$OI/ink-detection/optimized_inference"
  [ -f "$OID/entrypoint.py" ] || die "entrypoint.py missing after sparse checkout"
  N=$(grep -c "anon=False" "$OID/processing.py" || true)
  sed -i 's/anon=False/anon=True/g' "$OID/processing.py"
  say "oiprov: patched anon=False -> anon=True in processing.py ($N sites) for the public bucket"
  if [ ! -x "$OIVENV/bin/python" ]; then
    python3 -m venv --system-site-packages "$OIVENV" >> "$OUT/logs/oiprov.log" 2>&1 || die "venv failed"
  fi
  "$OIVENV/bin/python" -c 'import importlib
for m in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{m}=={importlib.import_module(m).__version__}")
    except Exception:
        pass' > "$VAR/constraints.txt"
  say "oiprov: pip constraints: $(tr '\n' ' ' < "$VAR/constraints.txt")"
  retry 3 timeout 2400 "$OIVENV/bin/pip" install -q --timeout 180 --retries 8 -c "$VAR/constraints.txt" -r "$OID/requirements.txt" >> "$OUT/logs/oiprov.log" 2>&1 || die "pip install -r requirements.txt failed - see logs/oiprov.log"
  "$OIVENV/bin/pip" install -q numpy scipy tifffile opencv-python-headless >> "$OUT/logs/oiprov.log" 2>&1 || true
  "$OIVENV/bin/python" - <<'PY' >> "$OUT/logs/oiprov.log" 2>&1 || { tail -14 "$OUT/logs/oiprov.log" | while read -r L; do say "import: $L"; done; die "optimized_inference import check failed"; }
import sys, os
sys.path.insert(0, os.path.join(os.environ["OI"], "ink-detection", "optimized_inference"))
import torch, zarr, s3fs, pytorch_lightning
import processing, inference
print("OI_IMPORT_OK torch", torch.__version__, "cuda", torch.cuda.is_available(), "zarr", zarr.__version__, "lightning", pytorch_lightning.__version__)
PY
  say "oiprov: $(grep OI_IMPORT_OK "$OUT/logs/oiprov.log" | tail -1)"
  stage_close oiprov
fi
OID="$OI/ink-detection/optimized_inference"

run_oi() { # run_oi <tag> <zarr path (local dir or s3://)> <start> <end> <reverse true|false>
  # villa optimized_inference is map/reduce: STEP=inference writes zarr partitions, STEP=reduce
  # blends them into a tiled TIFF at OUTPUT_PATH. Two villa caches are keyed WITHOUT the volume
  # identity (ZARR_CACHE_DIR LocalStore by chunk name; /tmp/partition_cache reused if present):
  # every run gets its own cache dir and both are wiped before and after (w059 pod, 2026-09-03).
  local TAG=$1 ZP=$2 S=$3 E=$4 REV=$5
  local OUTP="$PREDS/$TAG.tif" PARTS="$ROOT/parts_$TAG"
  if [ -s "$OUTP" ] && [ "$FORCE" != 1 ]; then say "oi skip (exists): $TAG"; return 0; fi
  say "OI OPEN $TAG: $ZP layers [$S,$E) reverse=$REV tile 256 stride 128 batch $BATCH_OI"
  local t0=$SECONDS
  local ZC="$ROOT/zcache_$TAG"
  rm -rf "$PARTS" "$ZC" /tmp/partition_cache "$OID/zarr_cache" 2>/dev/null || true
  mkdir -p "$PARTS" "$ZC"
  ( cd "$OID" && MODEL="$MODEL_REPO" MODEL_TYPE=resnet3d-152-3d-decoder STEP=inference NUM_PARTS=1 PART_ID=0 ZARR_OUTPUT_DIR="$PARTS" ZARR_CACHE_DIR="$ZC" \
      SURFACE_VOLUME_ZARR="$ZP" START_LAYER="$S" END_LAYER="$E" TILE_SIZE=256 STRIDE=128 BATCH_SIZE="$BATCH_OI" FORCE_REVERSE="$REV" \
      OUTPUT_PATH="$OUTP" COMPILE=0 PROFILING_LEVEL=basic "$OIVENV/bin/python" entrypoint.py > "$OUT/logs/oi_$TAG.log" 2>&1 ) || {
    tail -15 "$OUT/logs/oi_$TAG.log" | while read -r L; do say "oi_$TAG: $L"; done
    return 1; }
  say "OI inference $TAG done ($((SECONDS - t0))s); partitions $(du -sh "$PARTS" 2>/dev/null | cut -f1); reduce next"
  rm -rf /tmp/partition_cache 2>/dev/null || true
  ( cd "$OID" && MODEL="$MODEL_REPO" MODEL_TYPE=resnet3d-152-3d-decoder STEP=reduce NUM_PARTS=1 ZARR_OUTPUT_DIR="$PARTS" ZARR_CACHE_DIR="$ZC" \
      SURFACE_VOLUME_ZARR="$ZP" START_LAYER="$S" END_LAYER="$E" TILE_SIZE=256 STRIDE=128 FORCE_REVERSE="$REV" \
      OUTPUT_PATH="$OUTP" PROFILING_LEVEL=basic "$OIVENV/bin/python" entrypoint.py > "$OUT/logs/oi_${TAG}_reduce.log" 2>&1 ) || {
    tail -15 "$OUT/logs/oi_${TAG}_reduce.log" | while read -r L; do say "oi_${TAG}_reduce: $L"; done
    return 1; }
  [ -s "$OUTP" ] || { say "oi_$TAG: no output written after reduce; preds/: $(ls "$PREDS" | tr '\n' ' ')"; tail -8 "$OUT/logs/oi_${TAG}_reduce.log" | while read -r L; do say "oi_${TAG}_reduce: $L"; done; return 1; }
  rm -rf "$PARTS" "$ZC" /tmp/partition_cache /tmp/prediction_*.tif 2>/dev/null || true
  say "OI DONE $TAG ($((SECONDS - t0))s): $(du -h "$OUTP" | cut -f1) $(pyrun -c "import tifffile;print(tifffile.imread('$OUTP').shape)" 2>/dev/null)"
}

render_mesh() { # render_mesh <mesh dir> <volume s3 path> <out zarr>
  local MD=$1 VOL=$2 OZ=$3
  if [ -d "$OZ/0" ] && [ -f "$OZ/.render_done" ] && [ "$FORCE" != 1 ]; then say "render skip (exists): $OZ"; return 0; fi
  rm -rf "$OZ"
  local t0=$SECONDS
  say "RENDER OPEN $(basename "$MD"): $SLICES slices, tile $TILE_RENDER, volume $VOL"
  pyrun "$SCRIPTS/render_tifxyz_sv.py" "$MD" "$S3/$VOL" "$OZ" --num-slices "$SLICES" --slice-step 1.0 --tile "$TILE_RENDER" > "$OUT/logs/render_$(basename "$MD").log" 2>&1 || {
    tail -12 "$OUT/logs/render_$(basename "$MD").log" | while read -r L; do say "render: $L"; done; return 1; }
  touch "$OZ/.render_done"
  say "RENDER DONE $(basename "$MD") ($((SECONDS - t0))s): $(du -sh "$OZ" | cut -f1) $(grep -c 'row band' "$OUT/logs/render_$(basename "$MD").log" || true) row bands"
}

# ============================================================================
# STAGE meshes -- fetch the published 9.362 um meshes, transform them into the
# 2.4 um volumes (catalogue matrix for w035; our derived matrix for 1203), keep
# only the ctl sub-crop / the selected tiles. Writes $MESHES/<name>/ + geom.json.
# ============================================================================
if stage_done meshes; then
  say "=== STAGE meshes already done, skipping ==="
else
  stage_open meshes
  pyrun "$SCRIPTS/c2a_meshes.py" "$SCRIPTS/selection.json" "$MESHES" > "$OUT/logs/meshes.log" 2>&1 || { tail -20 "$OUT/logs/meshes.log" | while read -r L; do say "meshes: $L"; done; die "mesh preparation failed"; }
  grep -E "^MESH " "$OUT/logs/meshes.log" | while read -r L; do say "$L"; done
  stage_close meshes
fi

# ============================================================================
# STAGE render_ctl / ctl -- w035 sub-crop rendered from the 2.399 um volume,
# scored against the published map and the human labels.
# ============================================================================
if stage_done render_ctl; then
  say "=== STAGE render_ctl already done, skipping ==="
else
  stage_open render_ctl
  render_mesh "$MESHES/ctl_w035" "$VOL_0139_2399" "$SV/ctl_w035.zarr" || die "ctl render failed"
  stage_close render_ctl
fi

if stage_done ctl; then
  say "=== STAGE ctl already done, skipping ==="
else
  stage_open ctl
  [ -s "$DATA/ref_w035_A.tif" ] || retry 3 curl -fsSL --max-time 1200 -o "$DATA/ref_w035_A.tif" "$HTTPB/$REF_W035_A" || die "reference prediction download failed"
  say "ctl: reference prediction $(du -h "$DATA/ref_w035_A.tif" | cut -f1)"
  C=$(( (SLICES - 62) / 2 ))
  BEST=""; BESTR=0
  for S0 in "$C" 0 $(( SLICES - 62 )); do
    TAG="ctl_fwd_${S0}"
    run_oi "$TAG" "$SV/ctl_w035.zarr" "$S0" "$(( S0 + 62 ))" false || die "ctl inference failed for window [$S0,$((S0+62)))"
    RC=0; pyrun "$SCRIPTS/c2a_score.py" ctl "$PREDS/$TAG.tif" "$DATA/ref_w035_A.tif" "$MESHES/ctl_w035" "$TAG" || RC=$?
    R=$(pyrun -c "import json;print(json.load(open('$RESULTS/ctl_$TAG.json'))['r_ds4_joint'])")
    if pyrun -c "import sys; sys.exit(0 if float('$R') > float('$BESTR') else 1)"; then BEST="$S0"; BESTR=$R; fi
    if [ $RC = 0 ]; then break; fi
  done
  echo "$BEST" > "$VAR/window.txt"; echo "$BESTR" > "$VAR/ctl_r.txt"
  run_oi "ctl_rev_${BEST}" "$SV/ctl_w035.zarr" "$BEST" "$(( BEST + 62 ))" true || die "ctl reverse inference failed"
  pyrun "$SCRIPTS/c2a_score.py" ctlrev "$PREDS/ctl_fwd_${BEST}.tif" "$PREDS/ctl_rev_${BEST}.tif" "$MESHES/ctl_w035" "ctl_${BEST}" || die "ctl reverse scoring failed"
  if pyrun -c "import sys; sys.exit(0 if float('$BESTR') >= 0.90 else 1)"; then
    say "CTL PASSED: window start $BEST r=$BESTR -- renderer + model reproduce the published w035 map at 2.4 um; this window is used for 1203"
  else
    touch "$VAR/contract_unverified"
    say "CTL NOT REPRODUCED: best r=$BESTR at window start $BEST -- everything downstream is flagged CONTRACT_UNVERIFIED"
  fi
  stage_close ctl
fi
read -r WS < "$VAR/window.txt"; WE=$(( WS + 62 ))

# ============================================================================
# STAGE render_c2a / c2a -- the selected PHerc1203 tiles (sets H and L), rendered
# from the 2.403 um band, forward + reverse, per-tile statistics.
# ============================================================================
if stage_done render_c2a; then
  say "=== STAGE render_c2a already done, skipping ==="
else
  stage_open render_c2a
  for MD in "$MESHES"/seg_*; do
    [ -d "$MD" ] || continue
    render_mesh "$MD" "$VOL_1203_2403" "$SV/$(basename "$MD").zarr" || die "render failed for $(basename "$MD")"
  done
  stage_close render_c2a
fi

if stage_done c2a; then
  say "=== STAGE c2a already done, skipping ==="
else
  stage_open c2a
  BP=$(pyrun -c "import json;print(json.load(open('$RESULTS/ctl_ctl_fwd_${WS}.json'))['blank_p99'])")
  say "c2a: window [$WS,$WE), blank p99 from the ctl = $BP"
  for MD in "$MESHES"/seg_*; do
    [ -d "$MD" ] || continue
    N=$(basename "$MD")
    run_oi "${N}_fwd" "$SV/$N.zarr" "$WS" "$WE" false || die "forward failed for $N"
    run_oi "${N}_rev" "$SV/$N.zarr" "$WS" "$WE" true || die "reverse failed for $N"
    pyrun "$SCRIPTS/c2a_score.py" tiles "$PREDS/${N}_fwd.tif" "$PREDS/${N}_rev.tif" "$MD" "$BP" "$N" || die "tile scoring failed for $N"
  done
  stage_close c2a
fi

# ============================================================================
# STAGE finalize
# ============================================================================
stage_open finalize
pyrun - <<'PY' || die "finalize failed"
import json, os, time, glob
R = os.environ["RESULTS"]; V = os.environ["VAR"]; O = os.environ["OUT"]
def rd(p):
    return json.load(open(p)) if os.path.exists(p) else None
ctl = {os.path.basename(p)[4:-5]: rd(p) for p in glob.glob(os.path.join(R, "ctl_*.json"))}
tiles = {os.path.basename(p)[6:-5]: rd(p) for p in glob.glob(os.path.join(R, "tiles_*.json"))}
agg = dict(run="pod_betB_c2a v1", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           prereg=rd(os.path.join(O, "prereg.json")), selection=rd(os.path.join(os.environ["SCRIPTS"], "selection.json")),
           villa_sha=open(os.path.join(V, "villa_sha.txt")).read().strip(), villa_pin_sha=open(os.path.join(V, "villa_pin_sha.txt")).read().strip(),
           window_start=int(open(os.path.join(V, "window.txt")).read().strip()), ctl_best_r=float(open(os.path.join(V, "ctl_r.txt")).read()),
           contract_unverified=os.path.exists(os.path.join(V, "contract_unverified")), ctl=ctl, tiles=tiles,
           slices=int(os.environ["SLICES"]), meshes=rd(os.path.join(os.environ["MESHES"], "meshes.json")))
for p in (os.path.join(R, "results.json"), os.path.join(O, "results.json")):
    json.dump(agg, open(p, "w"), indent=1)
print("SUMMARY ctl best r", agg["ctl_best_r"], "window", agg["window_start"], "unverified", agg["contract_unverified"])
for k, v in ctl.items():
    print("SUMMARY ctl", k, json.dumps({kk: v[kk] for kk in v if kk in ("r_ds4_joint", "auc_fwd", "auc_rev", "blank_p99", "tripwire_components", "tripwire_on_ink")})[:300])
for k, v in tiles.items():
    s = v.get("summary", {})
    print("SUMMARY tiles", k, json.dumps(s)[:300])
PY
cp -f "$STATUS" "$OUT/status_at_done.txt" 2>/dev/null || true
( cd "$ROOT" && tar czf "$OUT/bundle.tgz.part" out/results out/maps out/logs out/previews out/prereg.json out/status_at_done.txt meshes/meshes.json ) && mv -f "$OUT/bundle.tgz.part" "$OUT/bundle.tgz" || say "bundle FAILED (non-fatal)"
say "bundle: $(du -h "$OUT/bundle.tgz" | cut -f1) (results + ds4/ds16 maps + previews + logs); full-res TIFFs stay in preds/ (fetch-files if wanted)"
stage_close finalize
say "ALL DONE -- results.json + bundle.tgz served on :$PORT; the laptop guard harvests and TERMINATES."
echo IDLE > "$VAR/stage"
if [ "$LINGER_EXIT" = 1 ]; then exit 0; fi
