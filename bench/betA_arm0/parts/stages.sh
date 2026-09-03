
KEPT="pherc1667-w013 pherc1667-w018 pherc1667-w023 pherc1667-w028 pherc1667-w029 pherc1667-w031 phercparis4-w00 phercparis4-w01 phercparis4-w02 phercparis4-w03 phercparis4-w05 phercparis4-w06 phercparis4-w07 phercparis4-w09 pherc0814-46527"
NATIVE5="w035 w039 w040 w041 w044"
if [ "$SMOKE_ONLY" = 1 ]; then STEPS=$SMOKE_STEPS; SAVE_EVERY=1000; SEEDS=$(echo $SEEDS | awk '{print $1}'); else STEPS=$FULL_STEPS; SAVE_EVERY=5000; fi
mkdir -p "$OUT/logs" "$OUT/cfg" "$RUNS" "$LABELS" "$VOLS/aligned9" "$NATIVE" "$DATA/level2"
say "MODE SMOKE_ONLY=$SMOKE_ONLY seeds='$SEEDS' steps=$STEPS save_every=$SAVE_EVERY fetch_threads=$FETCH_THREADS nproc=$(nproc)"

# ============================================================================
# DRY mode: machinery only.
# ============================================================================
if [ "$DRY" = 1 ]; then
  for st in provision ckpt trainer_check labels_fetch sv_fetch pool config_gen native_fetch ctl train_s42 eval_s42 ref; do
    if stage_done "$st"; then say "=== STAGE $st already done, skipping ==="; continue; fi
    stage_open "$st"; sleep 0.2; stage_close "$st"
  done
  stage_open finalize; say "DRY finalize: machinery only"; stage_close finalize
  say "ALL DONE (DRY)"; echo IDLE > "$VAR/stage"
  if [ "$LINGER_EXIT" = 1 ]; then exit 0; fi
  while :; do sleep 300; say "IDLE (DRY)"; done
fi

@@PROVISION_AND_CKPT@@

export WANDB_MODE=disabled

# ============================================================================
# STAGE trainer_check -- the trainer must import, and train 30 iterations on ONE
# real small label set (pherc0814-46527) + a random volume, BEFORE any big fetch.
# ============================================================================
if stage_done trainer_check; then
  say "=== STAGE trainer_check already done, skipping ==="
else
  stage_open trainer_check
  pyrun -c "import torch, accelerate; import vesuvius.ink_detection.training.train as T; import vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input as P; print('TRAIN_IMPORT_OK', torch.__version__, torch.cuda.get_device_name(0))" || die "trainer import failed (accelerate / vesuvius.ink_detection missing in villa-pin env)"
  say "trainer_check: TRAIN_IMPORT_OK"
  retry 3 pyrun "$SCRIPTS/hfsync.py" aligned-scrollprizeorg-21slices pherc0814-46527 || die "hfsync of pherc0814-46527 failed"
  pyrun "$SCRIPTS/cfggen.py" synthetic || die "synthetic config failed"
  if ! pyrun -m vesuvius.ink_detection.training.train "$OUT/cfg/syn.json" > "$OUT/logs/train_syn.log" 2>&1; then
    tail -30 "$OUT/logs/train_syn.log" | while read -r L; do say "train_syn: $L"; done
    die "synthetic 30-iteration training FAILED (logs/train_syn.log) -- fix the trainer contract before spending on data"
  fi
  ls "$RUNS/syn"/*.pth >/dev/null 2>&1 || die "synthetic training produced no checkpoint in $RUNS/syn"
  # Arms 1/2: exercise the flat inference path (arm 2 fits its whitener here) with the synthetic checkpoint
  # BEFORE the data fetch, so an eval-time failure cannot surface only after hours of training.
  SYN_CKPT=$(ls "$RUNS/syn"/*.pth | head -1)
  ( cd /workspace/villa/vesuvius && uv run --no-sync --extra models python -m vesuvius.ink_detection.inference.infer "$DATA/syn/volumes/aligned9/pherc0814-46527.zarr" "$SYN_CKPT" "$OUT/syn_infer.tif" --direction forward --batch-size 8 > "$OUT/logs/syn_infer.log" 2>&1 ) || { tail -20 "$OUT/logs/syn_infer.log" | while read -r L; do say "syn_infer: $L"; done; die "synthetic flat inference FAILED (arm $ARM) -- logs/syn_infer.log"; }
  [ -s "$OUT/syn_infer.tif" ] || die "synthetic flat inference wrote no output"
  say "trainer_check: synthetic flat inference OK (arm $ARM): $(du -h "$OUT/syn_infer.tif" | cut -f1) $(grep -c 'whitening fitted' "$OUT/logs/syn_infer.log" || true) whitening-fit lines"
  say "trainer_check: 30 synthetic iterations OK; checkpoints: $(ls "$RUNS/syn" | grep -c '\.pth$'); log tail: $(tail -1 "$OUT/logs/train_syn.log" | cut -c1-160)"
  stage_close trainer_check
fi

# ============================================================================
# STAGE labels_fetch -- the other 14 kept representations + the 5 native eval sets.
# ============================================================================
if stage_done labels_fetch; then
  say "=== STAGE labels_fetch already done, skipping ==="
else
  stage_open labels_fetch
  for S in $KEPT; do
    [ "$S" = pherc0814-46527 ] && continue
    retry 3 pyrun "$SCRIPTS/hfsync.py" aligned-scrollprizeorg-21slices "$S" || die "hfsync failed for $S"
  done
  retry 3 pyrun "$SCRIPTS/hfsync.py" native9-scrollprizeorg-21slices $NATIVE5 || die "hfsync failed for the native eval labels"
  stage_close labels_fetch
fi

# ============================================================================
# STAGE sv_fetch -- sparse level-2 plan (gated vs the manifest) + fetch.
# ============================================================================
if stage_done sv_fetch; then
  say "=== STAGE sv_fetch already done, skipping ==="
else
  stage_open sv_fetch
  pyrun "$SCRIPTS/svplan.py" plan || die "sparse plan out of band vs the manifest"
  for S in $KEPT; do
    retry 2 pyrun "$SCRIPTS/svplan.py" fetch "$S" || die "level-2 fetch failed for $S"
  done
  say "sv_fetch: level2 on disk $(du -sh "$DATA/level2" | cut -f1)"
  stage_close sv_fetch
fi

# ============================================================================
# STAGE pool -- 2.4um level-2 -> 21-slice ~9.6um (the recipe's input), gated,
# then the level-2 chunks are deleted.
# ============================================================================
if stage_done pool; then
  say "=== STAGE pool already done, skipping ==="
else
  stage_open pool
  # Smoke #5 (2026-09-03) hung here for 2 h: a bare `wait` also waits on the status server and
  # heartbeat loops (background jobs of this same shell) that never exit. Wait on the pool PIDs only.
  N=0; POOL_PIDS=""
  for S in $KEPT; do
    if [ -d "$VOLS/aligned9/$S.zarr" ]; then continue; fi
    # `timeout` cannot run the pyrun shell function (smoke #7: rc=127 on all 15 stores); wrap the real command.
    ( cd /workspace/villa/vesuvius && timeout -k 60 5400 uv run --no-sync --extra models python -m vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input "$DATA/level2/$S.zarr" "$VOLS/aligned9/$S.zarr" --level 2 --workers 6 > "$OUT/logs/pool_$S.log" 2>&1 || echo "POOL_FAIL $S rc=$?" >> "$OUT/logs/pool_failures.txt" ) &
    POOL_PIDS="$POOL_PIDS $!"
    N=$((N + 1))
    if [ "$N" -ge "$POOL_PAR" ]; then wait $POOL_PIDS; POOL_PIDS=""; N=0; say "pool: batch done ($(ls "$VOLS/aligned9" | wc -l)/15 pooled)"; fi
  done
  [ -z "$POOL_PIDS" ] || wait $POOL_PIDS
  [ ! -s "$OUT/logs/pool_failures.txt" ] || die "pooling failed for: $(cat "$OUT/logs/pool_failures.txt" | tr '\n' ' ')"
  for S in $KEPT; do
    pyrun "$SCRIPTS/svplan.py" check "$S" || die "pooled-volume gate failed for $S"
    rm -rf "$DATA/level2/$S.zarr"
  done
  say "pool: 15 volumes ready ($(du -sh "$VOLS/aligned9" | cut -f1)); level2 deleted"
  stage_close pool
fi

# ============================================================================
# STAGE config_gen -- khj1222's generator, --exclude-scroll 0139, per seed.
# ============================================================================
if stage_done config_gen; then
  say "=== STAGE config_gen already done, skipping ==="
else
  stage_open config_gen
  for SD in $SEEDS; do
    pyrun "$SCRIPTS/cfggen.py" loso "$SD" "$STEPS" "$SAVE_EVERY" || die "LOSO config generation failed for seed $SD"
  done
  stage_close config_gen
fi

# ============================================================================
# STAGE native_fetch -- the five held-out native crops + label planes.
# ============================================================================
if stage_done native_fetch; then
  say "=== STAGE native_fetch already done, skipping ==="
else
  stage_open native_fetch
  retry 2 pyrun "$SCRIPTS/natfetch.py" $NATIVE5 || die "native crop fetch failed"
  stage_close native_fetch
fi

# ============================================================================
# STAGE ctl -- harness certification with the RELEASED checkpoint (p2a_v3 arms).
# ============================================================================
if stage_done ctl; then
  say "=== STAGE ctl already done, skipping ==="
else
  stage_open ctl
  retry 3 pyrun "$SCRIPTS/ctl_build.py" fetch || die "ctl chunk fetch failed"
  pyrun "$SCRIPTS/ctl_build.py" build || die "ctl build failed"
  for CARM in ctl_native ctl_scalefault ctl_half; do
    run_infer "$DATA/$CARM.zarr" "$PREDS/$CARM.tif" both
  done
  RC=0
  pyrun "$SCRIPTS/ctl_score.py" || RC=$?
  if [ $RC = 31 ]; then die "HARNESS_BROKEN -- released checkpoint reads the w035 control below 0.95"
  elif [ $RC = 32 ]; then die "DEPTH-ORDER FAULT NOT REPRODUCED on the control"
  elif [ $RC != 0 ]; then die "ctl scoring failed rc=$RC"; fi
  stage_close ctl
fi

# ============================================================================
# ============================================================================
# STAGE measure -- input statistics with the arm-1 estimator (pooled vs native vs index targets);
# reported before any arm-1 training so the degradation's activity is known (prereg gate).
# ============================================================================
if stage_done measure; then
  say "=== STAGE measure already done, skipping ==="
else
  stage_open measure
  pyrun "$SCRIPTS/measure_inputs.py" "$VOLS/aligned9" "$NATIVE" "$SCRIPTS/k2b_index.json" 64 128 || die "input statistics failed"
  stage_close measure
fi

# STAGES train_s<seed> / eval_s<seed>
# ============================================================================
train_seed() { # train_seed <seed>
  local SD=$1 LOG="$OUT/logs/train_s$1.log" CFG="$OUT/cfg/loso_s$1.json"
  pyrun -m vesuvius.ink_detection.training.train "$CFG" > "$LOG" 2>&1 &
  local TP=$!
  while kill -0 "$TP" 2>/dev/null; do
    sleep 120
    say "TRAIN s$SD: $(tail -c 400 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-200) | ckpts=$(ls "$RUNS/s$SD" 2>/dev/null | grep -c 'ckpt_.*\.pth$')"
  done
  wait "$TP"
}
eval_seed() { # eval_seed <seed>: forward on every ckpt_*.pth, reverse on the best-of-grid
  local SD=$1 C TAG
  for C in $(ls "$RUNS/s$SD"/ckpt_*.pth | sort); do
    TAG="s${SD}_$(basename "$C" .pth | sed 's/ckpt_0*//')"
    pyrun "$SCRIPTS/evalf1.py" "$TAG" "$C" || die "eval failed for $TAG"
  done
  BEST=$(pyrun - "$SD" <<'PY'
import json, os, sys
ev = json.load(open(os.path.join(os.environ["RESULTS"], "eval.json")))
rows = {k: v for k, v in ev.items() if k.startswith(f"s{sys.argv[1]}_")}
print(max(rows, key=lambda k: rows[k]["native5_mean_best_f1"]))
PY
)
  say "eval s$SD: best-of-grid = $BEST -> reverse pass"
  pyrun "$SCRIPTS/evalf1.py" "$BEST" "$RUNS/s$SD/ckpt_$(printf %06d "${BEST#s${SD}_}").pth" --reverse || die "reverse eval failed for $BEST"
}
for SD in $SEEDS; do
  if stage_done "train_s$SD"; then
    say "=== STAGE train_s$SD already done, skipping ==="
  else
    stage_open "train_s$SD"
    if ! train_seed "$SD"; then
      tail -25 "$OUT/logs/train_s$SD.log" | while read -r L; do say "train_s$SD: $L"; done
      die "training seed $SD FAILED (logs/train_s$SD.log)"
    fi
    ls "$RUNS/s$SD"/ckpt_*.pth >/dev/null 2>&1 || die "training seed $SD produced no ckpt_*.pth"
    say "train_s$SD: done; checkpoints $(ls "$RUNS/s$SD" | grep 'ckpt_.*\.pth$' | tr '\n' ' ')"
    stage_close "train_s$SD"
  fi
  if stage_done "eval_s$SD"; then
    say "=== STAGE eval_s$SD already done, skipping ==="
  else
    stage_open "eval_s$SD"
    eval_seed "$SD"
    stage_close "eval_s$SD"
  fi
done

# ============================================================================
# STAGE ref -- the released in-scroll checkpoint on the same five crops.
# ============================================================================
if stage_done ref; then
  say "=== STAGE ref already done, skipping ==="
else
  stage_open ref
  pyrun "$SCRIPTS/evalf1.py" ref_released "$CKPT" --reverse || die "reference eval failed"
  stage_close ref
fi

# ============================================================================
# STAGE finalize -- inventory, gate (full runs only), bundles, ALL DONE.
# ============================================================================
stage_open finalize
pyrun "$SCRIPTS/finalize.py" || die "finalize refused (missing artifacts listed above)"
cp -f "$STATUS" "$OUT/status_at_done.txt" 2>/dev/null || true
for SD in $SEEDS; do cp -f "$RUNS/s$SD/sampling_observed.json" "$RESULTS/sampling_observed_s$SD.json" 2>/dev/null || true; done
( cd "$ROOT" && tar czf "$OUT/bundle.tgz.part" out/results out/cfg out/logs out/prereg.json out/status_at_done.txt out/sv_plan.json preds ) && mv -f "$OUT/bundle.tgz.part" "$OUT/bundle.tgz" || say "bundle tar FAILED (non-fatal)"
( cd "$RUNS" && tar czf "$OUT/ckpts_keep.tgz.part" $(for SD in $SEEDS; do ls s$SD/ckpt_*.pth 2>/dev/null; done) ) && mv -f "$OUT/ckpts_keep.tgz.part" "$OUT/ckpts_keep.tgz" || say "ckpts_keep tar FAILED (non-fatal)"
say "bundle: $(du -h "$OUT/bundle.tgz" 2>/dev/null | cut -f1) results+preds; ckpts_keep.tgz $(du -h "$OUT/ckpts_keep.tgz" 2>/dev/null | cut -f1)"
stage_close finalize
say "ALL DONE -- results.json, bundle.tgz, ckpts_keep.tgz served on :$PORT; the laptop guard harvests and TERMINATES."
echo IDLE > "$VAR/stage"
if [ "$LINGER_EXIT" = 1 ]; then exit 0; fi
while :; do sleep 300; say "IDLE -- ALL DONE; terminate the pod"; done
