
# ============================================================================
# DRY mode: exercise the machinery only (local validation; no network/GPU).
# ============================================================================
if [ "$DRY" = 1 ]; then
  for st in provision ckpt ctl_fetch ctl_build ctl_infer ctl_score p2a_fetch \
            p2a_build_win1 expa_baseline expa_rungs p2a_build_rest expb_infer \
            expb_score; do
    if stage_done "$st"; then say "=== STAGE $st already done, skipping ==="; continue; fi
    stage_open "$st"
    sleep 0.2
    stage_close "$st"
  done
  stage_open finalize
  say "DRY finalize: machinery only, skipping inventory (no real outputs)"
  stage_close finalize
  say "ALL DONE (DRY)"
  echo IDLE > "$VAR/stage"
  if [ "$LINGER_EXIT" = 1 ]; then exit 0; fi
  while :; do sleep 300; say "IDLE (DRY) -- terminate when done"; done
fi

@@PROVISION_AND_CKPT@@

# ============================================================================
# STAGE ctl_fetch / ctl_build / ctl_infer / ctl_score -- the in-domain
# positive controls on w035. They run FIRST: if this harness cannot read the
# control, or cannot reproduce the known depth-order fault, nothing measured
# on 500p2a afterwards would be interpretable, and the run dies here (~10 min,
# ~$0.15) instead of after the 500p2a fetch.
# ============================================================================
if stage_done ctl_fetch; then
  say "=== STAGE ctl_fetch already done, skipping ==="
else
  stage_open ctl_fetch
  retry 3 pyrun "$SCRIPTS/ctl_build.py" fetch || die "ctl chunk fetch failed"
  stage_close ctl_fetch
fi

if stage_done ctl_build; then
  say "=== STAGE ctl_build already done, skipping ==="
else
  stage_open ctl_build
  pyrun "$SCRIPTS/ctl_build.py" build || die "ctl build failed (an embedded-label sha256 or crop-stats gate tripping here means the harness input is wrong: evidence above)"
  stage_close ctl_build
fi

if stage_done ctl_infer; then
  say "=== STAGE ctl_infer already done, skipping ==="
else
  stage_open ctl_infer
  for ARM in ctl_native ctl_scalefault ctl_half; do
    run_infer "$DATA/$ARM.zarr" "$PREDS/$ARM.tif" both
  done
  stage_close ctl_infer
fi

if stage_done ctl_score; then
  say "=== STAGE ctl_score already done, skipping ==="
else
  stage_open ctl_score
  # '|| RC=$?' is REQUIRED: a bare call under 'set +e' still fires the ERR trap
  # (verified 2026-08-25), and fail_linger never returns.
  RC=0
  pyrun "$SCRIPTS/ctl_score.py" || RC=$?
  if [ $RC = 31 ]; then
    die "HARNESS_BROKEN -- ctl_native forward below the pre-registered 0.95 floor (on record 0.9991); the harness does not read the in-domain control, so no 500p2a number from it would mean anything (evidence in results/ctl.json + previews)"
  elif [ $RC = 32 ]; then
    die "DEPTH-ORDER FAULT NOT REPRODUCED -- ctl_native reverse above 0.80 (on record 0.5123); the harness cannot certify fault controls (evidence in results/ctl.json)"
  elif [ $RC != 0 ]; then
    die "ctl scoring failed rc=$RC"
  fi
  stage_close ctl_score
fi

# ============================================================================
# STAGE p2a_fetch -- 500p2a raster labels + the zarr chunks covering all three
# windows, from the public HF bucket (verified live 2026-08-24, re-verified
# 2026-09-01). 3675 chunks, measured 541 s on the c2c pod.
# ============================================================================
if stage_done p2a_fetch; then
  say "=== STAGE p2a_fetch already done, skipping ==="
else
  stage_open p2a_fetch
  retry 3 pyrun "$SCRIPTS/fetch_windows.py" || die "window fetch failed"
  stage_close p2a_fetch
fi

# ============================================================================
# STAGE p2a_build_win1 -- the anchor window first (native + iso + fit17).
# ============================================================================
if stage_done p2a_build_win1; then
  say "=== STAGE p2a_build_win1 already done, skipping ==="
else
  stage_open p2a_build_win1
  pyrun "$SCRIPTS/build_windows.py" win1 || die "win1 build/verify failed (a data gate tripping here means KILLED: evidence above)"
  stage_close p2a_build_win1
fi

# ============================================================================
# STAGE expa_baseline -- THE CORRECTED ANCHOR. ink_9um on win1 at the correct
# 2.215->9.36um resample, both depth modes, both directions, scored on the
# native 2.215um grid vs the human labels. The curve gate (0.85) decides only
# whether the rungs run; a low anchor is a RESULT and the run continues.
# ============================================================================
if stage_done expa_baseline; then
  say "=== STAGE expa_baseline already done, skipping ==="
else
  stage_open expa_baseline
  run_infer "$DATA/win1_iso.zarr" "$PREDS/expA_base.tif" both
  run_infer "$DATA/win1_fit17.zarr" "$PREDS/expA_base_fit17.tif" both
  RC=0
  pyrun "$SCRIPTS/score_expa.py" baseline || RC=$?
  if [ $RC = 21 ]; then
    touch "$VAR/gate_failed"
    say "CURVE GATE NOT MET -- the corrected anchor is below 0.85; rungs skipped; the anchor IS the result and win2/win3 + EXP B still run (evidence in results/expA_baseline.json + previews)"
  elif [ $RC != 0 ]; then
    die "baseline scoring failed rc=$RC"
  fi
  stage_close expa_baseline
fi
DIRECTION=$(cat "$VAR/direction.txt" 2>/dev/null || echo forward)
case $DIRECTION in forward) DIR_SHORT=fwd;; reverse) DIR_SHORT=rev;; *) DIR_SHORT=fwd;; esac

# ============================================================================
# STAGE expa_rungs -- iff the gate passed and RUN_RUNGS=1: build the degraded
# win1 volumes, infer (chosen direction), score the curve.
# ============================================================================
if [ "$RUN_RUNGS" = 1 ] && [ ! -f "$VAR/gate_failed" ]; then
  if stage_done expa_rungs; then
    say "=== STAGE expa_rungs already done, skipping ==="
  else
    stage_open expa_rungs
    pyrun "$SCRIPTS/build_expa.py" || die "rung build failed"
    for P in 3.24 4.32 5.5 6.5 8.0 12.0; do
      run_infer "$DATA/rung_p$P.zarr" "$PREDS/expA_p$P.tif" "$DIR_SHORT"
    done
    for K in 1 2 4 8; do
      run_infer "$DATA/rung_n$K.zarr" "$PREDS/expA_n$K.tif" "$DIR_SHORT"
    done
    run_infer "$DATA/rung_bit4.zarr" "$PREDS/expA_bit4.tif" "$DIR_SHORT"
    for S in 1.0 2.0; do
      run_infer "$DATA/rung_blur$S.zarr" "$PREDS/expA_blur$S.tif" "$DIR_SHORT"
    done
    pyrun "$SCRIPTS/score_expa.py" rungs || die "rung scoring failed"
    stage_close expa_rungs
  fi
else
  say "expa_rungs skipped (RUN_RUNGS=$RUN_RUNGS, gate_failed=$([ -f "$VAR/gate_failed" ] && echo yes || echo no))"
fi

# ============================================================================
# STAGE p2a_build_rest -- win2 + win3 (EXP B inputs), iso + fit17.
# ============================================================================
if stage_done p2a_build_rest; then
  say "=== STAGE p2a_build_rest already done, skipping ==="
else
  stage_open p2a_build_rest
  pyrun "$SCRIPTS/build_windows.py" win2 win3 || die "win2/win3 build/verify failed (a data gate tripping here means KILLED: evidence above)"
  stage_close p2a_build_rest
fi

# ============================================================================
# STAGE expb_infer -- fwd+rev on win2/win3, both depth modes (win1 reuses the
# EXP A baseline maps).
# ============================================================================
if stage_done expb_infer; then
  say "=== STAGE expb_infer already done, skipping ==="
else
  stage_open expb_infer
  for W in win2 win3; do
    for MODE in iso fit17; do
      run_infer "$DATA/${W}_${MODE}.zarr" "$PREDS/expB_${W}_${MODE}.tif" both
    done
  done
  stage_close expb_infer
fi

# ============================================================================
# STAGE expb_score -- real vs 40 matched translated nulls, gaps, verdicts.
# ============================================================================
if stage_done expb_score; then
  say "=== STAGE expb_score already done, skipping ==="
else
  stage_open expb_score
  pyrun "$SCRIPTS/score_expb.py" || die "expB scoring failed"
  stage_close expb_score
fi

# ============================================================================
# STAGE finalize -- aggregate + full inventory re-verification, then the
# result bundle the laptop guard harvests. ALL DONE is printed here and
# nowhere else.
# ============================================================================
stage_open finalize
pyrun "$SCRIPTS/finalize.py" || die "finalize inventory check refused (missing artifacts listed above)"
cp -f "$STATUS" "$OUT/status_at_done.txt" 2>/dev/null || true
if (cd "$ROOT" && tar czf "$OUT/bundle.tgz.part" out/results out/previews out/prereg.json out/status_at_done.txt preds); then
  mv -f "$OUT/bundle.tgz.part" "$OUT/bundle.tgz"
  say "bundle: $OUT/bundle.tgz ($(du -h "$OUT/bundle.tgz" | cut -f1)) -- results + previews + all prediction maps"
else
  rm -f "$OUT/bundle.tgz.part"
  say "bundle tar FAILED (non-fatal; results.json is still served at the root)"
fi
stage_close finalize
say "ALL DONE -- results.json + bundle.tgz served on :$PORT; the laptop guard harvests and TERMINATES."
echo IDLE > "$VAR/stage"

if [ "$LINGER_EXIT" = 1 ]; then exit 0; fi
while :; do
  sleep 300
  say "IDLE -- ALL DONE; fetch results and TERMINATE THE POD (it bills until you do)"
done
