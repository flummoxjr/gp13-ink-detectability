#!/bin/bash
# =============================================================================
# pod_p2a_v3.sh -- C3: the 500p2a transfer anchor RE-MEASURED AT THE CORRECT
# PITCH, with the fake-null positive controls the Aug-25 run lacked.      v3
#
#   WHY v3 EXISTS. The Aug-25 v2 run scored ink_9um at pixel AUC 0.5382 fwd /
#   0.5055 rev on 500p2a win1 and aborted as KILLED_BASELINE ("ink_9um reads
#   at chance on a clean foreign scroll"). On 2026-09-01 the surface volume's
#   pitch was re-derived and it is 2.215 um, NOT the 4.32 um that v1/v2 took
#   from the meta.json "volume" string:
#     (1) the mesh bbox (x <= 16037, z <= 27616) cannot fit the public 4.317um
#         volume (9423 x 15838) and fits only the 2.215um one (18209 x 28096);
#     (2) the surface-volume canvas 26239 x 16182 is 1:1 with those extents;
#     (3) label geometry: median letter-component height 2.49 mm and stroke
#         width 0.61 mm at 2.215 um (Herculaneum-typical) vs 4.86 mm / 1.19 mm
#         at 4.32 um (implausible).
#   Consequence: v2 resampled by 4.32/9.36 and fed the model data whose TRUE
#   pitch was 4.80 um -- 1.95x finer than its training pitch -- which is the
#   known "wrong in-plane level" fake-null fault (villa #1648/#1580: level-0
#   instead of level-2 costs ~0.31 AUC). The Aug-25 anchor is therefore VOID,
#   not a transfer result. This run re-measures it at the correct pitch and,
#   FIRST, proves on the in-domain control (PHerc0139 w035, on-record AUC
#   0.9991) that (a) this harness reproduces the control, (b) reproduces the
#   depth-order fault (on-record reverse 0.5123), and (c) reproduces the
#   SCALE fault by feeding w035 upsampled by exactly the factor v2 was wrong by.
#
#   EXPERIMENT CTL  positive controls on w035 (label-bbox crop, 3 arms x 2 dirs)
#   EXPERIMENT A    the corrected anchor: win1 at 2.215->9.36um, two depth
#                   modes, fwd+rev; the detectability curve iff the gate passes
#   EXPERIMENT B    hallucination audit (translation nulls) on win1/2/3
#
# Model under test: scrollprize/ink_9um hybrid_3d2d-seed42/step-075000.pth
# (flat, patch [17,128,128], robust_mad 1/99). Every input is uint8.
#
# OBSERVABILITY (v1/v2 machinery, verbatim): the status http server starts
# BEFORE anything else; BOOT / SERVE / PREREG are the first three lines;
# HEARTBEAT every 60 s; "=== STAGE x OPEN/DONE ===" per stage; "ALL DONE"
# only from finalize after a full inventory; any failure -> die() -> FAILED
# linger loop (ALL DONE unreachable). The pod script never calls the RunPod
# API; termination is the laptop guard's job (scripts/pod_guard.py).
#
# LAUNCH: as the pod dockerStartCmd via a gist raw URL (bench/p2a_v3/
# launch_v3.py); pod needs 1x RTX 5090 (or >= 24 GB), a runpod/pytorch
# image (python3+curl), >= 60 GB container disk, and port 8000/http exposed.
# Env knobs: PORT BATCH WORKERS FORCE=1 DRY=1 LINGER_EXIT=1 RUN_RUNGS=0
#
# COST (est.): provision 2-4m; ckpt <1m; ctl fetch 183 MB 1-3m; ctl build
# 3-6m; ctl infer 3-6m; p2a fetch 3675 chunks ~9m (measured 541 s on c2c);
# window builds 2-4m; 12 window inferences ~6m; rungs (if gate passes)
# ~15m; scoring 5-10m.  ~45-70 min => ~$0.55-0.85 on a $0.69/h 5090.
# =============================================================================
# PRE-REGISTRATION (locked in out/prereg.json before any data; the scoring
# code reads the same constants from curvelib.py)
#
# CTL (gates on the harness itself, decided BEFORE the anchor is looked at):
#   crop = w035 rows 512:2944, cols 384:3072 (the supervision-mask bbox padded
#   to 128-px chunk bounds); labels = the exact plane the 0.9991 control was
#   scored on (embedded, sha256-checked); pos = ink & sup (334,035 px),
#   neg = sup & ~ink (737,086 px).
#   ctl_native      the crop as released (28 layers; infer.py centre-crops 17)
#   ctl_scalefault  the crop upsampled by 4.32/2.215 = 1.9504 in-plane AND in
#                   depth (28 -> 55 layers), i.e. 4.80 um voxels presented as
#                   9.36 -- the exact v2 fault, applied to data the model reads
#   ctl_half        the crop downsampled by 0.5 (18.7 um voxels; 14 layers,
#                   infer.py zero-pads to 17) -- the opposite level fault
#   HARNESS GATE (fatal): ctl_native forward AUC >= 0.95 (on record 0.9991
#     on the full canvas). Below: HARNESS_BROKEN, the run dies, nothing else
#     is interpretable.
#   DEPTH-ORDER GATE (fatal): ctl_native reverse AUC <= 0.80 (on record
#     0.5123). Above: the harness does not reproduce the known depth-order
#     fault, so it cannot certify fault controls -- dies.
#   SCALE-FAULT VERDICT (non-fatal, verdict-bearing for the Aug-25 number):
#     FAULT_REPRODUCED  iff max(fwd,rev) of ctl_scalefault < 0.75  (the fault
#       class alone takes an in-domain 0.999 read to near chance -> the
#       Aug-25 0.538 carries no information about transfer);
#     FAULT_NOT_REPRODUCED iff >= 0.85 (a 1.95x scale error does NOT explain
#       the Aug-25 collapse; the corrected anchor must then be read on its own
#       and the Aug-25 number stands as unexplained until it is);
#     PARTIAL otherwise.
#
# EXP A (the corrected anchor):
#   pitch 2.215 um; windows unchanged (win1 y0=12416 x0=6912, 4438 px =
#   9.83 mm); resample 2.215 -> 9.36 (Gaussian AA sigma=(f-1)/2 source px,
#   linear, area-aligned grid_mode): in-plane 4438 -> 1050 px.
#   Depth: the released stack is 65 x 2.215 = 144 um, 91% of the model's
#   17 x 9.362 = 159 um window. Two modes, both run, both reported:
#     iso   (PRIMARY)  65 -> 15 layers at exactly 9.36 um; infer.py centres
#                      the short stack and zero-pads one layer each side;
#     fit17 (sensitivity) 65 -> 17 layers at 8.47 um (10% finer than trained,
#                      no padding).
#   Scoring: exact tie-corrected pixel rank-AUC on the NATIVE 2.215 um grid
#   (predictions bilinearly upsampled, 16-bit histogram bins); positives =
#   ink & mask (win1: 4,388,955), negatives = mask & ~ink (win1: 8,576,280);
#   win1's 469 out-of-mask ink px excluded from both classes.
#   THE CORRECTED ANCHOR = max(fwd, rev) of win1 iso. Reported alongside:
#   win1 fit17, win2/win3 both modes, both directions -- the run does NOT
#   abort on a low anchor (v2 did, which is why win2/win3 were never
#   measured). CURVE GATE: rungs run iff anchor >= 0.85 (unchanged from
#   v1/v2: the minimum that can order the rungs). Below 0.85 the anchor is
#   simply the result, with the CTL verdict saying how to read it.
#   Pre-stated reading of the anchor: >= 0.85 -> the Aug-25 "chance on a
#   clean foreign scroll" was an input fault and ink_9um does transfer
#   in-modality to an unseen scroll at curve-anchoring quality; 0.65-0.85 ->
#   partial transfer (Bet A's 500p2a gate rebases to anchor+0.05); < 0.65 ->
#   transfer failure confirmed at the correct pitch.
#   RUNGS (iff gate): pitch {3.24,4.32,5.5,6.5,8.0,12.0} via 2.215->P->9.36
#   (the 9.36 and 2.215 rows reuse baseline); noise k in {1,2,4,8} x
#   sigma_plate (adjacent-slice MAD over in-mask voxels of the iso stack);
#   bit4; blur sigma {1.0,2.0} px. DETECTABLE <=> AUC >= 0.75 AND retained
#   >= 0.50, retained = (AUC-0.5)/(AUC_base-0.5).
#
# EXP B (hallucination audit, v2 rules, annulus corrected for the true
#   window size): 40 rigid translations of the (label & mask) shape image per
#   window, seed 20260901+idx, |shift|_inf in [3.75 mm, 0.60*min(H,W)] --
#   v2's [4.4 mm, 0.40*size] is EMPTY at 2.215 um (4.4 mm = 1987 px > 0.40 x
#   4438 = 1775 px); 3.75 mm = 1.5 x the measured 2.49 mm median letter
#   height keeps translated strokes off their sources. >= 50,000 pseudo-
#   positive px per draw; pos_i = T_i(shapes) & mask & ~label; neg_i = mask &
#   ~label & ~T_i(shapes). SHAPE_CONFOUNDED iff null median >= 0.60 OR >= 20%
#   of draws >= 0.60; GENUINE iff not confounded AND real > max(null) AND
#   gap = real - median(null) >= 0.15; else INDETERMINATE. Audited cell per
#   window = the iso direction with the higher real AUC; fit17 cells reported.
#
# DATA GATES (exact, else FATAL): 500p2a raster counts and window counts equal
#   the embedded values; window volume subsample stats within +-0.5 DN of the
#   Aug-25 record; embedded w035 labels sha256-match; checkpoint size exact.
# =============================================================================

set -Eeuo pipefail

# ---------------------------------------------------------------- config ----
ROOT=${ROOT:-/workspace/p2a}
PORT=${PORT:-8000}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}
FORCE=${FORCE:-0}
DRY=${DRY:-0}
DRY_FAIL_STAGE=${DRY_FAIL_STAGE:-}
LINGER_EXIT=${LINGER_EXIT:-0}
RUN_RUNGS=${RUN_RUNGS:-1}
PYTHON_BIN=${PYTHON_BIN:-python3}
SEED=20260901

OUT=$ROOT/out;  VAR=$ROOT/var;  DATA=$ROOT/data;  PREDS=$ROOT/preds
SCRIPTS=$ROOT/scripts;  RESULTS=$OUT/results;  STATUS=$OUT/status.txt
export ROOT OUT VAR DATA PREDS SCRIPTS RESULTS STATUS SEED BATCH WORKERS
export RUN_RUNGS

# =================================================================== L1 ======
# The very first actions: make the served dir and write the BOOT line.
mkdir -p "$OUT" "$VAR" "$DATA" "$PREDS" "$SCRIPTS" "$RESULTS" "$OUT/previews" "$DATA/tmp"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(now) $*" >> "$STATUS"; echo "$(now) $*"; }
say "BOOT pod_p2a_v3 pid=$$ host=${HOSTNAME:-unknown} root=$ROOT -- status live; server next"

# =================================================================== L2 ======
# Serve the status dir BEFORE anything else (provisioning included).
SERVER_PID=""
start_server() {
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m http.server "$PORT" --bind 0.0.0.0 --directory "$OUT" \
      >/dev/null 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$VAR/server.pid"
  fi
}
start_server
PROBE="FAILED"
for _ in 1 2 3 4 5 6; do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/status.txt" 2>/dev/null; then
    PROBE="200"; break
  fi
  sleep 0.5
done
EXT_URL="(RUNPOD_POD_ID unset -- use 'ssh ... tail -f $STATUS' or http://<pod-ip>:$PORT/status.txt)"
[ -n "${RUNPOD_POD_ID:-}" ] && EXT_URL="https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net/status.txt"
if [ "$PROBE" = 200 ]; then
  say "SERVE http.server pid=${SERVER_PID:-none} 0.0.0.0:$PORT dir=$OUT external=$EXT_URL local_probe=200"
else
  say "SERVE DEGRADED -- local probe failed (python3 missing or port busy); status.txt still written; will retry at every heartbeat. external=$EXT_URL"
fi

# heartbeat + server watchdog, child of this script
( STARTED=$SECONDS
  while :; do
    sleep 60
    ST=$(cat "$VAR/stage" 2>/dev/null || echo boot)
    DF=$(df -h "$ROOT" 2>/dev/null | awk 'NR==2{print $4}' || echo '?')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) HEARTBEAT stage=$ST up=$((SECONDS))s free=$DF" >> "$STATUS"
    SP=$(cat "$VAR/server.pid" 2>/dev/null || true)
    if [ -n "$SP" ] && ! kill -0 "$SP" 2>/dev/null; then
      if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        "$PYTHON_BIN" -m http.server "$PORT" --bind 0.0.0.0 --directory "$OUT" >/dev/null 2>&1 &
        echo $! > "$VAR/server.pid"
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SERVE RESTART pid=$(cat "$VAR/server.pid")" >> "$STATUS"
      fi
    fi
  done
) &
HEART_PID=$!
echo boot > "$VAR/stage"

cleanup() {
  kill "$HEART_PID" 2>/dev/null || true
  SP=$(cat "$VAR/server.pid" 2>/dev/null || true)
  [ -n "$SP" ] && kill "$SP" 2>/dev/null || true
}
trap cleanup EXIT

# =================================================================== L3 ======
# Lock the pre-registration into the served dir before any provisioning.
cat > "$OUT/prereg.json" <<'PREREG_JSON'
{
  "locked_before": "any provisioning, download, or data contact",
  "version": "v3 -- C3: 500p2a anchor RE-MEASURED at the CORRECT pitch (2.215 um, not 4.32) with in-domain positive controls for the harness, the depth-order fault and the scale fault; v2's 0.5382/0.5055 is VOID (input fault: data at 4.80 um presented as 9.36)",
  "seed": 20260901,
  "pitch_correction": {
    "old_um": 4.32,
    "new_um": 2.215,
    "evidence": [
      "mesh bbox x<=16037, z<=27616 fits only the 2.215um volume (18209x28096), not the 4.317um one (9423x15838)",
      "surface-volume canvas 26239x16182 is 1:1 with the 2.215um voxel extents",
      "label geometry: median component height 2.49 mm, stroke width 0.61 mm at 2.215um vs 4.86/1.19 mm at 4.32um"
    ],
    "fault_factor": 1.9504
  },
  "ctl": {
    "input": "PHerc0139 w035 native 9.362um surface volume, crop rows 512:2944 cols 384:3072 (supervision bbox padded to 128); labels = the plane the on-record 0.9991 control was scored on, embedded + sha256-checked; pos = ink & sup = 334035 px, neg = sup & ~ink = 737086 px",
    "arms": {
      "ctl_native": "as released (28 layers, centre-cropped to 17 by infer.py)",
      "ctl_scalefault": "upsampled x1.9504 in-plane and depth (28->55 layers): 4.80um voxels presented as 9.36 -- the exact v2 fault",
      "ctl_half": "downsampled x0.5 (14 layers, zero-padded to 17 by infer.py)"
    },
    "harness_gate_fatal": "ctl_native forward AUC >= 0.95, else HARNESS_BROKEN and the run dies",
    "depth_order_gate_fatal": "ctl_native reverse AUC <= 0.80 (on record 0.5123), else the harness cannot certify fault controls and the run dies",
    "scale_fault_verdict": "FAULT_REPRODUCED iff max(fwd,rev) of ctl_scalefault < 0.75; FAULT_NOT_REPRODUCED iff >= 0.85; PARTIAL otherwise; non-fatal; decides how the Aug-25 0.538 is read"
  },
  "expA": {
    "anchor": "win1 (y0=12416, x0=6912, 4438 px @ 2.215um = 9.83 mm); resample 2.215->9.36um: in-plane 4438->1050 px",
    "depth_modes": {
      "iso": "PRIMARY: 65->15 layers at 9.36um, infer.py zero-pads 1 layer each side (stack is 144um vs the 159um window)",
      "fit17": "sensitivity: 65->17 layers at 8.47um, no padding"
    },
    "scoring": "exact tie-corrected pixel rank-AUC on the native 2.215um grid; predictions bilinearly upsampled; 16-bit histogram bins (map*257); positives = ink & mask (4388955), negatives = mask & ~ink (8576280); 469 out-of-mask ink px excluded from both classes",
    "corrected_anchor": "max(forward, reverse) of win1 iso; win1 fit17 and win2/win3 (both modes, both directions) reported alongside; the run does NOT abort on a low anchor",
    "curve_gate": "rungs run iff corrected anchor >= 0.85 (unchanged from v1/v2)",
    "prestated_reading": {
      ">=0.85": "Aug-25 chance result was an input fault; ink_9um transfers in-modality to an unseen scroll at curve-anchoring quality",
      "0.65-0.85": "partial transfer; Bet A's 500p2a gate rebases to anchor+0.05",
      "<0.65": "transfer failure confirmed at the correct pitch (read jointly with the CTL scale-fault verdict)"
    },
    "pitch_rungs_um": [3.24, 4.32, 5.5, 6.5, 8.0, 9.36, 12.0],
    "pitch_harness": "anti-aliased resample 2.215->P (Gaussian sigma=(f-1)/2 source px, linear, grid_mode) then regrid P->9.36 iso; the 9.36 row is baseline by construction; the 2.215 row is the native acquisition == baseline",
    "noise_rungs": "sigma_add = k*sigma_plate, k in {1,2,4,8}, on the 9.36 iso grid, clipped to uint8; sigma_plate = 1.4826*median|(I_z - I_{z+1})/sqrt(2)| over in-mask voxels of the iso stack",
    "bitdepth_rung": "bit4: uint8 -> 16 levels via rint(v/17)*17 (the released volume is already uint8, so bit8 == baseline)",
    "blur_rungs_sigma_px": [1.0, 2.0],
    "detectable_rule": "DETECTABLE <=> AUC_rung >= 0.75 AND (AUC_rung-0.5)/(AUC_base-0.5) >= 0.50; detectability limit = coarsest DETECTABLE pitch"
  },
  "expB": {
    "inputs": ["win1", "win2", "win3"],
    "cells": "iso fwd/rev (primary; the audited cell is the iso direction with the higher real AUC) and fit17 fwd/rev (reported, never verdict-bearing); win1 iso cells reuse the EXP A baseline maps",
    "real_auc": "pos = label & mask, neg = mask & ~label; win3's 621303 out-of-mask ink px excluded from both classes",
    "null": "40 rigid translations of the (label & mask) shape image, seed 20260901+window index, drawn once per window on the native grid; |shift|_inf in [3.75 mm, 0.60*min(H,W)] (v2's [4.4 mm, 0.40] is EMPTY at 2.215um: 1987 px > 1775 px; 3.75 mm = 1.5 x the measured 2.49 mm median letter height); accepted iff >= 50000 pseudo-positive px; pos_i = T_i(shapes) & mask & ~label; neg_i = mask & ~label & ~T_i(shapes)",
    "shape_confounded_threshold": "median of the 40 translated-label AUCs >= 0.60, OR >= 20% of the 40 individually >= 0.60",
    "genuine_rule": "GENUINE <=> not SHAPE_CONFOUNDED AND AUC_real > max of the 40 AND gap = AUC_real - median(translated) >= 0.15; otherwise INDETERMINATE",
    "headline": "gap = AUC_real - median(translated AUC), per cell"
  },
  "data_gates": "exact label-count asserts for the 500p2a raster and every window; window volume subsample stats within +-0.5 DN of the Aug-25 record; embedded w035 labels sha256-match; checkpoint 138360039 bytes with mode=flat crop=(17,128,128) norm=robust_mad; any mismatch -> FATAL with evidence"
}
PREREG_JSON
PRSHA=$(sha256sum "$OUT/prereg.json" | cut -c1-12)
say "PREREG locked prereg.json sha256=$PRSHA -- decision rules recorded before any provisioning, download, or data"

# ------------------------------------------------------------ machinery -----
CURRENT_STAGE=boot
fail_linger() {
  touch "$OUT/FAILED"
  echo FAILED > "$VAR/stage"
  say "FAILED -- run is dead; status + logs stay served on :$PORT; fix or fetch, then TERMINATE THE POD (it bills until you do)"
  if [ "$LINGER_EXIT" = 1 ]; then exit 1; fi
  while :; do sleep 300; say "FAILED (still lingering; terminate the pod when done reading)"; done
}
die() { say "FATAL stage=$CURRENT_STAGE $*"; fail_linger; }
on_err() {
  local ec=$?
  say "FATAL stage=$CURRENT_STAGE exit=$ec line=${BASH_LINENO[0]} cmd: ${BASH_COMMAND}"
  fail_linger
}
trap on_err ERR

stage_open() {
  CURRENT_STAGE=$1
  echo "$1" > "$VAR/stage"
  STAGE_T0=$SECONDS
  say "=== STAGE $1 OPEN ==="
  if [ "$DRY" = 1 ] && [ "$DRY_FAIL_STAGE" = "$1" ]; then die "DRY injected failure"; fi
}
stage_close() {
  touch "$VAR/done_$1"
  say "=== STAGE $1 DONE ($((SECONDS - STAGE_T0))s) ==="
}
stage_done() { [ "$FORCE" != 1 ] && [ -f "$VAR/done_$1" ]; }

retry() { # retry <tries> <cmd...>  backoff 10/30/90
  local tries=$1; shift
  local n=1 waits=(0 10 30 90 180)
  while :; do
    if "$@"; then return 0; fi
    if [ $n -ge "$tries" ]; then return 1; fi
    say "retry $n/$tries failed: $1 -- backoff ${waits[$n]}s"
    sleep "${waits[$n]}"; n=$((n + 1))
  done
}

pyrun() { (cd /workspace/villa/vesuvius && uv run --no-sync --extra models python "$@"); }

run_infer() { # run_infer <zarr> <out_base_path.tif> <direction fwd|rev|both>
  local zarr=$1 out=$2 dir=$3 d flag
  case $dir in fwd) flag=forward;; rev) flag=reverse;; both) flag=both;; esac
  local rev=${out%.tif}_reverse.tif
  if [ -s "$out" ] && { [ "$flag" != both ] || [ -s "$rev" ]; } && [ "$FORCE" != 1 ]; then
    say "infer skip (exists): $(basename "$out") [$flag]"; return 0
  fi
  local tmp=$PREDS/tmp_$(basename "$out")
  local tmprev=${tmp%.tif}_reverse.tif
  rm -f "$tmp" "$tmprev"
  say "infer OPEN $(basename "$zarr") -> $(basename "$out") [$flag]"
  local t0=$SECONDS
  if ! (cd /workspace/villa/vesuvius && uv run --no-sync --extra models \
        python -m vesuvius.ink_detection.inference.infer \
        "$zarr" "$CKPT" "$tmp" --direction "$flag" \
        --batch-size "$BATCH" --num-workers "$WORKERS" --gpus 0 --no-compile); then
    rm -f "$tmp" "$tmprev"
    say "infer FIRST ATTEMPT FAILED $(basename "$out") -- retrying once in 20s"
    sleep 20
    (cd /workspace/villa/vesuvius && uv run --no-sync --extra models \
        python -m vesuvius.ink_detection.inference.infer \
        "$zarr" "$CKPT" "$tmp" --direction "$flag" \
        --batch-size "$BATCH" --num-workers "$WORKERS" --gpus 0 --no-compile) \
      || { rm -f "$tmp" "$tmprev"; die "inference failed twice: $(basename "$out")"; }
  fi
  [ -s "$tmp" ] || die "inference produced no output: $(basename "$out")"
  mv -f "$tmp" "$out"
  if [ "$flag" = both ]; then
    [ -s "$tmprev" ] || die "direction both produced no reverse output: $(basename "$rev")"
    mv -f "$tmprev" "$rev"
  fi
  say "infer DONE $(basename "$out") [$flag] ($((SECONDS - t0))s)"
}

# ============================================================================
# The embedded python programs. Written before any stage runs so the exact
# analysis code is on disk (and served conventions locked) up front.
# ============================================================================


# ============================================================================
# The embedded programs. Written before any stage runs so the exact analysis
# code is on disk (and served) up front.
# ============================================================================
write_scripts() {

cat > "$SCRIPTS/curvelib.py" <<'PY_LIB'
"""Shared library for pod_curve_audit. Env: ROOT/OUT/DATA/PREDS/RESULTS/STATUS/SEED."""
import json, os, sys, time, hashlib
import numpy as np

ROOT = os.environ["ROOT"]; OUT = os.environ["OUT"]; DATA = os.environ["DATA"]
PREDS = os.environ["PREDS"]; RESULTS = os.environ["RESULTS"]
STATUS = os.environ["STATUS"]; SEED = int(os.environ.get("SEED", "20260824"))

def say(msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg
    with open(STATUS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# ---------------------------------------------------------------- expected --
# Every number below was measured against the live servers and the laptop
# ground-truth copies BEFORE this script was written. Parse, then assert --
# never assume.
MODEL_PITCH = 9.36

# 500p2a -- PITCH CORRECTED 2026-09-01. v1/v2 used 4.32 um, read off the
# meta.json "volume" string. Three independent measurements say 2.215 um:
#   (1) mesh bbox x<=16037, z<=27616 cannot fit the 4.317um volume
#       (9423 x 15838) and fits only the 2.215um one (18209 x 28096);
#   (2) the surface-volume canvas 26239 x 16182 is 1:1 with those extents;
#   (3) label geometry: median component height 2.49 mm, stroke width 0.61 mm
#       at 2.215um (Herculaneum-typical) vs 4.86 / 1.19 mm at 4.32um.
# See trackD/bench/P2A_PITCH_RESOLUTION.md.
BUCKET = ("https://huggingface.co/buckets/scrollprize/datasets/resolve/"
          "ink/unused/500p2a")
P2A_SHAPE = (65, 26239, 16182)             # .zarray, verified live
P2A_CHUNK = (65, 128, 128)
P2A_PITCH = 2.215                           # CORRECTED (was 4.32)
P2A_PITCH_WRONG = 4.32                      # the Aug-25 value; fault arithmetic only
P2A_LABEL_BYTES = 3009475
P2A_MASK_BYTES = 2983168
P2A_RASTER = dict(ink=12856732, mask=34871346, ink_and_mask=12230762,
                  ink_outside_mask=625970, annot_blank=22640584)
WINDOWS = {
    "win1": dict(y0=12416, x0=6912, size=4438, ink=4389424, ink_and_mask=4388955,
                 ink_outside_mask=469, annot_blank=8576280, mask=12965235,
                 vol_mean=76.114, vol_std=32.313, vol_zero_frac=0.0),
    "win2": dict(y0=18432, x0=7424, size=4438, ink=2862504, ink_and_mask=2862504,
                 ink_outside_mask=0, annot_blank=4778034, mask=7640538,
                 vol_mean=72.612, vol_std=32.042, vol_zero_frac=0.0),
    "win3": dict(y0=12160, x0=2432, size=4438, ink=2158076, ink_and_mask=1536773,
                 ink_outside_mask=621303, annot_blank=2237639, mask=3774412,
                 vol_mean=74.129, vol_std=31.922, vol_zero_frac=0.00023),
}
NZ_WIN, S_WIN = 65, 4438
# depth modes (asserted against rint_shape below, after it is defined)
ISO_NZ = 15          # 65 * 2.215 / 9.36 = 15.38 -> 15 (infer.py zero-pads to 17)
FIT17_NZ = 17        # 65 layers -> 17 at 8.47 um
S9 = 1050            # 4438 * 2.215 / 9.36 = 1050.2 -> 1050
DEPTH_MODES = {"iso": ISO_NZ, "fit17": FIT17_NZ}
PRIMARY_DEPTH = "iso"

# CTL -- PHerc0139 w035 native 9.362um surface volume (public S3, zarr v2,
# level 0 [28,5820,5240] uint8, chunks [28,128,128], compressor null,
# dimension_separator "/"; verified live 2026-09-01).
CTL_SV = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/"
          "segments/20260317000000-w035_2026031718/surface-volumes/"
          "9.362um-1.2m-113keV-volume-20250728140407.zarr")
CTL_SHAPE = (28, 5820, 5240)
CTL_CHUNK = (28, 128, 128)
CTL_CHUNK_BYTES = 28 * 128 * 128           # 458752, raw (compressor null)
CTL_PITCH = 9.362
CTL_CROP = (512, 2944, 384, 3072)          # y0, y1, x0, x1 (128-aligned)
CTL_N_POS = 334035                         # ink & sup inside the crop
CTL_N_NEG = 737086                         # sup & ~ink inside the crop
CTL_LABELS_SHA = "23ad57aed651ca8ff81e13bd4b829d359af7d6711fb904b68168c850e84aa4cf"
CTL_FAULT_FACTOR = P2A_PITCH_WRONG / P2A_PITCH   # 1.9504 -- exactly v2's error
CTL_HALF_FACTOR = 0.5
CTL_HARNESS_MIN_FWD = 0.95
CTL_DEPTHREV_MAX = 0.80
CTL_FAULT_REPRODUCED_MAX = 0.75
CTL_FAULT_NOT_REPRODUCED_MIN = 0.85

# ------------------------------------------------------------------ prereg --
GATE_BASELINE_AUC = 0.85
EXPA_ANCHOR = "win1"
DETECT_AUC_MIN = 0.75
DETECT_RETAIN_MIN = 0.50
PITCHES = [3.24, 4.32, 5.5, 6.5, 8.0, 9.36, 12.0]
NOISE_KS = [1, 2, 4, 8]
BLUR_SIGMAS = [1.0, 2.0]
N_TRANSLATIONS = 40
MIN_SHIFT_MM = 3.75           # 1.5 x the measured 2.49 mm median letter height
MAX_SHIFT_FRAC = 0.60         # v2's 0.40 leaves an EMPTY annulus at 2.215um
MIN_PSEUDO_POS = 50000
CONFOUND_MEDIAN_AUC = 0.60
CONFOUND_FRAC = 0.20          # OR-clause: >= 20% of nulls at/above 0.60
GAP_MIN = 0.15

# --------------------------------------------------------------- resampling -
def _aa_sigma(f):
    """skimage-convention anti-aliasing sigma for downscale factor f (>1)."""
    return max(0.0, (f - 1.0) / 2.0)

def zoom_to(arr, out_shape, prefilter=True):
    """Area-aligned linear resample of a float32 array to an EXACT shape,
    with per-axis Gaussian anti-alias prefilter on downsampled axes."""
    from scipy import ndimage
    arr = np.asarray(arr, dtype=np.float32)
    in_shape = arr.shape
    if prefilter:
        sigmas = []
        for i, o in zip(in_shape, out_shape):
            f = i / o
            sigmas.append(_aa_sigma(f) if f > 1.0 else 0.0)
        if any(s > 0 for s in sigmas):
            arr = ndimage.gaussian_filter(arr, sigma=sigmas, mode="nearest")
    factors = [o / i for i, o in zip(in_shape, out_shape)]
    out = ndimage.zoom(arr, factors, order=1, mode="nearest", grid_mode=True)
    if out.shape != tuple(out_shape):
        raise AssertionError(f"zoom shape {out.shape} != target {tuple(out_shape)}")
    return out.astype(np.float32)

def resample_stack(layer_get, n_in, H, W, out_shape, tmp_path, band=384, tag=""):
    """(n_in,H,W) -> float32 memmap of out_shape via separable two-pass
    (z first in y-bands, then per-slice yx). layer_get(z) -> (H,W) float-able."""
    from scipy import ndimage
    nz_out, Ht, Wt = out_shape
    fz = n_in / nz_out
    sz = _aa_sigma(fz) if fz > 1.0 else 0.0
    ztmp = tmp_path + ".zpass.f32"
    zmm = np.memmap(ztmp, dtype=np.float32, mode="w+", shape=(nz_out, H, W))
    for y0 in range(0, H, band):
        y1 = min(H, y0 + band)
        buf = np.empty((n_in, y1 - y0, W), dtype=np.float32)
        for z in range(n_in):
            buf[z] = layer_get(z)[y0:y1, :]
        if sz > 0:
            buf = ndimage.gaussian_filter1d(buf, sigma=sz, axis=0, mode="nearest")
        zb = ndimage.zoom(buf, (nz_out / n_in, 1, 1), order=1, mode="nearest",
                          grid_mode=True)
        assert zb.shape[0] == nz_out, (zb.shape, nz_out)
        zmm[:, y0:y1, :] = zb
    zmm.flush()
    out = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(nz_out, Ht, Wt))
    for z in range(nz_out):
        out[z] = zoom_to(np.asarray(zmm[z]), (Ht, Wt))
        if z % 8 == 0:
            say(f"resample {tag} slice {z + 1}/{nz_out}")
    out.flush()
    del zmm
    os.remove(ztmp)
    return out

def rint_shape(n, p_in, p_out):
    """Half-up rounding with epsilon: deterministic at exact .5 boundaries
    (65*3.24/9.36 = 22.5 exactly -> 23; np.rint would be FP-jitter fragile)."""
    return max(1, int(np.floor(n * p_in / p_out + 0.5 + 1e-9)))

# ------------------------------------------------------------------- zarr ---
def write_group_zarr(path, vol):
    """Write a Zarr-v2 group with level '0' (the volume) and a binary
    occupancy level '3' (YX max-pool by 8), atomically via <path>.tmp."""
    import shutil
    from numcodecs import Blosc
    from vesuvius.label_zarr import open_v2_group, create_v2_array
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    n, H, W = vol.shape
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    group = open_v2_group(tmp)
    a0 = create_v2_array(group, "0", shape=(n, H, W), chunks=(n, 256, 256),
                         dtype=vol.dtype, compressor=comp, fill_value=0)
    for y0 in range(0, H, 1024):
        a0[:, y0:min(H, y0 + 1024), :] = vol[:, y0:min(H, y0 + 1024), :]
    p = 8
    Hp, Wp = (H + p - 1) // p, (W + p - 1) // p
    a3 = create_v2_array(group, "3", shape=(n, Hp, Wp), chunks=(n, 256, 256),
                         dtype=np.uint8, compressor=comp, fill_value=0)
    for y0 in range(0, H, 4096):
        y1 = min(H, y0 + 4096)
        blk = np.asarray(vol[:, y0:y1, :])
        h = blk.shape[1]
        ph, pw = (-h) % p, (-W) % p
        if ph or pw:
            blk = np.pad(blk, ((0, 0), (0, ph), (0, pw)))
        pooled = blk.reshape(n, (h + ph) // p, p, (W + pw) // p, p).max(axis=(2, 4))
        a3[:, y0 // p: y0 // p + pooled.shape[1], :] = \
            ((pooled > 0) * np.uint8(255))
    if os.path.exists(path):
        shutil.rmtree(path)
    os.rename(tmp, path)

def read_zarr0(path):
    import zarr
    g = zarr.open_group(path, mode="r")
    return g["0"]

def quant4(v8):
    """uint8 -> 4-bit (16 levels {0,17,...,255}); the v2 bit-depth stressor.
    The released 4.32um 500p2a volume is ALREADY uint8, so the v1
    uint16->uint8 rung is the baseline by construction here."""
    x = np.rint(np.asarray(v8, dtype=np.float32) / 17.0) * 17.0
    return np.clip(x, 0, 255).astype(np.uint8)

# -------------------------------------------------------------------- AUC ---
NBINS = 65536

def quantize_map(m):
    """Map (uint8 native or float 0..255 upsampled) -> uint16 bins 0..65535."""
    q = np.rint(np.asarray(m, dtype=np.float32) * 257.0)
    return np.clip(q, 0, NBINS - 1).astype(np.uint16)

def masked_hist(q, mask):
    return np.bincount(q[mask], minlength=NBINS).astype(np.float64)

def hist_auc(hpos, hneg):
    """Exact tie-corrected rank AUC from per-bin histograms."""
    P, N = hpos.sum(), hneg.sum()
    if P == 0 or N == 0:
        return float("nan")
    cneg_below = np.concatenate([[0.0], np.cumsum(hneg)[:-1]])
    return float((hpos * (cneg_below + 0.5 * hneg)).sum() / (P * N))

def upsample_pred(pred, out_hw):
    """Bilinear, area-aligned upsample of a prediction map to the native grid."""
    return zoom_to(np.asarray(pred, dtype=np.float32), out_hw, prefilter=False)

# ------------------------------------------------------------ translations --
def draw_translations(shapes, blank, pitch_um, seed, n=N_TRANSLATIONS):
    """40 rigid (dy,dx) shifts, |shift|_inf in [4.4mm, 0.40*min(H,W)],
    each leaving >= MIN_PSEUDO_POS pseudo-positive px inside blank."""
    H, W = shapes.shape
    min_px = int(np.ceil(MIN_SHIFT_MM * 1000.0 / pitch_um))
    max_px = int(np.floor(MAX_SHIFT_FRAC * min(H, W)))
    if max_px <= min_px:
        raise AssertionError(f"translation annulus empty: {min_px}..{max_px}")
    rng = np.random.default_rng(seed)
    out, tried = [], 0
    while len(out) < n:
        tried += 1
        if tried > 4000:
            raise AssertionError(
                f"could not draw {n} valid translations (got {len(out)})")
        dy, dx = (int(v) for v in rng.integers(-max_px, max_px + 1, size=2))
        if max(abs(dy), abs(dx)) < min_px or (dy, dx) in out:
            continue
        if shifted_count(shapes, blank, dy, dx) < MIN_PSEUDO_POS:
            continue
        out.append((dy, dx))
    return out, min_px, max_px

def _shift_slices(H, W, dy, dx):
    sy0, sy1 = max(0, -dy), min(H, H - dy)
    dy0, dy1 = max(0, dy), min(H, H + dy)
    sx0, sx1 = max(0, -dx), min(W, W - dx)
    dx0, dx1 = max(0, dx), min(W, W + dx)
    return (slice(sy0, sy1), slice(sx0, sx1)), (slice(dy0, dy1), slice(dx0, dx1))

def shifted_count(shapes, blank, dy, dx):
    (ssy, ssx), (dsy, dsx) = _shift_slices(*shapes.shape, dy, dx)
    return int(np.count_nonzero(shapes[ssy, ssx] & blank[dsy, dsx]))

def translated_hist(q, shapes, blank, dy, dx):
    """Histogram of map values on T(shapes) & blank (the pseudo-positives)."""
    (ssy, ssx), (dsy, dsx) = _shift_slices(*shapes.shape, dy, dx)
    sel = shapes[ssy, ssx] & blank[dsy, dsx]
    return np.bincount(q[dsy, dsx][sel], minlength=NBINS).astype(np.float64)

# ----------------------------------------------------------------- preview --
def save_preview(arr2d, path, ds=4):
    from PIL import Image
    a = np.asarray(arr2d, dtype=np.float32)[::ds, ::ds]
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)

# ------------------------------------------------------- v3 additions -------
SCRIPTS_DIR = os.environ.get("SCRIPTS", os.path.dirname(os.path.abspath(__file__)))

def resample_pred(pred, out_hw):
    """Prediction map -> an exact target grid; anti-aliased iff downsampling.
    (upsample_pred is kept for the native-grid upsampling path.)"""
    pred = np.asarray(pred, dtype=np.float32)
    if tuple(pred.shape) == tuple(out_hw):
        return pred
    down = any(i > o for i, o in zip(pred.shape, out_hw))
    return zoom_to(pred, out_hw, prefilter=down)

def load_ctl_labels():
    """The embedded w035 crop labels (packbits -> zlib -> base64), verified by
    sha256 of the raw packed bits and by exact class counts."""
    import base64, zlib
    b64 = open(os.path.join(SCRIPTS_DIR, "ctl_labels.b64")).read().strip()
    raw = zlib.decompress(base64.b64decode(b64))
    got = hashlib.sha256(raw).hexdigest()
    assert got == CTL_LABELS_SHA, f"ctl labels sha256 {got} != {CTL_LABELS_SHA}"
    y0, y1, x0, x1 = CTL_CROP
    H, W = y1 - y0, x1 - x0
    a = np.unpackbits(np.frombuffer(raw, np.uint8))[:2 * H * W]
    a = a.reshape(2, H, W).astype(bool)
    ink, sup = a[0], a[1]
    pos, neg = ink & sup, sup & ~ink
    assert int(pos.sum()) == CTL_N_POS, int(pos.sum())
    assert int(neg.sum()) == CTL_N_NEG, int(neg.sum())
    return pos, neg

def ctl_arm_shape(factor):
    """Output (nz, H, W) of a CTL resample by `factor` (half-up rounding)."""
    y0, y1, x0, x1 = CTL_CROP
    nz = CTL_SHAPE[0]
    def r(n):
        return max(1, int(np.floor(n * factor + 0.5 + 1e-9)))
    return (r(nz), r(y1 - y0), r(x1 - x0))

# consistency asserts (fail at import time, i.e. before any stage runs)
assert rint_shape(NZ_WIN, P2A_PITCH, MODEL_PITCH) == ISO_NZ, \
    rint_shape(NZ_WIN, P2A_PITCH, MODEL_PITCH)
assert rint_shape(S_WIN, P2A_PITCH, MODEL_PITCH) == S9, \
    rint_shape(S_WIN, P2A_PITCH, MODEL_PITCH)
assert ctl_arm_shape(CTL_FAULT_FACTOR) == (55, 4743, 5243), \
    ctl_arm_shape(CTL_FAULT_FACTOR)
assert ctl_arm_shape(CTL_HALF_FACTOR) == (14, 1216, 1344), \
    ctl_arm_shape(CTL_HALF_FACTOR)
PY_LIB

cat > "$SCRIPTS/ctl_labels.b64" <<'B64_LABELS'
eNrt3UuOrUyX3nEQknGroumOZcbhFmPxDGoGILlh9zwd94pP1ahmeQhYJat6NqVqGMmI8HdumZtNBJvLAiIW/6f3nvfkydy/DCLWCm5JQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCyKNSDBhIxtoaBLnk1nYoyMVY26Mgl8paJlC5pNbaEQbJ6dNaGMRS/vBkgZecPq1tcBCcPq1tgRBKhqd09Wkp6AWbdzzllyMaJLFYPOWXdzxll3c8RbtNPIWXdzxFu3c8hcslPPEMuVzCU7ZcwlO2XKJ/F4rB84zyE0/Z8pP9ZNlyCU88g4zFE8/w2yPOv8u2R3jKtkd4yrZHeOIZcruJJ54ht+944oknngTPO1PhiSeeeBI88VQXiyeeeOJJTvDkOQOinjnjVrReKrnOVtLz57l5Dngxz1/nPmuQhDxLLrSV9ORGWVnPP5c2gSTjaaj0/Sn6zZ4ldyp4k/rW6QVPy51e3uS+C5D9nl9XhvKoQGfp02/0/LryjgXJ2aYPGz0Ltp68ybyNjt/z6/+oXZDS3V9pvHdweD2/pk+1C5LZP1AKL4vX8/twV7olkh5YGSrvOl357j/6PtyV3ptUHFgZ/Ou0zzO1r6l1ljx7B0rmV6k8i04+8VQ4QLMDpXXmV/F5FhNPhTOoOVBaLzwCzOdZTjwVlqBHNneNv3H0eU45FR7w1YEDr/AftR7P7M1TXQmaHTnwSn8j7vE0b57qVnhzZKBUfhRPvVRY5Qd8cWSzZ2FV8XiW757aevhD12kuoHg8q3dPZZug6ZGVIV34WrdnamfR5ZkfKa3Tha91j/ts7qlrQTJHSut0YZC5PfO5p66KvjhSCmYLg8ztaeaeuhb48kgvndnPJ4Wn/6uYe+pa4KsjK0O+2bOae+rqkA7NZPnCQev2nHPqKpjSQ0devjDIShe1o1zStWWXHRopZuFLnZ6OcklXAZodKgU3e+YuT00FaH6oFDQLg8zpaVyejVrPfr9njef8A25dGooFFOf67vRs1Xra/Z7tKs/iYZ7NHZ6aGqTiUK9SLqCs9+z1eo67PftVnuXDPDfWguVCL4Dn9rWhXFjKnJ6VVd7AF4fWhnJhZOO5/dirFkoDPLd/tqVLOZ/pWR7aPKsWZor1nqNiz20LfLUwU+C5fYGvFlTw3L7AL41sp6d9nGe/27PB0+G5ZbFNl2aK9Z5Ws+e427PD09VQ13s9+73rkW7PZq/ngKfrA27wzJZmimce79WRsw/Z0kyB5+YCNFsa2QWeWwvQbOk3wfjcXIBmS78Jl2f6QM9xt+eAp6uA2e1pOd5dnvVuz5rxKerZ4FkduTwrWypdn+l56HK3fKnUwnNzg5QvlVofPf+XYs/x5UTnfs/R6Vl7POtcr2f/MlK73Z72k2c2wVfs2bxcubi+4TRLU+8nz16zZ718r/Baz9ZVOXg8O82erwPqgGfn8vRMEK1Gz/RlITGbN0TM0t7UIz2z14V5s2extDf1ybNR71ke9rSuytbrmenzzF89860bdsVS7+/yNJO/mum7Qc5MCKvDns3cc/R4Jho9i8khXhz27OatkMdzVOk5nTLzjdNZuXDy6ZNnr9Kzmnhmhz3H1Z6dSs9pnZMe9vymcXpO76fD87NnO99Kcns2Gj3Ttzp8o2d1wLPW6Jm9eVaHPfvZ1ofneE80euZvnuVhz3Gl56jS830LpDjs+WXzwXN47T7xfGspHR1SvuzZq/Qs5D27dZ6das9eznNY59m8/hVt7aakp32yZ/XmaQQ867fW0u1Zqzze7Qme7SrPRKNneobn8EY3uhrUUaVn9r4oS3jat9p0cHkOKj3zUzzbFZ59onE9Msc8PVfLDVNrp2en0rM4xfPnAM2WPVuVnuW7ZyHiObwO/ck1DtNrGI02z9krDGQ8f2hVy571MzxLGc/x9cIFp2ei0tOe5Glrz4MCy0lNqs0zm3lWm86/+z1919yVkzVKm2d+ued0v0Cbp5l5brs+JFvl2To8u8n31/I8lmK2qX6Pp5bxWb57pmd4Ng7PVuX4rA565nh6tjPq1wE3yHrWDs8GT1HPejJ/K3leZXqfZ6LccwJ0vueI53L9iue03LnYc3iIp9nr+a8rPadbTttvyHuKZ7fNs3N18wo9i20f73s96spNnq1yz/Gw58JU6qh4m2n922n1LHd75ps8a+We08u993hmeDo8N05nL+tRssnzrf7V6pke8Kw2eI4P8cw2frzX/f3Cf65z1pENb99fq6c54Jlv8Owf4lkc8Mw2eHZvy1mr1LPc+PEm5/M2eLbaPftJv7LPs1zhmaj2TN/H537Pxr/ZNDxmfKbv49Ne62l2PPE+Js9068czzpuvV3sWePo86+WLQZ/pmR3y9HVIfs9SmWfy5mm23m0x9SzxnDYoxVbP6d1uZrdnrcSzcnsm+zw9E2j/WM/ymKdnAnV4Nm/fXqmnPehZfvTMnKc71HhOb5dJN1+dNb271bNl1/k8U3WP9556Zps9zVQkf7pn4bz7b7dn+vHy5Ilnps5zevvm9qtf3o73ZJtnvuMFQWEnF/YsV3q2Sj2zyRq8/WqNd0+zydMo97SHPbNPl89PPAur6/K6t8dd7Dh7++6ZPtwzcT78bL+ns0NqfOtRuf2AiMnTbD/7ULivz/NfXjfxrHR7Fgc8x/eCYY2n1e1Zbd+NnHlmGzxTxZ7j63A54Jns8uw0euY7dnvmntV6z1yxp33tbXZ4DrMdFvfldZnrYSxaTr+/emYegK2eZvnykNf6s7DaTh8tP29yl2e23rN6iOd4xDNZPB33eoFpYdVtzx/2dHQ41RrPfjqQ9XhWH87vbvcslk4f/VEcpt85Ue3ZH/LM13nm+w6IGI/37pBnuvjP/Xm7Wmqf49nuGN6df8i7PHdPMDF6Nsc8y6VfzxM962OexWZPPdsh6bFy3vkG6WxpuGcHJ+yneLb+Mf90z/GoZ7kwfaQHF8AIPYc9X976x2DycM8NR1/mJqk2eurZDsmOjZbc/UW5f/pIDxYUEXo2hz0T//TxQM8Nn863w156i8ujBUWEnhu+vPB45t7p+GhBEZ/nuMuz8U0Eb9Kp7nYzP/bpSu+kW/mmD9Xt5nmehW/6UN0euTx7EU/vpbKqy3nX9a/tLs/a14kOnz0b1Z5bPl3l9yw9B3Olufx0XV64z9M7k7SfPRPVnomMZ+L59ZSay08xz9H7T3/+joNqz0HK0/Pk4EJzueQ4+nopz8SNVWgulxye3S7PwTf2m88VhaLl3bHatmLj0zixnubZiI3PH/93XNGRJao9N42W5VXMuP4817y8zz23FYMfUCrHbJxpXt6Tg3tnH74sc4z2VHH37vDs9g3vfv+3TFR7tqd7loqnz+Tg3tkez0Lx9JkcLF6qHSpGb3c0XxzGyz0T1Z79+Z654ukzPVgMloeuedI3faYHe+lyzyyotnl3eG78+mKPZ6X2cJ95bv14u54nXWpd3eeeW6/UyPcctkbllSFOz63DJdvjmWs92uee9d6v39VEJAme7q/f1gZUOg92h+fmf6Da45lrHZ7vntvns3LXF2pc2h29yo4j0Oz6wn+ncS1yeHZ7/4Fu+6+hfoBns3fC0HjsCnju+BcqfeeAxDz3XOhmFO5qSHnuKQhzbVccHkp++NRDZrWu1cc9dx21+jaFxTx3/RMFy5HHs9/7TzB9Oj33FZGpyo0NAc+9i4qxOLo8mQRFPeEU9aTkEfWkIheJYXie48nsKXu811iIekKBJ554EjzxVFd/soOJJ554km0p8MQzAk+2P2VS4olnBJ6cQccTT/2p8DzFk9ObeOKJJ9kWLg/BMwbPFgo8w0uK5zmeDRZ4hpcMTzzxfExyPPHE83meNRZ44qk9Bk888cST4BlACjzP8YQCTzzxJJtScnsHnnjiSXal4nYZPPHEk+wKTwM8x5Pbj/DEU31SPPHE83me3H6EZ4DJ8MQzBs8WCzzxxJNsSY4nnnjiSfDEU69ng4VEeJsU45Px+UDPGgs8wwu3H+GJ5wM9ocATTzwJnjeG24/wxBNPgmcI4XYuPPHEk+CJp75w+xGeeD7Pk9tl8Aww3G54kmeLBZ544knwxFNLuH3zJM8GCzzxxJNsCbcfneRZY4FnwJ5Q4IknnmRTDJcr4onnY1Jw+RKeeOJJdqXkcjA8A06F5ymeXA6GZ8ieLRQi4XIGPPF8TFJOF+MZg2eNhUQyPPGMwRMKkXC6+BxPTsfhGbInpztkYvA8xZPteTxD9mR7Hs8QU7D9iSeej/NsoBBJiecpnjUUeAbsiYSoJ9t1sp5sL+EZsifbS3iG7Mn2kqwn7btQKjzP8KR9l/Wk3cQzZE8g8AzYk+0QWU/adzxD9qR9l+03ad/xDNmT9l3Wk/Ydz5A9ad9lPXEQ9aTdFEuBp2gM7aa8J+2mrCftkVhyPOU9aTfFktEeiSalPZL3hEEuFeWndMNJ+SnbcOIpW9BTLskWTCzvkmF5F55AmT4JIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQggh5IxkPD5bNIbnPYum4nU4kkl5nZhoct43Ij19MoEKpuT9V9LLEZ6C4fXq4ss7L8SR7I7wFC+XKEBlyyU8hT15f7VUCjzly3kaTjxDbo/wxBNPPMnmWDzxxPMpSfHEE088CZ54qkuGJ5544knwDMizhkIkOZ544oknwTMgTyTwxBNPgieeeBI88cQTTzzxxBNPgieeeBI8L/fkflg88XyoZ8rTGkQ9DWePRT0rHsck6snjg9Zl9Hu+zpgZl4usinEdxg7PnOevrUrpesqSw9PwPKZVqVyHscOzpP9ck9R5GJu5Z8X1dmuSOQ9jh+fPP2gA+7gcuYbd3PPXHYgtYB9SOJ9LOffkAZarlyPHOjP3zPFcFeucF+eePHBx9fLumBfnngUb9hs8e7dn/1Z+ssCvKZcc427uWVkW+BX5c+Ktcaz6k+P999+jg19Rfs6dZp4pnps8e6dnP5sX2BFZUc7PC6GZZ47nqvxett8XpJmnsZzxXN8ezRak0udJAbrSs132/DMvsGO30rNb9izd45g42/fZQlPheaTdnC00M0/PvEB8nnbZ0+K5pX1/X2j8njRIKz1bh2c3H8d4rvTs8BT1HJY8v/8eZzxWeo54Hk++0jP31KnE6zlZ4K3Xkw2RtZ7tgqfBc7Nnh6eo5zBvm748Czw3e9pVnmyALsW8ejazMurLs/T0pWTBs8XzaArrXJCWPBvUVnoOM8/W4dmittLTzpap9q1fwnOTZ42nqGezxpMNu4WU1rkgmYX5E8/VngOeB1NNPMd3z8bhyQboas8vvyVPNkDXe3ZvyxSexzyHNz6XJxtM6z0tnrKe3WdPNuwW8sb5Z/Guljxr2FZ7jis8G9hWe/7GskueNPAbPIfJH9eueRZPb9KZ5y/CRU8azi2e7UdPGs4tnj2e+5PNPcdX5sblScO5xfOHYbo4PvHc5Nl98qTh3OQ5vPyx05OGc5OnfbkKp3bWqTScvuTWecDns5FoLQX9Xs/hkycF/SZPO/dMXZtQZKVna5Y9KZh8MU7P4YMnBdM2T/vflj0pmDZ6zuAy92ll8pZinycFk6wnBdNBz9xSMK3J79NCJZ6insVGTwpQT35vG/mW+dHjSQG66DlmnzyNpQBdk99s6VbPGjpX0j/T4SfPwlLQb/Gs3J7D27pFQb/Ss/zg6bkMj0yT/WErPnh6LsMj0+R/dPKNnhSgy57ZRk8KUGe+n4q+7Jl6Gyfi2A7pE88C7/ekAF3w7HwbTX5PCtBlT3cH30/rADzXbS91ngtFFjwp6Jc900XP3HXRGPFsL/3EqbZ5UtD7t5d+ehZLnsZS0K9v339Ohvk2Twr6D56p+0q7SR1AQb9qO2R6S8dKTwomv2fzstY7PUtLwbR+O+SXZ7HgWVkKpq2eZpsnBZN3e+nX5oZrgW+ndRUF06r2/Zdn9tnz/1EwrfdM/Z5//te/4LmmfbfeSfLNs6MAXdO+209F5ledWrGjvN7TfPRseOrFmu0Q69tEmnkaCvr1npn1nSX6qlNzCtAV2yH27T+9nnXOAr+ifbfeLui97s84xbmi3bTeXaS3/zEdwixIHzxzn+d3ncqOyAbP1LcJX339J49lWbEdMiaeCXTuWbIgfW7fx/f+830P6fs/CxakDZ7Gs8f57WmYQD9vh4yzAmrqmX7/Z8Y5zs/t+zhrmLyeKRfZfW43x9kMMD2gX182wznOLZ7ZR8+KCfRj+z4kngO+ndap7XsJwDkkZ/s+zJcor2fBgrTF0/m6mdeXJRguuvnYbg6OP3tdv19flpCzwH9qNyfTYLmwHVLPViwW+E+emWN6fPVM8fzUbk6XacfyPTltx0Vhn9rNqWf5wZMCdJun8W6HjPMJFk9H+z7dGM5mWunE03CR3Yd2802l8l1tM8xbUhokR7v55ln4rmbo5y0pnp898/dq/e1dxjw5/UO7+eaZWs9pz27ektLAu8Zh755Xx7cVv5sVAHg6C6Peve4PbxNqO2/x8XStO737f/RuzxTP5XL+3dN4bj5q5x0pnq46s3cvVM2bZzP7QjydGx+d+/+8w9fzDh/PNZ6l+9lg9WzixdNVZs48s+le3PvJDYPnNs+fA3Sc/cV54Yrn0j7S5P91M/jR8ZvA09EefdjFzPAU9cxnex8Vnkvt5krP3tEJ4OloNz94mtnfwnOxPfrgWc7+VsH+51I5/+Gsbzn7WwbPpfLzg2c1+1s5nkvl5wfP+bUgGZ5Ly/uy5/xVsS9jG0/H8r7ZM8HTu8p89Mwc13pWXM+wUC4te+YOzxLPhXJp2dM4SveC65cWyqVlz8IxVRo8lzybrZ45ngvl57JntXQRM56bPV1LT4rnkme91TPB01cFfRqf7qFY4bngWa/5i52rG8Bzq6dxFlUFnjs9C+ekgOdez9Lpaeg3d3pWTs8czwXPVX1+7Sq38Nzombo9Uzz3eWaev4Sn33PE8zLP3ONZ4ent38c15fzoLKPwdHgOa8r5wfnneG70rJbvqcFzVgcte364pwbPGdQiSuq7ZizH0+vZrZkWWuf/YD9kNjEuouQ+zxRPZ8Hz4XSx1zPB01MILXoa7zmRCk+P1NLpo9K7p1fi6StA6zWzbO0c3y2Qc89VVYB7fOM5ryzHXX8px9Mz9IY1g3h0U+P5sTN3l0uj+/fR4Dhburs1RcDg/n3gOS9A2zVF6uD+feA5H3v1mvKzd1vXOM7mxmSfp8HTWQuNqzZNOvfaD+MMq9/pmXJ78fsB+9OrXdUedc7/ye1c82N2Vc3vUq/YnneM0XZVT+oqjArao909vsszp/zc71m7JguWo92eztmV3WRRT4JnIJ7MlHiG7EkjJOtJIyTrSWWEJ57P8WzBwDNgzwYMUc8aDDwD9sQCTzwf48n2kkzYXjrHk+0lWU/ad5nwWLVzPGnfZVLiKZqC7RDRGNp3PANOjucpDScSogU97btsAYqnbAGKp+wCz/aS7ALP9pLsgsR2iOyChKdsB9/iIDqBNjiIdkh4yg5Q2nfRAUo5L7vCU87LtkiUS7IHPAaEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEELI51g4gSCW1PzICIZTqp6ftkBCJsb/TYiGR4o8nR7zc7MkAFT/cGaCSqxEDVHj2/BGKUIFK/jU1IMeSTT2pQQ9h9q+rEQf88aWoqaaeFhSRypMJVCC5w5OKSahU+p0eFpFKnhZJuvRkQZItPX+nAUZwOWJB2h/j9KRDklzeWeD3p3R60nHKelIwSZafeAqXnxSgkrsh7IjgGXZ7RIOEZ8jtJg0nnmF70sBLbofgiWfYnmww4RlCCo8nG3Z44vkcTzZARdajv8NTzLN9bZfwPOzZ/Pjvig16Uc8CTynP+sd/p3iKen6dP66xEfE0eAp5/vqDjBMeop4JnrKeFSc8RD1LNpRFPMdpR88GqJCnwVPEc5j+CRt2eIbsyQbTQc8/E2ZOA49nBJ408DKeGZ6neNLA44nnczxbcPAMyLPD81RPNkDxDMmzxRPP4JJ7Pdmgl/WMe0P5rvJEg+ffuj5Wf7dnE6en+1zCbecXYvfMXZsN5X374X7POE4g/b5aoEtfypHsxv3byD3T2RMlqnvvkI7c00wfIVPcfsd55J6Tpx315v47ev2eMZzg9D2t475+JJ890SYmzxzPM1b3gO6YjNuzXPAc8dzVHHmD54HqM5RHnM09I7qjK1/0bALzrIP3NIueLZ6Cy9Htnkl8nouc9zRIMXumUXhm0XguL0f3NEgxe5oAPbOZZx6NZ4GnaCo8L1ze79kQidgzDdpznM3yeD7LM4vDs4jF04Tomc48y1j2lwo8L9wNwVO4/AzFM5on2Fk8ryyXQvGM5YmVWRyeaSyeOZ6Xlp83XdDwzpfF4lngeWn5eZNn9caXx/LEdIvnpeXSTR+gdD9uLXjPLDbPwO+Py/G8tly66QOU7sdVBu9ZBOpZ+DwDv7+4jMSzisSz+ujZheCZxvIGORuHZx6J5+fy86Ybut48i0g8P5efNz2Q680zljfE5nF4Zoo861vr4vG9qgvb83M5b5P7PVOrx3MIwDOP5g3wRaDl0tSzUuTZ3Os5vBd1YXt+bDfH5H7P4v7lUcyzv98ztfF4VmFWSxPPMoByQ8rztrOJ357GxuzZVWG8Lfjbs4ras67CKJy/PLOoXghdzY7vMozCxMT5gu1qVm1WYXsOUXk2CZ6SnnXwnn1MnmMSvGdUp+N+/LAWzyd5tjF5NniK7ockr573bjzkYW0f7vLsI/CsI/JsIvAMm3O6VZsE5JlF2W5OPIcIPIeIPLsIPEN/e4eZLZ1he4b+dpl8tnQGMvNnUZafr57vV14F6Rl4+fn6Yw8xeAZefr6ei+1i8ExCz+xQwlOmQRqTCDzDf9nE7D6UQDzzKMv5lwI0icEz/Jfx5b7ndATpGf7LImevYQzaswneM31vPPAUKZiaKDzr8D3LN72QPWN4N1fxVogE8rPncZZLf37w70KkCtgzhncXp2+FSCCeJsbdum/ANgrPJgbPYvqThuwZA+evij6JYf6M49XF6RQv4PHZR+H5owLtovBsk1gO+CQKzyaJMFUYk5WJs9uMyXNM8JT0HPAU9ezwFPVso/QM5OgqlCxHAXsmeEp6jniKevZ4inq2cXv2oXk2eIp6JnhKeg54inp2eIp6NngeSamjO/r27MLyjLSaD9ZzwFN0/uzxFB2fTeyebVieCZ6SnrFWn983eDVBedZ4SnpGu7oH6hnt4f49f958hFU6DvcwPfsk/uM9JM8WT0nPMVHgmYTjWeMpNo1HPjxD9Gw0eI7BeEY9PAP0jHp4ft2/PYbxa4251QzTs9XhOQTi2cXN+XUf1cBhIurZh+HZ4inpGf3wDMyzV+PZ4anJM9fiacJYCPA8xbPDU/THiN+zCGMXAs9TPKMv57/OezdB/Frj96zCOGeD5ymHiR7PMKadJnpPiyee4XvevVFWxX8lw8+keKr0tHjiiefpCeR0cRL7XTKBeaZ44onn+cmZPzV6qhufPZ4iMVxuc4pnh6cmz1yLZ4EnngGnxBNPPFd6DtF7VmF4Gm2eLZ6S+xCBePZ4Mj4Zn+cllKcFafHMbFCXf0bvmeN5xnEWimf0l88XNqjLk6P3LPE8pT3CU6VnG7tnKI/kUnI7QornOeU8nrLlfCCeTeSehQ3raZWxe5Z4nlMu4SlbLgXyNNXIPVM8T1reA3maKp54vsTgeVL5GcjTP2s88XSV83jiGXJ7hKcqzwxPPJfadzxlPQN5uneCp2jjiyeeTs8eT02eRptnh6do/YmnrGeLp6hng6cmz0KHZzCXM6jzTPAU/Bi33yetzXMI5ECJ3NOE8n5wJZ55KO9oLHU8niGUu+O0eGaBLO9aPNNQXnFbKXkcSygvvdTiWQZyGqxS8niGIpBhocUzD+RjaPHMArmLyqp6/FKNp+gEGkydEf/jl7IgRkWqxvOvB/z9s6cmzyyEa7BSq+VtpmH1vXjiGW4ZjCeeeD4hudXytl088SR44hlzjNXyNugwUuCJZ8Ap8cQzAs8aCpFUeOIZcCyeeIab1Gp52y6eGpPhiWfAyfE8xXOEQiQGT9EUeIqmxPMUzwEKPANMhSeeeOJJ8AwgFs9TPHso8MRTe1I8z/HkdgQ88cSTbEqGJ5544kl2JccTTzyf59ligSeeeBI88cSTLHo2WDA+GZ94EjwD8KyxwDO8ZHjiiSeeBE888SR44oknmXpCgWeA4XFW53jyuAs8gwyesqm4vRjPgFNy+yae4XtyubdQCjzP8GyRkInh8hA8A07O9ucZnkAIJaN9P8GTdlMqKe2RbCjn8Qw5Fe2RaErKefkGHgbRhpNySbZBwlO2AGV5ly2YWI5EFyS6d9kJlG6TEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQggh5GCyGgPBFJZXmArmxxuLeWWxXErLK98l85OTd0BLxfzy5J3vQql+eVrWeLnV6Ed4q7bk4c6KJLq6/0gLhuDhTgkqkvzbkxVeptf8Ciu85PRJSS/XHDGBCiV79WQCFaw+LZt2wtMnFb1c884CL13Ns8BLL0d0nBvaoPrzckTBtGEZ71csR5O/ZOzIdLpQta9Yjl4L0Jz5dHmabD4vR68FaMkW88Ls6Zkb35aj10H8Z8H/G/Tcy87wefp88fwt3eWs+R638fPh/lLQG5r6T13QisPd5UlT7xmHH6vP10WroAldnj5dLpXfs6QJ/bTp0biqUutpOCua0E+bHu3n5ej779CELlefLs987tk5qPF0Lzszl8LvmXmaevLt1n+s5vHcsBzNXSq/Z37RWToTY7fgG2dzzq+/c5VnFeEpq8zjkq70HE/9Vce32hmH5+jsNr89zTWeWYyzczH3ND+K+wA88xi3B8q5Z/HjMMvv9yxi3B6o5p7Vj/8w93uWMW4PONbpn0gBeFYRtl/p3DP7uXtX3O8ZY7vg8Mx/TlvlgmdxiWca4/5/Nvc0P6et+z0zJZ7Fz92m6nbPPMb9/3zuWQXl2UTv+Ws31MH5tdgWl/TvJsb91eA9+9g9fy6rbbrkaS7xLGI8wW9mLtk2zx7PZc88LM8xUs9+8ieta3vp65xdccn5uDLGuyDymWe5zbM9e6cmds+fH6PN13qeVx9GeUHP3NNu86zxXPRMFzwbh2eC56Lnr4mzNSs9T95eiq7hzN3nLr8962x+bBdXlEtxembuvY4vz/FVL5l7tqf/ZJF7/qr6uuK7vMyWPBs8nUfVl2c180xmc2VxxXKkw/P3f395tonjFGhx0dUMMW6Avnn+uROmfPk0xfvaU1xx9VIep2c17cONw9P4PXs8lz1/O3avjwHM37c+iiuuTo7Us5zKVA7P7L02Kq64GyGP84acYuL5Z3n68nydY5u554mDx8TpOT2Y8zfP8XXQ1nPPGk9PwdRNj/6JZ/FWaxZXPPImUs9k4vm1Ok3qS/NWaxZXXJxsIr0Br3rx/Kruu0l9mb/VRsUlNyMo8Mzcntmbp8FznefX7t2/TOv1t9rIXHHzURG3ZzuBevOsph+suOJmQw2exbvna5Vfzz3bCzzriD1Lj2cxXcuLK25+1+Bp3z3b7wlzuMkz4uP9eyf+n6ae2XSuNFc87CLy8dm8On15Nt9NVIvnVs/vPvLv3w62ctJamiuesFrG72m9nmZSuuP5+eduJrcU/+e3D5NOlgVzxcOX4vfM/Z7uVnC4wjOJ1vPlGpsMz+Oe5VbPHk/vz12/Tp8jnsc987A8q9g9DZ6ini/T57DOs8PT6zl54FKP52HP1zuOVnq2Qj9CtuA5xuqZ3+ZZOAd67J6Tm7TWeTYyP4H7oLaRe1Z3eWbukR655+SGzfZKT8/j2iP3nNxg+HU77LJnLfcDqPMsbvOsFh/fHKtnOb1x62rPXpunvc3Tw6bJs55d8nmeZ+pug1I8D3nWHs9Bg+fkLJ3fU6z8dHwnZZ7lZZ758vOGNXiOV3p6/rFMl2dxuWfzeM9R1LPV6/n9uN8LPAv35nQe6/sXSudzFK/zLN0DUZlnfrnn6PbsFHh2V3pW7osfNXm2t3h2OjwLp2d2mafnXnqjyLP58mwv9BzxlNsOmU2gmjzrezw714/VKvD8bk+60z0zz6ODFXmON3mOrrJDgefwfRie75l7ntVYxvr+2YA8a52e/bdnf7qn8Tw7WJFnd5dn52ibFHi2V3r63rWgyLP+9hwu9Rx1eiYzz7+phrM8S+te4KtIb4+bew4zT/NWy5zm2Wj07F88v58P1lzh2c23SeL37N490/e1V9Kzsu6XgVxxC/M1nu3L5xm/Su7+Cs9Bo2fz+nm++IaTPGc7B2/bTvF71u+e5Yme6by0mPz5GL/nZNzUieMCTcHrQ1L3Gxi+tp3i9xwnns2X3jmemfuNK4o8+4lnmzguBc1P8+z0eXZvno73aJzn2b9/i/g920kd07k2zs7zHN+/xRC9ZzP1TB2bk3hu8KwddfZ5nrn7uyvyTDye3SWezVsJEb3neK9nq81zuNbTeF7hfckjSk6JcX+gmzyHt99y9J7dtZ6e7sx3FXh8nu29nlabZ7PBsz5td0uPZ32z5+/jo9Ti+eePy2s8S+te4KO9HOxthAx3ew66PPstns3x715Z9wJf4SnjaXV5dl5Px35Ie4Zno8qz+bTwvnp2Z3j++j42Ws/SWS7NGsGTPJ035738eRO5p7exdu3P92d49po8h4s9U+fN4knEly9NPXvvxqTr/OYpnnby54/yHE7xrOP2rNxFZu7Z+Dnfs1Xk+b2IZwuecs/2yKxduh0iwsvBpp5NAJ79659H7ll7j8TRoXD84o3cuhd4JZ6jvzIcLvO0ejwHv2fvOkrP8WzUePb+zvpCzzbmy8ESz57c+wZT5/I8XM0Ya5cerxO7Z+vdd5o00um5nn3Mly8lPjTjbY++PQ/vVhTWLj3uLXbP2ju1jcl1nvb7zyP39JfavdOzPcezjvhym9Q3CBPfbvL5nu3XYhi35+BvnNxfc3iDvrTuBb5S4dl7P+twqefw5dlF7dl5a5nG/TX9OZ42sSo8W2/vklzr+Rcdno1vL230fM3hcqZye/5TvKeLM0/5OZHuL/b8vzo8vR+28Xgebq/th0TtOfqKwzG5y7OJ2dP3yofZqpBaoQ27P//QP6j07HzdS322p28ejfD0ZrYwWRlPVZRKfd6vp77kGj0b98cdktM8s69pWKNn7V4umtM9B9/GSNSeo7t9GZd6gIPrxfe7jzwHfBKxp6M4N+4SUNqzd194E6VnbpeacfdHOsHT3cmPMXuu33tIpfqXlwKiUOJpdtikVmj/x3z/K5k6z+Z6z+LlX1HiWeyoTVIrtMFUvBwZpfVd+x2pZ3K9Z/niaXR4lnuOLakD8tXTdylorJ7DDZ7Vi2eqzLO/2dPzIrtoPbs9nlbQs/DdKxdVqj2luZTn5Fvnujyb6z1T5yMIlXjWuzxrCc8m8UygbcSeye2eRoPnrtJZ6IRZNv1HMgWn44LwrGf/bPyefQCeVfynO/Zdeijk+f7coTL+8bnvUmOhFeP9OXhGkeemH706xzNTdHlIfYPn7L0Aejy3bWxUMh3MzLNSc3ndcIdn8e5ZqPHsg/A00XvuezSVrOc4+2ni99w285cynuW7Zxb9/tK+R3me5ZlG77nvVUaFqOeQ+Aqm+DyLXSdmC5kZrpp98+qZnuYszxJPUc8i9vV9n2d+lmceu2e5yzOT6bDnvVn6TM/0LM9EiWe/T+KYZ+rwrJ7pWUmckXB5lpHv1+30LCU9+8S3wMfnWe0r9AqJ60Nce1sm8vNxOz3zszyzyG+X2Tnxi7z+2vlY62d6/nUC/cvhC7Kde69l1J7pgcIkP3rBq3F5mqhv7zji+WMoteKeWdS3I6RHCpP0YEHj9Eyi9swOFXrVsQnO/c69MubL5495muEET6PhdPEtjUjhZMtibo/yO39y9zv3UjxFPV+2mOIr582djXLl/mUWCu7WDskzU3B3cRKQZ8QP/7zX07fsFPHeLROkZx7v3R13evovNa9iXd6/SpawPE2sy/ufkXDLj+6/dD+N9eb3QD1/zaARLu/JnZXzwjmoNM6TcTKngU7w/LFOxjh9pnd2ImbBM4/zcA/C0zkQYyyW9t4tc4VngufeXmLAU9Sz1+OZ37mTo9DThODZ6fNs7/Rs9Xk2d3o2ejyLEDxrfZ41nno8Ezwlv/mg0PPOb66oXLr1dMef4qLW5zne6KmoO7r3dMfvZrdV6HnPkpCpO9wTe7vnmCj0vGcOS9VNn+m9O2Z4ys/eHZ6i1Zqq5T27d0e30La85/d65ro2Q273THRthtx8+ugnaIcnWVwQ1PXQeOJJFj0bKMQaFDzlPWsoxDYkrLYmBU81sXjiGW5SfRew46koGZ54RuA5QIFnwJ49FHgGmBzPUzw5fSQTTsfJpsBTNCWnj/AMOBWeorGcjpNMiucp7RGesuU8nniG3B5xOQOeIbebeMq2R5x+l22POB2HZ8jtJp6y7SanN/HEU30yPM/x5HSxSHI88Qw4xnK6GE88n5ICTzwDTmm5nAHPcFPhiSeez/NssMAzvFhOv+MZbr5Od3A5A554qk+G5zmeXB6CJ57qk3M5A554PiaGyxnO8WR8iqRgfOIZcEo88Qw4FZ544okn2RWLJ57hJsUTTzwfkwxPPANOjqdoDJ544vmYfG3PczsXnnjiSTalxBNPPB+TCk888XxMLJ4nebZg4IknngRPPPEkeOKJJ548LgjPQFPhiSeej0mJJ54Bp8BTNAbPkzxrMASS44lnwMnwFE2Kp2zwPKnhxFO2QcJTtkHCU7agx1O2AIUCz5ALeihEC3oenyxb0OMpW9DjKVvQ83ha2YIeTzxDbpB4PIOsJ7cf4Rlyw9kiIdpw4inr2SAh2sDXSEg28LSbeAaaivZI3pPyUywl5ZJoCsol+YYTBlFPlnfZBp7lXbbhZHmXbTjpNmULUBBECyamT9kFiWqeEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEPDXGWtvCIJX8r5x2wEEq5Q9P2wAhk+wnp+2RkMkvTg542eFpbY2F1OL+M6zwgqsRE6jw9PnXiNdh//WBnOm3p2jFVD10Ss6/PTvxNe6BNYOxJxzwxXNrhsLKH/DfY/55A7R88ZSqmKrz1rjg8/LZpSqm3J60xsVVLlk7CmM+8IBPJ5/+wGjKfoxuO8/DVqRs8uE7mXnDys8hEZafBz58Zr0Zn+y5a7b720mNMMtzy/k9H76wn9I+2XPzgvSRU7CLjdBz62BKP3s+a0EqjpWL+WfPZ1Wg5bHV4/P0+bAF/t1z4wRaffZ81gJfHVo9VkyfD/fctnpkazwf1XEem+3yNZ7Nkz23HZ0Fnp88Nx2d5RrPJzVI6bHRVOH5aUFpD43upzec+aFPn+K53L5v/PQZnh89+0Oj++mexaHti3WeT9pgKg95GjxFPQs8P3qOh7766Z4Vnniq9azw/Ngw4ingmdp/3eNp8XSKDC+tOJ4Hkn7XnNWODWU83Rsa/etajaeIZ4Hn8eTfn9jsOOGBp9uze90rwnN/DJ5neWY7TshZ9pOdG25d8lKA4nnYs33F2eBZ4XmDZ/scz/KFcMfHL/G8wbN5jmf18omrszxrPLdNv1xP++ZZvw62DcvxqvPFI56ink+6v+P1iNzhmdFu+j2L7Z4p5byoZ0L56fIYd3tWlJ9eT7PDs6Rccsx/U88t6we3G8p6GpZ3r2d+zgXKzfM8h92eKZyOenyYFOebDtAKzblnv9+zoO4U9Uzp2X2e6a4FmSeCEUIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEBJmRgjEkv69tbbFQYrT/kyNhEx+cVqOeJnkvz054kVS/uFkgIqOzr+mh0NoLUJUKGbqaRtIBIentR0mR1K9ew6YCB7tDFCxtf0rsOxN5uKkqt+dwunJDCq1tlMyyc+ezKDChzsDVKz2ZIAeipeTLl5yOWKrXng5ogaV6jXZWT5jeWeJ35VyyZMVSbBc4oCXLZf+rPAVo1SmXPq1K2JY54XKpV89UsU0KlQu/RygKZ2nWLn0I/9A3SRWLnE26SRPNutFyk/qUOHyk50m2fKTRv4kTyr6dclWerLAy3rSIcm0mxRMsu0mBdM5nnTwQu07BSieMXhS0Atuh1DQi3tS0K9Khec9njRIeIbsScMp60nDuSoWTzw1eLIhgieeeJLXpOs9a7TwxBNP8p1svWeDFp544kl2ebZoiXqywAt7csZD1pMBKuzJAJXsj9hiEvekZJL15ICX9WRFEvZkgEqu76xI4p6sSLKeXLUoOn9ywEt7MkBlPbmMSdaTElTYswVN1JOSXtaTCVTWk4pJ2JMWSdaTClTWkxtnZD054IU9qZhkPamYZD1pOYU9WeFlPVnhZT1Z4YU9WeFlPVnhZT1Z4YU9W+Q82edJxSQ7PqmYZD3ZVBb2ZAIVnT+pQIU9OeCFPalAZT2ZQGU9qUBlPZlAhT0p6WU9WZGEPZlBZT054GU9Oesh7Mkuvayn8gGa7Vpxj3jq3hQpf64Q//ZCT9UrUvrzANw8qx3xVN0j/XiRUV9srgstE6g7Px58Pm4fNRbQTzDddZ56S6Zs54c85qm3iTc7P+RBT7VNfLFz7/ygp9olvto5qx31bJV67h00lgPeV83vKmOOeirtkbK9ZYxlAnUl3zurVUc9dZb0Zu9RWLEgfSqXNh2Fhz11btqVe4/CigVpDUt3mafKFj7dPWoqFviP5dKWUVNaFviP5dKGUXPcU+MCb3ZvpB331NhxFruPwuOeGgumcveoKSwF05ouvL/MU2HBlO4fNcc9FRZM+f5RI+BZq/M0+0eNsRSgqwbZdZ7dIzzryzz1LfDlfs/cMoGu2tRorvNUN4EeaKslPLVNoOnNnsMTPLvrPLVV9NmBbQoRT2UTaH63Z4unqKeyLTtzYJEQ8RzwFPVUtiAVB7YlZTxrPEU9dS3w5e2euhb46nbPTr+nxVNyO2StZ2YpQMPzVFWApvd7jg/wrC/0VFXQZ3iq89TUIOV4qvNs8MQzYE9NDbzBU51nhyeeAXv2eOLpSXH/foiqDSY89XmOeIp6atpgKvHU51njiecWz/FazwZPPAP2bPHE8wTPFM8gPRVt2FV44snxjueV63vP8Y4nno9ZjxR5WjzxDDcpnnji+ZgTxoxP2WTHzojjuc5zuNZzwBNPPC9IjqdCz1G7Z4/nLZ4VntMYPPGMz7PDE0889V2gjKdsCjyv8GxXfnmJJ55npsQTz/g8m2PTBZ47xyeejE88I0qFJ5544onnHZ4Gz2k8n6/GE88AkuJ5iWeC565keOKJ52M8czzxjM9zxDNuz1qJpzl4/TWeeD7BM9Ht2TM+96XAE8/4PDs88cQTTzzP9czxxPPEHLxcEU88g/bkgf7TVHgG5Sn1AKYETzwDPt6VezZ47os96Mnxjiee0XvWeOIZQFI8VXqOeOKJJ5547sqg3DPBk/HJ+MQTz13bl3jiiedjPEc89yXHE0888VwdPEMcnz2eeDpi8FQ5f3Z44nnG8Y4nng/wbPHEc73ngCeeIaTAE0888bzYs8ETT0dKPPHEE8+LPWslnhWeeOKJJ5544qnudtjDnimeeJ4Xi6dGzxFPPDd4jnjiiSeeJ3kOeOKJ5+lJ8VTp2XO848n4jMAzwxNPPPHEE88APHM88cQTTzx3eRo88cQTz03p8MQTTzxjC554crw/5vbi9PD5WzzxxBPPZ3pmgXg2eOKJJ5544olnQJ4FnnjiGUlyPPGM0LPHE0888cRzl2eHJ5544onnrt0hPGU9WzzxxBNPPHdpNHhG7VnjiecGjfWfr8QTTzwj90zwvMmzwpPxebXnyPjclxJPxmd843PAE088H+PZX+2Z4Claf+KJJ57Mn0o8u4s9Rzzx3KCBZ9yeg3LPdv2/YPEMz1P76ybwlPVs8MQTzz31BJ54btneFfFs8cRzg0aCZ9SeDZ54BuxZ6/bcsH2W4hmgZ6Lbc8BT1LO/2FPN9nyK5xWeHZ5Rew54inqq2f7MDrfTeAbo2Sn3bC72bPH88C/geadng6eop5rtkPzw58MzQM9EuWdyreeIJ57OmCA8B+WeI547UwThqaZ999yOcLVnh6eop5p203O59xbPnHbzo+eAZ9SeatrN46ffRV7PleAp6amnPTp++l3Cc8BT1LPHU9RTTzmfHf+AhnI+NM9au+eWAVNQLomOz4Jy6WNzc+347PEU9dSzvPsWk2s9WzxFPfUs7z6MLZ4ly/tHz+ZKz0GRZxmAZ4enqGer37O+0lPRcuR7+tKVnoq6oyA8+wd4CpQIj1yOQvDUxJkITGklh3tIni2ekse7qtXddzruwvHZ4SnpqWt4+k53XOfZ4yk6f+pajXynO4bLPBM8JT07ZZ7mXs8hwVPQc9TG6bPoL/HUx3mvZ63PsxTwNIzOT57dBZ6tRs/qNs8heZBne7KnymPdv123zTOnjP/k2ZzrOWjlTCU8DWNT1LN49A7Imu26Mz27J3pu6lzKpzdFwp4VldKnpfk8T9WHu3dp3uTJ4X6XZ9cmulMIeKYP3wJZ45ngeZtnhufH0hHP+zxzPD95jnjuS4VncJ4Gz0+eA573eRY0759aRTzxDCHpxZ7a589MwrPE85Nnz/i8z5Px+bG1wRNPPPXF4IlnhJ7dSZ6Ncs8CTzzxfIxniWfUnrVyz0rCs8Dzk2d7kmeCJ57heo5P9dy0Dhs8/8Re6zngieflnuuvX+qVc6YidSKesp7rr6ft8BT1bPHEU8JT5l953PZSdrFnjeeRqutx7fvFnurbzVzmg1e0R6KeJeX8cud9kmf3VM+NB2ZBuSTqaSiXlgfWRs+c5X3Zc+PCkbG8i3qmLEfLC/PWT870KetZ0W0uerZC/87DliMxz4LuaPE43eqZU80vem796BnTp6hnQvUp61lSfYp6GqbPJc/NlXfG4S7qmXK4i3p+7JD65Mme2/8l8/jeXdYzfXztKeuZsBglMk9bW1OBJnhKTqDDwz33fP784aX80qS3xzO9ceMzmCshJYvF6p5SPv3HgDarJT29B/x/P+/nzwPrGEQH1MXXJP+HMrhZOhX1zC6slcogdwhS2SF10eT5P4M93yfsac4fnlnQewSpbL2YnTxkPp31a5V5Juce7B9Pot6+wmfCB055amNUBn8KQNqzOHPy/HyKagzVU+qI7M+fmoLaxsqkf8/VaQvuqivOmjA9989DxVkfb90V0Hcv8Lm0Z/oyjEbJT7fugsguTM8D015xzmq78oLyPkzPVuRflByeK29wGsL0PDLvndP9RXJ/qJHvg80Zny1/rmd6xuG++gFkQXoe+y1X8kNl/fM06hA9h+PHpuxrdYu4PQ9WcYV0l7LhdX9NiJ63byPuK5aC9azD4tzwduS7h4KJ4bYrE7lnaFe+Vhs8+wA9AzvcN7xsOkjPIeLD/e4f3kRwaWEVt2dos+emw/3utdSEf1tLHrlnaMV8scnTBufZBOZZRu4Z2nK0jfPmYs8Ev7xnkXuGtrxvnD5vnq7y4Jf3KnLPwJb31EbuGVjznsfuGfn0efPxlYe+vFdxeZrAl/fN02do4zOw5SiL3TOwbtNs9uzC8gxseS9j94x9OQrMM7BTm9uXo8A8AyuXstg9AyuXitg9AyuXytg8s6DLpR3TZ2CeYZVLWfSekW8u3b4CZEEv70XsnoFtzpfReaZBL+/VDs8hJM/AdkNsdJ5JyMtRGrtn/N1mWJ6BdZt7yqW7d3SqgJcjE7tng6doiRcW565y/m7PIuD7ZMrIPXs8Reeo2C8FC8EzV7a8h+RZx98d3b6oZsEuRyZKzzTY5aiK0jMJdTnae7jbYIZBWN1Rttfz5lWgDLQ7MpF6FoFu1pWReppAq/kqUs880CsV93LevQykaq6sC2RZDbP6zKL1rHTcdxSMp4n+GUFheWZ69j7DaEtCPNz3l0v3e5YhPkUgYs8swMN9f/kZwDZEEeDLcGP2TMMbnmnMngEmwxPPgJPjiWfAMXjiGXAKPPEMOCWegWyH4CnbvuOJZ8jbIXjKtu944hly+46nbLuJJ55Be7bwSbbveOIZ8nYInniGvL2Ep+x2SHB39eLJ9hKeeMbi2eP3ngzPcDwH/N6T44mnVs8RPzzxjCkGz3A8LX544oknnnjiiSeeeOJ5Zwo88cQTTzx3pcQzIM8aQDzxZP7EE088A0yFZ0CerEd44oknnnjSH+HJ8Y4nnnjiiSeeV8TiiSeeeOKJJ5544oknnngSPPHEE0888VTi2QCIJ5544oknnnjiieeWpHjiiSeeeO5KhieeeD4mOZ544oknnrti8MQTTzzxxBNPPKdpEcQTT/ojPPHEE0888WR9xxNPPE+YP/HEE0888cQTTzzxxBNPPPHEE0888cQTTzzxxBNPPPHEE0888XymZ4cgnnjiiSeeeOKJJ55bYvDEE0888cQTTzzxxBNPPAmeeAafAk888cQTTzwDSIlnSJ49gnjiyfyJJ54SqfDEM+BY1iM88cQTzz1J8cQTTzzx3JUMTzzxxBNPPPHEE0888cST4IknnnjieSA5nnjiiSeeu2LwDMpzgBBPPKNJgSeeeD4mJZ544vmYVHjiiSeeeOKJJ554LsbiiSeeeOKJJ57akuKJJ554rs2IIZ544okn69HxZHjiqdmT+RPPE5PjiSeeeOKJJ5544oknnngSPPHE82ExeOKJJ5547kqBJ5544oknnnjiiSeeeD7X02KIJ5544oknnnjiiSeeeOKJJ554xpUSTzzxxBPPmzxrEPHEk/kTTzwPp8ITTzyZP/HEE088uR4MTzzxJDKePA+Q8Yknnngyf+LJ8Y4nnnjiiSeeeOKJJ5544oknnniyH3JzCjzxxBNPPPHEE0888cQTTyLj2WOIJ5544rkrHYZ44vkYzxZDPPF8jGeDIZ7nxeCJJ5544rkrh9+3W2OIJ56xJMMzLE8I8TwxKZ5BeXK591vwlE2FZ0ieXG7zloM3eHD6Hc9Tc3CDntPFeJ6agxugLYKiG0wNgqINPJ6ynmzXyTbweMo28PjhGXADz3aIbMPJdohsg0T7Ltsg0W7KNki0m7IFPZ6yBT3tpmwBSnskW4CiJ1qAUn7KFqCUn7IFE+WnbMHE8i67wGMnusCzHMku8CxHsgs83absDhPdkegCz+a81ALf/BzTTJ9CC/zw63dA9Sm0IHU/J10Od6kJ9Oc69G9w8yWl7LxxQeI4F51AKTpFJ1B6dtED/m+wEms5WYfkDvierfhN+blnl3kqJxqh7fnff478/8NZopO7eUhEV3uWIZHJ9Gf+GQyx1X7490iI9Z+sQYQQEkHSf2RPVbiB6JP/xM6qTHL62pO6BwaobHfbgiF4uLNTILK2s5N10tHOBCp7tDOBii7uTKDSsycn94UPdy7lkT3cWeBFV3cWeOHpkzP9stMnWyIHU3CJ1KnLEQXTsVRc+3zqckTBdCgZV5PjGXAMl5qe7dmiIlh+UtDLlp8U9LLlJ56Hwu1ieMbVHtFw4hlye0QDfyA5nme3RzTwsu0Rnnji+Zz2nQ07PEPeDsFT2rPFRXA7BE88Q94O4YQHnng+yZMTSHiGkQxPPANOjieeAcfgKZqC5wmLpsRTNBXHu2h4fjie0bWbeMq2R3gKe3KBnWh7hCeeIbebXLCIJ57P2Q7BE088NaaM3rOy/wNP2fUUT6mkoV3NErlnHtr2dxW3Zxna9dSRe36dnwnlXTxxe6bBXfNvo/bMgzuHGPf7kU1wd6XE7Rneg3Xj9qxCu2wgjXv+DO60Qhr1+MyCO08Tt2ce3I1TcXuWwZ1IzGL2zMK7EiNqzzK8WyWj9gzwUqGYPfMAS708Yk/fqa8WT8Hp897d+pg9fT/6nROoidfTWzqPeIoupRZP0anqzm3QiD0Lr2eLp+DyfusCX8TrWXk9ezwFy6VbCyaVniOeguXnrQVTvJ7ZgmeNp2D5iae0Z4OnXOV8a0FfBNizHfe8r6Avox2fBZ6XefZ4Crbvd3rGux5VNsSGE088o/QMvl5a4LxxQwRPPD9uL+Ep7GnxxDNkzxpPPG9Ntuh5209f4onnZ88WTzzxfIznbRv0lU7PnvGpw5PxiecKzwFPPPHEE89dniOeOjxLPPHEE88WTzwDXt/xxJP1iPkTTzzxxBNPPPH8TornhZ4WT8YnnnjiGZdnhSeeAa/veOKJ53M8LZ544oln6J5JmPUnnniu8AzveO/wxJP1iPkzlFR44onnU/ZDYvPs8HxWvVTiiWfAKeK63xBPPPHEE88nevZ44oknnnjiiSee9O94PqGej9fThOiZ4olnwPNnyvwpmgxPPPHEE8+neQZZL+XhdWx44onn5Z7Mn7I/FOMTTzxPmarwVONZ4Iknnte2ynjiiac2zxRPPPF8jGeC53WeI57KPQeOd8YnnnjiyXqEJ+s7nnjiiSeeeOIZiGeJJ5544onnrlQRe1Z44omn0ByEJ554nlOb4Iknnto8Czy1e6YxexrGJ554crzjKe5p8dTgmeDJ+IzTs8cTTzzxxNOfDM+rPDs8GZ944oknnnjG6ZnjieeKHz5Azw5PPOmP8MST/h1PdfVSi+fDPAOcPw2eeK7yrPHEE8+3FMyfeOKJJ567UrIeMT4D9mR84onnczwrPPHEE0888cQTTzwXs/ADNXg+zDPDk/GJp5BnjefDPLOo6iU88eR4xxNPPPHEk/qT/gjPwD2rmD0X70do8MQTTzzxxPOaFHjiGasn9adyzxpPPPHEE088A/BM8MQTTzzxVOoZ1/4Snjti1R7veGqYP6P2LPHEE88NSfV6sh5JFns3eS6MzxFPPPHEE89dngOeeOJ5UrGHpwrPjPXoKs/Yx2eLJ54Be3Z4is6feOr2jH197/HEE0888dzlect6muOJJ5544olnAJ4GTzzxxBPPB3paPJV7Fnji+SdpeJ4lnld5DnjiiSeewSQLz7PC8yrPHk888TzNc8QTTzzxvGmHBk888cQTz32eA54P88zxVO6ZRj1/xuXJ+MST4/1IDJ7Kx2eGJ54BH+944hmyZ44nnnjiedfzDSL3NHjiiSeeeO5KodhzwBNPPFd7tnjiebNnqdizxxPPgD3vWE8rPC/zbIL3zIN73q/FE088b/FsAvOsI/e84edP8cRzrWeCZ+yemWLPEU9RzyE0zwRPPO/dXsrj9szwvM6zw1PU84723Sj2bPCMvh3BUzaFYs8ET0nPEc/tiejpirF73nL1fxW5ZxXY2e4qtAlI7udvAjteYve8o1xKY/csw/rpFXv2eIrWe7csR1lwj3cV65cbPCU97zm4gtvf3prZ/u1f6l+TQBvGjxObZ+YclbfN/dF7pmH9yNo8WzxFG7z65p/GRO9ZBVWRxO9ZBvUTx+9ZBHVDSvyeJqgL2OL3zILacDChXa9yqGDq8BQtmAKbzaM83r8X+DYJ3DOO8WlCOntYxu+ZhbS7qMDzT4fU4in6GcLrfiP1zAI62aXB89eHaMP3bCPxzMM5F2s1eCZZHcpPosMzyGYNz7M9G3wObM7gebZnDRCeeCpKhiee0Xrig+f9Owl4XuQ54oPnvTGxXz6PJ55kdRRcHhKPZ4cPngF7tvhsTYnndZ4NPqKeNT5bU+F5nSc8op5sh+B5d2jfr/OkfZf1pN2U9aQ9kvWkPcIzZE90RD0pP2U9KT+3J6X8vMyTcknWs4ZHdP4ER9ST5UjWk+VI1pPdEFlPuk1ZT2xEPVmOdqXicL/Ek80QWU+ao30p6d2v8GR4inrSG+1NweJ+vieLu6wnrfvuGFYj0eTMnmd7srgfSEYtf7Yns+eBpBRLsmHj82RPNuYPpWL+PHlDhPJTtuGkXpJtOOneZRsk+iPZgp6CSbagx1O4YMJEtmDCRLZgoqCXLZgo6GUXeDxlF3gapEPB8+QFnoZTdkHCU3ZBooGXnUDxlK3o8ZStmNgQOZwcz/PWeDxlp1CuaNi1qOMpO2F2eMqlmrKVeB4vkWo8RdfzbjJcOeFxdD3v8RScPifHNZ4SWyDv/43nzmTvJ4rwlGgvWzxFPbtJ+YTn0Xa993jWCO2ZPr8XeDwPpXw/sPEUqJb8ng1EW5LPBiKeAqvRC1yGp8Th/lWA4nkgKZ6nVEsvBT2ex5uj14I+4wK7/anwPGn6/DrTjqdE9fntmeIpUH1+N/B4ikyfeIpW898NPJ4S1SeestUnngIpXTudeEpUny8bdngKVJ94Sk+fXzsfeIpMn1+eFZ4C1afPkxvkdk6feAo276+eJZ6Hm3c8hZvNF88CT4Fq6XslP8Wze9zh/uVpTrghtvz5e/lnxdVXZi/0/NHZ/p3u20WM1zOX9/yeQtr/qHRG9j87NZP3fEDL5X/WfCZ+A3yq/yFuud8zFffM9D+EqPC/qkPe0+h/iMbSq0/EPQv1TxnMljwrac9K/VOdikvHp/73BFRXeqb63+Brr/TM1L8SOV/yTKVfMZOrf49aeamnUf+eP3upp/r3emaL79XNpD097/FVfrh7PO1pnr3u4Xmep+896IPq4Xmep/VFyRRqr/XMvJ461vh8m2d91vfTcvajutjTeD1VFPWZvdizsKon0OKTZy7sWVrVE2h1tWfl9+wVH+42OcnTLiR+T3O1Z7rkWes93G/xbPQe7md5Zkuerd7DfTzJM1/yjH6BLy/3NEue0W+J2LA8Y+848xVDxch6FlZxwVSF5hl3wZTZ6z1LxZ5FcJ6N1sP9NM/Kqi1AF1uVHk/RVuWe8Rl1QV/eMT4XOaPesUvtHs8Gzz2dylmey7/EqBvOEs/LqqWXlUHWM1v2HLUe7l7PGs89C8NNnvFuiOT3eOZaPYsPH6y9x7NWerh/e+ai65FR6pnbneMTzx3FJ56yZfV5noVOT2Nvmj8/eTZRcmY21PEZp6e9zbNU6Jna+zyrtd9W09qOp/jw/HYrRD2tPk+D5/WHO56yh/tJnqk+z3ydZ42nTEWNp2zFcqpnps5z5fSJp+z0aX3lAJ57qk88hZcjPGWXo5M8c3WeKzlHPGUOOMbnKcs7nrLLO56yyzuekpt1rEfS5ZL1fUWL545y6TbPyG7wWLsb8nK8i3oaZZ7ZzZ6FMs88dM8WT9HyIjJPg+c95fyAZwyelTLPEs972iM8o/C0yjxXt5u970tO9mzwxHNFOjzP8UzxFNheOsnz8/evn+bZ4bljuw5PYc/2FM8MT9H58/P3T/AU3X7V6tngKfJ5zvU0eIp6FnjiKeJZn+JZ4olnwJ4ft7NH9Z6J5AYQnniefLwf8rR44hmu5+ftugFPPG9bjzI8Z541nnv26/CU9UzO8MzxFJ0/DZ54BuxZaPPccfsRnuF6lniKelbaPBM87/EcTvG06jwrPPV4pnjiebRieb8dQbB/Tzd93yhS3OmZ4TmfIvDctiOBZzSeuT7P/LBngueODTs8ZTdE/Os7nnsa+LvGZ3TvK6/wvKNBwlO2oO/O8DQKPc1mT4un6Pi81LNV75nKeRZ44imzvt/l2SitP/HEM+R+0+9ZnzrX1JFxrt0Pab07UgmehzxTPAX2P+/yTNR7Znge2+J5W2fxjNhzjM3TbPZM8RTwrPGU3A45x7PS51ke9RxP/eaxXV63+npFPEXb93M89V0+v/r+juQezx7PR3tmgXvGdnozx1ORp77T7yZwz1Zpe4TnVZ4DnjvazbvGZ6O03bxrfKr1rE/wTLd8W13tO554hty+4ynbvr9+MLEdi2zLMqir3cQTz5Dbd3/92Z/629TabuKJZ8jtu01OmD8/e44P8JQbn0adZxW4p9XqOeIp2r7jGYFngee8q8Fzz/YSnniGvF2HJ54hb9fd5DniKdrtRuZp8HyW54Cn6O7BAz07PLcULGd6Wjwv9uwf4JlLeaZ44hmw54puN7LbEUo87/c0F3q2Wj3tCZ65Os9KwLM907N5nmeH54aC+sz50+CJJ554xuqZ4qnJs8ATz4MNH554xuA54IknnnjiGYxniSee8p45nmF6Vnhe7VnjiSeeQsklPFs88bzds8czeE+LJ+PzdM/mTM9Eq2fn7d+bM493PJ/safBU5Jmq81x/+fzrKn6h5/gEzwJPPC/JhsvnG3nPFbtbA554BuwZ2e1xG25HqPHEE8+4bzfcdfvRu2eN5zFPgyeel2TX5clinjmeLs8zd19brZ4DnqKevX99P3P39RGeBs/Dnp3XYTzTs8ETz43byXgKeDYneK44O1Br9azxFPVM7vFM8MRzY7uJpy/prnbzQs8RTzy3tpt4HtjQvddz0OrZ4Rm+Z/lcz9bvOeCJ5znJQ/dU+7ae5gzP6rmeNZ54BuyZ+D17PDd7jnjiGbDnsOC5/xIjdbd3GDzv8ewXvqzFE8+gPHM8D3p2eMps6OJ5imeL52Wezd7v/uDbtZuFaRfPDRu6TjPG51HPGk9Rz+QMzwxPl2fN+NzqOZ7i+dzHM+Ap6zkseSZ44nlO1t6u3eN5meeI52bPDk9Rz3bBczjTM7LH21g8b/FsFjz3f+QcT8fAwnNLA+1sgfAM0tNsLiwCz+rLvZe+rsNzq+eI56meqczpTXWeay+vG/CMwbN4qme/VLc2eG717PAU9WyX9lHwlPWs8dywwLqHYCWxPb/Gs1XpORuCJZ5HPJMFzxHPLR/ITVaInJF4quewNLDx3OzZL3n2p377RqNnt1RodXhu9WyXNqZaPJ1lz6YPlYl84qd6zjugVKI90udZ7Sw/XzdEEjw3eo5LXzmee3jE5bm3/HyhGPDc6tkvHao9nls9u6XOvz3Xs46JM91bfr4UTA2eWz2bpS89eTmMyjPbW35+TxUDnls9x6Vjtcdzq6d7CBqB8zvP9HST5QL1jH2kZ7v0tQnj07mLua17/xpcx26/sE/0HJcGV4fnVs9uaXMowXOrZ7sw+fane0bVvpu93dGfA77Gc7Nn7R/dR2++0OaZH1iOLtreUjc+z7wBqLr1t3mPZ3vr+NTnWeMp6XnqB3qgZ4enqGeD5/oUN9crzxufA56inj2eosf7qdNn+jzPBE9Jz/5uT6vLs8NT1PPU6XPV6UBdnvXdnrrmz5M/zeM8BzxFPdvbPeOaP8tbp8/HeZ59sOUP8xzwFPXs8BT1bPEU9azxFPVM8JT0PP1RpuZZni2eop4JnpKePZ6inh2eop6nT5+rnl6ix7PGU9Lzgp3HR3kOeG5Odety9CzPBk9RzwRPSc8BT1HPHk9RzxZPUc8mDM9Ri2eC5/bc/Dke5NkH4jkoOd5bPEU9azwlPa/5GCs8ex2eLZ6injWeop5JKJ6dCs8BT1HPPhjPVoXnRZ/i7tvF8YzUsw7meK9VeCZ4SnqO4XgmGjyvKqLNQzy7YDzj2v70XR/SBOM5qPBM8JT0HMLx7DV4tniKetbheMa1HeIpAJNwPFsFnkNAnnG1R+4P1OEp+oHagDwTBZ4NnqIfqA7Hc1TgOSbheEbWHjlvmB7u/fbqPLuAPCNrj5wP7GkD8uwUeDb3fvuoPdN7S5RMWbvp8hxD8mzi9+xD8oys3XRd8N3dPN1o82wD8oytPXKd4Kzv/XVGXc47PMckIM8+Os/y3o9Q6So/HRv0XUiebXSe5t6Kr9JVfjo8b55uot5NduxIDCF5xlcuzTqUi1eAQtnyPquo65A841ve3yvAq48wo2w5el9hrz7Ccl3d+2xF6ELyjHA5ej/iri6gM23T59sIaULybGP0zG6dsVJth/v0E13/EbQNz+kCP9z63RXMnm8LfHfrd4+/+HxvUZpwPIdIOScL/L2/zfgXo7cF/oYxYZTNnpMltrn1txl9qzk75O4u16LeqZsfcsO9R8dL/kvEnN+H3C0VSqVrcL4OkdvLtcgrz7casL93ton5LJzzI90zLDJ1w/N3RX9XBV2pKTynNctdH6RQNjr/TKB3VdCZrsnzzwF/3/6D0dG3vw3QNoSGN/7a888YufNzpMqmz3A6XiWH++0pGJ7yR3yPg1yF0WFACCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEIIIYQQQgghhBBCCCGEEEKIlvx/DqHdxg==
B64_LABELS

cat > "$SCRIPTS/fetch_windows.py" <<'PY_FETCHW'
"""Fetch the 500p2a raster labels + the exact zarr chunks covering the three
windows from the HF bucket (public, verified live). Threaded, resumable."""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

CH = 128
CHUNKDIR = os.path.join(cl.DATA, "p2a_chunks")
UA = {"User-Agent": "curl/8"}

def http_get(url, tries=4):
    waits = [0, 5, 15, 45]
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(waits[i + 1])

def fetch_file(rel, dest, expect_bytes=None):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return False
    b = http_get(cl.BUCKET + "/" + rel)
    if expect_bytes is not None and len(b) != expect_bytes:
        raise AssertionError(f"{rel}: got {len(b)} bytes, expected {expect_bytes}")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, dest)
    return True

def chunk_keys():
    keys = set()
    for w in cl.WINDOWS.values():
        cy0, cy1 = w["y0"] // CH, (w["y0"] + w["size"] - 1) // CH
        cx0, cx1 = w["x0"] // CH, (w["x0"] + w["size"] - 1) // CH
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                keys.add(f"0.{cy}.{cx}")
    return sorted(keys)

def main():
    os.makedirs(CHUNKDIR, exist_ok=True)
    za = json.loads(http_get(cl.BUCKET + "/500p2a.zarr/0/.zarray").decode())
    assert za["shape"] == list(cl.P2A_SHAPE), za
    assert za["chunks"] == list(cl.P2A_CHUNK), za
    assert za["dtype"] == "|u1" and za["compressor"]["id"] == "blosc", za
    sep = za.get("dimension_separator", ".")
    assert sep == ".", f"unexpected dimension_separator {sep!r}"
    with open(os.path.join(cl.DATA, "p2a_zarray.json"), "w") as f:
        json.dump(za, f)
    cl.say("EXPB_FETCH .zarray parsed and matches embedded expectation "
           "(shape/chunks/dtype/compressor/separator)")
    fetch_file("500p2a_inklabels.tif",
               os.path.join(cl.DATA, "500p2a_inklabels.tif"),
               cl.P2A_LABEL_BYTES)
    fetch_file("500p2a_supervision_mask.tif",
               os.path.join(cl.DATA, "500p2a_supervision_mask.tif"),
               cl.P2A_MASK_BYTES)
    cl.say("EXPB_FETCH raster labels fetched (byte sizes exact)")
    keys = chunk_keys()
    todo = [k for k in keys
            if not (os.path.exists(os.path.join(CHUNKDIR, k))
                    and os.path.getsize(os.path.join(CHUNKDIR, k)) > 0)]
    cl.say(f"EXPB_FETCH {len(keys)} chunks total, {len(todo)} to fetch "
           f"(8 threads, resumable)")
    done = [0]
    def one(k):
        fetch_file("500p2a.zarr/0/" + k, os.path.join(CHUNKDIR, k))
        done[0] += 1
        if done[0] % 200 == 0:
            cl.say(f"EXPB_FETCH progress {done[0]}/{len(todo)}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, todo))
    missing = [k for k in keys
               if not (os.path.exists(os.path.join(CHUNKDIR, k))
                       and os.path.getsize(os.path.join(CHUNKDIR, k)) > 0)]
    assert not missing, f"missing chunks after fetch: {missing[:5]}..."
    cl.say(f"EXPB_FETCH complete: {len(keys)} chunks on disk")

if __name__ == "__main__":
    main()
PY_FETCHW

cat > "$SCRIPTS/build_windows.py" <<'PY_BUILDW'
"""Assemble requested windows (argv names; default all three) from fetched
chunks, assert every embedded fingerprint, write the two 9.36um depth-mode
zarrs (iso: 15 layers; fit17: 17 layers), the scoring label crops, the
9.36um mask, and (anchor window only) the native 2.215um zarr for the rungs."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

CH = 128
CHUNKDIR = os.path.join(cl.DATA, "p2a_chunks")

def main(names):
    import numcodecs
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    codec = numcodecs.Blosc()
    ink_full = np.array(Image.open(
        os.path.join(cl.DATA, "500p2a_inklabels.tif"))) > 0
    msk_full = np.array(Image.open(
        os.path.join(cl.DATA, "500p2a_supervision_mask.tif"))) > 0
    assert ink_full.shape == cl.P2A_SHAPE[1:], ink_full.shape
    r = dict(ink=int(ink_full.sum()), mask=int(msk_full.sum()),
             ink_and_mask=int((ink_full & msk_full).sum()),
             ink_outside_mask=int((ink_full & ~msk_full).sum()),
             annot_blank=int((msk_full & ~ink_full).sum()))
    assert r == cl.P2A_RASTER, f"raster label counts drifted: {r}"
    cl.say("P2A_BUILD raster label counts EXACT "
           f"(ink_outside_mask={r['ink_outside_mask']})")
    S, NZ, S9 = cl.S_WIN, cl.NZ_WIN, cl.S9
    for name in names:
        w = cl.WINDOWS[name]
        y0, x0 = w["y0"], w["x0"]
        ink = ink_full[y0:y0 + S, x0:x0 + S]
        msk = msk_full[y0:y0 + S, x0:x0 + S]
        got = dict(ink=int(ink.sum()), ink_and_mask=int((ink & msk).sum()),
                   ink_outside_mask=int((ink & ~msk).sum()),
                   annot_blank=int((msk & ~ink).sum()), mask=int(msk.sum()))
        exp = {k: w[k] for k in got}
        assert got == exp, f"{name} label counts drifted: {got} != {exp}"
        np.save(os.path.join(cl.DATA, f"{name}_ink.npy"), ink)
        np.save(os.path.join(cl.DATA, f"{name}_mask.npy"), msk)
        m9 = cl.zoom_to(msk.astype(np.float32), (S9, S9),
                        prefilter=False) >= 0.5
        np.save(os.path.join(cl.DATA, f"{name}_mask936.npy"), m9)
        npy = os.path.join(cl.DATA, "tmp", f"{name}.u8")
        vol = np.memmap(npy, dtype=np.uint8, mode="w+", shape=(NZ, S, S))
        cy0, cx0 = y0 // CH, x0 // CH
        for cy in range(cy0, (y0 + S - 1) // CH + 1):
            for cx in range(cx0, (x0 + S - 1) // CH + 1):
                raw = open(os.path.join(CHUNKDIR, f"0.{cy}.{cx}"), "rb").read()
                arr = np.frombuffer(codec.decode(raw), np.uint8)
                arr = arr.reshape(NZ, CH, CH)
                ys, xs = cy * CH - y0, cx * CH - x0
                ty0, ty1 = max(0, ys), min(S, ys + CH)
                tx0, tx1 = max(0, xs), min(S, xs + CH)
                vol[:, ty0:ty1, tx0:tx1] = \
                    arr[:, ty0 - ys:ty1 - ys, tx0 - xs:tx1 - xs]
        sub = np.asarray(vol[::8, ::4, ::4]).astype(np.float32)
        stats = (float(sub.mean()), float(sub.std()),
                 float((sub == 0).mean()), int(sub.max()))
        assert abs(stats[0] - w["vol_mean"]) < 0.5, (name, stats)
        assert abs(stats[1] - w["vol_std"]) < 0.5, (name, stats)
        assert abs(stats[2] - w["vol_zero_frac"]) < 0.002, (name, stats)
        assert stats[3] == 255, (name, stats)
        cl.say(f"P2A_BUILD {name} volume stats match the Aug-25 record "
               f"(mean={stats[0]:.3f} sd={stats[1]:.3f} zf={stats[2]:.5f})")
        if name == cl.EXPA_ANCHOR:
            zn = os.path.join(cl.DATA, f"{name}_native.zarr")
            if not os.path.exists(zn):
                cl.write_group_zarr(zn, np.asarray(vol))
        for mode, nz9 in cl.DEPTH_MODES.items():
            z9 = os.path.join(cl.DATA, f"{name}_{mode}.zarr")
            if os.path.exists(z9):
                continue
            tmp = os.path.join(cl.DATA, "tmp", f"{name}_{mode}.f32")
            mm = cl.resample_stack(lambda z: np.asarray(vol[z]), NZ, S, S,
                                   (nz9, S9, S9), tmp, tag=f"{name}{mode}")
            v = np.clip(np.rint(np.asarray(mm)), 0, 255).astype(np.uint8)
            del mm; os.remove(tmp)
            cl.write_group_zarr(z9, v)
            cl.say(f"P2A_BUILD {name} {mode}: ({nz9},{S9},{S9}) at "
                   f"{cl.P2A_PITCH * NZ / nz9:.2f} um depth pitch")
        cl.save_preview(np.asarray(vol[NZ // 2]),
                        os.path.join(cl.OUT, "previews", f"{name}_midslice.png"))
        del vol; os.remove(npy)
        cl.say(f"P2A_BUILD {name} zarrs ready (native 65x{S}x{S} @ "
               f"{cl.P2A_PITCH} um -> iso {cl.ISO_NZ}x{S9}x{S9}, "
               f"fit17 {cl.FIT17_NZ}x{S9}x{S9})")

if __name__ == "__main__":
    names = sys.argv[1:] or list(cl.WINDOWS)
    for n in names:
        assert n in cl.WINDOWS, f"unknown window {n}"
    main(names)
PY_BUILDW

cat > "$SCRIPTS/ctl_build.py" <<'PY_CTLB'
"""CTL arms on PHerc0139 w035 (the in-domain control). argv[1]: fetch | build.
fetch: the 399 raw S3 chunks covering the label-bbox crop (threaded,
resumable; a 404 chunk is an all-zero chunk under fill_value 0 and is
recorded as such). build: ctl_native (as released), ctl_scalefault (x1.9504
in-plane and depth -- exactly v2's pitch error), ctl_half (x0.5)."""
import json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

CH = 128
CHUNKDIR = os.path.join(cl.DATA, "ctl_chunks")
UA = {"User-Agent": "curl/8"}
Y0, Y1, X0, X1 = cl.CTL_CROP
H, W = Y1 - Y0, X1 - X0
NZ = cl.CTL_SHAPE[0]

def http_get(url, tries=4):
    waits = [0, 5, 15, 45]
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == tries - 1:
                raise
            time.sleep(waits[i + 1])
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(waits[i + 1])

def chunk_keys():
    keys = []
    for cy in range(Y0 // CH, (Y1 - 1) // CH + 1):
        for cx in range(X0 // CH, (X1 - 1) // CH + 1):
            keys.append((cy, cx))
    return keys

def dest(cy, cx):
    return os.path.join(CHUNKDIR, f"{cy}_{cx}")

def fetch():
    os.makedirs(CHUNKDIR, exist_ok=True)
    za = json.loads(http_get(cl.CTL_SV + "/0/.zarray").decode())
    assert za["shape"] == list(cl.CTL_SHAPE), za
    assert za["chunks"] == list(cl.CTL_CHUNK), za
    assert za["dtype"] == "|u1" and za["compressor"] is None, za
    assert za.get("dimension_separator") == "/", za
    assert za.get("fill_value", 0) == 0, za
    cl.say("CTL_FETCH .zarray parsed and matches embedded expectation "
           "(shape/chunks/dtype/raw/separator)")
    keys = chunk_keys()
    assert len(keys) == 399, len(keys)
    todo = [k for k in keys if not os.path.exists(dest(*k))]
    cl.say(f"CTL_FETCH {len(keys)} chunks total, {len(todo)} to fetch "
           f"(8 threads, resumable)")
    done = [0]; missing = [0]
    def one(k):
        cy, cx = k
        b = http_get(f"{cl.CTL_SV}/0/0/{cy}/{cx}")
        tmp = dest(cy, cx) + ".part"
        if b is None:
            missing[0] += 1
            open(tmp, "wb").close()           # zero-length = all-zero chunk
        else:
            assert len(b) == cl.CTL_CHUNK_BYTES, (k, len(b))
            with open(tmp, "wb") as f:
                f.write(b)
        os.replace(tmp, dest(cy, cx))
        done[0] += 1
        if done[0] % 100 == 0:
            cl.say(f"CTL_FETCH progress {done[0]}/{len(todo)}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, todo))
    left = [k for k in keys if not os.path.exists(dest(*k))]
    assert not left, f"missing chunks after fetch: {left[:5]}..."
    cl.say(f"CTL_FETCH complete: {len(keys)} chunks on disk "
           f"({missing[0]} absent-on-server = all-zero)")

def assemble():
    vol = np.zeros((NZ, H, W), dtype=np.uint8)
    for cy, cx in chunk_keys():
        raw = open(dest(cy, cx), "rb").read()
        if not raw:
            continue
        arr = np.frombuffer(raw, np.uint8).reshape(NZ, CH, CH)
        ys, xs = cy * CH - Y0, cx * CH - X0
        ty0, ty1 = max(0, ys), min(H, ys + CH)
        tx0, tx1 = max(0, xs), min(W, xs + CH)
        vol[:, ty0:ty1, tx0:tx1] = arr[:, ty0 - ys:ty1 - ys, tx0 - xs:tx1 - xs]
    return vol

def build():
    pos, neg = cl.load_ctl_labels()        # asserts sha256 + exact counts
    cl.say(f"CTL_BUILD embedded labels verified: pos={int(pos.sum())} "
           f"neg={int(neg.sum())} crop={H}x{W}")
    vol = assemble()
    sub = vol[::4, ::4, ::4].astype(np.float32)
    stats = dict(mean=float(sub.mean()), std=float(sub.std()),
                 zero_frac=float((sub == 0).mean()), max=int(sub.max()))
    json.dump(stats, open(os.path.join(cl.RESULTS, "ctl_volume_stats.json"),
                          "w"), indent=1)
    cl.say(f"CTL_BUILD crop stats mean={stats['mean']:.2f} sd={stats['std']:.2f} "
           f"zero_frac={stats['zero_frac']:.4f} max={stats['max']}")
    assert stats["zero_frac"] < 0.20, ("crop is mostly empty", stats)
    assert stats["max"] == 255, ("crop never reaches 255", stats)
    zn = os.path.join(cl.DATA, "ctl_native.zarr")
    if not os.path.exists(zn):
        cl.write_group_zarr(zn, vol)
    cl.save_preview(vol[NZ // 2], os.path.join(cl.OUT, "previews",
                                               "ctl_native_midslice.png"), ds=2)
    cl.say("CTL_BUILD ctl_native.zarr ready (28x{}x{})".format(H, W))
    for name, factor in (("ctl_scalefault", cl.CTL_FAULT_FACTOR),
                         ("ctl_half", cl.CTL_HALF_FACTOR)):
        zp = os.path.join(cl.DATA, f"{name}.zarr")
        if os.path.exists(zp):
            continue
        shape = cl.ctl_arm_shape(factor)
        cl.say(f"CTL_BUILD {name}: x{factor:.4f} -> {shape}")
        tmp = os.path.join(cl.DATA, "tmp", f"{name}.f32")
        mm = cl.resample_stack(lambda z: vol[z], NZ, H, W, shape, tmp,
                               tag=name)
        v = np.clip(np.rint(np.asarray(mm)), 0, 255).astype(np.uint8)
        del mm; os.remove(tmp)
        cl.write_group_zarr(zp, v)
        cl.save_preview(v[shape[0] // 2], os.path.join(
            cl.OUT, "previews", f"{name}_midslice.png"), ds=2)
        del v
        cl.say(f"CTL_BUILD {name}.zarr ready {shape}")

if __name__ == "__main__":
    {"fetch": fetch, "build": build}[sys.argv[1]]()
PY_CTLB

cat > "$SCRIPTS/ctl_score.py" <<'PY_CTLS'
"""CTL scoring: three arms x two directions on the embedded w035 crop labels.
Exit 31 = HARNESS_BROKEN (native forward below 0.95); exit 32 = the known
depth-order fault did not reproduce (native reverse above 0.80). Both are
fatal by pre-registration. The scale-fault verdict is recorded, never fatal."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

def auc_of(tif, pos, neg):
    import tifffile
    pred = tifffile.imread(tif)
    m = cl.resample_pred(pred, pos.shape)
    q = cl.quantize_map(m)
    return cl.hist_auc(cl.masked_hist(q, pos), cl.masked_hist(q, neg)), pred

def main():
    pos, neg = cl.load_ctl_labels()
    arms = {}
    for arm in ("ctl_native", "ctl_scalefault", "ctl_half"):
        arms[arm] = {}
        for d, suffix in (("forward", ""), ("reverse", "_reverse")):
            tif = os.path.join(cl.PREDS, f"{arm}{suffix}.tif")
            auc, pred = auc_of(tif, pos, neg)
            arms[arm][d] = float(auc)
            arms[arm][f"{d}_map_shape"] = list(pred.shape)
            cl.save_preview(pred, os.path.join(cl.OUT, "previews",
                                               f"{arm}_{d}.png"), ds=2)
            cl.say(f"CTL_SCORE {arm} {d}: AUC={auc:.4f} (map {pred.shape})")
    nat_f, nat_r = arms["ctl_native"]["forward"], arms["ctl_native"]["reverse"]
    sf = max(arms["ctl_scalefault"]["forward"], arms["ctl_scalefault"]["reverse"])
    hf = max(arms["ctl_half"]["forward"], arms["ctl_half"]["reverse"])
    if sf < cl.CTL_FAULT_REPRODUCED_MAX:
        verdict = "FAULT_REPRODUCED"
    elif sf >= cl.CTL_FAULT_NOT_REPRODUCED_MIN:
        verdict = "FAULT_NOT_REPRODUCED"
    else:
        verdict = "PARTIAL"
    res = dict(arms=arms, n_pos=int(pos.sum()), n_neg=int(neg.sum()),
               crop=list(cl.CTL_CROP), fault_factor=cl.CTL_FAULT_FACTOR,
               harness_gate=dict(min_forward=cl.CTL_HARNESS_MIN_FWD,
                                 passed=bool(nat_f >= cl.CTL_HARNESS_MIN_FWD)),
               depth_order_gate=dict(max_reverse=cl.CTL_DEPTHREV_MAX,
                                     passed=bool(nat_r <= cl.CTL_DEPTHREV_MAX)),
               scale_fault=dict(best=sf, verdict=verdict,
                                reproduced_max=cl.CTL_FAULT_REPRODUCED_MAX,
                                not_reproduced_min=cl.CTL_FAULT_NOT_REPRODUCED_MIN),
               half_scale=dict(best=hf),
               on_record=dict(forward_full_canvas=0.9991, reverse_full_canvas=0.5123))
    json.dump(res, open(os.path.join(cl.RESULTS, "ctl.json"), "w"), indent=1)
    if not res["harness_gate"]["passed"]:
        cl.say(f"CTL HARNESS_BROKEN: ctl_native forward {nat_f:.4f} < "
               f"{cl.CTL_HARNESS_MIN_FWD} (on record 0.9991) -- nothing downstream "
               f"is interpretable; dying")
        sys.exit(31)
    if not res["depth_order_gate"]["passed"]:
        cl.say(f"CTL DEPTH-ORDER FAULT NOT REPRODUCED: ctl_native reverse "
               f"{nat_r:.4f} > {cl.CTL_DEPTHREV_MAX} (on record 0.5123) -- the "
               f"harness cannot certify fault controls; dying")
        sys.exit(32)
    cl.say(f"CTL GATES PASSED: native fwd={nat_f:.4f} rev={nat_r:.4f}; "
           f"scale-fault best={sf:.4f} -> {verdict}; half-scale best={hf:.4f}")

if __name__ == "__main__":
    main()
PY_CTLS

cat > "$SCRIPTS/build_expa.py" <<'PY_EXPA'
"""EXPERIMENT A rung builder -- all degraded win1 volumes, from the native
2.215um window (iso depth throughout). Rungs (pre-registered): pitch
{3.24,4.32,5.5,6.5,8.0,12.0}um via anti-aliased resample 2.215->P then
regrid P->9.36 (the 9.36 and 2.215 rows reuse baseline); noise k in
{1,2,4,8} x sigma_plate on the 9.36 iso grid; bit4; blur sigma {1.0,2.0} px."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

NZ, S = cl.NZ_WIN, cl.S_WIN
NZ9, S9 = cl.ISO_NZ, cl.S9
A = cl.EXPA_ANCHOR

def main():
    base = np.asarray(cl.read_zarr0(os.path.join(cl.DATA, f"{A}_iso.zarr")))
    assert base.shape == (NZ9, S9, S9) and base.dtype == np.uint8, \
        (base.shape, base.dtype)
    m9 = np.load(os.path.join(cl.DATA, f"{A}_mask936.npy"))
    assert m9.shape == (S9, S9), m9.shape
    slab = base.astype(np.float32)                 # all 15 iso layers
    d = (slab[:-1] - slab[1:]) / np.sqrt(2.0)
    dm = np.broadcast_to(m9, d.shape)
    sigma_plate = 1.4826 * float(np.median(np.abs(d[dm])))
    json.dump(dict(sigma_plate=sigma_plate, dtype="uint8", n_layers=int(NZ9)),
              open(os.path.join(cl.RESULTS, "expA_sigma.json"), "w"), indent=1)
    cl.say(f"RUNGS sigma_plate={sigma_plate:.3f} DN (uint8, in-mask, all "
           f"{NZ9} iso layers, adjacent-slice MAD estimator)")
    for k in cl.NOISE_KS:
        p = os.path.join(cl.DATA, f"rung_n{k}.zarr")
        if os.path.exists(p):
            continue
        rng = np.random.default_rng([cl.SEED, k])
        noisy = base.astype(np.float32) + rng.normal(
            0.0, k * sigma_plate, size=base.shape).astype(np.float32)
        cl.write_group_zarr(p, np.clip(np.rint(noisy), 0, 255).astype(np.uint8))
        cl.say(f"RUNGS built noise k={k}")
    p = os.path.join(cl.DATA, "rung_bit4.zarr")
    if not os.path.exists(p):
        cl.write_group_zarr(p, cl.quant4(base))
        cl.say("RUNGS built bit4 (uint8 -> 16 levels via rint(v/17)*17)")
    from scipy import ndimage
    for sg in cl.BLUR_SIGMAS:
        p = os.path.join(cl.DATA, f"rung_blur{sg}.zarr")
        if os.path.exists(p):
            continue
        b = ndimage.gaussian_filter(base.astype(np.float32), sigma=sg,
                                    mode="nearest")
        cl.write_group_zarr(p, np.clip(np.rint(b), 0, 255).astype(np.uint8))
        cl.say(f"RUNGS built blur sigma={sg}")
    vol = None
    for P in cl.PITCHES:
        if abs(P - cl.MODEL_PITCH) < 1e-9:
            continue
        p = os.path.join(cl.DATA, f"rung_p{P}.zarr")
        if os.path.exists(p):
            continue
        if vol is None:
            vol = np.asarray(cl.read_zarr0(
                os.path.join(cl.DATA, f"{A}_native.zarr")))
            assert vol.shape == (NZ, S, S) and vol.dtype == np.uint8, \
                (vol.shape, vol.dtype)
        nzP = cl.rint_shape(NZ, cl.P2A_PITCH, P)
        SP = cl.rint_shape(S, cl.P2A_PITCH, P)
        cl.say(f"RUNGS pitch {P}um stage A -> ({nzP},{SP},{SP})")
        tmpA = os.path.join(cl.DATA, "tmp", f"p{P}_A.f32")
        mmA = cl.resample_stack(lambda z: vol[z], NZ, S, S,
                                (nzP, SP, SP), tmpA, tag=f"p{P}A")
        tmpB = os.path.join(cl.DATA, "tmp", f"p{P}_B.f32")
        mmB = cl.resample_stack(lambda z: np.asarray(mmA[z]), nzP, SP, SP,
                                (NZ9, S9, S9), tmpB, tag=f"p{P}B")
        del mmA; os.remove(tmpA)
        v = np.clip(np.rint(np.asarray(mmB)), 0, 255).astype(np.uint8)
        del mmB; os.remove(tmpB)
        cl.write_group_zarr(p, v)
        cl.say(f"RUNGS built pitch {P}um")

if __name__ == "__main__":
    main()
PY_EXPA

cat > "$SCRIPTS/score_expa.py" <<'PY_SCOREA'
"""Experiment A scoring on the 500p2a win1 anchor at the CORRECTED pitch.
argv[1]: baseline | rungs.
baseline: AUC fwd/rev for BOTH depth modes on the native 2.215um grid; the
corrected anchor = max(fwd, rev) of the iso mode; curve gate 0.85 -> exit 21
when the gate fails (the caller records it and CONTINUES: the anchor is the
result, only the rungs are skipped). rungs: AUC + retained + DETECTABLE."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR = os.environ["VAR"]
A = cl.EXPA_ANCHOR
W1 = cl.WINDOWS[A]

def native_labels():
    ink = np.load(os.path.join(cl.DATA, f"{A}_ink.npy"))
    msk = np.load(os.path.join(cl.DATA, f"{A}_mask.npy"))
    pos, neg = ink & msk, msk & ~ink
    assert int(pos.sum()) == W1["ink_and_mask"], int(pos.sum())
    assert int(neg.sum()) == W1["annot_blank"], int(neg.sum())
    return pos, neg

def map_auc(tif_path, pos, neg):
    import tifffile
    pred = tifffile.imread(tif_path)
    up = cl.upsample_pred(pred, pos.shape)
    q = cl.quantize_map(up)
    return cl.hist_auc(cl.masked_hist(q, pos), cl.masked_hist(q, neg)), pred

def baseline():
    pos, neg = native_labels()
    modes = {}
    for mode in cl.DEPTH_MODES:
        stem = "expA_base" if mode == cl.PRIMARY_DEPTH else f"expA_base_{mode}"
        aucs = {}
        for d, suffix in (("forward", ""), ("reverse", "_reverse")):
            p = os.path.join(cl.PREDS, f"{stem}{suffix}.tif")
            aucs[d], pred = map_auc(p, pos, neg)
            cl.save_preview(pred, os.path.join(cl.OUT, "previews",
                                               f"expA_base_{mode}_{d}.png"), ds=2)
            cl.say(f"EXPA_BASELINE {A}@500p2a [{mode}] AUC[{d}] = {aucs[d]:.4f} "
                   f"(pos={int(pos.sum())} neg={int(neg.sum())})")
        best = "forward" if aucs["forward"] >= aucs["reverse"] else "reverse"
        modes[mode] = dict(auc_forward=aucs["forward"],
                           auc_reverse=aucs["reverse"], direction=best,
                           best=aucs[best])
    prim = modes[cl.PRIMARY_DEPTH]
    anchor = prim["best"]
    if anchor >= 0.85:
        reading = "Aug-25 chance result was an input fault; ink_9um transfers in-modality to an unseen scroll at curve-anchoring quality"
    elif anchor >= 0.65:
        reading = "partial transfer; Bet A's 500p2a gate rebases to anchor+0.05"
    else:
        reading = "transfer failure confirmed at the correct pitch (read jointly with the CTL scale-fault verdict)"
    res = dict(anchor=A, pitch_um=cl.P2A_PITCH, primary_depth=cl.PRIMARY_DEPTH,
               modes=modes, corrected_anchor_auc=anchor,
               direction=prim["direction"], gate=cl.GATE_BASELINE_AUC,
               gate_passed=bool(anchor >= cl.GATE_BASELINE_AUC),
               prestated_reading=reading,
               n_pos=int(pos.sum()), n_neg=int(neg.sum()),
               aug25_void=dict(auc_forward=0.5382, auc_reverse=0.5055,
                               pitch_assumed=cl.P2A_PITCH_WRONG,
                               true_effective_pitch_um=round(
                                   cl.MODEL_PITCH * cl.P2A_PITCH / cl.P2A_PITCH_WRONG, 3)))
    json.dump(res, open(os.path.join(cl.RESULTS, "expA_baseline.json"), "w"),
              indent=1)
    open(os.path.join(VAR, "direction.txt"), "w").write(prim["direction"])
    cl.say(f"CORRECTED ANCHOR ({A} iso, {prim['direction']}): AUC = {anchor:.4f} "
           f"[fit17 best {modes['fit17']['best']:.4f}] -- {reading}")
    if not res["gate_passed"]:
        cl.say(f"EXPA_BASELINE curve gate not met: {anchor:.4f} < "
               f"{cl.GATE_BASELINE_AUC}; rungs are skipped; the anchor stands "
               f"as the result (win2/win3 + EXP B still run)")
        sys.exit(21)
    cl.say(f"EXPA_BASELINE curve gate PASSED: AUC_base={anchor:.4f} "
           f"direction={prim['direction']} -- rungs will use it")

def rungs():
    pos, neg = native_labels()
    base = json.load(open(os.path.join(cl.RESULTS, "expA_baseline.json")))
    direction = base["direction"]
    auc_b = base["corrected_anchor_auc"]
    rows = []
    def add(rung_id, family, x, tif, reused=False, note=None):
        auc = auc_b if reused else map_auc(tif, pos, neg)[0]
        retained = (auc - 0.5) / max(auc_b - 0.5, 1e-9)
        det = bool(auc >= cl.DETECT_AUC_MIN and retained >= cl.DETECT_RETAIN_MIN)
        row = dict(rung=rung_id, family=family, x=x, auc=auc,
                   retained=retained, detectable=det, reused=reused)
        if note:
            row["note"] = note
        rows.append(row)
        cl.say(f"EXPA_SCORE {rung_id:<12} AUC={auc:.4f} retained={retained:.3f}"
               f" DETECTABLE={det}{' (reused baseline)' if reused else ''}")
    add(f"pitch_{cl.P2A_PITCH}", "pitch", cl.P2A_PITCH, None, reused=True,
        note="native acquisition == baseline")
    for Pu in cl.PITCHES:
        if abs(Pu - cl.MODEL_PITCH) < 1e-9:
            add(f"pitch_{Pu}", "pitch", Pu, None, reused=True,
                note="2.215->9.36 == baseline by construction")
        else:
            add(f"pitch_{Pu}", "pitch", Pu,
                os.path.join(cl.PREDS, f"expA_p{Pu}.tif"))
    for k in cl.NOISE_KS:
        add(f"noise_k{k}", "noise", k, os.path.join(cl.PREDS, f"expA_n{k}.tif"))
    add("bit8", "bitdepth", 8, None, reused=True,
        note="released volume is already uint8; baseline by construction")
    add("bit4", "bitdepth", 4, os.path.join(cl.PREDS, "expA_bit4.tif"))
    for sg in cl.BLUR_SIGMAS:
        add(f"blur_{sg}", "blur", sg,
            os.path.join(cl.PREDS, f"expA_blur{sg}.tif"))
    pit = [r for r in rows if r["family"] == "pitch" and r["detectable"]]
    limit = max((r["x"] for r in pit), default=None)
    sigma = json.load(open(os.path.join(cl.RESULTS, "expA_sigma.json")))
    out = dict(anchor=A, baseline=base, sigma_plate=sigma, direction=direction,
               rungs=rows, detectability_limit_um=limit,
               rule=dict(auc_min=cl.DETECT_AUC_MIN,
                         retain_min=cl.DETECT_RETAIN_MIN))
    json.dump(out, open(os.path.join(cl.RESULTS, "expA_curve.json"), "w"),
              indent=1)
    cl.say(f"EXPA_SCORE curve complete; detectability limit = {limit} um "
           f"(coarsest DETECTABLE pitch)")

if __name__ == "__main__":
    {"baseline": baseline, "rungs": rungs}[sys.argv[1]]()
PY_SCOREA

cat > "$SCRIPTS/score_expb.py" <<'PY_SCOREB'
"""Experiment B scoring: real AUC vs 40 rigid-translation matched nulls, all
inside (mask==1 AND label==0); per-cell gap + pre-registered verdicts.
Inputs win1/win2/win3 at the corrected 2.215um pitch. Cells: iso fwd/rev
(primary; the audited cell is the iso direction with the higher real AUC)
and fit17 fwd/rev (reported, never verdict-bearing)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR = os.environ["VAR"]
IDX = {"win1": 0, "win2": 1, "win3": 2}   # fixed seed offsets

def load_input(name):
    ink = np.load(os.path.join(cl.DATA, f"{name}_ink.npy"))
    msk = np.load(os.path.join(cl.DATA, f"{name}_mask.npy"))
    shapes = ink & msk           # positives; out-of-mask ink excluded here
    blank = msk & ~ink           # blank = (mask==1 AND label==0), ALWAYS
    return shapes, blank, cl.P2A_PITCH

def cell_maps(name):
    cells = []
    for mode in cl.DEPTH_MODES:
        if name == cl.EXPA_ANCHOR:
            stem = "expA_base" if mode == cl.PRIMARY_DEPTH else f"expA_base_{mode}"
        else:
            stem = f"expB_{name}_{mode}"
        cells.append((mode, "forward", os.path.join(cl.PREDS, stem + ".tif")))
        cells.append((mode, "reverse",
                      os.path.join(cl.PREDS, stem + "_reverse.tif")))
    return cells

def score_cell(tif, shapes, blank, offsets):
    import tifffile
    pred = tifffile.imread(tif)
    if pred.shape != shapes.shape:
        pred = cl.upsample_pred(pred, shapes.shape)
    q = cl.quantize_map(pred)
    h_shapes = cl.masked_hist(q, shapes)
    h_blank = cl.masked_hist(q, blank)
    auc_real = cl.hist_auc(h_shapes, h_blank)
    null_aucs = []
    for dy, dx in offsets:
        h_pos = cl.translated_hist(q, shapes, blank, dy, dx)
        h_neg = h_blank - h_pos
        null_aucs.append(cl.hist_auc(h_pos, h_neg))
    return auc_real, np.array(null_aucs), pred

def main():
    inputs = ["win1", "win2", "win3"]
    results = {}
    for name in inputs:
        shapes, blank, pitch = load_input(name)
        offsets, min_px, max_px = cl.draw_translations(
            shapes, blank, pitch, seed=cl.SEED + IDX[name])
        cl.say(f"EXPB_SCORE {name}: 40 translations drawn "
               f"(|shift| in [{min_px},{max_px}] px @ {pitch}um, "
               f">= {cl.MIN_PSEUDO_POS} pseudo-pos each)")
        cells = {}
        for mode, direction, tif in cell_maps(name):
            auc_real, nulls, pred = score_cell(tif, shapes, blank, offsets)
            gap = float(auc_real - np.median(nulls))
            frac_hi = float((nulls >= cl.CONFOUND_MEDIAN_AUC).mean())
            cells[f"{mode}_{direction}"] = dict(
                depth_mode=mode, direction=direction,
                auc_real=float(auc_real),
                null_median=float(np.median(nulls)),
                null_max=float(nulls.max()), null_min=float(nulls.min()),
                null_p95=float(np.percentile(nulls, 95)),
                null_spread_sd=float(nulls.std()),
                null_frac_ge_060=frac_hi,
                cell_confounded=bool(
                    np.median(nulls) >= cl.CONFOUND_MEDIAN_AUC
                    or frac_hi >= cl.CONFOUND_FRAC),
                gap=gap,
                beats_all_40=bool(auc_real > nulls.max()),
                rank_p=1.0 / (len(nulls) + 1),
                null_aucs=[float(a) for a in nulls])
            cl.say(f"EXPB_SCORE {name} {mode} {direction}: "
                   f"AUC_real={auc_real:.4f} null_med="
                   f"{np.median(nulls):.4f} null_max={nulls.max():.4f} "
                   f"GAP={gap:.4f}")
            cl.save_preview(pred, os.path.join(
                cl.OUT, "previews", f"expB_{name}_{mode}_{direction}.png"))
        prim = {k: v for k, v in cells.items()
                if v["depth_mode"] == cl.PRIMARY_DEPTH}
        aud_key = max(prim, key=lambda k: prim[k]["auc_real"])
        aud = cells[aud_key]
        confounded = bool(aud["cell_confounded"])
        genuine = bool(not confounded and aud["beats_all_40"]
                       and aud["gap"] >= cl.GAP_MIN)
        verdict = ("SHAPE_CONFOUNDED" if confounded
                   else "GENUINE" if genuine else "INDETERMINATE")
        results[name] = dict(
            pitch_um=pitch, audited_cell=aud_key, cells=cells,
            offsets=[[int(a), int(b)] for a, b in offsets],
            min_shift_px=min_px, max_shift_px=max_px,
            n_pos=int(shapes.sum()), n_blank=int(blank.sum()),
            verdict=verdict, shape_confounded=confounded, genuine=genuine)
        cl.say(f"EXPB_SCORE {name} VERDICT={verdict} "
               f"(audited {aud_key}: real={aud['auc_real']:.4f} "
               f"null_med={aud['null_median']:.4f} gap={aud['gap']:.4f})")
    overall = ("SHAPE_CONFOUNDED"
               if any(r["shape_confounded"] for r in results.values())
               else "GENUINE"
               if all(r["genuine"] for r in results.values())
               else "MIXED")
    out = dict(prereg=dict(confound_median_auc=cl.CONFOUND_MEDIAN_AUC,
                           confound_frac=cl.CONFOUND_FRAC,
                           gap_min=cl.GAP_MIN, n_translations=cl.N_TRANSLATIONS,
                           min_shift_mm=cl.MIN_SHIFT_MM,
                           max_shift_frac=cl.MAX_SHIFT_FRAC, seed=cl.SEED),
               inputs=results, overall=overall,
               overall_note=("overall verdict over win1/win2/win3 iso cells; "
                             "fit17 cells are reported, never verdict-bearing"))
    json.dump(out, open(os.path.join(cl.RESULTS,
                                     "expB_hallucination.json"), "w"), indent=1)
    cl.say(f"EXPB_SCORE complete; overall={overall}")

if __name__ == "__main__":
    main()
PY_SCOREB

cat > "$SCRIPTS/finalize.py" <<'PY_FINAL'
"""Aggregate results, verify the full inventory, refuse to bless a partial run.
Writes results/results.json AND out/results.json (the laptop guard harvests
the latter by name)."""
import json, os, sys, time
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR = os.environ["VAR"]
RUN_RUNGS = os.environ.get("RUN_RUNGS", "1") == "1"

def main():
    missing = []
    gate_failed = os.path.exists(os.path.join(VAR, "gate_failed"))
    rungs_expected = RUN_RUNGS and not gate_failed
    for f in ["ctl.json", "ctl_volume_stats.json", "expA_baseline.json",
              "expB_hallucination.json"]:
        if not os.path.exists(os.path.join(cl.RESULTS, f)):
            missing.append("results/" + f)
    if rungs_expected:
        for f in ["expA_sigma.json", "expA_curve.json"]:
            if not os.path.exists(os.path.join(cl.RESULTS, f)):
                missing.append("results/" + f)
    preds = []
    for arm in ("ctl_native", "ctl_scalefault", "ctl_half"):
        preds += [f"{arm}.tif", f"{arm}_reverse.tif"]
    preds += ["expA_base.tif", "expA_base_reverse.tif",
              "expA_base_fit17.tif", "expA_base_fit17_reverse.tif"]
    for w in ("win2", "win3"):
        for mode in cl.DEPTH_MODES:
            preds += [f"expB_{w}_{mode}.tif", f"expB_{w}_{mode}_reverse.tif"]
    if rungs_expected:
        preds += ["expA_bit4.tif"]
        preds += [f"expA_p{Pu}.tif" for Pu in cl.PITCHES
                  if abs(Pu - cl.MODEL_PITCH) >= 1e-9]
        preds += [f"expA_n{k}.tif" for k in cl.NOISE_KS]
        preds += [f"expA_blur{sg}.tif" for sg in cl.BLUR_SIGMAS]
    for f in preds:
        if not os.path.exists(os.path.join(cl.PREDS, f)):
            missing.append("preds/" + f)
    stages = ["provision", "ckpt", "ctl_fetch", "ctl_build", "ctl_infer",
              "ctl_score", "p2a_fetch", "p2a_build_win1", "expa_baseline",
              "p2a_build_rest", "expb_infer", "expb_score"]
    if rungs_expected:
        stages.append("expa_rungs")
    for st in stages:
        if not os.path.exists(os.path.join(VAR, "done_" + st)):
            missing.append("stage:" + st)
    if missing:
        cl.say("FINALIZE REFUSED -- missing: " + ", ".join(missing))
        sys.exit(3)
    ctl = json.load(open(os.path.join(cl.RESULTS, "ctl.json")))
    a = json.load(open(os.path.join(cl.RESULTS, "expA_baseline.json")))
    b = json.load(open(os.path.join(cl.RESULTS, "expB_hallucination.json")))
    curve = None
    if rungs_expected:
        curve = json.load(open(os.path.join(cl.RESULTS, "expA_curve.json")))
    agg = dict(
        run="pod_p2a_v3 (C3: 500p2a anchor at the corrected 2.215um pitch + CTL positive controls)",
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        prereg=json.load(open(os.path.join(cl.OUT, "prereg.json"))),
        ctl=ctl, expA_baseline=a, expA_curve=curve,
        curve_gate_failed=gate_failed, expB=b)
    for p in (os.path.join(cl.RESULTS, "results.json"),
              os.path.join(cl.OUT, "results.json")):
        json.dump(agg, open(p, "w"), indent=1)
    na = ctl["arms"]["ctl_native"]; sf = ctl["scale_fault"]
    cl.say(f"SUMMARY ctl: native fwd={na['forward']:.4f} rev={na['reverse']:.4f}; "
           f"scale-fault best={sf['best']:.4f} -> {sf['verdict']}; "
           f"half best={ctl['half_scale']['best']:.4f}")
    m = a["modes"]
    cl.say(f"SUMMARY anchor (win1 @ 2.215um): iso fwd={m['iso']['auc_forward']:.4f} "
           f"rev={m['iso']['auc_reverse']:.4f} | fit17 fwd={m['fit17']['auc_forward']:.4f} "
           f"rev={m['fit17']['auc_reverse']:.4f} | CORRECTED ANCHOR = "
           f"{a['corrected_anchor_auc']:.4f} ({a['direction']}) | "
           f"Aug-25 void 0.5382/0.5055")
    cl.say(f"SUMMARY reading: {a['prestated_reading']}")
    if curve:
        cl.say(f"SUMMARY curve: detectability limit {curve['detectability_limit_um']} um; "
               f"sigma_plate={curve['sigma_plate']['sigma_plate']:.2f} DN")
    else:
        cl.say("SUMMARY curve: not run (gate not met or RUN_RUNGS=0)")
    for name, r in b["inputs"].items():
        aud = r["cells"][r["audited_cell"]]
        cl.say(f"SUMMARY expB {name}: {r['verdict']} real="
               f"{aud['auc_real']:.4f} null_med={aud['null_median']:.4f} "
               f"gap={aud['gap']:.4f}")
    cl.say(f"SUMMARY expB overall: {b['overall']}")

if __name__ == "__main__":
    main()
PY_FINAL

}
write_scripts
say "scripts written to $SCRIPTS (analysis code locked before provisioning)"

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

# ============================================================================
# STAGE provision -- villa + uv env (the proven zroll/runbook recipe, verbatim
# where it matters). The status server has been up since second ~2.
# ============================================================================
export PATH="$HOME/.local/bin:$PATH"
if stage_done provision; then
  say "=== STAGE provision already done, skipping ==="
else
  stage_open provision
  cd /workspace
  apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1 || true
  command -v git >/dev/null || die "git unavailable after apt"
  # Pinned snapshot repo (27 MB) instead of the full villa monorepo: the previous
  # pod spent 2 h failing to fetch villa over a flaky link. timeout caps each try.
  if [ ! -d villa ]; then
    retry 3 timeout 300 git clone --depth 1 https://github.com/flummoxjr/villa-pin-37e300d3.git villa >> "$OUT/provision.log" 2>&1 || die "villa-pin clone failed - see provision.log"
  fi
  VSHA=$(cd villa && git rev-parse --short HEAD)
  say "provision: villa @ $VSHA"
  cd /workspace/villa/vesuvius
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  say "provision: uv sync starting (full log at /provision.log on :8000)"
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

# ============================================================================
# STAGE ckpt -- ink_9um seed42/step-075000 (public; expected 138360039 bytes).
# ============================================================================
CKPT=/workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth
if stage_done ckpt; then
  say "=== STAGE ckpt already done, skipping ==="
else
  stage_open ckpt
  if [ ! -s "$CKPT" ]; then
    fetch_ckpt() {
      pyrun - <<'PY_CKPT'
from huggingface_hub import hf_hub_download
p = hf_hub_download("scrollprize/ink_9um", "hybrid_3d2d-seed42/step-075000.pth",
                    local_dir="/workspace/ckpts/ink_9um")
print("CKPT", p)
PY_CKPT
    }
    retry 4 fetch_ckpt || die "checkpoint download failed"
  fi
  SZ=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
  [ "$SZ" = 138360039 ] || die "checkpoint size $SZ != expected 138360039"
  pyrun - <<'PY_CFG' || die "checkpoint config check failed"
import torch
from vesuvius.ink_detection.config import InkConfig
payload = torch.load("/workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth",
                     map_location="cpu", weights_only=False)
c = InkConfig.from_mapping(payload["config"])
mode, crop, norm = c.data.mode, tuple(c.model.crop_size), c.data.normalization.mode
print(f"CKPT_CFG mode={mode} crop_size={crop} norm={norm}")
assert mode == "flat" and crop == (17, 128, 128) and norm == "robust_mad", \
    (mode, crop, norm)
PY_CFG
  say "ckpt verified: size exact, mode=flat crop=(17,128,128) norm=robust_mad"
  stage_close ckpt
fi

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
