#!/bin/bash
# =============================================================================
# pod_curve_audit.sh -- BOTH heavy experiments on one RunPod GPU pod.   v1
#
#   EXPERIMENT A  DETECTABILITY CURVE   Frag1 (PHercParis2Fr47, optical GT)
#   EXPERIMENT B  HALLUCINATION AUDIT   three 500p2a windows + Frag1
#
# Model under test: scrollprize/ink_9um  hybrid_3d2d-seed42/step-075000.pth
# (flat mode, patch [17,128,128], robust_mad 1/99 -- the tifxyz_robust path in
# villa infer.py casts ANY input dtype to float32, verified against source, so
# uint16 inputs are legal; a live probe still checks it before relying on it.)
#
# -----------------------------------------------------------------------------
# OBSERVABILITY IS THE FIRST REQUIREMENT (the zroll pod burned $2.30 unobserved
# because its status server started only AFTER provisioning; here the server
# starts before ANYTHING else).
#
#   THE FIRST THREE STATUS LINES, AND WHEN THEY APPEAR (measured, not planned;
#   t0 = the moment this script starts -- which is container start when used as
#   the pod's start command):
#     L1  t ~= 0-1 s   "<utc> BOOT pod_curve_audit v1 pid=... host=... root=..."
#                      written by the 4th statement of this script, before any
#                      other action of any kind.
#     L2  t ~= 1-3 s   "<utc> SERVE http.server pid=... 0.0.0.0:8000 dir=...
#                      external=https://<POD_ID>-8000.proxy.runpod.net/status.txt
#                      local_probe=200"  -- written only after a successful
#                      localhost curl of status.txt through the server.
#     L3  t ~= 2-4 s   "<utc> PREREG locked prereg.json sha256=... -- decision
#                      rules recorded before any provisioning, download, or data"
#   Provisioning opens at L5 (t ~= 3-5 s). A HEARTBEAT line with a UTC
#   timestamp, current stage, uptime and free disk follows every 60 s for the
#   entire life of the pod (first beat at t ~= 60 s). The heartbeat is also a
#   watchdog: if the http server has died it restarts it and says so.
#   Every stage opens with "=== STAGE <name> OPEN ===" and closes with
#   "=== STAGE <name> DONE (<sec>s) ===".  "ALL DONE" is printed by the
#   finalize stage ONLY after it has re-verified every stage marker, every
#   expected prediction file, and every results JSON -- any failure anywhere
#   exits through die() into a FAILED linger loop instead, so ALL DONE is
#   unreachable on a failed run.
#
# -----------------------------------------------------------------------------
# LAUNCH (from the laptop; pod needs: 1x RTX 5090 (or >=16GB CUDA), a
# runpod/pytorch image (python3+curl present), volume >= 100 GB at /workspace,
# and HTTP PORT 8000 EXPOSED in the pod config -- that last one is what makes
# the status page reachable at https://<POD_ID>-8000.proxy.runpod.net/status.txt):
#
#   scp pod_curve_audit.sh root@POD:/workspace/pod_curve_audit.sh
#   ssh root@POD 'sed -i "s/\r$//" /workspace/pod_curve_audit.sh;
#                 nohup bash /workspace/pod_curve_audit.sh </dev/null \
#                   >/workspace/curve_nohup.out 2>&1 & disown; echo KICKED'
#   # watch (either):
#   curl -s https://<POD_ID>-8000.proxy.runpod.net/status.txt | tail -30
#   ssh root@POD 'tail -f /workspace/curve/out/status.txt'
#   # fetch results when status shows ALL DONE:
#   scp -r root@POD:/workspace/curve/out ./curve_out
#   # THEN TERMINATE THE POD. It lingers (serving results) and bills until you do.
#
# Env knobs: PORT=8000  BATCH=16  WORKERS=8  FORCE=1 (rebuild everything)
#   NATIVE_WINDOWS=1 (0 skips the native-4.32um sensitivity passes, ~-$0.30)
#   AUTOSTOP=0 (1 = 'runpodctl stop pod' 30 min after ALL DONE, if available)
#   DRY=1 (machinery test only: no network/GPU; used for local validation)
# This script never calls the RunPod API except the optional AUTOSTOP above.
#
# COST (estimated before launch; a 5090 was $0.69-0.89/hr in our survey):
#   provision 10-15m; frag1 fetch 6.7GB 12-25m; ckpt 1m; buildA 8-12m;
#   baseline infer+gate 5-8m; rung build+infer+score 25-40m; p2a fetch 4-8m;
#   window build 6-10m; window infer 25-40m (both scales) / ~12m (9.36 only);
#   scoring 6-12m; total ~1.8-2.9h  => ~$1.40-$2.60.
#   Early aborts: ckpt/config or data-gate failure ~25-45m (~$0.35-0.60);
#   KILLED_BASELINE ~60-80m (~$0.75-1.10) -- that abort IS a result.
#
# =============================================================================
# PRE-REGISTRATION (locked here and in out/prereg.json BEFORE any data exists;
# the scoring code reads these same constants from the environment block below)
#
# EXPERIMENT A -- detectability curve on Frag1.
#   Scoring: exact tie-corrected pixel rank-AUC on the NATIVE 3.24um grid
#   (8181x6330); predictions are bilinearly upsampled to native; maps are
#   quantized to 16-bit bins (x257) so the AUC is closed-form on histograms.
#   positives = inklabels & mask (expected exactly 5,339,364 px),
#   negatives = mask & ~inklabels (expected exactly 23,803,476 px).  blank is
#   ALWAYS (mask==1 AND label==0).
#   BASELINE GATE: baseline = anti-aliased 3.24->9.36um resample (the model's
#   native pitch), inferred forward AND reverse. AUC_base = max(fwd,rev).
#   If AUC_base < 0.85 the whole run aborts as KILLED_BASELINE (both
#   experiments; ALL DONE unreachable). Justification: 0.85 leaves 0.35 of
#   dynamic range above chance -- the minimum that can order 13 degradation
#   rungs; an instrument that cannot clearly read the undegraded plate cannot
#   support any claim about degraded ones. The winning direction (tie->forward)
#   is used for every rung.
#   RUNG RULE: retained = (AUC_rung - 0.5) / (AUC_base - 0.5).
#   DETECTABLE <=> AUC_rung >= 0.75 AND retained >= 0.50. The detectability
#   limit is the coarsest pitch rung still DETECTABLE. All rungs report AUC,
#   retained, DETECTABLE; nothing else is verdict-bearing.
#   RUNGS (13): pitch 3.24->{4.0,5.0,6.5,8.0,9.36,12.0}um simulated by
#   anti-aliased resample to pitch P (Gaussian prefilter sigma=(f-1)/2 source
#   px, linear interp, area-aligned grid_mode) THEN regridded P->9.36um so the
#   model always operates at its trained pitch: the curve isolates ACQUISITION
#   information content, not model scale sensitivity. The 9.36 rung is by
#   construction identical to baseline and is reported as reused. | noise:
#   Gaussian sigma_add = k*sigma_plate, k in {1,2,4,8}, seed 20260824, added on
#   the 9.36 grid, clipped to dtype; sigma_plate = 1.4826*median|(I_z -
#   I_{z+1})/sqrt(2)| over in-mask voxels of the model's central 17 slices of
#   baseline -- the plate's OWN in-mask noise, so the four rungs are SNR x
#   {0.71,0.45,0.24,0.12} of the plate as scanned. | bit depth: uint16 ->
#   uint8 = rint(v/257), the full-range released-volume windowing (env
#   BIT_LO/BIT_HI override; if the uint16 probe fails and baseline falls back
#   to uint8, this rung is flagged DEGENERATE and reuses baseline). | blur:
#   3D isotropic Gaussian sigma in {1.0, 2.0} px at 9.36um (PSF-like; FWHM
#   22/44um).
#
# EXPERIMENT B -- hallucination audit (P1 passed; licensed).
#   Inputs: 500p2a win1/win2/win3 (coords embedded below) and Frag1.
#   PRIMARY scale = resampled to 9.36um (the model's operating pitch; verdicts
#   are issued only there). Native-4.32um window passes are sensitivity-only,
#   never verdict-bearing (at native pitch the model's 17 slices span 73um vs
#   159um trained -- off-domain, reported as such). Directions: forward AND
#   reverse are run and reported; the audited cell per input is the direction
#   with the higher REAL AUC (the map a user would actually keep).
#   REAL AUC: positives = label & mask, negatives = mask & ~label. win3:
#   positives are ink AND mask ONLY -- its 621,303 out-of-mask ink px (asserted
#   exactly) are excluded from BOTH classes, as everywhere.
#   MATCHED NULL: 40 rigid translations T_i of the (label & mask) shape image,
#   seed 20260824, per-input; |shift|_inf in [4.4mm, 0.40*min(H,W) px];
#   accepted only if >= 50,000 pseudo-positive px survive; scored ENTIRELY
#   inside annotated blank: pos_i = T_i(shapes) & mask & ~label,
#   neg_i = mask & ~label & ~T_i(shapes). The same 40 offsets are reused across
#   directions and scales of one input, drawn once on the native grid.
#   SHAPE-CONFOUNDED DECLARATION (the pre-registered threshold): the model is
#   declared shape-confounded on an input if, in the audited cell, EITHER
#   (i) the median of the 40 translated-label AUCs >= 0.60, OR (ii) at least
#   20% of the 40 (>= 8) are individually >= 0.60. Justification: a rigid
#   translation carries ZERO information about true ink locations, and the min
#   shift of 4.4mm (>= ~1.5-2 letter heights of 2-3mm Herculaneum script)
#   guarantees the translated strokes do not overlap their sources, so ANY
#   systematic discrimination of translated letterforms on ink-free parchment
#   measures shape/texture confound, not detection; 0.60 is 0.10 above chance
#   -- beyond any plausible translation-sampling spread (the empirical spread
#   of the 40 is reported beside the verdict so this premise is checkable),
#   yet far below genuine-detection AUCs, so real signal cannot trip it.
#   Clause (ii) exists because text is quasi-PERIODIC (line spacing): a
#   confound that fires on text-row structure re-aligns only when the shift is
#   near a multiple of the line spacing, inflating the null MAX while leaving
#   the median at or below 0.5 -- demonstrated numerically in pre-launch
#   validation (synthetic row-confound: null median 0.44, null max 0.96, ~20%
#   of draws aligned). A median-only rule provably misses that presentation;
#   under a clean null, 8 of 40 draws exceeding 0.60 has ~zero probability.
#   GENUINE-GAP: a cell shows genuine ink tracking iff AUC_real > max of the 40
#   translated AUCs (empirical rank p = 1/41 ~= 0.024 -- every positive beats a
#   matched null) AND gap = AUC_real - median(translated) >= 0.15. Cells that
#   are neither GENUINE nor SHAPE-CONFOUNDED are INDETERMINATE. The gap is the
#   reported headline number for every cell.
#   DATA GATES (exact, else FATAL): window label counts equal the embedded
#   values from p2a_windows.json; window volume subsample stats within +-0.5 DN;
#   Frag1 label PNGs byte-identical (sha256) to the laptop copies; class counts
#   equal the embedded values. If a premise fails verification the run reports
#   KILLED with the evidence and stops.
# =============================================================================

set -Eeuo pipefail

# ---------------------------------------------------------------- config ----
ROOT=${ROOT:-/workspace/curve}
PORT=${PORT:-8000}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}
FORCE=${FORCE:-0}
DRY=${DRY:-0}
DRY_FAIL_STAGE=${DRY_FAIL_STAGE:-}
LINGER_EXIT=${LINGER_EXIT:-0}
NATIVE_WINDOWS=${NATIVE_WINDOWS:-1}
AUTOSTOP=${AUTOSTOP:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
SEED=20260824

OUT=$ROOT/out;  VAR=$ROOT/var;  DATA=$ROOT/data;  PREDS=$ROOT/preds
SCRIPTS=$ROOT/scripts;  RESULTS=$OUT/results;  STATUS=$OUT/status.txt
export ROOT OUT VAR DATA PREDS SCRIPTS RESULTS STATUS SEED BATCH WORKERS
export NATIVE_WINDOWS

# =================================================================== L1 ======
# The very first actions: make the served dir and write the BOOT line.
mkdir -p "$OUT" "$VAR" "$DATA" "$PREDS" "$SCRIPTS" "$RESULTS" "$OUT/previews" "$DATA/tmp"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(now) $*" >> "$STATUS"; echo "$(now) $*"; }
say "BOOT pod_curve_audit v1 pid=$$ host=${HOSTNAME:-unknown} root=$ROOT -- status live; server next"

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
  "seed": 20260824,
  "expA": {
    "scoring": "exact tie-corrected pixel rank-AUC on the native 3.24um grid; predictions bilinearly upsampled to native; 16-bit histogram bins (map*257)",
    "positives": "inklabels & mask (expected 5339364 px)",
    "negatives": "mask & ~inklabels (expected 23803476 px)",
    "baseline_gate": "AUC_base = max(forward, reverse) on the undegraded 3.24->9.36um resample; if AUC_base < 0.85 the ENTIRE run aborts as KILLED_BASELINE",
    "baseline_gate_justification": "0.85 keeps 0.35 of range above chance, the minimum that can order 13 rungs; an instrument that cannot read the undegraded plate supports no degradation claim",
    "rung_direction": "the baseline argmax direction (tie -> forward) for every rung",
    "detectable_rule": "DETECTABLE <=> AUC_rung >= 0.75 AND (AUC_rung-0.5)/(AUC_base-0.5) >= 0.50; detectability limit = coarsest DETECTABLE pitch",
    "pitch_rungs_um": [4.0, 5.0, 6.5, 8.0, 9.36, 12.0],
    "pitch_harness": "anti-aliased resample 3.24->P (Gaussian sigma=(f-1)/2 source px, linear, grid_mode) then regrid P->9.36 so the model always runs at its trained pitch; measures acquisition information, not model scale sensitivity; the 9.36 rung is identical to baseline by construction and reported as reused",
    "noise_rungs": "sigma_add = k*sigma_plate for k in {1,2,4,8} at the 9.36 grid, clipped to dtype; sigma_plate = 1.4826*median|(I_z - I_{z+1})/sqrt(2)| over in-mask voxels of the central 17 baseline slices; SNR falls to {0.71,0.45,0.24,0.12} of the plate as scanned",
    "bitdepth_rung": "uint8 = rint(uint16/257), full-range released-volume windowing; DEGENERATE-flagged if baseline had to fall back to uint8",
    "blur_rungs_sigma_px": [1.0, 2.0],
    "blur_note": "3D isotropic Gaussian at 9.36um; FWHM 22 and 44 um (PSF-like)"
  },
  "expB": {
    "inputs": ["win1", "win2", "win3", "frag1"],
    "primary_scale": "resampled to 9.36um; native-4.32um passes are sensitivity only, never verdict-bearing",
    "audited_cell": "per input, the direction (fwd/rev) with the higher REAL AUC at the primary scale; both directions reported",
    "real_auc": "pos = label & mask, neg = mask & ~label; win3 positives are ink AND mask only, its 621303 out-of-mask ink px excluded from both classes",
    "null": "40 rigid translations of the (label & mask) shape image, seed 20260824, drawn once per input on the native grid; |shift|_inf in [4.4mm, 0.40*min(H,W)]; accepted iff >= 50000 pseudo-positive px; pos_i = T_i(shapes) & mask & ~label; neg_i = mask & ~label & ~T_i(shapes) -- the whole comparison lives inside (mask==1 AND label==0)",
    "shape_confounded_threshold": "in the audited cell: median of the 40 translated-label AUCs >= 0.60, OR >= 20% of the 40 individually >= 0.60",
    "shape_confounded_justification": "a rigid translation carries zero information about true ink; min shift 4.4mm >= ~1.5-2 letter heights so translated strokes cannot overlap their sources; 0.60 is 0.10 above chance, beyond any plausible translation-sampling spread (empirical spread reported beside the verdict), yet far below genuine detection, so real signal cannot trip it. The 20%-of-draws OR-clause catches PERIODIC confounds: text-row structure re-aligns only at shifts near multiples of the line spacing, inflating the null max while leaving the median at or below 0.5 (demonstrated in pre-launch synthetic validation: row-confound null median 0.44, max 0.96); under a clean null, 8 of 40 draws above 0.60 has ~zero probability",
    "genuine_rule": "GENUINE <=> not SHAPE-CONFOUNDED, AND AUC_real > max of the 40 translated AUCs (rank p = 1/41 ~= 0.024; every positive beats a matched null), AND gap = AUC_real - median(translated) >= 0.15; otherwise INDETERMINATE",
    "headline": "gap = AUC_real - median(translated AUC), reported for every cell"
  },
  "data_gates": "exact label-count and sha256 asserts (embedded expectations); any mismatch -> KILLED with evidence"
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
# Every number below was measured on 2026-08-24 against the live servers and
# the laptop ground-truth copies BEFORE this script was written. Parse, then
# assert -- never assume.
FRAG1_BASE = ("https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/"
              "working/54keV_exposed_surface")
FRAG1_H, FRAG1_W, FRAG1_NZ = 8181, 6330, 65
FRAG1_SBC = 103571460                      # StripByteCounts = H*W*2, every layer
FRAG1_LABEL_SHA = "bc36bd54e84b423faa44d3b62ca0d3a3c28403cb714b154acb382f94241d39b6"
FRAG1_MASK_SHA = "1fdc13f7dc149dc4e00272af597ba236b650e54bb2dca38865f211207cee069f"
FRAG1_INK_PX = 5339364
FRAG1_MASK_PX = 29142840
FRAG1_BLANK_PX = 23803476
FRAG1_INK_OUTSIDE_MASK = 0
NATIVE_PITCH = 3.24
MODEL_PITCH = 9.36

BUCKET = ("https://huggingface.co/buckets/scrollprize/datasets/resolve/"
          "ink/unused/500p2a")
P2A_SHAPE = (65, 26239, 16182)             # .zarray, verified live
P2A_CHUNK = (65, 128, 128)
P2A_PITCH = 4.32
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

# ------------------------------------------------------------------ prereg --
GATE_BASELINE_AUC = 0.85
DETECT_AUC_MIN = 0.75
DETECT_RETAIN_MIN = 0.50
PITCHES = [4.0, 5.0, 6.5, 8.0, 9.36, 12.0]
NOISE_KS = [1, 2, 4, 8]
BLUR_SIGMAS = [1.0, 2.0]
N_TRANSLATIONS = 40
MIN_SHIFT_MM = 4.4
MAX_SHIFT_FRAC = 0.40
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

def to_u8_released(v16):
    """uint16 -> uint8 with the full-range released-volume windowing."""
    lo = float(os.environ.get("BIT_LO", "0"))
    hi = float(os.environ.get("BIT_HI", "65535"))
    x = (np.asarray(v16, dtype=np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)

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
PY_LIB

cat > "$SCRIPTS/verify_frag1.py" <<'PY_VERIFY'
"""Parse every Frag1 layer's OWN IFD (StripOffsets is PER LAYER: 260 for 00-09,
262 for 10-64 -- but we PARSE each file, never assume), cross-check tifffile,
fingerprint the labels, and write data/frag1_manifest.json."""
import json, os, struct, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

TIFDIR = os.path.join(cl.DATA, "frag1")

def parse_ifd(path):
    """Hand-rolled TIFF IFD parse: both byte orders, front or end-of-file IFD."""
    with open(path, "rb") as f:
        head = f.read(8)
        if head[:2] == b"II":
            end = "<"
        elif head[:2] == b"MM":
            end = ">"
        else:
            raise AssertionError(f"{path}: not a TIFF")
        ifd_off = struct.unpack(end + "I", head[4:8])[0]
        f.seek(ifd_off)
        n = struct.unpack(end + "H", f.read(2))[0]
        blob = f.read(n * 12)
        tags = {}
        for i in range(n):
            tag, typ, cnt = struct.unpack(end + "HHI", blob[i*12:i*12+8])
            raw = blob[i*12+8:i*12+12]
            if typ == 3 and cnt == 1:
                tags[tag] = struct.unpack(end + "H", raw[:2])[0]
            elif typ == 4 and cnt == 1:
                tags[tag] = struct.unpack(end + "I", raw)[0]
        return end, ifd_off, tags

def main():
    import tifffile
    man = {}
    so_hist = {}
    for L in range(cl.FRAG1_NZ):
        p = os.path.join(TIFDIR, f"{L:02d}.tif")
        end, ifd_off, t = parse_ifd(p)
        W, H = t.get(256), t.get(257)
        bits, comp = t.get(258), t.get(259)
        so, sbc = t.get(273), t.get(279)
        assert (H, W) == (cl.FRAG1_H, cl.FRAG1_W), f"L{L:02d} dims {H}x{W}"
        assert bits == 16 and comp == 1, f"L{L:02d} bits={bits} comp={comp}"
        assert sbc == cl.FRAG1_SBC, f"L{L:02d} StripByteCounts={sbc}"
        size = os.path.getsize(p)
        assert size == so + sbc, f"L{L:02d} size {size} != SO {so} + SBC {sbc}"
        with tifffile.TiffFile(p) as tf:
            page = tf.pages[0]
            assert page.dataoffsets[0] == so, \
                f"L{L:02d} tifffile offset {page.dataoffsets[0]} != parsed {so}"
            assert page.shape == (cl.FRAG1_H, cl.FRAG1_W)
            arr_rows = page.asarray()[[0, cl.FRAG1_H - 1], :]
        mm = np.memmap(p, dtype=(end + "u2"), mode="r", offset=so,
                       shape=(cl.FRAG1_H, cl.FRAG1_W))
        assert np.array_equal(np.asarray(mm[[0, cl.FRAG1_H - 1], :]), arr_rows), \
            f"L{L:02d} memmap rows != tifffile rows"
        del mm
        man[f"{L:02d}"] = dict(so=int(so), endian=end, size=int(size))
        so_hist[so] = so_hist.get(so, 0) + 1
        if L % 10 == 0:
            cl.say(f"FRAG1_VERIFY layer {L:02d} SO={so} endian="
                   f"{'MM' if end == '>' else 'II'} ok")
    for name, sha, exp in (("inklabels.png", cl.FRAG1_LABEL_SHA, None),
                           ("mask.png", cl.FRAG1_MASK_SHA, None)):
        got = cl.sha256_file(os.path.join(TIFDIR, name))
        assert got == sha, f"{name} sha256 {got} != laptop ground truth {sha}"
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    ink = np.array(Image.open(os.path.join(TIFDIR, "inklabels.png"))) > 0
    msk = np.array(Image.open(os.path.join(TIFDIR, "mask.png"))) > 0
    counts = dict(ink=int(ink.sum()), mask=int(msk.sum()),
                  ink_and_mask=int((ink & msk).sum()),
                  blank=int((msk & ~ink).sum()),
                  ink_outside_mask=int((ink & ~msk).sum()))
    assert counts["ink_and_mask"] == cl.FRAG1_INK_PX, counts
    assert counts["mask"] == cl.FRAG1_MASK_PX, counts
    assert counts["blank"] == cl.FRAG1_BLANK_PX, counts
    assert counts["ink_outside_mask"] == cl.FRAG1_INK_OUTSIDE_MASK, counts
    with open(os.path.join(cl.DATA, "frag1_manifest.json"), "w") as f:
        json.dump(dict(layers=man, labels=counts), f, indent=1)
    cl.say(f"FRAG1_VERIFY all 65 layers parsed per-file; observed StripOffsets "
           f"histogram {so_hist} (expected {{260:10, 262:55}} -- parsed, not "
           f"assumed); labels sha256 + class counts EXACT match")

if __name__ == "__main__":
    main()
PY_VERIFY

cat > "$SCRIPTS/build_frag1.py" <<'PY_BUILD'
"""Build Frag1 zarrs. argv[1]: baseline | planb | rungs."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

TIFDIR = os.path.join(cl.DATA, "frag1")
MAN = os.path.join(cl.DATA, "frag1_manifest.json")
BASE_PTR = os.path.join(os.environ["VAR"], "baseline_path.txt")
NZ9 = cl.rint_shape(cl.FRAG1_NZ, cl.NATIVE_PITCH, cl.MODEL_PITCH)   # 23 (22.5 half-up)
H9 = cl.rint_shape(cl.FRAG1_H, cl.NATIVE_PITCH, cl.MODEL_PITCH)     # 2832
W9 = cl.rint_shape(cl.FRAG1_W, cl.NATIVE_PITCH, cl.MODEL_PITCH)     # 2191

def layer_getter():
    man = json.load(open(MAN))["layers"]
    def get(z):
        m = man[f"{z:02d}"]
        return np.memmap(os.path.join(TIFDIR, f"{z:02d}.tif"),
                         dtype=(m["endian"] + "u2"), mode="r", offset=m["so"],
                         shape=(cl.FRAG1_H, cl.FRAG1_W))
    return get

def active_baseline():
    return open(BASE_PTR).read().strip()

def build_baseline():
    path = os.path.join(cl.DATA, "frag1_base_u16.zarr")
    if not os.path.exists(path):
        cl.say(f"BUILD baseline: 3.24 -> 9.36um, target ({NZ9},{H9},{W9}) uint16")
        tmp = os.path.join(cl.DATA, "tmp", "base.f32")
        mm = cl.resample_stack(layer_getter(), cl.FRAG1_NZ, cl.FRAG1_H,
                               cl.FRAG1_W, (NZ9, H9, W9), tmp, tag="baseline")
        v16 = np.clip(np.rint(np.asarray(mm)), 0, 65535).astype(np.uint16)
        del mm; os.remove(tmp)
        cl.write_group_zarr(path, v16)
        cl.save_preview(v16[NZ9 // 2], os.path.join(cl.OUT, "previews",
                                                    "frag1_base_midslice.png"))
    with open(BASE_PTR, "w") as f:
        f.write(path)
    # tiny probe zarr for the uint16-acceptance check
    probe = os.path.join(cl.DATA, "probe_u16.zarr")
    if not os.path.exists(probe):
        v = np.asarray(cl.read_zarr0(path)[:, :256, :256])
        cl.write_group_zarr(probe, v)
    # downsampled labels for sigma_plate region
    lp = os.path.join(cl.DATA, "frag1_mask936.npy")
    if not os.path.exists(lp):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        msk = (np.array(Image.open(os.path.join(TIFDIR, "mask.png"))) > 0)
        m9 = cl.zoom_to(msk.astype(np.float32), (H9, W9), prefilter=False) >= 0.5
        np.save(lp, m9)
    cl.say(f"BUILD baseline done: {path}")

def build_planb():
    """uint16 probe failed -> fall back to uint8 baseline; bitdepth degenerate."""
    src = os.path.join(cl.DATA, "frag1_base_u16.zarr")
    path = os.path.join(cl.DATA, "frag1_base_u8.zarr")
    if not os.path.exists(path):
        v = cl.to_u8_released(np.asarray(cl.read_zarr0(src)))
        cl.write_group_zarr(path, v)
    with open(BASE_PTR, "w") as f:
        f.write(path)
    open(os.path.join(os.environ["VAR"], "bitdepth_degenerate"), "w").write("1")
    cl.say("BUILD PLAN B: baseline swapped to uint8 (released windowing); "
           "bitdepth rung flagged DEGENERATE")

def build_rungs():
    base_path = active_baseline()
    base = np.asarray(cl.read_zarr0(base_path))
    dt = base.dtype
    dmax = 65535 if dt == np.uint16 else 255
    m9 = np.load(os.path.join(cl.DATA, "frag1_mask936.npy"))
    z0 = NZ9 // 2 - 17 // 2
    slab = base[z0:z0 + 17].astype(np.float32)
    d = (slab[:-1] - slab[1:]) / np.sqrt(2.0)
    dm = np.broadcast_to(m9, d.shape)
    sigma_plate = 1.4826 * float(np.median(np.abs(d[dm])))
    json.dump(dict(sigma_plate=sigma_plate, dtype=str(dt), z0=int(z0)),
              open(os.path.join(cl.RESULTS, "expA_sigma.json"), "w"), indent=1)
    cl.say(f"RUNGS sigma_plate={sigma_plate:.3f} DN (dtype={dt}, in-mask, "
           f"central 17 slices, adjacent-slice MAD estimator)")
    # noise rungs (fresh per-k RNG so a resumed run draws identical noise)
    for k in cl.NOISE_KS:
        p = os.path.join(cl.DATA, f"rung_n{k}.zarr")
        if os.path.exists(p):
            continue
        rng = np.random.default_rng([cl.SEED, k])
        noisy = base.astype(np.float32) + rng.normal(
            0.0, k * sigma_plate, size=base.shape).astype(np.float32)
        cl.write_group_zarr(p, np.clip(np.rint(noisy), 0, dmax).astype(dt))
        cl.say(f"RUNGS built noise k={k}")
    # bit-depth rung
    if not os.path.exists(os.path.join(os.environ["VAR"], "bitdepth_degenerate")):
        p = os.path.join(cl.DATA, "rung_bit8.zarr")
        if not os.path.exists(p):
            cl.write_group_zarr(p, cl.to_u8_released(base))
            cl.say("RUNGS built bit8 (uint16 -> uint8 released windowing)")
    # blur rungs
    from scipy import ndimage
    for s in cl.BLUR_SIGMAS:
        p = os.path.join(cl.DATA, f"rung_blur{s}.zarr")
        if os.path.exists(p):
            continue
        b = ndimage.gaussian_filter(base.astype(np.float32), sigma=s,
                                    mode="nearest")
        cl.write_group_zarr(p, np.clip(np.rint(b), 0, dmax).astype(dt))
        cl.say(f"RUNGS built blur sigma={s}")
    # pitch rungs (9.36 reuses baseline; built via two-stage resample otherwise)
    get = layer_getter()
    for P in cl.PITCHES:
        if abs(P - cl.MODEL_PITCH) < 1e-9:
            continue
        p = os.path.join(cl.DATA, f"rung_p{P}.zarr")
        if os.path.exists(p):
            continue
        nzP = cl.rint_shape(cl.FRAG1_NZ, cl.NATIVE_PITCH, P)
        HP = cl.rint_shape(cl.FRAG1_H, cl.NATIVE_PITCH, P)
        WP = cl.rint_shape(cl.FRAG1_W, cl.NATIVE_PITCH, P)
        cl.say(f"RUNGS pitch {P}um stage A -> ({nzP},{HP},{WP})")
        tmpA = os.path.join(cl.DATA, "tmp", f"p{P}_A.f32")
        mmA = cl.resample_stack(get, cl.FRAG1_NZ, cl.FRAG1_H, cl.FRAG1_W,
                                (nzP, HP, WP), tmpA, tag=f"p{P}A")
        tmpB = os.path.join(cl.DATA, "tmp", f"p{P}_B.f32")
        mmB = cl.resample_stack(lambda z: np.asarray(mmA[z]), nzP, HP, WP,
                                (NZ9, H9, W9), tmpB, tag=f"p{P}B")
        del mmA; os.remove(tmpA)
        v = np.clip(np.rint(np.asarray(mmB)), 0, dmax).astype(dt)
        del mmB; os.remove(tmpB)
        cl.write_group_zarr(p, v)
        cl.say(f"RUNGS built pitch {P}um")

if __name__ == "__main__":
    {"baseline": build_baseline, "planb": build_planb,
     "rungs": build_rungs}[sys.argv[1]]()
PY_BUILD

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
"""Assemble the three windows from fetched chunks, assert every embedded
fingerprint, write native + 9.36um zarrs and the scoring label crops."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

CH = 128
CHUNKDIR = os.path.join(cl.DATA, "p2a_chunks")

def main():
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
    cl.say("EXPB_BUILD raster label counts EXACT "
           f"(ink_outside_mask={r['ink_outside_mask']})")
    S = 4438
    NZ, S9 = 65, cl.rint_shape(4438, cl.P2A_PITCH, cl.MODEL_PITCH)   # 2048
    NZ9 = cl.rint_shape(65, cl.P2A_PITCH, cl.MODEL_PITCH)            # 30
    for name, w in cl.WINDOWS.items():
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
        cl.say(f"EXPB_BUILD {name} volume stats match recorded "
               f"(mean={stats[0]:.3f} sd={stats[1]:.3f} zf={stats[2]:.5f})")
        zn = os.path.join(cl.DATA, f"{name}_native.zarr")
        if not os.path.exists(zn):
            cl.write_group_zarr(zn, np.asarray(vol))
        z9 = os.path.join(cl.DATA, f"{name}_936.zarr")
        if not os.path.exists(z9):
            tmp = os.path.join(cl.DATA, "tmp", f"{name}_936.f32")
            mm = cl.resample_stack(lambda z: np.asarray(vol[z]), NZ, S, S,
                                   (NZ9, S9, S9), tmp, tag=f"{name}936")
            v = np.clip(np.rint(np.asarray(mm)), 0, 255).astype(np.uint8)
            del mm; os.remove(tmp)
            cl.write_group_zarr(z9, v)
        cl.save_preview(np.asarray(vol[NZ // 2]),
                        os.path.join(cl.OUT, "previews", f"{name}_midslice.png"))
        del vol; os.remove(npy)
        cl.say(f"EXPB_BUILD {name} zarrs ready (native 65x{S}x{S}, "
               f"9.36um {NZ9}x{S9}x{S9})")

if __name__ == "__main__":
    main()
PY_BUILDW

cat > "$SCRIPTS/score_expa.py" <<'PY_SCOREA'
"""Experiment A scoring. argv[1]: baseline | rungs.
baseline: AUC fwd/rev on native grid, gate at 0.85, pick direction (exit 21 on
gate failure). rungs: AUC + retained + DETECTABLE per rung, curve JSON."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR = os.environ["VAR"]

def native_labels():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    d = os.path.join(cl.DATA, "frag1")
    ink = np.array(Image.open(os.path.join(d, "inklabels.png"))) > 0
    msk = np.array(Image.open(os.path.join(d, "mask.png"))) > 0
    return ink & msk, msk & ~ink          # positives, negatives(blank)

def map_auc(tif_path, pos, neg):
    import tifffile
    pred = tifffile.imread(tif_path)
    up = cl.upsample_pred(pred, pos.shape)
    q = cl.quantize_map(up)
    return cl.hist_auc(cl.masked_hist(q, pos), cl.masked_hist(q, neg)), pred

def baseline():
    pos, neg = native_labels()
    aucs = {}
    for d, p in (("forward", os.path.join(cl.PREDS, "expA_base.tif")),
                 ("reverse", os.path.join(cl.PREDS, "expA_base_reverse.tif"))):
        aucs[d], pred = map_auc(p, pos, neg)
        cl.save_preview(pred, os.path.join(cl.OUT, "previews",
                                           f"expA_base_{d}.png"), ds=2)
        cl.say(f"EXPA_BASELINE AUC[{d}] = {aucs[d]:.4f} "
               f"(pos={int(pos.sum())} neg={int(neg.sum())})")
    best = "forward" if aucs["forward"] >= aucs["reverse"] else "reverse"
    res = dict(auc_forward=aucs["forward"], auc_reverse=aucs["reverse"],
               direction=best, gate=cl.GATE_BASELINE_AUC,
               gate_passed=bool(aucs[best] >= cl.GATE_BASELINE_AUC))
    json.dump(res, open(os.path.join(cl.RESULTS, "expA_baseline.json"), "w"),
              indent=1)
    open(os.path.join(VAR, "direction.txt"), "w").write(best)
    if not res["gate_passed"]:
        cl.say(f"EXPA_BASELINE GATE FAILED: best AUC {aucs[best]:.4f} < "
               f"{cl.GATE_BASELINE_AUC} -- pre-registered verdict "
               f"KILLED_BASELINE; the whole run aborts (evidence in "
               f"results/expA_baseline.json + previews)")
        sys.exit(21)
    cl.say(f"EXPA_BASELINE GATE PASSED: AUC_base={aucs[best]:.4f} "
           f"direction={best} -- rungs will use {best}")

def rungs():
    pos, neg = native_labels()
    base = json.load(open(os.path.join(cl.RESULTS, "expA_baseline.json")))
    direction = base["direction"]
    auc_b = base[f"auc_{direction}"]
    degen = os.path.exists(os.path.join(VAR, "bitdepth_degenerate"))
    rows = []
    def add(rung_id, family, x, tif, reused=False, degenerate=False):
        if degenerate:
            auc = auc_b
        elif reused:
            auc = auc_b
        else:
            auc, _ = map_auc(tif, pos, neg)
        retained = (auc - 0.5) / max(auc_b - 0.5, 1e-9)
        det = bool(auc >= cl.DETECT_AUC_MIN and retained >= cl.DETECT_RETAIN_MIN)
        rows.append(dict(rung=rung_id, family=family, x=x, auc=auc,
                         retained=retained, detectable=det, reused=reused,
                         degenerate=degenerate))
        cl.say(f"EXPA_SCORE {rung_id:<12} AUC={auc:.4f} retained={retained:.3f}"
               f" DETECTABLE={det}{' (reused baseline)' if reused else ''}"
               f"{' (DEGENERATE)' if degenerate else ''}")
    add("pitch_3.24", "pitch", 3.24, None, reused=True)   # baseline = native acq
    for P in cl.PITCHES:
        if abs(P - cl.MODEL_PITCH) < 1e-9:
            add(f"pitch_{P}", "pitch", P, None, reused=True)
        else:
            add(f"pitch_{P}", "pitch", P,
                os.path.join(cl.PREDS, f"expA_p{P}.tif"))
    for k in cl.NOISE_KS:
        add(f"noise_k{k}", "noise", k, os.path.join(cl.PREDS, f"expA_n{k}.tif"))
    add("bit8", "bitdepth", 8,
        None if degen else os.path.join(cl.PREDS, "expA_bit8.tif"),
        degenerate=degen)
    for s in cl.BLUR_SIGMAS:
        add(f"blur_{s}", "blur", s,
            os.path.join(cl.PREDS, f"expA_blur{s}.tif"))
    pit = [r for r in rows if r["family"] == "pitch" and r["detectable"]]
    limit = max((r["x"] for r in pit), default=None)
    sigma = json.load(open(os.path.join(cl.RESULTS, "expA_sigma.json")))
    out = dict(baseline=base, sigma_plate=sigma, rungs=rows,
               detectability_limit_um=limit,
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
inside (mask==1 AND label==0); per-cell gap + pre-registered verdicts."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

NATIVE = os.environ.get("NATIVE_WINDOWS", "1") == "1"

def load_input(name):
    if name == "frag1":
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        d = os.path.join(cl.DATA, "frag1")
        ink = np.array(Image.open(os.path.join(d, "inklabels.png"))) > 0
        msk = np.array(Image.open(os.path.join(d, "mask.png"))) > 0
        pitch = cl.NATIVE_PITCH
    else:
        ink = np.load(os.path.join(cl.DATA, f"{name}_ink.npy"))
        msk = np.load(os.path.join(cl.DATA, f"{name}_mask.npy"))
        pitch = cl.P2A_PITCH
    shapes = ink & msk           # positives; win3 out-of-mask ink excluded here
    blank = msk & ~ink           # blank = (mask==1 AND label==0), ALWAYS
    return shapes, blank, pitch

def cell_maps(name):
    cells = []
    if name == "frag1":
        cells.append(("s936", "forward", os.path.join(cl.PREDS, "expA_base.tif")))
        cells.append(("s936", "reverse",
                      os.path.join(cl.PREDS, "expA_base_reverse.tif")))
    else:
        for sc in (["s936", "native"] if NATIVE else ["s936"]):
            base = {"s936": f"expB_{name}_s936", "native": f"expB_{name}_native"}[sc]
            cells.append((sc, "forward", os.path.join(cl.PREDS, base + ".tif")))
            cells.append((sc, "reverse",
                          os.path.join(cl.PREDS, base + "_reverse.tif")))
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
    null_aucs = np.array(null_aucs)
    return auc_real, null_aucs, pred

def main():
    inputs = ["win1", "win2", "win3", "frag1"]
    results = {}
    for idx, name in enumerate(inputs):
        shapes, blank, pitch = load_input(name)
        offsets, min_px, max_px = cl.draw_translations(
            shapes, blank, pitch, seed=cl.SEED + idx)
        cl.say(f"EXPB_SCORE {name}: 40 translations drawn "
               f"(|shift| in [{min_px},{max_px}] px @ {pitch}um, "
               f">= {cl.MIN_PSEUDO_POS} pseudo-pos each)")
        cells = {}
        for scale, direction, tif in cell_maps(name):
            auc_real, nulls, pred = score_cell(tif, shapes, blank, offsets)
            gap = float(auc_real - np.median(nulls))
            frac_hi = float((nulls >= cl.CONFOUND_MEDIAN_AUC).mean())
            cells[f"{scale}_{direction}"] = dict(
                scale=scale, direction=direction,
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
            cl.say(f"EXPB_SCORE {name} {scale} {direction}: "
                   f"AUC_real={auc_real:.4f} null_med="
                   f"{np.median(nulls):.4f} null_max={nulls.max():.4f} "
                   f"GAP={gap:.4f}")
            cl.save_preview(pred, os.path.join(
                cl.OUT, "previews", f"expB_{name}_{scale}_{direction}.png"))
        prim = {k: v for k, v in cells.items() if v["scale"] == "s936"}
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
                           min_shift_mm=cl.MIN_SHIFT_MM, seed=cl.SEED),
               inputs=results, overall=overall,
               native_note=("native-4.32um cells are sensitivity only: the "
                            "model's 17 slices span 73um there vs 159um "
                            "trained -- never verdict-bearing"))
    json.dump(out, open(os.path.join(cl.RESULTS,
                                     "expB_hallucination.json"), "w"), indent=1)
    cl.say(f"EXPB_SCORE complete; overall={overall}")

if __name__ == "__main__":
    main()
PY_SCOREB

cat > "$SCRIPTS/finalize.py" <<'PY_FINAL'
"""Aggregate results, verify the full inventory, refuse to bless a partial run."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR = os.environ["VAR"]
NATIVE = os.environ.get("NATIVE_WINDOWS", "1") == "1"

def main():
    missing = []
    need_results = ["expA_baseline.json", "expA_sigma.json", "expA_curve.json",
                    "expB_hallucination.json"]
    for f in need_results:
        if not os.path.exists(os.path.join(cl.RESULTS, f)):
            missing.append("results/" + f)
    dpath = os.path.join(VAR, "direction.txt")
    direction = open(dpath).read().strip() if os.path.exists(dpath) else None
    if direction not in ("forward", "reverse"):
        missing.append("var/direction.txt")
    degen = os.path.exists(os.path.join(VAR, "bitdepth_degenerate"))
    preds = ["expA_base.tif", "expA_base_reverse.tif"]
    for P in cl.PITCHES:
        if abs(P - cl.MODEL_PITCH) >= 1e-9:
            preds.append(f"expA_p{P}.tif")
    preds += [f"expA_n{k}.tif" for k in cl.NOISE_KS]
    if not degen:
        preds.append("expA_bit8.tif")
    preds += [f"expA_blur{s}.tif" for s in cl.BLUR_SIGMAS]
    for w in ("win1", "win2", "win3"):
        preds += [f"expB_{w}_s936.tif", f"expB_{w}_s936_reverse.tif"]
        if NATIVE:
            preds += [f"expB_{w}_native.tif", f"expB_{w}_native_reverse.tif"]
    for f in preds:
        if not os.path.exists(os.path.join(cl.PREDS, f)):
            missing.append("preds/" + f)
    stages = ["provision", "ckpt", "frag1", "expa_build", "expa_baseline",
              "expa_rungs", "expb_fetch", "expb_build", "expb_infer",
              "expb_score"]
    for s in stages:
        if not os.path.exists(os.path.join(VAR, "done_" + s)):
            missing.append("stage:" + s)
    if missing:
        cl.say("FINALIZE REFUSED -- missing: " + ", ".join(missing))
        sys.exit(3)
    agg = dict(
        run="pod_curve_audit v1",
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        prereg=json.load(open(os.path.join(cl.OUT, "prereg.json"))),
        direction=direction,
        bitdepth_degenerate=degen,
        expA=json.load(open(os.path.join(cl.RESULTS, "expA_curve.json"))),
        expB=json.load(open(os.path.join(cl.RESULTS,
                                         "expB_hallucination.json"))),
    )
    json.dump(agg, open(os.path.join(cl.RESULTS, "results.json"), "w"),
              indent=1)
    a = agg["expA"]; b = agg["expB"]
    cl.say(f"SUMMARY expA: AUC_base={a['baseline'][f'auc_{direction}']:.4f} "
           f"({direction}); detectability limit {a['detectability_limit_um']} um; "
           f"sigma_plate={a['sigma_plate']['sigma_plate']:.2f} DN")
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
  for st in provision ckpt frag1 expa_build expa_baseline expa_rungs \
            expb_fetch expb_build expb_infer expb_score; do
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
  retry 2 uv pip install "torch==2.11.0" torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/provision.log" 2>&1 || die "torch pin install failed - see provision.log"
  uv pip install tqdm scipy scikit-image pandas einops opencv-python-headless \
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
# STAGE frag1 -- 65 uint16 TIFFs + labels from dl.ash2txt.org (public,
# verified live 2026-08-24). 2.5s sleeps, resumable, then per-file IFD parse.
# ============================================================================
FRAG1_BASE="https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface"
FRAG1_DIR=$DATA/frag1
if stage_done frag1; then
  say "=== STAGE frag1 already done, skipping ==="
else
  stage_open frag1
  mkdir -p "$FRAG1_DIR"
  fetch_one() { # fetch_one <url> <dest>
    local url=$1 dest=$2
    [ -s "$dest" ] && return 0
    curl -fSs --connect-timeout 30 --max-time 1800 --retry 5 --retry-delay 5 \
      -C - -o "$dest.part" "$url" && mv -f "$dest.part" "$dest"
  }
  for L in $(seq 0 64); do
    LL=$(printf "%02d" "$L")
    if [ -s "$FRAG1_DIR/$LL.tif" ]; then continue; fi
    retry 3 fetch_one "$FRAG1_BASE/surface_volume/$LL.tif" \
      "$FRAG1_DIR/$LL.tif" || die "frag1 layer $LL download failed"
    [ $(( L % 10 )) = 0 ] && say "frag1 fetch: layer $LL done"
    sleep 2.5
  done
  retry 3 fetch_one "$FRAG1_BASE/inklabels.png" "$FRAG1_DIR/inklabels.png" \
    || die "inklabels.png download failed"
  sleep 2.5
  retry 3 fetch_one "$FRAG1_BASE/mask.png" "$FRAG1_DIR/mask.png" \
    || die "mask.png download failed"
  say "frag1 fetch complete ($(du -sh "$FRAG1_DIR" | cut -f1)); verifying every IFD"
  pyrun "$SCRIPTS/verify_frag1.py" || die "frag1 verification failed (evidence above)"
  stage_close frag1
fi

# ============================================================================
# STAGE expa_build -- baseline 9.36um zarr (uint16) + uint16-acceptance probe.
# ============================================================================
if stage_done expa_build; then
  say "=== STAGE expa_build already done, skipping ==="
else
  stage_open expa_build
  pyrun "$SCRIPTS/build_frag1.py" baseline || die "baseline build failed"
  say "probe: running inference on a uint16 crop to confirm dtype acceptance"
  if (cd /workspace/villa/vesuvius && uv run --no-sync --extra models \
      python -m vesuvius.ink_detection.inference.infer \
      "$DATA/probe_u16.zarr" "$CKPT" "$PREDS/probe.tif" \
      --direction forward --batch-size 4 --num-workers 2 --gpus 0 --no-compile) \
      && [ -s "$PREDS/probe.tif" ]; then
    say "probe PASSED: uint16 accepted (tifxyz_robust casts to float32) -- PLAN A"
  else
    say "probe FAILED: falling back to PLAN B (uint8 baseline, bitdepth rung DEGENERATE)"
    pyrun "$SCRIPTS/build_frag1.py" planb || die "plan B rebuild failed"
  fi
  stage_close expa_build
fi
BASE_ZARR=$(cat "$VAR/baseline_path.txt")

# ============================================================================
# STAGE expa_baseline -- infer both directions, score on native grid, GATE.
# =============================================================================
if stage_done expa_baseline; then
  say "=== STAGE expa_baseline already done, skipping ==="
else
  stage_open expa_baseline
  run_infer "$BASE_ZARR" "$PREDS/expA_base.tif" both
  set +e
  pyrun "$SCRIPTS/score_expa.py" baseline
  RC=$?
  set -e
  if [ $RC = 21 ]; then
    die "KILLED_BASELINE -- undegraded Frag1 AUC below the pre-registered 0.85 gate; everything aborts (this abort is the result; evidence in results/expA_baseline.json)"
  elif [ $RC != 0 ]; then
    die "baseline scoring failed rc=$RC"
  fi
  stage_close expa_baseline
fi
DIRECTION=$(cat "$VAR/direction.txt")
case $DIRECTION in forward) DIR_SHORT=fwd;; reverse) DIR_SHORT=rev;; esac

# ============================================================================
# STAGE expa_rungs -- build all 12 degraded volumes, infer (chosen direction),
# score the curve.
# ============================================================================
if stage_done expa_rungs; then
  say "=== STAGE expa_rungs already done, skipping ==="
else
  stage_open expa_rungs
  pyrun "$SCRIPTS/build_frag1.py" rungs || die "rung build failed"
  for P in 4.0 5.0 6.5 8.0 12.0; do
    run_infer "$DATA/rung_p$P.zarr" "$PREDS/expA_p$P.tif" "$DIR_SHORT"
  done
  for K in 1 2 4 8; do
    run_infer "$DATA/rung_n$K.zarr" "$PREDS/expA_n$K.tif" "$DIR_SHORT"
  done
  if [ ! -f "$VAR/bitdepth_degenerate" ]; then
    run_infer "$DATA/rung_bit8.zarr" "$PREDS/expA_bit8.tif" "$DIR_SHORT"
  fi
  for S in 1.0 2.0; do
    run_infer "$DATA/rung_blur$S.zarr" "$PREDS/expA_blur$S.tif" "$DIR_SHORT"
  done
  pyrun "$SCRIPTS/score_expa.py" rungs || die "rung scoring failed"
  stage_close expa_rungs
fi

# ============================================================================
# STAGE expb_fetch / expb_build -- the three 500p2a windows.
# ============================================================================
if stage_done expb_fetch; then
  say "=== STAGE expb_fetch already done, skipping ==="
else
  stage_open expb_fetch
  retry 3 pyrun "$SCRIPTS/fetch_windows.py" || die "window fetch failed"
  stage_close expb_fetch
fi

if stage_done expb_build; then
  say "=== STAGE expb_build already done, skipping ==="
else
  stage_open expb_build
  pyrun "$SCRIPTS/build_windows.py" || die "window build/verify failed (a data gate tripping here means KILLED: evidence above)"
  stage_close expb_build
fi

# ============================================================================
# STAGE expb_infer -- fwd+rev on each window at 9.36um (primary) and native
# 4.32um (sensitivity). Frag1 9.36um fwd+rev already exists from the baseline.
# ============================================================================
if stage_done expb_infer; then
  say "=== STAGE expb_infer already done, skipping ==="
else
  stage_open expb_infer
  for W in win1 win2 win3; do
    run_infer "$DATA/${W}_936.zarr" "$PREDS/expB_${W}_s936.tif" both
    if [ "$NATIVE_WINDOWS" = 1 ]; then
      run_infer "$DATA/${W}_native.zarr" "$PREDS/expB_${W}_native.tif" both
    fi
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
# STAGE finalize -- aggregate + full inventory re-verification. ALL DONE is
# printed here and nowhere else; finalize.py exits nonzero on ANY missing
# artifact, and every earlier failure died before reaching this line.
# ============================================================================
stage_open finalize
pyrun "$SCRIPTS/finalize.py" || die "finalize inventory check refused (missing artifacts listed above)"
cp -f "$STATUS" "$OUT/status_at_done.txt" 2>/dev/null || true
stage_close finalize
say "ALL DONE -- results at results/results.json (served on :$PORT). scp -r root@POD:$OUT ./curve_out ; then TERMINATE THE POD."
echo IDLE > "$VAR/stage"

if [ "$AUTOSTOP" = 1 ] && command -v runpodctl >/dev/null 2>&1; then
  say "AUTOSTOP armed: stopping pod in 30 min (results persist on the volume)"
  sleep 1800
  runpodctl stop pod "${RUNPOD_POD_ID:-}" || say "AUTOSTOP failed -- terminate manually"
fi
if [ "$LINGER_EXIT" = 1 ]; then exit 0; fi
while :; do
  sleep 300
  say "IDLE -- ALL DONE; fetch results and TERMINATE THE POD (it bills until you do)"
done
