#!/bin/bash
# =============================================================================
# pod_betA_arm0.sh -- Bet A arm 0: leave-PHerc0139-out retrain of the published
# ink_9um recipe, scored on the transfer benchmark.                       v1
#
#   Plan: trackD/bench/betA_arm0/PLAN.md (stages, data manifest, cost, khj1222's
#   anchor). Prereg: trackD/PREREG_BET_A_DRAFT.md sec.4 (corrected 2026-09-02).
#
#   FIRST LAUNCH IS SMOKE_ONLY=1: everything up to and including a 2,000-step
#   training run of seed 42 plus its evaluation on the five held-out native
#   PHerc0139 crops (~40-60 min, ~$0.5-0.8). It validates the whole pipeline on
#   real data -- HF label sync, sparse level-2 fetch, pooling, config generation,
#   the trainer, inference, the F1 sweep and the AUC scorer -- before the
#   8-11 h full run (SMOKE_ONLY=0: 78,125 steps x seeds 42 and 43).
#
#   ORDER OF FAILURE-COST: the trainer is exercised on ONE real small label set
#   (pherc0814-46527, 460 files) + a random volume for 30 iterations BEFORE the
#   29 GB fetch; the released checkpoint certifies the harness on the w035
#   control crop (the p2a_v3 CTL arms) BEFORE any of our checkpoints are scored.
#
# OBSERVABILITY: the v1/v2/v3 machinery verbatim -- status server first, BOOT /
# SERVE / PREREG, HEARTBEAT every 60 s (with the newest trainer log line),
# stage markers, ALL DONE only from finalize. Never calls the RunPod API.
#
# Env knobs: SMOKE_ONLY=1|0  SEEDS="42 43"  SMOKE_STEPS=2000  FULL_STEPS=78125
#            FETCH_THREADS=32  POOL_PAR=3  RUN_P2A=0  PORT DRY LINGER_EXIT FORCE
# =============================================================================
set -Eeuo pipefail

ROOT=${ROOT:-/workspace/betA}
PORT=${PORT:-8000}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}
FORCE=${FORCE:-0}
DRY=${DRY:-0}
DRY_FAIL_STAGE=${DRY_FAIL_STAGE:-}
LINGER_EXIT=${LINGER_EXIT:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
SMOKE_ONLY=${SMOKE_ONLY:-1}
ARM=${ARM:-0}                 # Bet A arm: 0 = LOSO baseline, 1 = input degradation, 2 = PSD whitening
VILLA_PIN_REF=${VILLA_PIN_REF:-master}   # fork branch to clone: master = arm-0 snapshot a3f2c29; betA-arms = arms 1/2
SEEDS=${SEEDS:-"42 43"}
SMOKE_STEPS=${SMOKE_STEPS:-2000}
FULL_STEPS=${FULL_STEPS:-78125}
FETCH_THREADS=${FETCH_THREADS:-32}
POOL_PAR=${POOL_PAR:-4}
RUN_P2A=${RUN_P2A:-0}
SEED=20260902

OUT=$ROOT/out;  VAR=$ROOT/var;  DATA=$ROOT/data;  PREDS=$ROOT/preds
SCRIPTS=$ROOT/scripts;  RESULTS=$OUT/results;  STATUS=$OUT/status.txt
LABELS=$DATA/labels;  VOLS=$DATA/volumes;  NATIVE=$DATA/native;  RUNS=$OUT/runs
export ROOT OUT VAR DATA PREDS SCRIPTS RESULTS STATUS SEED BATCH WORKERS
export LABELS VOLS NATIVE RUNS SMOKE_ONLY SMOKE_STEPS FULL_STEPS FETCH_THREADS ARM VILLA_PIN_REF

# =================================================================== L1 ======
# The very first actions: make the served dir and write the BOOT line.
mkdir -p "$OUT" "$VAR" "$DATA" "$PREDS" "$SCRIPTS" "$RESULTS" "$OUT/previews" "$DATA/tmp"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(now) $*" >> "$STATUS"; echo "$(now) $*"; }
say "BOOT pod_betA_arm0 pid=$$ host=${HOSTNAME:-unknown} root=$ROOT -- status live; server next"

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
 "version": "Bet A arm 0 v1 -- leave-PHerc0139-out retrain of the published ink_9um recipe (aligned21_hybrid_3d2d + aligned21_fixed_scroll_prior, villa PR #1608 generator), 2 seeds, scored on the five held-out native PHerc0139 segments; SMOKE_ONLY runs are pipeline validation only and carry no verdict | ARMS 1/2 v1 (2026-09-03 21:50 UTC, after the arm-0 anchor PASS): same recipe, seeds, schedule and evaluation; arm 1 adds input_degradation, arm 2 adds input_whitening (fork branch betA-arms @ 4516beddbf4309517a66715eb579fad9c332cf81)",
 "seed": 20260902,
 "prereg_document": "trackD/PREREG_BET_A_DRAFT.md sec.4 as corrected 2026-09-02 from bench/betA_arm0/PLAN.md sec.8-9",
 "arm": "0 -- recipe unchanged; the 14 PHerc0139 representations (9 aligned + 5 native) removed from training; quotas renormalised {1667: 40, Paris4: 20, 0814: 4} of batch 64; 78,125 iterations; checkpoints every 5,000",
 "anchor": {
  "published_native5_mean_best_f1": 0.653,
  "published_seeds": {
   "42": 0.627,
   "43": 0.653
  },
  "floor_mean": 0.541,
  "margin": 0.112,
  "per_segment": {
   "w035": 0.679,
   "w039": 0.585,
   "w040": 0.652,
   "w041": 0.703,
   "w044": 0.646
  },
  "gate": "PASS iff best-of-both-seeds native-5 mean best-F1 >= 0.603 AND mean-of-seeds margin over the same floors >= +0.06 AND the trajectory peaks at 10-30k with 75k below the peak on both seeds; else the recipe reproduction is wrong and arm comparisons are void",
  "scorer": "khj1222 replica: prediction = infer uint8 TIFF (forward, centred 17-of-28 window); pixel positive iff score >= t, t swept 0..255 from 256-bin histograms over the supervision-mask region; best F1; floor = 2p/(1+p)"
 },
 "benchmark": "pixel AUC (tie-corrected, native grid, pos = ink & sup, neg = sup & ~ink) both directions on the best-of-grid checkpoint per seed; the arm-0 AUC and its seed spread become the baseline and noise floor for sec.5 of the prereg",
 "harness_controls": "the p2a_v3 CTL arms with the RELEASED seed42 step-075000 checkpoint on the w035 crop: native forward >= 0.95 and reverse <= 0.80 (fatal), scale-fault x1.9504 and x0.5 reported",
 "data_gates": "sha256 of every label-store metadata file and every source-volume .zarray/.zattrs against the embedded manifest; sparse level-2 chunk plan within [0.5x, 1.25x] of the manifest count per representation; absent-chunk fraction < 60% per store; pooled shape == label shape; 20 random supervised patches per representation non-zero after pooling; generator quotas exactly {1667: 40, Paris4: 20, 0814: 4} with 15 kept / 14 held out",
 "smoke": "SMOKE_ONLY=1: seed 42 only, num_iterations 2000, save_every 1000; the 2,000-step checkpoint and the released checkpoint are scored on the five crops; no gate verdict is issued",
 "arms_1_2": {
  "locked": "2026-09-03 before any arm-1/2 pod; arm-0 results (seeds 42/43: best native-5 F1 0.627 @ 20k / 0.631 @ 30k, mean AUC fwd 0.7459 / 0.7566, mean 0.7513, seed spread 0.011) were known when this was written",
  "code": "flummoxjr/villa-pin-37e300d3 branch betA-arms @ 4516bedd = arm-0 snapshot a3f2c29 + vesuvius/ink_detection/data/degradation.py + config/dataset/infer hooks; absent keys keep arm 0 byte-identical",
  "arm_1_input_degradation": {
   "where": "every TRAINING flat crop (pooled 2.4->9.6 um representations), before robust_mad normalisation; never on validation crops or at inference",
   "target_draw": "per crop: one scroll uniformly from the 14-scroll k2b index (trackD out/k2b_index), then one of its ROIs uniformly -> (bandwidth_cyc_px, snr_q025, dn_headroom)",
   "steps": [
    "Gaussian in-plane blur with sigma^2 = ln(P(q_t) / (2 (N + n_add))) / (4 pi^2 q_t^2) so the crop's radial PSD crosses 2x its floor at the target bandwidth q_t (0 if already below)",
    "additive white Gaussian noise so that the residual-floor structural SNR at q = 0.25 cyc/px equals the target (two measure-and-add passes)",
    "contrast scaling about the crop mean so the in-mask p99.5 - p0.5 DN spread equals the target headroom, then clip to [0, 255]"
   ],
   "estimator": "2-D per-crop transposition of the k2b definitions (Hanning radial PSD, 3x3 uniform high-pass residual floor from the 0.35-0.48 band); index targets were measured in 3-D with air (where found) or residual references, so residual-referenced SNR is a lower bound and the added noise is conservative",
   "activity_gate": "stage `measure` (results/input_stats.json) reports the pooled sources' snr_q025 / bandwidth / headroom with the same estimator BEFORE training; the fraction of pooled stores whose SNR (bandwidth) exceeds the index-target median is the fraction of crops where the noise (blur) step can act. If both fractions are 0 the arm-1 degradation reduces to the headroom match and the result is read as such Activity is judged AFTER calibration (pooled 2-D stats vs scaled targets).",
   "target_calibration": "AMENDED 2026-09-03 23:45 UTC before any arm-1 training (the first four arm pods died in the measure stage on a zarr group/array bug; their input statistics never ran): the index targets were measured on 3-D ROIs, the per-crop estimator is 2-D; on the native PHerc0139 crop the 2-D estimator reads snr_q025 ~51 vs the index's 115.5 (bandwidth 0.369 vs 0.386, headroom 167 vs 151). Each arm-1 pod therefore multiplies every drawn (bandwidth, snr, headroom) target by target_scale = median over the five native-0139 crops of the 2-D value / the index's PHerc0139 median (computed in the measure stage, which now runs BEFORE config generation, and recorded in the config under input_degradation.calibration). If input_stats.json is missing the scale is (1,1,1) and the run is flagged uncalibrated."
  },
  "arm_2_input_whitening": {
   "where": "every flat crop of every volume at training AND validation AND flat inference (fitted per eval volume from 64 of its blocks in infer.py)",
   "filter": "in-plane radial gain g(q) = sqrt(PSD(q_ref) / PSD(q)) with q_ref = 0.02 cyc/px, PSD = median radial PSD of 64 random 128x128 in-plane windows of the volume's central slices; gain clipped to [1/8, 8]; DC and crop mean preserved; applied per slice by rFFT before robust_mad"
  },
  "decision_rule_frozen": {
   "primary": "arm 1 OR arm 2 beats arm 0 by >= +0.05 mean native-5 forward AUC at the best-of-grid checkpoint, both seeds (arm-0 baseline 0.7513; the improvement must also exceed 2 x the arm-0 seed spread = 0.022) -> the arm's mean AUC over its two seeds must be >= 0.8013",
   "secondary": "500p2a win1 (iso) >= 0.65 for the passing arm (A3 = 0.5211), measured afterwards with bench/p2a_v3 on the saved best checkpoints; not computed on these pods",
   "reverse_direction": "reported (depth-order asymmetry must persist: reverse AUC near 0.5) but not a gate",
   "KILLED": "otherwise; the augmentation code, configs, input statistics and held-out numbers ship either way",
   "combination": "two pods per arm (SEEDS=42 / SEEDS=43), combined locally with combine_verdict.py's rule extended by the +0.05 AUC clause (arm_verdict.py)"
  },
  "cost_cap": "4 pods x <= 15 h guard; about $5 per pod on a community 5090"
 }
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
# Embedded programs and data (locked before provisioning).
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
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
B64_LABELS

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

cat > "$SCRIPTS/manifest.json" <<'JSON_MANIFEST'
{
 "manifest_version": "betA_arm0 data manifest v1",
 "generated_utc": "2026-09-02T03:57:55+00:00",
 "purpose": "Every store the LOSO(no-0139) retrain pod must fetch: 15 kept training representations (labels + sparse level-2 source volumes), 5 native PHerc0139 eval volumes (crops), code pins and the released checkpoint for harness controls.",
 "conventions": {
  "s3_base": "https://vesuvius-challenge-open-data.s3.amazonaws.com/",
  "hf_labels_base": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/",
  "chunk_key": "level/<z>{sep}<y>{sep}<x> with sep = the store's .zarray dimension_separator ('.' for pherc1667-w013 level 2, '/' for every other store below); copy .zarray/.zattrs/.zgroup verbatim so the local store honours the same separator and fill_value",
  "http_404_on_chunk": "absent chunk = all fill_value (0); write a zero-length marker (p2a_v3 ctl_build.py convention) and count it; FATAL if > 60% of the planned chunks of a store are absent (the planned set lies inside the supervision bbox, where data exists)",
  "aligned_label_all_zero_chunk": {
   "bytes": 78,
   "xetHash": "50cce7bbd06eea2a0367c31de5fb3ac212044946ebc1055005e49f634c82ea6a",
   "note": "blosc-compressed 21x128x128 zeros; skipping it is content-identical to fetching it (fill_value 0). Filter by xetHash, not by size."
  },
  "native_label_all_zero_chunk": {
   "bytes": 134,
   "xetHash": "4f3a32296d2d0709242deb9790ff0079d6174b177386edca5ed9de03e00fcef4",
   "note": "134-byte chunks with a different xetHash exist (few nonzero px) -- size alone is NOT an emptiness test"
  },
  "hf_tree_api": "https://huggingface.co/api/buckets/scrollprize/datasets/tree/<path>?limit=1000 (paginated via Link rel=next cursor); each entry carries path, size, xetHash",
  "training_chunk_rule": "level-2 chunks covering the union of training patches: for every supervised pixel (y,x) at label plane z=10, patch corner = (y//32*32, x//32*32), patch = [corner, corner+128); union of covering 128x128 chunk columns, dilated by 1 chunk (128 px) as margin. The counts below used a superset (supervision mask max-filtered +-128 px); the pod recomputes from the downloaded supervision masks and asserts every patch bbox of the trainer's patch index lies inside the fetched set.",
  "pooling": "vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input <local sparse level-2 group> <out.zarr> --level 2 : centered 84 of 109 planes = z[13:97], 4x mean -> 21 slices, uint8, chunks (21,128,128), blosc zstd clevel 5 bitshuffle, attrs.format 'level2-zmean4-21slice-v1'; matches the label .zattrs (source_z_slice [13,97], z_pool '21 complete means of 4 source planes')"
 },
 "code": {
  "villa_pin": {
   "repo": "https://github.com/flummoxjr/villa-pin-37e300d3",
   "branch": "master",
   "sha": "a3f2c292f61b76d1471db4f6bedcb54cb36f4e90",
   "upstream": "ScrollPrize/villa vesuvius @ 37e300d3 (pinned 2026-08-24)",
   "train_entry": "python -m vesuvius.ink_detection.training.train <config.json>",
   "pool_entry": "python -m vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input <in> <out> --level 2",
   "infer_entry": "python -m vesuvius.ink_detection.inference.infer <zarr> <ckpt> <out.tif> --direction forward|reverse|both --batch-size 16 --num-workers 8 --gpus 0 --no-compile",
   "offline_checks_2026-09-02": "generated LOSO config parsed by TrainingConfig.from_mapping(resolve_training_mapping(..)) and stage_training_request() at this sha; make_model: 34.55 M params, peak 5.43 GiB allocated / 7.23 GiB reserved at batch 64 fp16 (17,128,128), 360 ms/step compute-only on an RTX 4090 laptop"
  },
  "reference_upstream": {
   "repo": "https://github.com/ScrollPrize/villa",
   "branch": "merge-ink-pipelines",
   "tip": "3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e",
   "package": "ink-detection/koine_machines (pyproject: torch==2.10.0, torchvision==0.25.0, zarr==2.18.7, numcodecs==0.15.1, numpy<=2.2, python>=3.11, uv.lock present, depends on ../vesuvius editable)",
   "note": "khj1222 trained the published LOSO arms here (PR #1608: tested at c61cc9f, branched from 3ea17f5). villa-pin carries the same pipeline under vesuvius.ink_detection (the merged copy). Not byte-identical code paths; the anchor-gate tolerance absorbs that."
  },
  "generator": {
   "path": "trackD/bench/vendor/make_holdout_config.py",
   "sha256": "1361f28852da0538d4597546f4dacec18c899a4d10a220de115f095c5390335d",
   "source": "villa PR #1608 @ dc9edb6 (khj1222)",
   "invocation": "--labels-root <labels> --volumes-root <volumes> --exclude-scroll 0139 --seed <42|43> --recipe vendor/configs/aligned21_hybrid_3d2d.json --contract vendor/configs/aligned21_fixed_scroll_prior.json --out <cfg> --run-dir <run>",
   "expected": "15 kept / 14 held out; quotas {'1667': 40, 'Paris4': 20, '0814': 4}; 3 dataset entries (0814, 1667, Paris4)"
  },
  "recipe": {
   "path": "trackD/bench/vendor/configs/aligned21_hybrid_3d2d.json",
   "sha256": "e5036d8ed38ec00a2dbc824276c362c1e11329d210d98a3dd41a42bb3250a003"
  },
  "contract": {
   "path": "trackD/bench/vendor/configs/aligned21_fixed_scroll_prior.json",
   "sha256": "0e20230f4d2517987f31baed4c6fc2fa8110d230ce2d14a60dfe5d6a838c68d1",
   "note": "byte-identical to villa-pin's vesuvius/src/vesuvius/ink_detection/configs/aligned21_fixed_scroll_prior.json"
  },
  "released_checkpoint_for_controls": {
   "hf_repo": "scrollprize/ink_9um",
   "file": "hybrid_3d2d-seed42/step-075000.pth",
   "bytes": 138360039,
   "note": "used ONLY for the benchmark's four positive controls (harness certification) and the in-scroll 'ref' row; seed43 files are 138360231 bytes"
  },
  "hf_dataset_readme": {
   "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/README.md",
   "bytes": 10739,
   "sha256": "c9a88ff970594fcef3788eb588bb75dbbd0e6436f4812f89074e26c0d9fef7a5"
  }
 },
 "labels_hf": {
  "kept_aligned": {
   "pherc1667-w013": {
    "scroll": "1667",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w013/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w013/pherc1667-w013_inklabels.zarr",
      "shape": [
       21,
       10400,
       19900
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       82,
       156
      ],
      "chunk_files": 12792,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w013/pherc1667-w013_inklabels.zarr/0/.zarray",
       "bytes": 368,
       "sha256": "5b1aa3ac9416ee2f3db51c06b85485139d9dea3f4a7c625ae7895fe3bed48fb2"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w013/pherc1667-w013_inklabels.zarr/.zattrs",
       "bytes": 1235,
       "sha256": "cb3cf48f7f288af5861554fad0c7de63bcb784edea2dcf4b0b6344edb2a4bd66"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w013/pherc1667-w013_supervision_mask.zarr",
      "shape": [
       21,
       10400,
       19900
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       82,
       156
      ],
      "chunk_files": 12792,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w013/pherc1667-w013_supervision_mask.zarr/0/.zarray",
       "bytes": 368,
       "sha256": "5b1aa3ac9416ee2f3db51c06b85485139d9dea3f4a7c625ae7895fe3bed48fb2"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w013/pherc1667-w013_supervision_mask.zarr/.zattrs",
       "bytes": 1235,
       "sha256": "cb3cf48f7f288af5861554fad0c7de63bcb784edea2dcf4b0b6344edb2a4bd66"
      }
     },
     "validation_mask": null
    },
    "supervision_plane_stats": {
     "sup_px": 3358655,
     "ink_px": 1012042,
     "sup_frac": 0.01623,
     "sup_bbox_yx": [
      669,
      7361,
      514,
      4857
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "pherc1667-w018": {
    "scroll": "1667",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w018/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w018/pherc1667-w018_inklabels.zarr",
      "shape": [
       21,
       10595,
       24525
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       83,
       192
      ],
      "chunk_files": 15936,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w018/pherc1667-w018_inklabels.zarr/0/.zarray",
       "bytes": 368,
       "sha256": "250fe1f6374039fa1871a13af5b3cd34e10e16006e1c4a4b965ec7f1ac82fab2"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w018/pherc1667-w018_inklabels.zarr/.zattrs",
       "bytes": 1235,
       "sha256": "f8d047f8eb787a2860589f7ba78bf5f09aed7336650993f21b385f5cb3ae0a67"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w018/pherc1667-w018_supervision_mask.zarr",
      "shape": [
       21,
       10595,
       24525
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       83,
       192
      ],
      "chunk_files": 15936,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w018/pherc1667-w018_supervision_mask.zarr/0/.zarray",
       "bytes": 368,
       "sha256": "250fe1f6374039fa1871a13af5b3cd34e10e16006e1c4a4b965ec7f1ac82fab2"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w018/pherc1667-w018_supervision_mask.zarr/.zattrs",
       "bytes": 1235,
       "sha256": "f8d047f8eb787a2860589f7ba78bf5f09aed7336650993f21b385f5cb3ae0a67"
      }
     },
     "validation_mask": null
    },
    "supervision_plane_stats": {
     "sup_px": 13899340,
     "ink_px": 3580927,
     "sup_frac": 0.05349,
     "sup_bbox_yx": [
      962,
      8193,
      192,
      24057
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "pherc1667-w023": {
    "scroll": "1667",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w023/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w023/pherc1667-w023_inklabels.zarr",
      "shape": [
       21,
       10465,
       25590
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       82,
       200
      ],
      "chunk_files": 16400,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w023/pherc1667-w023_inklabels.zarr/0/.zarray",
       "bytes": 368,
       "sha256": "34274061d16ac94dd178fddb33eaa304318566089033f9fb5576ca67290e69aa"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w023/pherc1667-w023_inklabels.zarr/.zattrs",
       "bytes": 1235,
       "sha256": "4f338c48be7f08042ef9eaa077773e30632bd967ac95d26973bf78526ddfe13d"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w023/pherc1667-w023_supervision_mask.zarr",
      "shape": [
       21,
       10465,
       25590
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       82,
       200
      ],
      "chunk_files": 16400,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w023/pherc1667-w023_supervision_mask.zarr/0/.zarray",
       "bytes": 368,
       "sha256": "34274061d16ac94dd178fddb33eaa304318566089033f9fb5576ca67290e69aa"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w023/pherc1667-w023_supervision_mask.zarr/.zattrs",
       "bytes": 1235,
       "sha256": "4f338c48be7f08042ef9eaa077773e30632bd967ac95d26973bf78526ddfe13d"
      }
     },
     "validation_mask": null
    },
    "supervision_plane_stats": {
     "sup_px": 5404437,
     "ink_px": 1419591,
     "sup_frac": 0.02018,
     "sup_bbox_yx": [
      1603,
      7949,
      1330,
      25264
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "pherc1667-w028": {
    "scroll": "1667",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w028/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w028/pherc1667-w028_inklabels.zarr",
      "shape": [
       21,
       9355,
       7585
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       74,
       60
      ],
      "chunk_files": 4440,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w028/pherc1667-w028_inklabels.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "4441304e06f67efaf4563605510b1c09122e05e4f2c2777d74eb88dd22736ffa"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w028/pherc1667-w028_inklabels.zarr/.zattrs",
       "bytes": 1236,
       "sha256": "7fee18c15f1a142c7a762728df0cd62823d0eaf789ac085201a40793b8789b5f"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w028/pherc1667-w028_supervision_mask.zarr",
      "shape": [
       21,
       9355,
       7585
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       74,
       60
      ],
      "chunk_files": 4440,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w028/pherc1667-w028_supervision_mask.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "4441304e06f67efaf4563605510b1c09122e05e4f2c2777d74eb88dd22736ffa"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w028/pherc1667-w028_supervision_mask.zarr/.zattrs",
       "bytes": 1236,
       "sha256": "7fee18c15f1a142c7a762728df0cd62823d0eaf789ac085201a40793b8789b5f"
      }
     },
     "validation_mask": null
    },
    "supervision_plane_stats": {
     "sup_px": 844786,
     "ink_px": 331857,
     "sup_frac": 0.01191,
     "sup_bbox_yx": [
      2251,
      5519,
      560,
      7502
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "pherc1667-w029": {
    "scroll": "1667",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029/pherc1667-w029_inklabels.zarr",
      "shape": [
       21,
       9500,
       7830
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       75,
       62
      ],
      "chunk_files": 4650,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029/pherc1667-w029_inklabels.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "52e50d0371f434ea084ad8f1fb02f31a0e1c4cc11b6e2f250faf63febb8b024a"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029/pherc1667-w029_inklabels.zarr/.zattrs",
       "bytes": 1235,
       "sha256": "1a749c0c9d290f24424d9018c0a98ed80a3abe4f99424a0f773d66c7c8774cd6"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029/pherc1667-w029_supervision_mask.zarr",
      "shape": [
       21,
       9500,
       7830
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       75,
       62
      ],
      "chunk_files": 4650,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029/pherc1667-w029_supervision_mask.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "52e50d0371f434ea084ad8f1fb02f31a0e1c4cc11b6e2f250faf63febb8b024a"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029/pherc1667-w029_supervision_mask.zarr/.zattrs",
       "bytes": 1235,
       "sha256": "1a749c0c9d290f24424d9018c0a98ed80a3abe4f99424a0f773d66c7c8774cd6"
      }
     },
     "validation_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029/pherc1667-w029_validation_mask.zarr",
      "shape": [
       21,
       9500,
       7830
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       75,
       62
      ],
      "chunk_files": 4650,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029/pherc1667-w029_validation_mask.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "52e50d0371f434ea084ad8f1fb02f31a0e1c4cc11b6e2f250faf63febb8b024a"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w029/pherc1667-w029_validation_mask.zarr/.zattrs",
       "bytes": 1235,
       "sha256": "1a749c0c9d290f24424d9018c0a98ed80a3abe4f99424a0f773d66c7c8774cd6"
      }
     }
    },
    "supervision_plane_stats": {
     "sup_px": 1213836,
     "ink_px": 444385,
     "sup_frac": 0.01632,
     "sup_bbox_yx": [
      1621,
      7012,
      1745,
      7163
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "pherc1667-w031": {
    "scroll": "1667",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w031/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w031/pherc1667-w031_inklabels.zarr",
      "shape": [
       21,
       9370,
       8045
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       74,
       63
      ],
      "chunk_files": 4662,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w031/pherc1667-w031_inklabels.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "071c6d76b2541f8c08eb4b8e9a03a7cb626212aeaccb46ddd0314234a2c69776"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w031/pherc1667-w031_inklabels.zarr/.zattrs",
       "bytes": 1229,
       "sha256": "6d728535b01f5b3ff88a20c89b10df06a2ded2ccce50c2af170f93c9f6d49980"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w031/pherc1667-w031_supervision_mask.zarr",
      "shape": [
       21,
       9370,
       8045
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       74,
       63
      ],
      "chunk_files": 4662,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w031/pherc1667-w031_supervision_mask.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "071c6d76b2541f8c08eb4b8e9a03a7cb626212aeaccb46ddd0314234a2c69776"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w031/pherc1667-w031_supervision_mask.zarr/.zattrs",
       "bytes": 1229,
       "sha256": "6d728535b01f5b3ff88a20c89b10df06a2ded2ccce50c2af170f93c9f6d49980"
      }
     },
     "validation_mask": null
    },
    "supervision_plane_stats": {
     "sup_px": 2621780,
     "ink_px": 778204,
     "sup_frac": 0.03478,
     "sup_bbox_yx": [
      2797,
      7070,
      1447,
      5949
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "phercparis4-w00": {
    "scroll": "Paris4",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w00/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w00/phercparis4-w00_inklabels.zarr",
      "shape": [
       21,
       7990,
       12990
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       63,
       102
      ],
      "chunk_files": 6426,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w00/phercparis4-w00_inklabels.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "73153d95cbe890f3b99ebfa1c0723a51390e312cdcf9a5fd50cd2051cce71fd7"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w00/phercparis4-w00_inklabels.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "62e482ef983b21ee941321118e304f2fe76d45da9061f6e78f1232da5958db78"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w00/phercparis4-w00_supervision_mask.zarr",
      "shape": [
       21,
       7990,
       12990
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       63,
       102
      ],
      "chunk_files": 6426,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w00/phercparis4-w00_supervision_mask.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "73153d95cbe890f3b99ebfa1c0723a51390e312cdcf9a5fd50cd2051cce71fd7"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w00/phercparis4-w00_supervision_mask.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "62e482ef983b21ee941321118e304f2fe76d45da9061f6e78f1232da5958db78"
      }
     },
     "validation_mask": null
    }
   },
   "phercparis4-w01": {
    "scroll": "Paris4",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w01/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w01/phercparis4-w01_inklabels.zarr",
      "shape": [
       21,
       12650,
       9100
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       99,
       72
      ],
      "chunk_files": 7128,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w01/phercparis4-w01_inklabels.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "56b3617182c41c16ae15082f94967b4594de36a07be4b960f25c2fbaa88cbb8c"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w01/phercparis4-w01_inklabels.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "34a657aa586d2adeeb63e5d06d2a7f82f45172ca19768f8d56f40bbf855be3b9"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w01/phercparis4-w01_supervision_mask.zarr",
      "shape": [
       21,
       12650,
       9100
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       99,
       72
      ],
      "chunk_files": 7128,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w01/phercparis4-w01_supervision_mask.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "56b3617182c41c16ae15082f94967b4594de36a07be4b960f25c2fbaa88cbb8c"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w01/phercparis4-w01_supervision_mask.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "34a657aa586d2adeeb63e5d06d2a7f82f45172ca19768f8d56f40bbf855be3b9"
      }
     },
     "validation_mask": null
    },
    "supervision_plane_stats": {
     "sup_px": 8268843,
     "ink_px": 2163941,
     "sup_frac": 0.07183,
     "sup_bbox_yx": [
      389,
      12594,
      407,
      8032
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "phercparis4-w02": {
    "scroll": "Paris4",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w02/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w02/phercparis4-w02_inklabels.zarr",
      "shape": [
       21,
       8545,
       11005
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       67,
       86
      ],
      "chunk_files": 5762,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w02/phercparis4-w02_inklabels.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "a65f9bf41b1c21fc0d439581d18915c212de958945f76653107dbe9c1baef1b3"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w02/phercparis4-w02_inklabels.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "f911d4ea285000fbcc02c7d87770dcb07be359dacf0e0eb21538d72bdafb4be9"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w02/phercparis4-w02_supervision_mask.zarr",
      "shape": [
       21,
       8545,
       11005
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       67,
       86
      ],
      "chunk_files": 5762,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w02/phercparis4-w02_supervision_mask.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "a65f9bf41b1c21fc0d439581d18915c212de958945f76653107dbe9c1baef1b3"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w02/phercparis4-w02_supervision_mask.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "f911d4ea285000fbcc02c7d87770dcb07be359dacf0e0eb21538d72bdafb4be9"
      }
     },
     "validation_mask": null
    },
    "supervision_plane_stats": {
     "sup_px": 4647609,
     "ink_px": 1087282,
     "sup_frac": 0.04942,
     "sup_bbox_yx": [
      668,
      8420,
      2209,
      10956
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "phercparis4-w03": {
    "scroll": "Paris4",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w03/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w03/phercparis4-w03_inklabels.zarr",
      "shape": [
       21,
       8575,
       12810
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       67,
       101
      ],
      "chunk_files": 6767,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w03/phercparis4-w03_inklabels.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "08541cb71dd81c3464a32997d8f4ec724c45437c3474deab104527f553a8e179"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w03/phercparis4-w03_inklabels.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "af8d6c654e47d757ae16aaa8c089975f8724fdf3418c05ca9309cf063bf182f9"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w03/phercparis4-w03_supervision_mask.zarr",
      "shape": [
       21,
       8575,
       12810
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       67,
       101
      ],
      "chunk_files": 6767,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w03/phercparis4-w03_supervision_mask.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "08541cb71dd81c3464a32997d8f4ec724c45437c3474deab104527f553a8e179"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w03/phercparis4-w03_supervision_mask.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "af8d6c654e47d757ae16aaa8c089975f8724fdf3418c05ca9309cf063bf182f9"
      }
     },
     "validation_mask": null
    },
    "supervision_plane_stats": {
     "sup_px": 6169508,
     "ink_px": 2125255,
     "sup_frac": 0.05617,
     "sup_bbox_yx": [
      744,
      7962,
      1449,
      9743
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "phercparis4-w05": {
    "scroll": "Paris4",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w05/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w05/phercparis4-w05_inklabels.zarr",
      "shape": [
       21,
       12700,
       18930
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       100,
       148
      ],
      "chunk_files": 14800,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w05/phercparis4-w05_inklabels.zarr/0/.zarray",
       "bytes": 368,
       "sha256": "d66a3860e233941be0786904ebe04cc3afb3ea646088da974b81a2fba30807a3"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w05/phercparis4-w05_inklabels.zarr/.zattrs",
       "bytes": 1197,
       "sha256": "05718f2d7eca260f61fe7b630aacf18674f32e61e35cffd4d095f44c9914ac63"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w05/phercparis4-w05_supervision_mask.zarr",
      "shape": [
       21,
       12700,
       18930
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       100,
       148
      ],
      "chunk_files": 14800,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w05/phercparis4-w05_supervision_mask.zarr/0/.zarray",
       "bytes": 368,
       "sha256": "d66a3860e233941be0786904ebe04cc3afb3ea646088da974b81a2fba30807a3"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w05/phercparis4-w05_supervision_mask.zarr/.zattrs",
       "bytes": 1197,
       "sha256": "05718f2d7eca260f61fe7b630aacf18674f32e61e35cffd4d095f44c9914ac63"
      }
     },
     "validation_mask": null
    },
    "supervision_plane_stats": {
     "sup_px": 31930770,
     "ink_px": 6737328,
     "sup_frac": 0.13282,
     "sup_bbox_yx": [
      1792,
      12112,
      3051,
      18095
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "phercparis4-w06": {
    "scroll": "Paris4",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w06/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w06/phercparis4-w06_inklabels.zarr",
      "shape": [
       21,
       12750,
       9995
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       100,
       79
      ],
      "chunk_files": 7900,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w06/phercparis4-w06_inklabels.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "dac94e3783f9a3f309015623fc0d7c9c7a49f706617eaccad5f59a43baa40c9b"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w06/phercparis4-w06_inklabels.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "a362caf3047d6be7f6279995461d30e1c9afdd99958f4b4de58c13592462bd3c"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w06/phercparis4-w06_supervision_mask.zarr",
      "shape": [
       21,
       12750,
       9995
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       100,
       79
      ],
      "chunk_files": 7900,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w06/phercparis4-w06_supervision_mask.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "dac94e3783f9a3f309015623fc0d7c9c7a49f706617eaccad5f59a43baa40c9b"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w06/phercparis4-w06_supervision_mask.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "a362caf3047d6be7f6279995461d30e1c9afdd99958f4b4de58c13592462bd3c"
      }
     },
     "validation_mask": null
    },
    "supervision_plane_stats": {
     "sup_px": 11074717,
     "ink_px": 3443040,
     "sup_frac": 0.0869,
     "sup_bbox_yx": [
      518,
      12571,
      1845,
      8865
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "phercparis4-w07": {
    "scroll": "Paris4",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w07/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w07/phercparis4-w07_inklabels.zarr",
      "shape": [
       21,
       13010,
       27035
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       102,
       212
      ],
      "chunk_files": 21624,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w07/phercparis4-w07_inklabels.zarr/0/.zarray",
       "bytes": 368,
       "sha256": "e152e26112f31cdba91c06c62fe9056897fb3bbc6be280ab12c1e46b21a291bc"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w07/phercparis4-w07_inklabels.zarr/.zattrs",
       "bytes": 1197,
       "sha256": "946366c031d057043c7731b30f801ed2823ad4d6c8737f36e07f097410aa42f7"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w07/phercparis4-w07_supervision_mask.zarr",
      "shape": [
       21,
       13010,
       27035
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       102,
       212
      ],
      "chunk_files": 21624,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w07/phercparis4-w07_supervision_mask.zarr/0/.zarray",
       "bytes": 368,
       "sha256": "e152e26112f31cdba91c06c62fe9056897fb3bbc6be280ab12c1e46b21a291bc"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w07/phercparis4-w07_supervision_mask.zarr/.zattrs",
       "bytes": 1197,
       "sha256": "946366c031d057043c7731b30f801ed2823ad4d6c8737f36e07f097410aa42f7"
      }
     },
     "validation_mask": null
    },
    "supervision_plane_stats": {
     "sup_px": 16003870,
     "ink_px": 5022838,
     "sup_frac": 0.0455,
     "sup_bbox_yx": [
      1748,
      12561,
      2382,
      25087
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   },
   "phercparis4-w09": {
    "scroll": "Paris4",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w09/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w09/phercparis4-w09_inklabels.zarr",
      "shape": [
       21,
       8020,
       18950
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       63,
       149
      ],
      "chunk_files": 9387,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w09/phercparis4-w09_inklabels.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "2635e05427d6aa985b2ffa8ba2bb11571fd1aef3107be03bdf51719da480a69c"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w09/phercparis4-w09_inklabels.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "0a6b6bea4d1505b65bbe6d42c3df8477647cc019a624d827efdfd5f21cb81beb"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w09/phercparis4-w09_supervision_mask.zarr",
      "shape": [
       21,
       8020,
       18950
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       63,
       149
      ],
      "chunk_files": 9387,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w09/phercparis4-w09_supervision_mask.zarr/0/.zarray",
       "bytes": 367,
       "sha256": "2635e05427d6aa985b2ffa8ba2bb11571fd1aef3107be03bdf51719da480a69c"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w09/phercparis4-w09_supervision_mask.zarr/.zattrs",
       "bytes": 1196,
       "sha256": "0a6b6bea4d1505b65bbe6d42c3df8477647cc019a624d827efdfd5f21cb81beb"
      }
     },
     "validation_mask": null
    }
   },
   "pherc0814-46527": {
    "scroll": "0814",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_inklabels.zarr",
      "shape": [
       21,
       2130,
       3455
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       17,
       27
      ],
      "chunk_files": 459,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_inklabels.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "6d35e50273e289cb032f1c7fb22f034467f17f559b4bc636ef0b31dc414cf223"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_inklabels.zarr/.zattrs",
       "bytes": 1221,
       "sha256": "a45a86e16a22c07b7a99566766bca3df0cb922efe42afff504e3393da2a31ac7"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_supervision_mask.zarr",
      "shape": [
       21,
       2130,
       3455
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       17,
       27
      ],
      "chunk_files": 459,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_supervision_mask.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "6d35e50273e289cb032f1c7fb22f034467f17f559b4bc636ef0b31dc414cf223"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_supervision_mask.zarr/.zattrs",
       "bytes": 1221,
       "sha256": "a45a86e16a22c07b7a99566766bca3df0cb922efe42afff504e3393da2a31ac7"
      }
     },
     "validation_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_validation_mask.zarr",
      "shape": [
       21,
       2130,
       3455
      ],
      "chunks": [
       21,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 5,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       17,
       27
      ],
      "chunk_files": 459,
      "annotated_plane_z": 10,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_validation_mask.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "6d35e50273e289cb032f1c7fb22f034467f17f559b4bc636ef0b31dc414cf223"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_validation_mask.zarr/.zattrs",
       "bytes": 1221,
       "sha256": "a45a86e16a22c07b7a99566766bca3df0cb922efe42afff504e3393da2a31ac7"
      }
     }
    },
    "supervision_plane_stats": {
     "sup_px": 428993,
     "ink_px": 187170,
     "sup_frac": 0.05829,
     "sup_bbox_yx": [
      1149,
      1761,
      277,
      1427
     ],
     "source": "local cache D:/vesuvius-data/trackD/ink9um_planes (extracted 2026-08-24)"
    }
   }
  },
  "heldout_native_eval": {
   "w035": {
    "scroll": "0139",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w035/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w035/w035_inklabels.zarr",
      "shape": [
       28,
       5820,
       5240
      ],
      "chunks": [
       28,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 3,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       46,
       41
      ],
      "chunk_files": 1886,
      "annotated_plane_z": 14,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w035/w035_inklabels.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "1df6c0c53d73e53f625ef02fc6bc2efadb304474b7dd23e11abd380d4e68016a"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w035/w035_inklabels.zarr/.zattrs",
       "bytes": 3744,
       "sha256": "a6f81fbe400b6f9c43ec3d48b422662033bb1639afeb0551b4f8527848f01603"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w035/w035_supervision_mask.zarr",
      "shape": [
       28,
       5820,
       5240
      ],
      "chunks": [
       28,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 3,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       46,
       41
      ],
      "chunk_files": 1886,
      "annotated_plane_z": 14,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w035/w035_supervision_mask.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "1df6c0c53d73e53f625ef02fc6bc2efadb304474b7dd23e11abd380d4e68016a"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w035/w035_supervision_mask.zarr/.zattrs",
       "bytes": 3797,
       "sha256": "e7945fb3839de67232979426ce8c0b51550432b7149f213e3404c0be1c420cd3"
      }
     }
    },
    "eval": {
     "sup_bbox_rows_cols": [
      680,
      2770,
      564,
      2910
     ],
     "crop_rows_cols_pad128": [
      512,
      2944,
      384,
      3072
     ],
     "crop_px": 6537216,
     "pos_ink_and_sup": 334035,
     "neg_sup_not_ink": 737086,
     "ink_frac": 0.3119,
     "floor_f1_2p_over_1p": 0.4754,
     "khj1222_scored_px": 1069617,
     "khj1222_floor": 0.4755
    }
   },
   "w039": {
    "scroll": "0139",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w039/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w039/w039_inklabels.zarr",
      "shape": [
       28,
       8560,
       7720
      ],
      "chunks": [
       28,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 3,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       67,
       61
      ],
      "chunk_files": 4087,
      "annotated_plane_z": 14,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w039/w039_inklabels.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "cb7c5185b209ae1992ed510da107c2dddf040b662697bdea359b94215c32564c"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w039/w039_inklabels.zarr/.zattrs",
       "bytes": 3743,
       "sha256": "b19a57000d6d473697064569f86f6f692d2243cd6058c928107d3c7625b35bfc"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w039/w039_supervision_mask.zarr",
      "shape": [
       28,
       8560,
       7720
      ],
      "chunks": [
       28,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 3,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       67,
       61
      ],
      "chunk_files": 4087,
      "annotated_plane_z": 14,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w039/w039_supervision_mask.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "cb7c5185b209ae1992ed510da107c2dddf040b662697bdea359b94215c32564c"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w039/w039_supervision_mask.zarr/.zattrs",
       "bytes": 3795,
       "sha256": "d8186471374a881c27bfc90656f5da2ad9cd49872a94b1605f631a90509319fd"
      }
     }
    },
    "eval": {
     "sup_bbox_rows_cols": [
      3829,
      4844,
      2453,
      5153
     ],
     "crop_rows_cols_pad128": [
      3584,
      4992,
      2304,
      5376
     ],
     "crop_px": 4325376,
     "pos_ink_and_sup": 182429,
     "neg_sup_not_ink": 393791,
     "ink_frac": 0.3166,
     "floor_f1_2p_over_1p": 0.4809,
     "khj1222_scored_px": 574928,
     "khj1222_floor": 0.4817
    }
   },
   "w040": {
    "scroll": "0139",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w040/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w040/w040_inklabels.zarr",
      "shape": [
       28,
       6400,
       7980
      ],
      "chunks": [
       28,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 3,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       50,
       63
      ],
      "chunk_files": 3150,
      "annotated_plane_z": 14,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w040/w040_inklabels.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "2b9e57043448af2583a60da580b9e3ecd70a0f687b590f5bd5fda8edd10447cf"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w040/w040_inklabels.zarr/.zattrs",
       "bytes": 3744,
       "sha256": "b6e9f7b89ed5acb10495ed2ef534f046a1404eee9a9e4d782d68e63b2c6b97fb"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w040/w040_supervision_mask.zarr",
      "shape": [
       28,
       6400,
       7980
      ],
      "chunks": [
       28,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 3,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       50,
       63
      ],
      "chunk_files": 3150,
      "annotated_plane_z": 14,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w040/w040_supervision_mask.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "2b9e57043448af2583a60da580b9e3ecd70a0f687b590f5bd5fda8edd10447cf"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w040/w040_supervision_mask.zarr/.zattrs",
       "bytes": 3798,
       "sha256": "390300e469aa9ccb9d834721657877ba901b863a97f55b49bc4fa136b79e7696"
      }
     }
    },
    "eval": {
     "sup_bbox_rows_cols": [
      644,
      5977,
      1552,
      7744
     ],
     "crop_rows_cols_pad128": [
      512,
      6144,
      1408,
      7936
     ],
     "crop_px": 36765696,
     "pos_ink_and_sup": 716725,
     "neg_sup_not_ink": 913009,
     "ink_frac": 0.4398,
     "floor_f1_2p_over_1p": 0.6109,
     "khj1222_scored_px": 1626899,
     "khj1222_floor": 0.6115
    }
   },
   "w041": {
    "scroll": "0139",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w041/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w041/w041_inklabels.zarr",
      "shape": [
       28,
       6200,
       8020
      ],
      "chunks": [
       28,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 3,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       49,
       63
      ],
      "chunk_files": 3087,
      "annotated_plane_z": 14,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w041/w041_inklabels.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "267c6654e5a3f9439e4cde3f416fc5a4297a661ce414775276aacdbbfe202201"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w041/w041_inklabels.zarr/.zattrs",
       "bytes": 3743,
       "sha256": "542b064b000c06d8a9aa401fefa3526bff1d3e0faa3124028bb14f8a47f0c47b"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w041/w041_supervision_mask.zarr",
      "shape": [
       28,
       6200,
       8020
      ],
      "chunks": [
       28,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 3,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       49,
       63
      ],
      "chunk_files": 3087,
      "annotated_plane_z": 14,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w041/w041_supervision_mask.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "267c6654e5a3f9439e4cde3f416fc5a4297a661ce414775276aacdbbfe202201"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w041/w041_supervision_mask.zarr/.zattrs",
       "bytes": 3795,
       "sha256": "b6e2884096185e3bfa9b6b30a3f58f0dc7f10d994405efab9c5f57e7ce632f39"
      }
     }
    },
    "eval": {
     "sup_bbox_rows_cols": [
      1836,
      5515,
      558,
      4592
     ],
     "crop_rows_cols_pad128": [
      1664,
      5760,
      384,
      4736
     ],
     "crop_px": 17825792,
     "pos_ink_and_sup": 229884,
     "neg_sup_not_ink": 325745,
     "ink_frac": 0.4137,
     "floor_f1_2p_over_1p": 0.5853,
     "khj1222_scored_px": 554742,
     "khj1222_floor": 0.5859
    }
   },
   "w044": {
    "scroll": "0139",
    "dir": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w044/",
    "stores": {
     "inklabels": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w044/w044_inklabels.zarr",
      "shape": [
       28,
       6040,
       8160
      ],
      "chunks": [
       28,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 3,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       48,
       64
      ],
      "chunk_files": 3072,
      "annotated_plane_z": 14,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w044/w044_inklabels.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "ee134fbb3bf3d522e58fefdaa8976822e791845d99ec2b15e8ed84a747069961"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w044/w044_inklabels.zarr/.zattrs",
       "bytes": 3743,
       "sha256": "0cce62e9da5a9296a97a56ce7c432f8330dd92a91b76b8e271ac04af92eeaf8d"
      }
     },
     "supervision_mask": {
      "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w044/w044_supervision_mask.zarr",
      "shape": [
       28,
       6040,
       8160
      ],
      "chunks": [
       28,
       128,
       128
      ],
      "dtype": "|u1",
      "compressor": {
       "blocksize": 0,
       "clevel": 3,
       "cname": "zstd",
       "id": "blosc",
       "shuffle": 2
      },
      "dimension_separator": ".",
      "chunk_grid_yx": [
       48,
       64
      ],
      "chunk_files": 3072,
      "annotated_plane_z": 14,
      "zarray_0": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w044/w044_supervision_mask.zarr/0/.zarray",
       "bytes": 366,
       "sha256": "ee134fbb3bf3d522e58fefdaa8976822e791845d99ec2b15e8ed84a747069961"
      },
      "zattrs": {
       "url": "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w044/w044_supervision_mask.zarr/.zattrs",
       "bytes": 3797,
       "sha256": "e71698e6616de096ecf8bbfdb972d6530fb930abf26e1c6b0229136531d800be"
      }
     }
    },
    "eval": {
     "sup_bbox_rows_cols": [
      3834,
      4759,
      1854,
      4413
     ],
     "crop_rows_cols_pad128": [
      3584,
      4992,
      1664,
      4608
     ],
     "crop_px": 4145152,
     "pos_ink_and_sup": 451297,
     "neg_sup_not_ink": 739010,
     "ink_frac": 0.3791,
     "floor_f1_2p_over_1p": 0.5498,
     "khj1222_scored_px": 1189591,
     "khj1222_floor": 0.5501
    }
   }
  },
  "measured_inventories": {
   "0814_sup": {
    "name": "0814_sup",
    "path": "ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_supervision_mask.zarr/0",
    "n_files": 460,
    "bytes": 38008,
    "empty_like": 440,
    "nonempty": 20,
    "pages": 1
   },
   "0814_ink": {
    "name": "0814_ink",
    "path": "ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0814-46527/pherc0814-46527_inklabels.zarr/0",
    "n_files": 460,
    "bytes": 39974,
    "empty_like": 427,
    "nonempty": 33,
    "pages": 1
   },
   "w00_sup": {
    "name": "w00_sup",
    "path": "ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w00/phercparis4-w00_supervision_mask.zarr/0",
    "n_files": 6427,
    "bytes": 520066,
    "empty_like": 6308,
    "nonempty": 119,
    "pages": 7
   },
   "w09_sup": {
    "name": "w09_sup",
    "path": "ink_9um/labels/aligned-scrollprizeorg-21slices/phercparis4-w09/phercparis4-w09_supervision_mask.zarr/0",
    "n_files": 9388,
    "bytes": 750301,
    "empty_like": 9248,
    "nonempty": 140,
    "pages": 10
   },
   "native_w035_sup": {
    "name": "native_w035_sup",
    "path": "ink_9um/labels/native9-scrollprizeorg-21slices/w035/w035_supervision_mask.zarr/0",
    "n_files": 1888,
    "bytes": 264996,
    "empty_like": 1765,
    "nonempty": 123,
    "pages": 2
   }
  }
 },
 "training_surface_volumes_s3_level2_sparse": {
  "pherc1667-w013": {
   "scroll": "1667",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304141531-w013_20240304141531_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    10400,
    19900
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": {
    "id": "blosc",
    "clevel": 3
   },
   "dimension_separator": ".",
   "chunk_grid_zyx": [
    1,
    82,
    156
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 12792,
   "bytes_per_chunk_measured_mean": 1603054,
   "full_fetch_GB": 20.5,
   "chunks_planned_sparse": 565,
   "sparse_fetch_GB_est": 0.91,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1784361,
    1783566,
    1780688,
    1773520,
    1781806,
    1774703,
    1785872,
    1772884,
    7272,
    1785872
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304141531-w013_20240304141531_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zattrs",
    "bytes": 3467,
    "sha256": "377f91ef8c5f43c66234e7699176698d34e0f945bccb47a88c916b115e408207"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304141531-w013_20240304141531_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/2/.zarray",
    "bytes": 232,
    "sha256": "746b1818331ab29c1be9a84d121873338b061409194319973f2924a9748eb6c2",
    "shape": [
     109,
     10400,
     19900
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": {
     "id": "blosc",
     "clevel": 3
    },
    "dimension_separator": "."
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304141531-w013_20240304141531_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/pherc1667-w013.zarr",
    "array": "0",
    "shape": [
     21,
     10400,
     19900
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "pherc1667-w018": {
   "scroll": "1667",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    10595,
    24525
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    83,
    192
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 15936,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 28.5,
   "chunks_planned_sparse": 2233,
   "sparse_fetch_GB_est": 3.99,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    null,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zattrs",
    "bytes": 3467,
    "sha256": "05563ff67f5ff9ab414ba4d9d762432983eca2da6ad6bd50322a612a66594d60"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/2/.zarray",
    "bytes": 208,
    "sha256": "01191b2e63f97b9db41a76cd68ebf72b4e4a927c21cdd7f7ecb380cf9cd72e30",
    "shape": [
     109,
     10595,
     24525
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304144031-w018_20240304144031_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/pherc1667-w018.zarr",
    "array": "0",
    "shape": [
     21,
     10595,
     24525
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "pherc1667-w023": {
   "scroll": "1667",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304161941-w023_20240304161941_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    10465,
    25590
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    82,
    200
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 16400,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 29.3,
   "chunks_planned_sparse": 935,
   "sparse_fetch_GB_est": 1.67,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    null,
    null,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304161941-w023_20240304161941_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zattrs",
    "bytes": 3468,
    "sha256": "ffde33ec7da4123dbda2c6e4d4bb2aea158cca0c3871ec23afe58db4fc66b508"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304161941-w023_20240304161941_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/2/.zarray",
    "bytes": 208,
    "sha256": "68eab84c37edec6960ad2ca0989a13507de6c85f29f4873cbba6dc9b0e84bcf6",
    "shape": [
     109,
     10465,
     25590
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20240304161941-w023_20240304161941_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/pherc1667-w023.zarr",
    "array": "0",
    "shape": [
     21,
     10465,
     25590
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "pherc1667-w028": {
   "scroll": "1667",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251208130119-w028_20251208130119156_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    9355,
    7585
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    74,
    60
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 4440,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 7.9,
   "chunks_planned_sparse": 182,
   "sparse_fetch_GB_est": 0.33,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    null,
    null,
    1785856,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251208130119-w028_20251208130119156_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zattrs",
    "bytes": 3467,
    "sha256": "067bf0fab127b2f626c427072163524d49155f9e4d139bf49fe5385746174fbf"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251208130119-w028_20251208130119156_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/2/.zarray",
    "bytes": 206,
    "sha256": "78f11b16a54282c3ca45c04175bc21096956a019d95d7a8d33c3ca6ddc8d6efd",
    "shape": [
     109,
     9355,
     7585
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251208130119-w028_20251208130119156_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/pherc1667-w028.zarr",
    "array": "0",
    "shape": [
     21,
     9355,
     7585
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "pherc1667-w029": {
   "scroll": "1667",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251212185248-w029_20251212185248662_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    9500,
    7830
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    75,
    62
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 4650,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 8.3,
   "chunks_planned_sparse": 299,
   "sparse_fetch_GB_est": 0.53,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    null,
    null,
    null,
    null
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251212185248-w029_20251212185248662_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zattrs",
    "bytes": 3467,
    "sha256": "f36a32c04fb7fe1b045fd45d117fc6991433c6286577d0fb87a1128c8484e414"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251212185248-w029_20251212185248662_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/2/.zarray",
    "bytes": 206,
    "sha256": "6e33b6b5d93db9cc803d66590a5a41efd960abeb46281f5595b6b7f1dc5a8b9a",
    "shape": [
     109,
     9500,
     7830
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251212185248-w029_20251212185248662_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/pherc1667-w029.zarr",
    "array": "0",
    "shape": [
     21,
     9500,
     7830
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "pherc1667-w031": {
   "scroll": "1667",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251223230000-w031_2025122323_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    9370,
    8045
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    74,
    63
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 4662,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 8.3,
   "chunks_planned_sparse": 447,
   "sparse_fetch_GB_est": 0.8,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    null,
    null,
    1785856,
    null
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251223230000-w031_2025122323_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zattrs",
    "bytes": 3467,
    "sha256": "7e9c1685e03409998f1fb9e27ac412d9d96c5ae0e9ef13f97094931cb2d03fcf"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251223230000-w031_2025122323_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/2/.zarray",
    "bytes": 206,
    "sha256": "745c344e47881fd368cdb4c68724126979ee3fadde9a3bca9679ae0fdccc5b5f",
    "shape": [
     109,
     9370,
     8045
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/20251223230000-w031_2025122323_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/pherc1667-w031.zarr",
    "array": "0",
    "shape": [
     21,
     9370,
     8045
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "phercparis4-w00": {
   "scroll": "Paris4",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231016151002/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    7990,
    12990
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    63,
    102
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 6426,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 11.5,
   "chunks_planned_sparse": 583,
   "sparse_fetch_GB_est": 1.04,
   "sparse_plan_source": "HF tree listing of supervision_mask nonempty chunks, +1-chunk dilation",
   "sampled_chunk_bytes": [
    null,
    1785856,
    1785856,
    1785856,
    1785856,
    null,
    1785856,
    1785856,
    1785856,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231016151002/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zattrs",
    "bytes": 3399,
    "sha256": "cc8823194a641db0a487a201b2b333013f325aa49382bb597c43b06d20d45225"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231016151002/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/2/.zarray",
    "bytes": 207,
    "sha256": "245fd92049dfb7b5e2b7cbb674cec584c5df1218df9df063766d1a677c7a610e",
    "shape": [
     109,
     7990,
     12990
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231016151002/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/phercparis4-w00.zarr",
    "array": "0",
    "shape": [
     21,
     7990,
     12990
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "phercparis4-w01": {
   "scroll": "Paris4",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20230702185753/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    12650,
    9100
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    99,
    72
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 7128,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 12.7,
   "chunks_planned_sparse": 1407,
   "sparse_fetch_GB_est": 2.51,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20230702185753/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zattrs",
    "bytes": 3399,
    "sha256": "5155f017823ac6cce5f0dda37a0a330fbfbcd98971a64b4c6b1f42fef6030e0b"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20230702185753/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/2/.zarray",
    "bytes": 207,
    "sha256": "ceda97f8680230e792bc786cc84ac9ee3ab4d8c70ce1204a9563eac85a3dcb50",
    "shape": [
     109,
     12650,
     9100
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20230702185753/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/phercparis4-w01.zarr",
    "array": "0",
    "shape": [
     21,
     12650,
     9100
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "phercparis4-w02": {
   "scroll": "Paris4",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231031143852/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    8545,
    11005
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    67,
    86
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 5762,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 10.3,
   "chunks_planned_sparse": 811,
   "sparse_fetch_GB_est": 1.45,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    null,
    1785856,
    null,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231031143852/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zattrs",
    "bytes": 3399,
    "sha256": "fb2ae133b069c3bc952609b01a814649a767b1ffe31181745e71c24f03a8fd93"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231031143852/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/2/.zarray",
    "bytes": 207,
    "sha256": "95ff2319cd1e00171b725f16edcd352bc8dae2b77025ec1ba9c3fd588b6ec7c5",
    "shape": [
     109,
     8545,
     11005
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231031143852/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/phercparis4-w02.zarr",
    "array": "0",
    "shape": [
     21,
     8545,
     11005
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "phercparis4-w03": {
   "scroll": "Paris4",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231106155351/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    8575,
    12810
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    67,
    101
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 6767,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 12.1,
   "chunks_planned_sparse": 1033,
   "sparse_fetch_GB_est": 1.84,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    null,
    1785856,
    null,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231106155351/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zattrs",
    "bytes": 3399,
    "sha256": "5988437428028cfa072d27bea424ec697645931e50ec574a92907ac47567f97b"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231106155351/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/2/.zarray",
    "bytes": 207,
    "sha256": "b4e25fb78ab621c14aae0aa4b63960069cb8d7b51a7e205ecb08e341779e7b24",
    "shape": [
     109,
     8575,
     12810
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231106155351/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/phercparis4-w03.zarr",
    "array": "0",
    "shape": [
     21,
     8575,
     12810
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "phercparis4-w05": {
   "scroll": "Paris4",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231012184424/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    12700,
    18930
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    100,
    148
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 14800,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 26.4,
   "chunks_planned_sparse": 3044,
   "sparse_fetch_GB_est": 5.44,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231012184424/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zattrs",
    "bytes": 3399,
    "sha256": "776b22d857d03c6c94157e1e8ae0392745fdd6a56d8220fe7eead986141aa184"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231012184424/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/2/.zarray",
    "bytes": 208,
    "sha256": "b2da39cb18ca0382f7b356a368834bf4d6ac95bcb817eb50faefd87152ac787e",
    "shape": [
     109,
     12700,
     18930
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231012184424/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/phercparis4-w05.zarr",
    "array": "0",
    "shape": [
     21,
     12700,
     18930
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "phercparis4-w06": {
   "scroll": "Paris4",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231210121321/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    12750,
    9995
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    100,
    79
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 7900,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 14.1,
   "chunks_planned_sparse": 1748,
   "sparse_fetch_GB_est": 3.12,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231210121321/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zattrs",
    "bytes": 3399,
    "sha256": "cf95285c9f7a7ff4752abf533647d5fca0145daed9e7e0db1d4e6c0ca2da8576"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231210121321/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/2/.zarray",
    "bytes": 207,
    "sha256": "08b54aa77cdc720a892848bcc0a6d0b4976c1968d1702ebcee8949c3b3f77042",
    "shape": [
     109,
     12750,
     9995
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231210121321/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/phercparis4-w06.zarr",
    "array": "0",
    "shape": [
     21,
     12750,
     9995
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "phercparis4-w07": {
   "scroll": "Paris4",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231007101619/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    13010,
    27035
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    102,
    212
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 21624,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 38.6,
   "chunks_planned_sparse": 2475,
   "sparse_fetch_GB_est": 4.42,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    null,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231007101619/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zattrs",
    "bytes": 3400,
    "sha256": "549d42677e49060290cc625669aabf6a30ec2a5306e75e8b06a080b35ec8cd8c"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231007101619/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/2/.zarray",
    "bytes": 208,
    "sha256": "636a6c2eefca42107c4f89a9a207e3bc3fca330da1b859a57d6954ff185c3b80",
    "shape": [
     109,
     13010,
     27035
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231007101619/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/phercparis4-w07.zarr",
    "array": "0",
    "shape": [
     21,
     13010,
     27035
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "phercparis4-w09": {
   "scroll": "Paris4",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20230929220926/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    8020,
    18950
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    63,
    149
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 9387,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 16.8,
   "chunks_planned_sparse": 645,
   "sparse_fetch_GB_est": 1.15,
   "sparse_plan_source": "HF tree listing of supervision_mask nonempty chunks, +1-chunk dilation",
   "sampled_chunk_bytes": [
    null,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    null,
    1785856,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20230929220926/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zattrs",
    "bytes": 3399,
    "sha256": "94ace5a986d22414e621239d292df6aceb8ddebf10dbc16481c9e0b06e1082ba"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20230929220926/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/2/.zarray",
    "bytes": 207,
    "sha256": "964bbfa3713ed5efa6f5f9be3dc9aa2c2ecc28445f071b0270726152be24d605",
    "shape": [
     109,
     8020,
     18950
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20230929220926/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/phercparis4-w09.zarr",
    "array": "0",
    "shape": [
     21,
     8020,
     18950
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  },
  "pherc0814-46527": {
   "scroll": "0814",
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0814/segments/20260226000000-46527_2um_try2/surface-volumes/2.399um-0.22m-78keV-volume-20260309142202.zarr",
   "level": 2,
   "level_scale_um_zyx": [
    2.399,
    9.596,
    9.596
   ],
   "shape": [
    109,
    2130,
    3455
   ],
   "chunks": [
    109,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "chunk_grid_zyx": [
    1,
    17,
    27
   ],
   "z_slice_for_pooling": [
    13,
    97
   ],
   "chunks_total": 459,
   "bytes_per_chunk_measured_mean": 1785856,
   "full_fetch_GB": 0.8,
   "chunks_planned_sparse": 75,
   "sparse_fetch_GB_est": 0.13,
   "sparse_plan_source": "local label plane, supervision max-filter +-128 px",
   "sampled_chunk_bytes": [
    1785856,
    null,
    1785856,
    null,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856,
    1785856
   ],
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0814/segments/20260226000000-46527_2um_try2/surface-volumes/2.399um-0.22m-78keV-volume-20260309142202.zarr/.zattrs",
    "bytes": 3467,
    "sha256": "b2f95a926b15c680a6b8a957a56f08a0633773a5bb38c78a985262fdf29c23db"
   },
   "zarray_level2": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0814/segments/20260226000000-46527_2um_try2/surface-volumes/2.399um-0.22m-78keV-volume-20260309142202.zarr/2/.zarray",
    "bytes": 206,
    "sha256": "fa8593eb1bacda3ef5dd64a4e50641e9e8016d628804db783bb5ab35bab65eb0",
    "shape": [
     109,
     2130,
     3455
    ],
    "chunks": [
     109,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0814/segments/20260226000000-46527_2um_try2/surface-volumes/2.399um-0.22m-78keV-volume-20260309142202.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "pooled_output": {
    "path": "volumes/aligned9/pherc0814-46527.zarr",
    "array": "0",
    "shape": [
     21,
     2130,
     3455
    ],
    "dtype": "uint8",
    "chunks": [
     21,
     128,
     128
    ]
   }
  }
 },
 "eval_native_volumes_s3_level0": {
  "w035": {
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr",
   "level": 0,
   "scale_um": 9.362,
   "shape": [
    28,
    5820,
    5240
   ],
   "chunks": [
    28,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "bytes_per_chunk": 458752,
   "chunks_total": 1886,
   "full_fetch_GB": 0.87,
   "eval_crop_rows_cols": [
    512,
    2944,
    384,
    3072
   ],
   "eval_crop_chunks": 399,
   "eval_crop_fetch_MB": 183.0,
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zattrs",
    "bytes": 3481,
    "sha256": "b95dec627c88cde2f89b5c8b685b73f0ad242838f798d597a2d426909d7fd8f3"
   },
   "zarray_level0": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/0/.zarray",
    "bytes": 204,
    "sha256": "198fe201477e275420a6100c383d382f6312b3c48ad8859bcea6c3aef34a8e5e",
    "shape": [
     28,
     5820,
     5240
    ],
    "chunks": [
     28,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "note": "w035 crop rows 512:2944 cols 384:3072 (399 chunks) is the p2a_v3 control crop: on-record fwd 0.9991 / rev 0.5118 with the released seed42 step-075000 checkpoint"
  },
  "w039": {
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260302000000-w039_2026030210/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr",
   "level": 0,
   "scale_um": 9.362,
   "shape": [
    28,
    8560,
    7720
   ],
   "chunks": [
    28,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "bytes_per_chunk": 458752,
   "chunks_total": 4087,
   "full_fetch_GB": 1.87,
   "eval_crop_rows_cols": [
    3584,
    4992,
    2304,
    5376
   ],
   "eval_crop_chunks": 264,
   "eval_crop_fetch_MB": 121.1,
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260302000000-w039_2026030210/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zattrs",
    "bytes": 3481,
    "sha256": "6f96a330f4a2d1e35d28b746a11d5a51ebbb26be7e0362fc127535cfa97dc145"
   },
   "zarray_level0": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260302000000-w039_2026030210/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/0/.zarray",
    "bytes": 204,
    "sha256": "b3f0eeff1059ba1ac5e7b8e664f3c7512379c87e1e28abeb768ee10d4850fed2",
    "shape": [
     28,
     8560,
     7720
    ],
    "chunks": [
     28,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260302000000-w039_2026030210/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "note": "w035 crop rows 512:2944 cols 384:3072 (399 chunks) is the p2a_v3 control crop: on-record fwd 0.9991 / rev 0.5118 with the released seed42 step-075000 checkpoint"
  },
  "w040": {
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20250831000000-w040_2025083102/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr",
   "level": 0,
   "scale_um": 9.362,
   "shape": [
    28,
    6400,
    7980
   ],
   "chunks": [
    28,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "bytes_per_chunk": 458752,
   "chunks_total": 3150,
   "full_fetch_GB": 1.45,
   "eval_crop_rows_cols": [
    512,
    6144,
    1408,
    7936
   ],
   "eval_crop_chunks": 2244,
   "eval_crop_fetch_MB": 1029.4,
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20250831000000-w040_2025083102/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zattrs",
    "bytes": 3481,
    "sha256": "cde869a04a8b490a960812af1d8a7c2f8eae2d471575da85c4f2b31803fb47d4"
   },
   "zarray_level0": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20250831000000-w040_2025083102/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/0/.zarray",
    "bytes": 204,
    "sha256": "063ef0a16a27b7b4aca77b64b50058ee65387a3881ac2a0603f49cbce380b353",
    "shape": [
     28,
     6400,
     7980
    ],
    "chunks": [
     28,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20250831000000-w040_2025083102/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "note": "w035 crop rows 512:2944 cols 384:3072 (399 chunks) is the p2a_v3 control crop: on-record fwd 0.9991 / rev 0.5118 with the released seed42 step-075000 checkpoint"
  },
  "w041": {
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260108000000-w041_2026010816/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr",
   "level": 0,
   "scale_um": 9.362,
   "shape": [
    28,
    6200,
    8020
   ],
   "chunks": [
    28,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "bytes_per_chunk": 458752,
   "chunks_total": 3087,
   "full_fetch_GB": 1.42,
   "eval_crop_rows_cols": [
    1664,
    5760,
    384,
    4736
   ],
   "eval_crop_chunks": 1088,
   "eval_crop_fetch_MB": 499.1,
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260108000000-w041_2026010816/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zattrs",
    "bytes": 3481,
    "sha256": "800bd4060bc7f7cff48b655a16823fac76a5b2b3f7d1d186943cc2aabba0e4b3"
   },
   "zarray_level0": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260108000000-w041_2026010816/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/0/.zarray",
    "bytes": 204,
    "sha256": "4b6065d478699d3dc225ed72ed7bc4b1e703a9ef73307e9145dd1c538b047ef2",
    "shape": [
     28,
     6200,
     8020
    ],
    "chunks": [
     28,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260108000000-w041_2026010816/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "note": "w035 crop rows 512:2944 cols 384:3072 (399 chunks) is the p2a_v3 control crop: on-record fwd 0.9991 / rev 0.5118 with the released seed42 step-075000 checkpoint"
  },
  "w044": {
   "store": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260115000000-w044_2026011522/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr",
   "level": 0,
   "scale_um": 9.362,
   "shape": [
    28,
    6040,
    8160
   ],
   "chunks": [
    28,
    128,
    128
   ],
   "dtype": "|u1",
   "fill_value": 0,
   "compressor": null,
   "dimension_separator": "/",
   "bytes_per_chunk": 458752,
   "chunks_total": 3072,
   "full_fetch_GB": 1.41,
   "eval_crop_rows_cols": [
    3584,
    4992,
    1664,
    4608
   ],
   "eval_crop_chunks": 253,
   "eval_crop_fetch_MB": 116.1,
   "zattrs": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260115000000-w044_2026011522/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zattrs",
    "bytes": 3481,
    "sha256": "fb8f9872638330a673e4066a0426c14732042337c2817148bb8ff887733d2ad2"
   },
   "zarray_level0": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260115000000-w044_2026011522/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/0/.zarray",
    "bytes": 204,
    "sha256": "64b36b3feb1f064de76c4ebdac6e1b4afaf332cc2aeae66b78aa55d200c21153",
    "shape": [
     28,
     6040,
     8160
    ],
    "chunks": [
     28,
     128,
     128
    ],
    "dtype": "|u1",
    "compressor": null,
    "dimension_separator": "/"
   },
   "zgroup": {
    "url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260115000000-w044_2026011522/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zgroup",
    "bytes": 19,
    "sha256": "2b31f1f542fb152b20ff3afe0da0ed76e15b34cd4320854e2d1e6d8671b29d85"
   },
   "note": "w035 crop rows 512:2944 cols 384:3072 (399 chunks) is the p2a_v3 control crop: on-record fwd 0.9991 / rev 0.5118 with the released seed42 step-075000 checkpoint"
  }
 },
 "totals": {
  "training_level2_chunks_planned": 16482,
  "training_level2_sparse_GB_est": 29.3,
  "training_level2_full_chunks": 139133,
  "training_level2_full_GB": 246.1,
  "eval_native_crop_chunks": 4248,
  "eval_native_crop_GB": 1.95,
  "eval_native_full_GB": 7.01,
  "labels_nonempty_chunk_files_est": "about 2 x 15,300 aligned + 2 x 600 native (skip all-zero chunks by xetHash); 3 metadata files per store",
  "released_checkpoint_MB": 138.4,
  "pod_disk_peak_GB_est": "about 20 env + 1 labels + 27-40 sparse level-2 (delete after pooling) + 8-12 pooled + 2 native crops + 4.4 checkpoints (2 seeds x 16 x 138 MB) -> request 120 GB container disk"
 }
}
JSON_MANIFEST

cat > "$SCRIPTS/make_holdout_config.py" <<'PY_HOLDOUT'
#!/usr/bin/env python3
"""Expand the aligned-21 sampling contract into a runnable training config.

``configs/aligned21_hybrid_3d2d.json`` ships one ``datasets`` entry with
``/path/to/...`` placeholders, and the 29 representations it is meant to train
on live separately in ``configs/aligned21_fixed_scroll_prior.json``. Joining the
two by hand means writing 29 entries, each with its own label directory and
surface-volume path, and keeping ``fixed_scroll_prior.target_batch_counts``
consistent with whatever subset you kept.

This script does that join by locating each representation under the roots you
give it. Holding scrolls or segments out turns the same recipe into a
generalisation probe: nothing on the held-out scroll is ever sampled, so its
whole supervision mask stays honest held-out ground truth. Per-scroll batch
quotas are renormalised over the survivors, because
``FixedScrollPriorStratifiedBatchSampler`` rejects a batch whose quotas do not sum
to ``batch_size``, rejects a non-positive quota, and rejects quota keys that do not
exactly match the scrolls the patches came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_RECIPE = CONFIG_DIR / "aligned21_hybrid_3d2d.json"
DEFAULT_CONTRACT = CONFIG_DIR / "aligned21_fixed_scroll_prior.json"


def as_posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def walk_dirs(root: Path):
    """Yield directories under root without descending into Zarr stores."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = [entry for entry in current.iterdir() if entry.is_dir()]
        except OSError:
            continue
        for entry in entries:
            yield entry
            if entry.suffix != ".zarr":
                stack.append(entry)


def find_label_dir(labels_root: Path, segment: str) -> Path | None:
    """A segment's label directory is named after it and holds its inklabels."""
    for candidate in walk_dirs(labels_root):
        if candidate.name == segment and (candidate / f"{segment}_inklabels.zarr").exists():
            return candidate
    return None


def find_volume(volumes_root: Path, segment: str) -> Path | None:
    """Accept either <segment>.zarr or <segment>/<one>.zarr."""
    holder = None
    for candidate in walk_dirs(volumes_root):
        if candidate.suffix == ".zarr" and candidate.stem == segment:
            return candidate
        if candidate.name == segment and candidate.suffix != ".zarr":
            holder = candidate
    if holder is None:
        return None
    stores = sorted(store for store in holder.iterdir() if store.suffix == ".zarr")
    if len(stores) != 1:
        sys.exit(f"error: expected exactly one *.zarr under {holder}, found {len(stores)}")
    return stores[0]


def renormalise(quotas: dict, keep: set, batch_size: int) -> dict:
    """Spread the recipe's quotas over the surviving scrolls, summing to batch_size."""
    live = {scroll: value for scroll, value in quotas.items() if scroll in keep}
    if not live:
        sys.exit("error: every scroll was excluded")
    if batch_size < len(live):
        # Every survivor is floored to one slot below, because the sampler rejects a
        # zero quota outright, so a smaller batch than the scroll count has no answer.
        sys.exit(f"error: batch_size {batch_size} cannot cover {len(live)} scrolls; "
                 "FixedScrollPriorStratifiedBatchSampler needs at least one slot per "
                 "scroll")
    total = sum(live.values())
    exact = {scroll: batch_size * value / total for scroll, value in live.items()}
    out = {scroll: max(1, int(value)) for scroll, value in exact.items()}
    # Largest-remainder top-up so the quotas land exactly on batch_size.
    order = sorted(live, key=lambda s: exact[s] - int(exact[s]), reverse=True)
    index = 0
    while sum(out.values()) < batch_size:
        out[order[index % len(order)]] += 1
        index += 1
    while sum(out.values()) > batch_size:
        victim = max((s for s in order if out[s] > 1), key=lambda s: out[s])
        out[victim] -= 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels-root", type=Path, required=True,
                        help="Directory the per-segment label folders live under.")
    parser.add_argument("--volumes-root", type=Path, required=True,
                        help="Directory the ~9 um surface volumes live under.")
    parser.add_argument("--out", type=Path, required=True, help="Config path to write.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training out_dir.")
    parser.add_argument("--exclude-scroll", action="append", default=[], metavar="SCROLL",
                        help="Hold out every representation of this scroll. Repeatable.")
    parser.add_argument("--exclude-segment", action="append", default=[], metavar="SEGMENT",
                        help="Hold out one representation by segment name. Repeatable.")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE,
                        help=f"Base training config. Default: {DEFAULT_RECIPE.name}")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT,
                        help=f"Sampling contract. Default: {DEFAULT_CONTRACT.name}")
    parser.add_argument("--seed", type=int, default=None, help="Override seed and sampler seed.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--allow-missing", action="store_true",
                        help="Write the config even if some inputs are not prepared yet.")
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text())
    contract = json.loads(args.contract.read_text())

    excluded_scrolls = set(args.exclude_scroll)
    excluded_segments = set(args.exclude_segment)
    known_scrolls = {rep["scroll"] for rep in contract["representations"]}
    unknown = excluded_scrolls - known_scrolls
    if unknown:
        sys.exit(f"error: unknown scroll(s) {sorted(unknown)}; known: {sorted(known_scrolls)}")
    known_segments = {rep["segment"] for rep in contract["representations"]}
    unknown = excluded_segments - known_segments
    if unknown:
        sys.exit(f"error: unknown segment(s) {sorted(unknown)}")

    kept, dropped = [], []
    for rep in contract["representations"]:
        if rep["scroll"] in excluded_scrolls or rep["segment"] in excluded_segments:
            dropped.append(rep)
        else:
            kept.append(rep)
    if not kept:
        sys.exit("error: the exclusions removed every representation")

    groups = OrderedDict()
    missing = []
    for rep in kept:
        segment = rep["segment"]
        labels = find_label_dir(args.labels_root, segment)
        volume = find_volume(args.volumes_root, segment)
        if labels is None:
            missing.append(f"labels for {segment} under {args.labels_root}")
            continue
        if volume is None:
            missing.append(f"surface volume for {segment} under {args.volumes_root}")
            continue
        entry = groups.setdefault((labels.parent, rep["scroll"]), {
            "segments_path": as_posix(labels.parent),
            "segments": [],
            "surface_volume_paths": {},
            "volume_scale": 0,
            "sampling_scroll": rep["scroll"],
            "sampling_physical_segment_keys": {},
            "sampling_representation_keys": {},
        })
        entry["segments"].append(segment)
        entry["surface_volume_paths"][segment] = as_posix(volume)
        entry["sampling_physical_segment_keys"][segment] = rep["physical_segment_key"]
        entry["sampling_representation_keys"][segment] = rep["representation_key"]

    if missing and not args.allow_missing:
        sys.exit("error: could not locate:\n  " + "\n  ".join(sorted(set(missing))))

    batch_size = args.batch_size or int(recipe["batch_size"])
    seed = args.seed if args.seed is not None else int(recipe["seed"])
    surviving = {entry["sampling_scroll"] for entry in groups.values()}
    if not surviving:
        # Only reachable under --allow-missing: without it the locate failure above exits.
        sys.exit(f"error: none of the {len(kept)} representations could be located under "
                 f"{args.labels_root} and {args.volumes_root}")
    quotas = renormalise(contract["target_batch_counts"], surviving, batch_size)
    held_out = sorted({rep["segment"] for rep in dropped})

    config = dict(recipe)
    config["batch_size"] = batch_size
    config["seed"] = seed
    config["fixed_scroll_prior"] = {"seed": seed, "target_batch_counts": quotas}
    config["out_dir"] = as_posix(args.run_dir)
    config["datasets"] = list(groups.values())
    for key, value in (("num_iterations", args.iterations),
                       ("save_every", args.save_every),
                       ("val_every", args.val_every)):
        if value is not None:
            config[key] = value
    arm = f"held out {sorted(excluded_scrolls) or held_out}" if dropped else "no held-out representations"
    kept_count = sum(len(entry["segments"]) for entry in groups.values())
    config["description"] = (f"{recipe['description'].split('.')[0]}. Arm: {arm}; "
                             f"{kept_count} representations, quotas {quotas}.")
    config["held_out_representations"] = held_out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"  representations : {kept_count} kept, {len(dropped)} held out")
    if dropped:
        print(f"  held out        : {', '.join(held_out)}")
    print(f"  quotas          : {quotas} (batch {batch_size})")
    print(f"  dataset entries : {len(groups)}")
    if missing:
        print(f"  WARNING         : {len(set(missing))} input(s) not located")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY_HOLDOUT

cat > "$SCRIPTS/aligned21_hybrid_3d2d.json" <<'JSON_RECIPE'
{
  "description": "Hybrid local-3D-stem / 2D U-Net on the aligned 21-slice corpus: 17-of-21 jittered Z window, fixed scroll-prior batch sampling, robust-MAD normalization, smoothed BCE + Dice. Fill in out_dir and the datasets block for your data layout; the full corpus contract (29 representations across 4 scrolls) is aligned21_fixed_scroll_prior.json.",
  "mode": "flat",
  "model_type": "vesuvius_unet_3d_stem_2d",
  "model_config": {
    "autoconfigure": true,
    "basic_encoder_block": "BasicBlockD",
    "basic_decoder_block": "ConvBlock",
    "stem_channels": 16,
    "z_projection_mode": "none"
  },
  "targets": {
    "ink": {
      "activation": "none",
      "out_channels": 1,
      "z_projection_mode": "none"
    }
  },
  "in_channels": 1,
  "patch_size": [17, 128, 128],
  "patch_overlap": 0.25,
  "patch_min_labeled_coverage": 0.0,
  "flat_z_window_jitter": {
    "enabled": true,
    "window_depth": 17,
    "max_offset": 2,
    "probability": 1.0,
    "padding": "forbidden"
  },
  "image_normalization": {
    "mode": "robust_mad",
    "percentile_lower": 1.0,
    "percentile_upper": 99.0
  },
  "augmentation_preset": "default",
  "augmentation_rotation_axes": [0],
  "loss": {
    "bce_label_smoothing": 0.5,
    "dice_label_smoothing": 0.0
  },
  "optimizer": "sgd",
  "learning_rate": 0.01,
  "weight_decay": 3e-05,
  "scheduler": {"name": "diffusers_cosine_warmup"},
  "warmup_steps": 1000,
  "grad_clip": 1.0,
  "mixed_precision": "fp16",
  "batch_size": 64,
  "num_iterations": 78125,
  "seed": 42,
  "sampling_strategy": "fixed_scroll_prior_stratified",
  "fixed_scroll_prior": {
    "seed": 42,
    "target_batch_counts": {
      "0139": 29,
      "1667": 22,
      "Paris4": 11,
      "0814": 2
    }
  },
  "sampling_audit_every": 200,
  "best_checkpoint_metric": "val_balanced_accuracy",
  "save_every": 5000,
  "val_every": 5000,
  "val_steps": 32,
  "val_preview_batches": 1,
  "log_every": 50,
  "dataloader_workers": 12,
  "pin_memory": true,
  "verify_finite_gradients_steps": 0,
  "max_amp_overflow_events": 0,
  "out_dir": "/path/to/run-output",
  "datasets": [
    {
      "segments_path": "/path/to/aligned-21slice-data",
      "segments": ["pherc0139-w016"],
      "surface_volume_paths": {
        "pherc0139-w016": "/path/to/aligned-21slice-data/pherc0139-w016/surface-volume.zarr"
      },
      "volume_scale": 0,
      "sampling_scroll": "0139",
      "sampling_physical_segment_keys": {
        "pherc0139-w016": "0139:w016"
      },
      "sampling_representation_keys": {
        "pherc0139-w016": "public_2p4_level2_zmean4:pherc0139-w016"
      }
    }
  ]
}
JSON_RECIPE

cat > "$SCRIPTS/aligned21_fixed_scroll_prior.json" <<'JSON_CONTRACT'
{
  "batch_size": 64,
  "description": "Fixed scroll prior for aligned21. Physical segments are explicit; duplicate public/native representations share one physical-segment budget.",
  "representations": [
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc0139-w016", "scroll": "0139", "physical_segment_key": "0139:w016", "representation_key": "public_2p4_level2_zmean4:pherc0139-w016"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc0139-w017", "scroll": "0139", "physical_segment_key": "0139:w017", "representation_key": "public_2p4_level2_zmean4:pherc0139-w017"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc0139-w028", "scroll": "0139", "physical_segment_key": "0139:w028", "representation_key": "public_2p4_level2_zmean4:pherc0139-w028"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc0139-w029", "scroll": "0139", "physical_segment_key": "0139:w029", "representation_key": "public_2p4_level2_zmean4:pherc0139-w029"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc0139-w035", "scroll": "0139", "physical_segment_key": "0139:w035", "representation_key": "public_2p4_level2_zmean4:pherc0139-w035"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc0139-w039", "scroll": "0139", "physical_segment_key": "0139:w039", "representation_key": "public_2p4_level2_zmean4:pherc0139-w039"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc0139-w040", "scroll": "0139", "physical_segment_key": "0139:w040", "representation_key": "public_2p4_level2_zmean4:pherc0139-w040"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc0139-w041", "scroll": "0139", "physical_segment_key": "0139:w041", "representation_key": "public_2p4_level2_zmean4:pherc0139-w041"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc0139-w043", "scroll": "0139", "physical_segment_key": "0139:w043", "representation_key": "public_2p4_level2_zmean4:pherc0139-w043"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc0814-46527", "scroll": "0814", "physical_segment_key": "0814:46527", "representation_key": "public_2p4_level2_zmean4:pherc0814-46527"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc1667-w013", "scroll": "1667", "physical_segment_key": "1667:w013", "representation_key": "public_2p4_level2_zmean4:pherc1667-w013"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc1667-w018", "scroll": "1667", "physical_segment_key": "1667:w018", "representation_key": "public_2p4_level2_zmean4:pherc1667-w018"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc1667-w023", "scroll": "1667", "physical_segment_key": "1667:w023", "representation_key": "public_2p4_level2_zmean4:pherc1667-w023"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc1667-w028", "scroll": "1667", "physical_segment_key": "1667:w028", "representation_key": "public_2p4_level2_zmean4:pherc1667-w028"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc1667-w029", "scroll": "1667", "physical_segment_key": "1667:w029", "representation_key": "public_2p4_level2_zmean4:pherc1667-w029"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "pherc1667-w031", "scroll": "1667", "physical_segment_key": "1667:w031", "representation_key": "public_2p4_level2_zmean4:pherc1667-w031"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "phercparis4-w00", "scroll": "Paris4", "physical_segment_key": "Paris4:w00", "representation_key": "public_2p4_level2_zmean4:phercparis4-w00"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "phercparis4-w01", "scroll": "Paris4", "physical_segment_key": "Paris4:w01", "representation_key": "public_2p4_level2_zmean4:phercparis4-w01"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "phercparis4-w02", "scroll": "Paris4", "physical_segment_key": "Paris4:w02", "representation_key": "public_2p4_level2_zmean4:phercparis4-w02"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "phercparis4-w03", "scroll": "Paris4", "physical_segment_key": "Paris4:w03", "representation_key": "public_2p4_level2_zmean4:phercparis4-w03"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "phercparis4-w05", "scroll": "Paris4", "physical_segment_key": "Paris4:w05", "representation_key": "public_2p4_level2_zmean4:phercparis4-w05"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "phercparis4-w06", "scroll": "Paris4", "physical_segment_key": "Paris4:w06", "representation_key": "public_2p4_level2_zmean4:phercparis4-w06"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "phercparis4-w07", "scroll": "Paris4", "physical_segment_key": "Paris4:w07", "representation_key": "public_2p4_level2_zmean4:phercparis4-w07"},
    {"source_family": "public_2p4_level2_zmean4", "segment": "phercparis4-w09", "scroll": "Paris4", "physical_segment_key": "Paris4:w09", "representation_key": "public_2p4_level2_zmean4:phercparis4-w09"},
    {"source_family": "native_9p362_level0", "segment": "w035", "scroll": "0139", "physical_segment_key": "0139:w035", "representation_key": "native_9p362_level0:w035"},
    {"source_family": "native_9p362_level0", "segment": "w039", "scroll": "0139", "physical_segment_key": "0139:w039", "representation_key": "native_9p362_level0:w039"},
    {"source_family": "native_9p362_level0", "segment": "w040", "scroll": "0139", "physical_segment_key": "0139:w040", "representation_key": "native_9p362_level0:w040"},
    {"source_family": "native_9p362_level0", "segment": "w041", "scroll": "0139", "physical_segment_key": "0139:w041", "representation_key": "native_9p362_level0:w041"},
    {"source_family": "native_9p362_level0", "segment": "w044", "scroll": "0139", "physical_segment_key": "0139:w044", "representation_key": "native_9p362_level0:w044"}
  ],
  "schema_version": 1,
  "seed": 42,
  "strategy": "fixed_scroll_prior_stratified",
  "target_batch_counts": {
    "0139": 29,
    "1667": 22,
    "Paris4": 11,
    "0814": 2
  }
}
JSON_CONTRACT

cat > "$SCRIPTS/hfsync.py" <<'PY_HFSYNC'
"""Mirror ink_9um label stores from the HF bucket, skipping the all-zero chunks.

  python hfsync.py <family> <segment> [<segment> ...]
     family = aligned-scrollprizeorg-21slices | native9-scrollprizeorg-21slices

For every store of the segment named in the manifest (inklabels, supervision_mask,
validation_mask if present): list the bucket tree (paginated via the Link header),
download every file whose xetHash is not the all-zero chunk, preserving the store
layout under $LABELS/<family>/<segment>/; verify .zattrs / .zgroup / 0/.zarray
sha256 against the manifest; report the annotated-plane non-zero count."""
import hashlib, json, os, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

MAN = json.load(open(os.path.join(os.environ["SCRIPTS"], "manifest.json")))
LABELS = os.environ["LABELS"]
TREE = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/"
RESOLVE = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
ZERO_HASHES = {MAN["conventions"]["aligned_label_all_zero_chunk"]["xetHash"],
               MAN["conventions"]["native_label_all_zero_chunk"]["xetHash"]}
UA = {"User-Agent": "curl/8"}
THREADS = int(os.environ.get("FETCH_THREADS", "32"))


def http(url, tries=5, timeout=120):
    waits = [0, 3, 10, 30, 60]
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, {}
            if e.code == 429 or i == tries - 1:
                if i == tries - 1:
                    raise
            time.sleep(waits[i + 1] * (2 if e.code == 429 else 1))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(waits[i + 1])


def list_tree(path):
    url = TREE + path + "?limit=1000&recursive=true"
    out = []
    while url:
        body, hdr = http(url, timeout=180)
        if body is None:
            raise RuntimeError(f"tree 404: {path}")
        out.extend(json.loads(body.decode()))
        link = hdr.get("Link") or hdr.get("link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return [e for e in out if e.get("type") == "file"]


def sync_store(family, seg, store_name, expect):
    rel_store = f"ink_9um/labels/{family}/{seg}/{seg}_{store_name}.zarr"
    local_store = os.path.join(LABELS, family, seg, f"{seg}_{store_name}.zarr")
    entries = list_tree(rel_store)
    keep = [e for e in entries if e.get("xetHash") not in ZERO_HASHES]
    skipped = len(entries) - len(keep)
    todo = []
    for e in keep:
        rel = e["path"][len(rel_store) + 1:]
        dest = os.path.join(local_store, rel)
        if os.path.exists(dest) and os.path.getsize(dest) == e["size"]:
            continue
        todo.append((e["path"], dest, e["size"]))
    cl.say(f"HFSYNC {seg}/{store_name}: {len(entries)} files listed, {skipped} all-zero skipped, "
           f"{len(todo)} to fetch")
    done = [0]
    failed = []

    def one(item):
        path, dest, size = item
        try:
            body = None
            for attempt in range(4):
                body, _ = http(RESOLVE + path)
                if body is not None:
                    break
                time.sleep(5 * (attempt + 1))       # listed-but-404: xet propagation lag, retry
            if body is None:
                raise RuntimeError(f"404 on listed file {path}")
            if len(body) != size:
                raise RuntimeError(f"size mismatch {path}: {len(body)} != {size}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, dest)
            done[0] += 1
            if done[0] % 500 == 0:
                cl.say(f"HFSYNC {seg}/{store_name}: {done[0]}/{len(todo)}")
        except Exception as e:                      # collect; retried below at low concurrency
            failed.append((item, f"{type(e).__name__}: {str(e)[:160]}"))
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        list(ex.map(one, todo))
    if failed:
        cl.say(f"HFSYNC {seg}/{store_name}: {len(failed)} files failed at {THREADS} threads "
               f"(first: {failed[0][1]}); retrying them at 4 threads after 30 s")
        time.sleep(30)
        retry_items = [f[0] for f in failed]
        failed.clear()
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(one, retry_items))
        if failed:
            cl.say(f"HFSYNC {seg}/{store_name}: STILL FAILING {len(failed)}: {failed[0][1]}")
            raise RuntimeError(f"{seg}/{store_name}: {len(failed)} files could not be fetched")
    # metadata gates
    for fn, key in ((".zattrs", "zattrs"), ("0/.zarray", "zarray_0")):
        p = os.path.join(local_store, fn)
        assert os.path.exists(p), f"{seg}/{store_name}: missing {fn}"
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        exp = expect.get(key, {}).get("sha256")
        if exp:
            assert got == exp, f"{seg}/{store_name}/{fn} sha256 {got[:12]} != manifest {exp[:12]}"
    assert os.path.exists(os.path.join(local_store, ".zgroup")), f"{seg}/{store_name}: missing .zgroup"
    return local_store, len(entries), skipped


def plane_stats(local_store, z):
    import zarr
    a = zarr.open(local_store, mode="r")["0"]
    plane = a[z]
    return int((plane > 0).sum()), list(plane.shape)


def main():
    family = sys.argv[1]
    segs = sys.argv[2:]
    block = MAN["labels_hf"]["kept_aligned"] if family.startswith("aligned") else MAN["labels_hf"]["heldout_native_eval"]
    report = {}
    for seg in segs:
        spec = block[seg]
        rep = {}
        for store_name, expect in spec["stores"].items():
            if expect is None:
                continue
            local_store, n, skipped = sync_store(family, seg, store_name, expect)
            zplane = int(expect.get("annotated_plane_z", 10))
            nz, shape = plane_stats(local_store, zplane)
            rep[store_name] = dict(files=n, skipped_zero=skipped, plane_z=zplane, plane_nonzero=nz, shape=shape)
            cl.say(f"HFSYNC {seg}/{store_name}: OK sha256 metadata; plane z={zplane} nonzero={nz} shape={shape}")
        assert rep["inklabels"]["shape"] == rep["supervision_mask"]["shape"], (seg, rep)
        report[seg] = rep
    p = os.path.join(cl.RESULTS, f"labels_{family.split('-')[0]}.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(report)
    json.dump(old, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
PY_HFSYNC

cat > "$SCRIPTS/svplan.py" <<'PY_SVPLAN'
"""Sparse level-2 source-volume fetch for the 15 kept representations.
  python svplan.py plan            -> out/sv_plan.json (chunk columns per rep, gated vs manifest)
  python svplan.py fetch <seg>...  -> $DATA/level2/<seg>.zarr (group: .zgroup .zattrs 2/.zarray + chunks)
  python svplan.py check <seg>     -> pooled volume gates (shape == label; 20 supervised patches non-zero)

The trainer picks patches from the supervision mask only (corner = (y//32*32, x//32*32),
128x128), never from the image, so fetching only the chunk columns those patches touch
(+1 chunk margin) is exact. Absent chunks (HTTP 404) are fill_value 0 and are NOT written
(zarr reads fill for missing keys); they are counted and capped at 60% per store."""
import hashlib, json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

MAN = json.load(open(os.path.join(os.environ["SCRIPTS"], "manifest.json")))
SV = MAN["training_surface_volumes_s3_level2_sparse"]
LAB = MAN["labels_hf"]["kept_aligned"]
LABELS, DATA, VOLS = os.environ["LABELS"], os.environ["DATA"], os.environ["VOLS"]
L2 = os.path.join(DATA, "level2")
UA = {"User-Agent": "curl/8"}
THREADS = int(os.environ.get("FETCH_THREADS", "32"))
CH = 128


def http(url, tries=5, timeout=300):
    waits = [0, 3, 10, 30, 60]
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
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


def sup_plane(seg):
    import zarr
    st = os.path.join(LABELS, "aligned-scrollprizeorg-21slices", seg, f"{seg}_supervision_mask.zarr")
    a = zarr.open(st, mode="r")["0"]
    z = int(LAB[seg]["stores"]["supervision_mask"].get("annotated_plane_z", 10))
    return np.asarray(a[z]) > 0


def plan_one(seg):
    sup = sup_plane(seg)
    ys, xs = np.nonzero(sup)
    if len(ys) == 0:
        return [], 0
    corners = np.unique(np.stack([ys // 32 * 32, xs // 32 * 32], 1), axis=0)
    cols = set()
    for cy0, cx0 in corners:
        for cy in range(cy0 // CH, (cy0 + 127) // CH + 1):
            for cx in range(cx0 // CH, (cx0 + 127) // CH + 1):
                cols.add((int(cy), int(cx)))
    H, W = sup.shape
    gy, gx = (H + CH - 1) // CH, (W + CH - 1) // CH
    dil = set()
    for cy, cx in cols:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = cy + dy, cx + dx
                if 0 <= y < gy and 0 <= x < gx:
                    dil.add((y, x))
    return sorted(dil), len(corners)


def plan():
    out, total = {}, 0
    for seg in SV:
        cols, ncorner = plan_one(seg)
        exp = int(SV[seg]["chunks_planned_sparse"])
        # Our rule (patch corners +128 px, then +1 chunk of dilation) is systematically
        # ~1.15-1.26x the manifest's "+-128 px max-filter" estimate (measured on the
        # 2026-09-03 smoke: w013 1.17, w018 1.18, w023 1.19, w028 1.18, w029 1.26).
        # The band is a sanity check against a broken plan, not a budget: the budget
        # is the 45 GB total cap below. Smoke #4 (2026-09-03) died here on
        # phercparis4-w00 at 1.80x (the manifest's estimate is poor for the Paris4
        # geometry) after 59 min of label sync, so a miss is now a logged WARNING;
        # only a grossly broken plan (>4x or <0.25x) or the total cap is fatal.
        lo, hi = 0.5 * exp, 1.6 * exp
        ok = lo <= len(cols) <= hi
        sane = 0.25 * exp <= len(cols) <= 4.0 * exp
        cl.say(f"SVPLAN {seg}: {ncorner} patch corners -> {len(cols)} chunk columns "
               f"(manifest {exp}; {'OK' if ok else ('WARN out of the 0.5-1.6x band, ratio %.2f' % (len(cols) / max(exp, 1)))})")
        assert sane, f"{seg}: planned {len(cols)} outside the 0.25-4x sanity bound of the manifest count {exp}"
        out[seg] = dict(chunks=[list(c) for c in cols], n=len(cols), manifest=exp,
                        gb_est=round(len(cols) * SV[seg]["bytes_per_chunk_measured_mean"] / 1e9, 2))
        total += out[seg]["gb_est"]
    assert total <= 45.0, f"planned {total:.1f} GB > 45 GB cap"
    json.dump(out, open(os.path.join(cl.OUT, "sv_plan.json"), "w"), indent=0)
    cl.say(f"SVPLAN total {sum(v['n'] for v in out.values())} chunks, ~{total:.1f} GB (cap 45)")


def fetch(seg):
    spec = SV[seg]
    store, sep = spec["store"], spec["dimension_separator"]
    local = os.path.join(L2, f"{seg}.zarr")
    os.makedirs(os.path.join(local, "2"), exist_ok=True)
    for fn, key in ((".zgroup", "zgroup"), (".zattrs", "zattrs"), ("2/.zarray", "zarray_level2")):
        b = http(f"{store}/{fn}")
        assert b is not None, f"{seg}: {fn} missing on S3"
        got = hashlib.sha256(b).hexdigest()
        assert got == spec[key]["sha256"], f"{seg}/{fn} sha256 {got[:12]} != manifest {spec[key]['sha256'][:12]}"
        open(os.path.join(local, fn), "wb").write(b)
    plan_ = json.load(open(os.path.join(cl.OUT, "sv_plan.json")))[seg]["chunks"]
    absent_p = os.path.join(local, "absent.json")
    absent = set(tuple(a) for a in json.load(open(absent_p))) if os.path.exists(absent_p) else set()

    def dest_of(cy, cx):
        key = sep.join(["0", str(cy), str(cx)])
        return os.path.join(local, "2", *key.split("/")) if sep == "/" else os.path.join(local, "2", key), key
    todo = [(cy, cx) for cy, cx in plan_ if (cy, cx) not in absent and not os.path.exists(dest_of(cy, cx)[0])]
    cl.say(f"SVFETCH {seg}: {len(plan_)} planned, {len(todo)} to fetch, {len(absent)} known absent (sep '{sep}')")
    done = [0]; miss = []

    def one(c):
        cy, cx = c
        dest, key = dest_of(cy, cx)
        b = http(f"{store}/2/{key}")
        if b is None:
            miss.append((cy, cx))
        else:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(b)
            os.replace(tmp, dest)
        done[0] += 1
        if done[0] % 200 == 0:
            cl.say(f"SVFETCH {seg}: {done[0]}/{len(todo)}")
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        list(ex.map(one, todo))
    absent |= set(miss)
    json.dump(sorted(list(a) for a in absent), open(absent_p, "w"))
    frac = len(absent) / max(1, len(plan_))
    cl.say(f"SVFETCH {seg}: done; absent {len(absent)}/{len(plan_)} = {frac:.1%}")
    assert frac < 0.60, f"{seg}: {frac:.0%} of planned chunks absent -- store or plan wrong"


def check(seg):
    import zarr
    lab = LAB[seg]["stores"]["inklabels"]["shape"]
    pooled = zarr.open(os.path.join(VOLS, "aligned9", f"{seg}.zarr"), mode="r")["0"]
    assert list(pooled.shape) == list(lab), f"{seg}: pooled {pooled.shape} != label {lab}"
    sup = sup_plane(seg)
    ys, xs = np.nonzero(sup)
    rng = np.random.default_rng(cl.SEED)
    idx = rng.choice(len(ys), size=min(20, len(ys)), replace=False)
    zero = 0
    for i in idx:
        y0, x0 = int(ys[i]) // 32 * 32, int(xs[i]) // 32 * 32
        patch = np.asarray(pooled[2:19, y0:y0 + 128, x0:x0 + 128])
        if patch.max() == 0:
            zero += 1
    cl.say(f"SVCHECK {seg}: pooled shape OK {list(pooled.shape)}; {zero}/{len(idx)} sampled supervised patches all-zero")
    assert zero == 0, f"{seg}: {zero} supervised patches read as zeros after pooling (fetch plan miss)"


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd == "fetch":
        for s in sys.argv[2:]:
            fetch(s)
    elif cmd == "check":
        for s in sys.argv[2:]:
            check(s)
PY_SVPLAN

cat > "$SCRIPTS/natfetch.py" <<'PY_NATFETCH'
"""Native PHerc0139 eval crops (the held-out tier): for each of w035/w039/w040/w041/w044,
crop = supervision bbox (plane z=14) padded to 128-multiples (w035 forced to the p2a_v3
control crop rows 512:2944 cols 384:3072), fetched as raw level-0 chunks from S3 and
written as a 28-layer zarr; the crop's ink/sup label planes saved as npy."""
import json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

MAN = json.load(open(os.path.join(os.environ["SCRIPTS"], "manifest.json")))
EV = MAN["eval_native_volumes_s3_level0"]
LAB = MAN["labels_hf"]["heldout_native_eval"]
LABELS, NATIVE = os.environ["LABELS"], os.environ["NATIVE"]
UA = {"User-Agent": "curl/8"}
THREADS = int(os.environ.get("FETCH_THREADS", "32"))
CH = 128


def http(url, tries=5, timeout=300):
    waits = [0, 3, 10, 30, 60]
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
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


def label_planes(w):
    import zarr
    fam = "native9-scrollprizeorg-21slices"
    z = int(LAB[w]["stores"]["inklabels"].get("annotated_plane_z", 14))
    ink = np.asarray(zarr.open(os.path.join(LABELS, fam, w, f"{w}_inklabels.zarr"), mode="r")["0"][z]) > 0
    sup = np.asarray(zarr.open(os.path.join(LABELS, fam, w, f"{w}_supervision_mask.zarr"), mode="r")["0"][z]) > 0
    return ink, sup


def crop_for(w, sup):
    if w == "w035":
        return tuple(EV["w035"]["eval_crop_rows_cols"])
    ys, xs = np.nonzero(sup)
    H, W = sup.shape
    y0 = max(0, (int(ys.min()) - 128) // CH * CH); x0 = max(0, (int(xs.min()) - 128) // CH * CH)
    y1 = min(H, -(-(int(ys.max()) + 129) // CH) * CH); x1 = min(W, -(-(int(xs.max()) + 129) // CH) * CH)
    return (y0, y1, x0, x1)


def build(w):
    spec = EV[w]
    store, nz = spec["store"], spec["shape"][0]
    ink, sup = label_planes(w)
    y0, y1, x0, x1 = crop_for(w, sup)
    Hc, Wc = y1 - y0, x1 - x0
    keys = [(cy, cx) for cy in range(y0 // CH, (y1 - 1) // CH + 1) for cx in range(x0 // CH, (x1 - 1) // CH + 1)]
    cdir = os.path.join(NATIVE, f"{w}_chunks"); os.makedirs(cdir, exist_ok=True)
    todo = [k for k in keys if not os.path.exists(os.path.join(cdir, f"{k[0]}_{k[1]}"))]
    cl.say(f"NATIVE {w}: crop rows {y0}:{y1} cols {x0}:{x1} ({Hc}x{Wc}), {len(keys)} chunks, {len(todo)} to fetch")
    absent = [0]

    def one(k):
        cy, cx = k
        b = http(f"{store}/0/0/{cy}/{cx}")
        tmp = os.path.join(cdir, f"{cy}_{cx}.part")
        if b is None:
            absent[0] += 1; open(tmp, "wb").close()
        else:
            assert len(b) == spec["bytes_per_chunk"], (w, k, len(b))
            open(tmp, "wb").write(b)
        os.replace(tmp, os.path.join(cdir, f"{cy}_{cx}"))
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        list(ex.map(one, todo))
    vol = np.zeros((nz, Hc, Wc), np.uint8)
    for cy, cx in keys:
        raw = open(os.path.join(cdir, f"{cy}_{cx}"), "rb").read()
        if not raw:
            continue
        arr = np.frombuffer(raw, np.uint8).reshape(nz, CH, CH)
        ys, xs = cy * CH - y0, cx * CH - x0
        ty0, ty1, tx0, tx1 = max(0, ys), min(Hc, ys + CH), max(0, xs), min(Wc, xs + CH)
        vol[:, ty0:ty1, tx0:tx1] = arr[:, ty0 - ys:ty1 - ys, tx0 - xs:tx1 - xs]
    zp = os.path.join(NATIVE, f"{w}_crop.zarr")
    if not os.path.exists(zp):
        cl.write_group_zarr(zp, vol)
    ci, cs = ink[y0:y1, x0:x1], sup[y0:y1, x0:x1]
    np.save(os.path.join(NATIVE, f"{w}_ink.npy"), ci); np.save(os.path.join(NATIVE, f"{w}_sup.npy"), cs)
    rec = dict(crop=[y0, y1, x0, x1], shape=[nz, Hc, Wc], n_pos=int((ci & cs).sum()), n_neg=int((cs & ~ci).sum()),
               zero_frac=float((vol[nz // 2] == 0).mean()), absent_chunks=absent[0])
    cl.say(f"NATIVE {w}: zarr ready {rec['shape']}, pos={rec['n_pos']} neg={rec['n_neg']} zero_frac={rec['zero_frac']:.3f}")
    assert rec["n_pos"] > 1000 and rec["zero_frac"] < 0.5, (w, rec)
    return rec


if __name__ == "__main__":
    res = {}
    for w in sys.argv[1:]:
        res[w] = build(w)
    p = os.path.join(cl.RESULTS, "native_crops.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(res); json.dump(old, open(p, "w"), indent=1)
PY_NATFETCH

cat > "$SCRIPTS/cfggen.py" <<'PY_CFGGEN'
"""Training configs.
  python cfggen.py synthetic          -> out/cfg/syn.json (pherc0814 real labels + random volume, 30 it)
  python cfggen.py loso <seed> <steps> <save_every> -> out/cfg/loso_s<seed>.json via make_holdout_config.py
"""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

S = os.environ["SCRIPTS"]; OUT = cl.OUT; LABELS = os.environ["LABELS"]; VOLS = os.environ["VOLS"]; RUNS = os.environ["RUNS"]
RECIPE = os.path.join(S, "aligned21_hybrid_3d2d.json")
CONTRACT = os.path.join(S, "aligned21_fixed_scroll_prior.json")
CFG = os.path.join(OUT, "cfg"); os.makedirs(CFG, exist_ok=True)
NPROC = os.cpu_count() or 8
ARM = int(os.environ.get("ARM", "0"))


def arm_keys():
    """Bet A arm config keys (trackD PREREG_BET_A): 0 -> none (recipe byte-identical),
    1 -> input_degradation from the k2b index, 2 -> input_whitening."""
    if ARM == 1:
        idx = json.load(open(os.path.join(S, "k2b_index.json")))
        # Calibrate the 3-D index targets to the 2-D per-crop estimator on the one scroll present in both:
        # PHerc0139 (native-5 eval crops measured by the measure stage vs the index's PHerc0139 medians).
        scale, cal = (1.0, 1.0, 1.0), {"status": "uncalibrated (input_stats.json or PHerc0139 missing)"}
        sp = os.path.join(cl.RESULTS, "input_stats.json")
        if os.path.exists(sp) and "PHerc0139" in idx:
            st = json.load(open(sp)); nat = [v for v in st.get("native", {}).values() if v.get("n")]
            if nat:
                med = lambda key: float(np.median([v[key][0] for v in nat]))
                ref = idx["PHerc0139"]
                scale = (med("bandwidth_med_iqr") / ref["bandwidth_med_iqr"][0], med("snr_q025_med_iqr") / ref["snr_q025_med_iqr"][0],
                         med("dn_headroom_med_iqr") / ref["dn_headroom_med_iqr"][0])
                cal = {"status": "calibrated on PHerc0139 native-5 crops (2-D) vs index PHerc0139 (3-D)", "native_2d": [med("bandwidth_med_iqr"), med("snr_q025_med_iqr"), med("dn_headroom_med_iqr")],
                       "index_3d": [ref["bandwidth_med_iqr"][0], ref["snr_q025_med_iqr"][0], ref["dn_headroom_med_iqr"][0]], "n_native_stores": len(nat)}
        cl.say(f"CFG arm 1 target_scale (bw, snr, headroom) = {tuple(round(x, 4) for x in scale)} -- {cal['status']}")
        return {"input_degradation": {"enabled": True, "index": idx, "probability": 1.0, "target_scale": list(scale), "calibration": cal,
                                      "apply_blur": True, "apply_noise": True, "apply_headroom": True, "seed": 1}}
    if ARM == 2:
        return {"input_whitening": {"enabled": True, "n_samples": 64, "sample_size": 128, "q_ref": 0.02, "max_gain": 8.0, "seed": 2}}
    return {}


def synthetic():
    import zarr
    seg = "pherc0814-46527"
    fam = "aligned-scrollprizeorg-21slices"
    lab = zarr.open(os.path.join(LABELS, fam, seg, f"{seg}_inklabels.zarr"), mode="r")["0"]
    shape = tuple(int(x) for x in lab.shape)
    vol_dir = os.path.join(cl.DATA, "syn", "volumes", "aligned9"); os.makedirs(vol_dir, exist_ok=True)
    vp = os.path.join(vol_dir, f"{seg}.zarr")
    if not os.path.exists(vp):
        rng = np.random.default_rng(1)
        v = rng.integers(40, 200, size=shape, dtype=np.uint8)
        cl.write_group_zarr(vp, v)
    r = json.load(open(RECIPE))
    r.pop("fixed_scroll_prior", None)
    r.update(sampling_strategy="uniform", num_iterations=30, save_every=30, val_every=30, val_steps=2,
             batch_size=8, dataloader_workers=min(2, max(0, NPROC - 2)), warmup_steps=5, log_every=5,
             out_dir=os.path.join(RUNS, "syn"), seed=42,
             datasets=[{"segments_path": os.path.join(LABELS, fam), "segments": [seg],
                        "surface_volume_paths": {seg: vp}, "volume_scale": 0,
                        "sampling_scroll": "0814",
                        "sampling_physical_segment_keys": {seg: "0814:46527"},
                        "sampling_representation_keys": {seg: f"public_2p4_level2_zmean4:{seg}"}}])
    r.update(arm_keys())
    p = os.path.join(CFG, "syn.json"); json.dump(r, open(p, "w"), indent=1)
    cl.say(f"CFG synthetic: {seg} labels ({shape}) + random volume, 30 iterations, batch 8, arm {ARM} -> {p}")


def loso(seed, steps, save_every):
    out = os.path.join(CFG, f"loso_s{seed}.json")
    run_dir = os.path.join(RUNS, f"s{seed}")
    cmd = [sys.executable, os.path.join(S, "make_holdout_config.py"), "--labels-root", os.path.join(LABELS, "aligned-scrollprizeorg-21slices"),
           "--volumes-root", os.path.join(VOLS, "aligned9"), "--exclude-scroll", "0139", "--seed", str(seed),
           "--recipe", RECIPE, "--contract", CONTRACT, "--out", out, "--run-dir", run_dir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    cl.say("CFG generator: " + " | ".join(l.strip() for l in (r.stdout + r.stderr).strip().splitlines()[-5:]))
    assert r.returncode == 0, "make_holdout_config failed"
    c = json.load(open(out))
    q = c["fixed_scroll_prior"]["target_batch_counts"]
    assert q == {"1667": 40, "Paris4": 20, "0814": 4}, q
    kept = sorted(s for d in c["datasets"] for s in d["segments"])
    assert len(kept) == 15 and len(c["datasets"]) == 3, (len(kept), len(c["datasets"]))
    for d in c["datasets"]:
        for s, vp in d["surface_volume_paths"].items():
            assert os.path.exists(vp), f"volume missing for {s}: {vp}"
    c.update(num_iterations=int(steps), save_every=int(save_every), val_every=int(save_every),
             dataloader_workers=min(12, max(2, NPROC - 2)), out_dir=run_dir, seed=int(seed))
    c["fixed_scroll_prior"]["seed"] = int(seed)
    c.update(arm_keys())
    json.dump(c, open(out, "w"), indent=1)
    cl.say(f"CFG loso seed {seed}: 15 kept, quotas {q}, {steps} iterations, save every {save_every}, "
           f"workers {c['dataloader_workers']}, ARM {ARM} ({', '.join(arm_keys().keys()) or 'baseline'}) -> {out}")


if __name__ == "__main__":
    if sys.argv[1] == "synthetic":
        synthetic()
    else:
        loso(sys.argv[2], sys.argv[3], sys.argv[4])
PY_CFGGEN

cat > "$SCRIPTS/evalf1.py" <<'PY_EVALF1'
"""Score checkpoints on the five native PHerc0139 crops.
  python evalf1.py <tag> <ckpt.pth> [--reverse]
For each crop: run infer (forward; reverse too if asked), then
  * khj1222 replica: 256-bin histograms of the uint8 prediction over the supervision
    region split by ink; F1 at every threshold t (positive iff score >= t); best F1;
    floor = 2p/(1+p); margin = best - floor;
  * benchmark: tie-corrected pixel AUC (curvelib.hist_auc), pos = ink & sup, neg = sup & ~ink.
Appends to results/eval.json under <tag>."""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

NATIVE, PREDS = os.environ["NATIVE"], cl.PREDS
CROPS = ["w035", "w039", "w040", "w041", "w044"]


def infer(zarr_path, ckpt, out_tif, direction):
    if os.path.exists(out_tif) and os.path.getsize(out_tif) > 0:
        return
    cmd = ["uv", "run", "--no-sync", "--extra", "models", "python", "-m", "vesuvius.ink_detection.inference.infer",
           zarr_path, ckpt, out_tif, "--direction", direction, "--batch-size", os.environ.get("BATCH", "16"),
           "--num-workers", os.environ.get("WORKERS", "8"), "--gpus", "0", "--no-compile"]
    r = subprocess.run(cmd, cwd="/workspace/villa/vesuvius", capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_tif):
        cl.say("INFER FAILED: " + (r.stderr or r.stdout)[-600:].replace("\n", " | "))
        raise RuntimeError(f"infer failed for {out_tif}")


def f1_sweep(pred, ink, sup):
    q = pred.astype(np.int64)
    hp = np.bincount(q[ink & sup], minlength=256).astype(np.float64)
    hn = np.bincount(q[sup & ~ink], minlength=256).astype(np.float64)
    P = hp.sum(); N = hn.sum()
    tp = np.cumsum(hp[::-1])[::-1]          # positives with score >= t
    fp = np.cumsum(hn[::-1])[::-1]
    fn = P - tp
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    t = int(np.argmax(f1))
    p_ink = P / max(P + N, 1)
    floor = 2 * p_ink / (1 + p_ink)
    return dict(best_f1=float(f1[t]), threshold=t, floor=float(floor), margin=float(f1[t] - floor),
                n_pos=int(P), n_neg=int(N))


def main():
    tag, ckpt = sys.argv[1], sys.argv[2]
    do_rev = "--reverse" in sys.argv
    import tifffile
    p = os.path.join(cl.RESULTS, "eval.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    row = res.get(tag, {"ckpt": ckpt, "crops": {}})
    for w in CROPS:
        z = os.path.join(NATIVE, f"{w}_crop.zarr")
        ink = np.load(os.path.join(NATIVE, f"{w}_ink.npy")); sup = np.load(os.path.join(NATIVE, f"{w}_sup.npy"))
        cell = {}
        for d, suffix in (("forward", ""), ("reverse", "_reverse")):
            if d == "reverse" and not do_rev:
                continue
            tif = os.path.join(PREDS, f"{tag}_{w}{suffix}.tif")
            if d == "forward":
                infer(z, ckpt, tif, "forward")
            else:
                infer(z, ckpt, os.path.join(PREDS, f"{tag}_{w}_r.tif"), "reverse")
                tif = os.path.join(PREDS, f"{tag}_{w}_r.tif")
            pred = tifffile.imread(tif)
            if pred.shape != ink.shape:
                pred = np.clip(np.rint(cl.resample_pred(pred, ink.shape)), 0, 255).astype(np.uint8)
            f1 = f1_sweep(pred, ink, sup)
            q = cl.quantize_map(pred.astype(np.float32))
            auc = cl.hist_auc(cl.masked_hist(q, ink & sup), cl.masked_hist(q, sup & ~ink))
            cell[d] = dict(f1=f1, auc=float(auc))
            cl.say(f"EVAL {tag} {w} {d}: bestF1={f1['best_f1']:.4f}@{f1['threshold']} floor={f1['floor']:.3f} "
                   f"margin={f1['margin']:+.3f} AUC={auc:.4f}")
        row["crops"][w] = cell
    f = [row["crops"][w]["forward"]["f1"] for w in CROPS]
    row["native5_mean_best_f1"] = float(np.mean([x["best_f1"] for x in f]))
    row["native5_mean_floor"] = float(np.mean([x["floor"] for x in f]))
    row["native5_mean_margin"] = float(np.mean([x["margin"] for x in f]))
    row["native5_mean_auc_forward"] = float(np.mean([row["crops"][w]["forward"]["auc"] for w in CROPS]))
    res[tag] = row
    json.dump(res, open(p, "w"), indent=1)
    cl.say(f"EVAL {tag}: native-5 mean bestF1={row['native5_mean_best_f1']:.4f} (floor {row['native5_mean_floor']:.3f}, "
           f"margin {row['native5_mean_margin']:+.3f}); mean AUC fwd={row['native5_mean_auc_forward']:.4f}")


if __name__ == "__main__":
    main()
PY_EVALF1

cat > "$SCRIPTS/finalize.py" <<'PY_FINAL'
"""Aggregate results, verify the inventory, apply the prereg gate when both seeds are
complete (never in SMOKE_ONLY), write results/results.json and out/results.json."""
import glob, json, os, re, sys, time
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR, RUNS = os.environ["VAR"], os.environ["RUNS"]
SMOKE = os.environ.get("SMOKE_ONLY", "1") == "1"
SEEDS = os.environ.get("SEEDS", "42 43").split()


def train_summary(seed):
    log = os.path.join(cl.OUT, "logs", f"train_s{seed}.log")
    if not os.path.exists(log):
        return None
    txt = open(log, errors="replace").read()
    steps = re.findall(r"step[ =:/]+(\d+)", txt)
    its = re.findall(r"(\d+\.\d+)\s*it/s", txt)
    ck = sorted(glob.glob(os.path.join(RUNS, f"s{seed}", "ckpt_*.pth")))
    return dict(log_bytes=len(txt), last_step=int(steps[-1]) if steps else None,
                it_per_s_last=float(its[-1]) if its else None, checkpoints=[os.path.basename(c) for c in ck],
                sampling_observed=os.path.exists(os.path.join(RUNS, f"s{seed}", "sampling_observed.json")))


def main():
    missing = []
    for f in ["ctl.json", "native_crops.json", "eval.json"]:
        if not os.path.exists(os.path.join(cl.RESULTS, f)):
            missing.append("results/" + f)
    # 2026-09-03: hard-coded "train_s42"/"eval_s42" refused a SEEDS=43-only full run after 7 h of
    # work (pod nxcv6ufppr8t6m); the required train/eval stages are exactly those of SEEDS.
    stages = ["provision", "ckpt", "trainer_check", "labels_fetch", "sv_fetch", "pool", "config_gen",
              "native_fetch", "ctl", "ref"]
    stages += [f"train_s{s}" for s in SEEDS] + [f"eval_s{s}" for s in SEEDS]
    for st in dict.fromkeys(stages):
        if not os.path.exists(os.path.join(VAR, "done_" + st)):
            missing.append("stage:" + st)
    if missing:
        cl.say("FINALIZE REFUSED -- missing: " + ", ".join(missing))
        sys.exit(3)
    ev = json.load(open(os.path.join(cl.RESULTS, "eval.json")))
    ctl = json.load(open(os.path.join(cl.RESULTS, "ctl.json")))
    trains = {s: train_summary(s) for s in SEEDS}
    verdict = {"mode": "SMOKE_ONLY -- pipeline validation; no gate verdict"} if SMOKE else {}
    if not SMOKE:
        per_seed = {}
        for s in SEEDS:
            rows = {k: v for k, v in ev.items() if k.startswith(f"s{s}_")}
            if rows:
                best = max(rows.items(), key=lambda kv: kv[1]["native5_mean_best_f1"])
                per_seed[s] = dict(best_tag=best[0], best_f1=best[1]["native5_mean_best_f1"],
                                   margin=best[1]["native5_mean_margin"], auc=best[1]["native5_mean_auc_forward"],
                                   trajectory={k: v["native5_mean_best_f1"] for k, v in sorted(rows.items())})
        if len(per_seed) == len(SEEDS):
            best_of_both = max(v["best_f1"] for v in per_seed.values())
            mean_margin = sum(v["margin"] for v in per_seed.values()) / len(per_seed)
            def peak_ok(tr):
                steps = {int(re.search(r"_(\d+)$", k).group(1)): v for k, v in tr.items()}
                if not steps:
                    return False
                pk = max(steps, key=steps.get)
                return 10000 <= pk <= 30000 and (75000 not in steps or steps[75000] < steps[pk])
            peaks = all(peak_ok(v["trajectory"]) for v in per_seed.values())
            verdict = dict(best_of_both=best_of_both, mean_margin=mean_margin, peak_rule=peaks,
                           PASS=bool(best_of_both >= 0.603 and mean_margin >= 0.06 and peaks),
                           rule="best-of-both >= 0.603 AND mean margin >= +0.06 AND peak at 10-30k with 75k below")
        verdict["per_seed"] = per_seed
    stats_p = os.path.join(cl.RESULTS, "input_stats.json")
    agg = dict(run="pod_betA_arm0 v1", arm=int(os.environ.get("ARM", "0")), villa_pin_ref=os.environ.get("VILLA_PIN_REF"), smoke_only=SMOKE,
               input_stats=json.load(open(stats_p)) if os.path.exists(stats_p) else None, finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg=json.load(open(os.path.join(cl.OUT, "prereg.json"))), ctl=ctl,
               native_crops=json.load(open(os.path.join(cl.RESULTS, "native_crops.json"))),
               eval=ev, training=trains, verdict=verdict)
    for p in (os.path.join(cl.RESULTS, "results.json"), os.path.join(cl.OUT, "results.json")):
        json.dump(agg, open(p, "w"), indent=1)
    na = ctl["arms"]["ctl_native"]
    cl.say(f"SUMMARY ctl (released ckpt): native fwd={na['forward']:.4f} rev={na['reverse']:.4f}; "
           f"scale-fault {ctl['scale_fault']['best']:.4f} -> {ctl['scale_fault']['verdict']}")
    for tag, row in ev.items():
        cl.say(f"SUMMARY {tag}: native-5 bestF1 {row['native5_mean_best_f1']:.4f} (margin {row['native5_mean_margin']:+.3f}), "
               f"AUC fwd {row['native5_mean_auc_forward']:.4f}")
    for s, t in trains.items():
        if t:
            cl.say(f"SUMMARY train s{s}: last step {t['last_step']}, {t['it_per_s_last']} it/s, ckpts {t['checkpoints']}")
    cl.say(f"SUMMARY verdict: {json.dumps(verdict)[:300]}")


if __name__ == "__main__":
    main()
PY_FINAL

cat > "$SCRIPTS/measure_inputs.py" <<'PY_MEASURE'
"""Bet A input statistics (prereg gate for arm 1): measure the pooled 2.4->9.6 um training volumes
and the five native 9.36 um eval crops with the SAME per-crop 2-D estimator the arm-1 degradation
uses (vesuvius.ink_detection.data.degradation.measure_2d: Hanning radial PSD, residual-based white
floor from the 0.35-0.48 cyc/px band, structural SNR at q = 0.25, bandwidth = max q with PSD >= 2x floor,
DN headroom = p99.5 - p0.5 in-mask). Writes results/input_stats.json and says a summary line per store.

  python measure_inputs.py <aligned9 dir> <native dir> <k2b_index.json> [n_windows=64] [size=128]

Reading: arm 1 can only degrade a crop whose measured SNR/bandwidth exceeds the drawn target. If the
pooled sources already sit at or below the index targets, the noise/blur steps are inactive and arm 1
reduces to the headroom match -- which is reported here BEFORE any arm-1 pod spends on training."""
import glob, json, os, sys
import numpy as np
import zarr
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl
from vesuvius.ink_detection.data.degradation import measure_2d, sample_inplane_windows


def open_level0(path):
    """Pooled aligned9 volumes and native crops are zarr GROUPS with the full-resolution array at "0"
    (prepare_9um_isotropic_input: group.create_array("0")); a bare array is accepted too."""
    g = zarr.open(path, mode="r")
    if hasattr(g, "shape"):
        return g
    for key in ("0", "s0", "level0"):
        if key in g:
            return g[key]
    keys = sorted(k for k in g.array_keys()) if hasattr(g, "array_keys") else []
    if not keys:
        raise ValueError(f"no array in zarr group {path}")
    return g[keys[0]]


def stats_for(volume, name, n, size, rng):
    wins = sample_inplane_windows(volume, n, size, rng)
    rows = []
    for w in wins:
        q, p, nz = measure_2d(w)
        i25 = int(np.argmin(np.abs(q - 0.25)))
        above = (q > 0.02) & (p / nz >= 2.0)
        dn = w[w > 0].astype(np.float32)
        rows.append(dict(snr_q025=float(p[i25] / nz), bandwidth_cyc_px=float(q[above].max()) if above.any() else 0.0,
                         dn_headroom=float(np.percentile(dn, 99.5) - np.percentile(dn, 0.5)), mean_dn=float(dn.mean())))
    if not rows:
        return dict(name=name, n=0)
    def med_iqr(k):
        v = np.array([r[k] for r in rows]); return [round(float(np.median(v)), 4), round(float(np.percentile(v, 25)), 4), round(float(np.percentile(v, 75)), 4)]
    return dict(name=name, n=len(rows), shape=list(volume.shape), snr_q025_med_iqr=med_iqr("snr_q025"),
                bandwidth_med_iqr=med_iqr("bandwidth_cyc_px"), dn_headroom_med_iqr=med_iqr("dn_headroom"), mean_dn=med_iqr("mean_dn"))


def main():
    aligned, native, index_path = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    size = int(sys.argv[5]) if len(sys.argv) > 5 else 128
    rng = np.random.default_rng(20260903)
    out = dict(estimator="2-D per-window, residual floor (conservative); index targets are 3-D air/residual", window=size, n_per_volume=n, pooled={}, native={})
    for kind, root in (("pooled", aligned), ("native", native)):
        for z in sorted(glob.glob(os.path.join(root, "*.zarr"))):
            name = os.path.basename(z)[:-5]
            try:
                st = stats_for(open_level0(z), name, n, size, rng)
            except Exception as e:  # one bad store must not sink a reporting stage
                st = dict(name=name, n=0, error=f"{type(e).__name__}: {e}"[:200])
            out[kind][name] = st
            if st.get("n"):
                cl.say(f"INPUTSTAT {kind} {name}: snr25 {st['snr_q025_med_iqr'][0]:.1f} [{st['snr_q025_med_iqr'][1]:.1f},{st['snr_q025_med_iqr'][2]:.1f}] "
                       f"bw {st['bandwidth_med_iqr'][0]:.3f} head {st['dn_headroom_med_iqr'][0]:.0f} (n={st['n']})")
            else:
                cl.say(f"INPUTSTAT {kind} {name}: FAILED {st.get('error', 'no windows')}")
    idx = json.load(open(index_path))
    tgt_snr = sorted(r["snr_q025"] for rec in idx.values() for r in rec["rois"])
    tgt_bw = sorted(r["bandwidth_cyc_px"] for rec in idx.values() for r in rec["rois"])
    pooled_snr = [v["snr_q025_med_iqr"][0] for v in out["pooled"].values() if v.get("n")]
    pooled_bw = [v["bandwidth_med_iqr"][0] for v in out["pooled"].values() if v.get("n")]
    out["index_targets"] = dict(n=len(tgt_snr), snr_q025_median=float(np.median(tgt_snr)), bandwidth_median=float(np.median(tgt_bw)))
    out["arm1_active_fraction"] = dict(
        noise=float(np.mean([s > np.median(tgt_snr) for s in pooled_snr])) if pooled_snr else None,
        blur=float(np.mean([b > np.median(tgt_bw) for b in pooled_bw])) if pooled_bw else None)
    json.dump(out, open(os.path.join(cl.RESULTS, "input_stats.json"), "w"), indent=1)
    cl.say(f"INPUTSTAT summary: pooled snr25 median {np.median(pooled_snr) if pooled_snr else float('nan'):.1f} vs index-target median {np.median(tgt_snr):.1f}; "
           f"pooled bw median {np.median(pooled_bw) if pooled_bw else float('nan'):.3f} vs target median {np.median(tgt_bw):.3f}; "
           f"fraction of pooled stores where arm-1 noise / blur is active: {out['arm1_active_fraction']}")


if __name__ == "__main__":
    main()
PY_MEASURE

cat > "$SCRIPTS/k2b_index.json" <<'JSON_K2B'
{
 "PHerc0125": {
  "n_pap_rois": 2,
  "rois": [
   {
    "origin": [
     3704,
     2928,
     4352
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 116.1,
    "dn_headroom": 219.0
   },
   {
    "origin": [
     3952,
     3368,
     4136
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 112.35,
    "dn_headroom": 223.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4958,
   0.4958,
   0.4958
  ],
  "snr_q025_med_iqr": [
   114.225,
   113.2875,
   115.1625
  ],
  "dn_headroom_med_iqr": [
   221.0,
   220.0,
   222.0
  ]
 },
 "PHerc0191": {
  "n_pap_rois": 4,
  "rois": [
   {
    "origin": [
     5072,
     4456,
     4592
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 68.85,
    "dn_headroom": 229.0
   },
   {
    "origin": [
     3272,
     5224,
     4088
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 74.5,
    "dn_headroom": 237.0
   },
   {
    "origin": [
     5368,
     4336,
     4632
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 124.63,
    "dn_headroom": 229.0
   },
   {
    "origin": [
     3816,
     4880,
     4240
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 139.59,
    "dn_headroom": 226.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4958,
   0.4958,
   0.4958
  ],
  "snr_q025_med_iqr": [
   99.565,
   73.0875,
   128.37
  ],
  "dn_headroom_med_iqr": [
   229.0,
   228.25,
   231.0
  ]
 },
 "PHerc0211": {
  "n_pap_rois": 5,
  "rois": [
   {
    "origin": [
     15184,
     1664,
     2976
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 97.76,
    "dn_headroom": 215.0
   },
   {
    "origin": [
     14232,
     1960,
     5832
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 144.52,
    "dn_headroom": 233.0
   },
   {
    "origin": [
     14408,
     2720,
     1784
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 123.66,
    "dn_headroom": 223.0
   },
   {
    "origin": [
     15776,
     4968,
     5056
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 106.64,
    "dn_headroom": 218.0
   },
   {
    "origin": [
     12800,
     3928,
     2280
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 91.32,
    "dn_headroom": 221.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4958,
   0.4958,
   0.4958
  ],
  "snr_q025_med_iqr": [
   106.64,
   97.76,
   123.66
  ],
  "dn_headroom_med_iqr": [
   221.0,
   218.0,
   223.0
  ]
 },
 "PHerc0257": {
  "n_pap_rois": 4,
  "rois": [
   {
    "origin": [
     3144,
     7296,
     4456
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 34.26,
    "dn_headroom": 217.0
   },
   {
    "origin": [
     3160,
     4024,
     5856
    ],
    "bandwidth_cyc_px": 0.4449,
    "snr_q025": 24.83,
    "dn_headroom": 150.0
   },
   {
    "origin": [
     6504,
     4240,
     2840
    ],
    "bandwidth_cyc_px": 0.4449,
    "snr_q025": 20.23,
    "dn_headroom": 135.0
   },
   {
    "origin": [
     4160,
     3472,
     5240
    ],
    "bandwidth_cyc_px": 0.4449,
    "snr_q025": 20.59,
    "dn_headroom": 137.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4449,
   0.4449,
   0.4576
  ],
  "snr_q025_med_iqr": [
   22.71,
   20.5,
   27.1875
  ],
  "dn_headroom_med_iqr": [
   143.5,
   136.5,
   166.75
  ]
 },
 "PHerc0268": {
  "n_pap_rois": 5,
  "rois": [
   {
    "origin": [
     10744,
     3672,
     7736
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 13.94,
    "dn_headroom": 183.0
   },
   {
    "origin": [
     11072,
     4072,
     8336
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 18.61,
    "dn_headroom": 209.0
   },
   {
    "origin": [
     3880,
     7600,
     1952
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 37.0,
    "dn_headroom": 173.0
   },
   {
    "origin": [
     11672,
     4920,
     7544
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 35.46,
    "dn_headroom": 211.0
   },
   {
    "origin": [
     11104,
     5016,
     7504
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 23.6,
    "dn_headroom": 214.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4958,
   0.4958,
   0.4958
  ],
  "snr_q025_med_iqr": [
   23.6,
   18.61,
   35.46
  ],
  "dn_headroom_med_iqr": [
   209.0,
   183.0,
   211.0
  ]
 },
 "PHerc0358": {
  "n_pap_rois": 5,
  "rois": [
   {
    "origin": [
     3592,
     4096,
     3248
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 143.65,
    "dn_headroom": 248.0
   },
   {
    "origin": [
     7768,
     3520,
     4680
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 57.91,
    "dn_headroom": 219.0
   },
   {
    "origin": [
     6728,
     3400,
     4320
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 91.82,
    "dn_headroom": 225.0
   },
   {
    "origin": [
     7256,
     3496,
     4560
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 85.77,
    "dn_headroom": 232.0
   },
   {
    "origin": [
     2456,
     3952,
     3200
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 163.37,
    "dn_headroom": 244.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4958,
   0.4958,
   0.4958
  ],
  "snr_q025_med_iqr": [
   91.82,
   85.77,
   143.65
  ],
  "dn_headroom_med_iqr": [
   232.0,
   225.0,
   244.0
  ]
 },
 "PHerc0800": {
  "n_pap_rois": 3,
  "rois": [
   {
    "origin": [
     4768,
     6096,
     3296
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 69.94,
    "dn_headroom": 192.0
   },
   {
    "origin": [
     19424,
     2416,
     3664
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 20.08,
    "dn_headroom": 146.0
   },
   {
    "origin": [
     19056,
     2656,
     2896
    ],
    "bandwidth_cyc_px": 0.4195,
    "snr_q025": 17.25,
    "dn_headroom": 130.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4958,
   0.4577,
   0.4958
  ],
  "snr_q025_med_iqr": [
   20.08,
   18.665,
   45.01
  ],
  "dn_headroom_med_iqr": [
   146.0,
   138.0,
   169.0
  ],
  "noise_ref": "air",
  "pap_mean_dn": 137.5,
  "air_rois": [
   {
    "origin": [
     23504,
     4896,
     9680
    ],
    "roi": 128,
    "sub_origin": [
     0,
     4,
     56
    ],
    "sub_n": 64,
    "mean_dn": 43.4,
    "psd_flatness": 11.7
   }
  ]
 },
 "PHerc0813": {
  "n_pap_rois": 2,
  "rois": [
   {
    "origin": [
     6784,
     3160,
     4600
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 163.16,
    "dn_headroom": 225.0
   },
   {
    "origin": [
     6936,
     3416,
     4552
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 156.02,
    "dn_headroom": 217.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4958,
   0.4958,
   0.4958
  ],
  "snr_q025_med_iqr": [
   159.59,
   157.805,
   161.375
  ],
  "dn_headroom_med_iqr": [
   221.0,
   219.0,
   223.0
  ]
 },
 "PHerc0826": {
  "n_pap_rois": 5,
  "rois": [
   {
    "origin": [
     3344,
     2760,
     4000
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 80.43,
    "dn_headroom": 222.0
   },
   {
    "origin": [
     4456,
     2648,
     4520
    ],
    "bandwidth_cyc_px": 0.4534,
    "snr_q025": 27.15,
    "dn_headroom": 152.0
   },
   {
    "origin": [
     4704,
     3560,
     4776
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 72.2,
    "dn_headroom": 222.0
   },
   {
    "origin": [
     3808,
     2584,
     5240
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 22.58,
    "dn_headroom": 151.0
   },
   {
    "origin": [
     6336,
     4128,
     4176
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 160.6,
    "dn_headroom": 232.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4958,
   0.4958,
   0.4958
  ],
  "snr_q025_med_iqr": [
   72.2,
   27.15,
   80.43
  ],
  "dn_headroom_med_iqr": [
   222.0,
   152.0,
   222.0
  ]
 },
 "PHerc1218": {
  "n_pap_rois": 4,
  "rois": [
   {
    "origin": [
     18888,
     3480,
     3504
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 21.13,
    "dn_headroom": 217.0
   },
   {
    "origin": [
     17392,
     3368,
     3688
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 24.87,
    "dn_headroom": 220.0
   },
   {
    "origin": [
     17856,
     3328,
     3664
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 23.83,
    "dn_headroom": 210.0
   },
   {
    "origin": [
     16952,
     3440,
     3784
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 27.52,
    "dn_headroom": 229.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4958,
   0.4958,
   0.4958
  ],
  "snr_q025_med_iqr": [
   24.35,
   23.155,
   25.5325
  ],
  "dn_headroom_med_iqr": [
   218.5,
   215.25,
   222.25
  ]
 },
 "PHerc1447": {
  "n_pap_rois": 5,
  "rois": [
   {
    "origin": [
     19136,
     4016,
     5704
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 10.28,
    "dn_headroom": 119.0
   },
   {
    "origin": [
     19120,
     3976,
     5432
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 10.89,
    "dn_headroom": 118.0
   },
   {
    "origin": [
     18504,
     3816,
     6624
    ],
    "bandwidth_cyc_px": 0.3941,
    "snr_q025": 8.45,
    "dn_headroom": 134.0
   },
   {
    "origin": [
     19056,
     3544,
     6824
    ],
    "bandwidth_cyc_px": 0.3856,
    "snr_q025": 7.65,
    "dn_headroom": 125.0
   },
   {
    "origin": [
     18920,
     3392,
     6216
    ],
    "bandwidth_cyc_px": 0.3771,
    "snr_q025": 8.12,
    "dn_headroom": 141.0
   }
  ],
  "bandwidth_med_iqr": [
   0.3941,
   0.3856,
   0.4958
  ],
  "snr_q025_med_iqr": [
   8.45,
   8.12,
   10.28
  ],
  "dn_headroom_med_iqr": [
   125.0,
   119.0,
   134.0
  ]
 },
 "PHerc1545": {
  "n_pap_rois": 5,
  "rois": [
   {
    "origin": [
     16032,
     3408,
     3368
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 132.73,
    "dn_headroom": 220.0
   },
   {
    "origin": [
     16960,
     3536,
     2400
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 83.59,
    "dn_headroom": 223.0
   },
   {
    "origin": [
     16352,
     3776,
     1760
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 132.14,
    "dn_headroom": 223.0
   },
   {
    "origin": [
     16704,
     3480,
     2424
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 112.23,
    "dn_headroom": 225.0
   },
   {
    "origin": [
     7040,
     2448,
     5280
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 86.72,
    "dn_headroom": 228.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4958,
   0.4958,
   0.4958
  ],
  "snr_q025_med_iqr": [
   112.23,
   86.72,
   132.14
  ],
  "dn_headroom_med_iqr": [
   223.0,
   223.0,
   225.0
  ]
 },
 "PHerc1203": {
  "n_pap_rois": 3,
  "rois": [
   {
    "origin": [
     4480,
     3432,
     1488
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 112.66,
    "dn_headroom": 236.0
   },
   {
    "origin": [
     15344,
     3224,
     1512
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 87.24,
    "dn_headroom": 194.0
   },
   {
    "origin": [
     15560,
     1880,
     3160
    ],
    "bandwidth_cyc_px": 0.4958,
    "snr_q025": 70.11,
    "dn_headroom": 212.0
   }
  ],
  "bandwidth_med_iqr": [
   0.4958,
   0.4958,
   0.4958
  ],
  "snr_q025_med_iqr": [
   87.24,
   78.675,
   99.95
  ],
  "dn_headroom_med_iqr": [
   212.0,
   203.0,
   224.0
  ]
 },
 "PHerc0139": {
  "n_pap_rois": 3,
  "rois": [
   {
    "origin": [
     13184,
     3200,
     2272
    ],
    "bandwidth_cyc_px": 0.3856,
    "snr_q025": 115.54,
    "dn_headroom": 151.0
   },
   {
    "origin": [
     12704,
     3168,
     3760
    ],
    "bandwidth_cyc_px": 0.3856,
    "snr_q025": 148.14,
    "dn_headroom": 155.0
   },
   {
    "origin": [
     13728,
     3440,
     2480
    ],
    "bandwidth_cyc_px": 0.3856,
    "snr_q025": 100.06,
    "dn_headroom": 143.0
   }
  ],
  "bandwidth_med_iqr": [
   0.3856,
   0.3856,
   0.3856
  ],
  "snr_q025_med_iqr": [
   115.54,
   107.8,
   131.84
  ],
  "dn_headroom_med_iqr": [
   151.0,
   147.0,
   153.0
  ],
  "noise_ref": "residual",
  "pap_mean_dn": 126.7,
  "noise_note": "no genuine air window passed validation; noise ref is the PSD of (papyrus ROI - its 3x3x3 uniform smooth), |1-U|^2-corrected white floor \u2014 SNR and bandwidth are conservative estimates"
 }
}
JSON_K2B

}
write_scripts
say "scripts written to $SCRIPTS (analysis code + manifest locked before provisioning)"

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
    retry 3 timeout 300 git clone --depth 1 --branch "$VILLA_PIN_REF" https://github.com/flummoxjr/villa-pin-37e300d3.git villa >> "$OUT/provision.log" 2>&1 || die "villa-pin clone failed - see provision.log"
  fi
  VSHA=$(cd villa && git rev-parse --short HEAD)
  say "provision: villa @ $VSHA (villa-pin ref $VILLA_PIN_REF, Bet A arm $ARM)"
  cd /workspace/villa/vesuvius
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  say "provision: PREFLIGHT (index reachability + throughput before uv sync)"
  PF_URL="https://pypi.nvidia.com/nvidia-curand/nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_x86_64.whl"
  PF_SPEED=$(curl -s -L --max-time 40 -r 0-8388607 -o /dev/null -w "%{speed_download}" "$PF_URL" || echo 0)
  PF_MBS=$(awk -v s="$PF_SPEED" 'BEGIN{printf "%.2f", s/1048576}')
  say "PREFLIGHT pypi.nvidia.com: ${PF_MBS} MB/s on an 8 MB range of nvidia-curand"
  for U in https://pypi.org/simple/uv/ https://huggingface.co/api/models/scrollprize/ink_9um https://vesuvius-challenge-open-data.s3.amazonaws.com/; do
    C=$(curl -s -o /dev/null --max-time 20 -w "%{http_code}" "$U" || echo 000); say "PREFLIGHT $U -> http $C"
  done
  if awk -v s="$PF_MBS" 'BEGIN{exit !(s < 1.0)}'; then die "PREFLIGHT: pypi.nvidia.com ${PF_MBS} MB/s (< 1 MB/s) from this host - the 2.2 GB of CUDA wheels would not arrive; relaunch on another host/cloud"; fi
  say "provision: uv sync starting (full log at /provision.log on :8000)"
  export UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=6
  retry 3 timeout 2400 uv sync --extra models >> "$OUT/provision.log" 2>&1 || die "uv sync failed - see provision.log"
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
# STAGE measure -- input statistics with the arm-1 estimator (pooled vs native vs index targets);
# reported before any arm-1 training so the degradation's activity is known (prereg gate).
# ============================================================================
if stage_done measure; then
  say "=== STAGE measure already done, skipping ==="
else
  stage_open measure
  # A reporting stage must never kill a $5 run (arm-2 s42 pod 28nacbva6r9l8p died here 2026-09-03 on a
  # zarr group vs array mismatch): log the failure, keep going; the traceback is kept in logs/measure.log.
  pyrun "$SCRIPTS/measure_inputs.py" "$VOLS/aligned9" "$NATIVE" "$SCRIPTS/k2b_index.json" 64 128 > "$OUT/logs/measure.log" 2>&1 || { tail -12 "$OUT/logs/measure.log" | while read -r L; do say "measure: $L"; done; say "measure: FAILED (non-fatal) -- input_stats.json absent; see logs/measure.log"; }
  stage_close measure
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
