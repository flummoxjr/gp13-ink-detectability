#!/bin/bash
# =============================================================================
# pod_w059_c0.sh -- one pod, two pre-registered questions:                  v1
#   C0   Bet B's contract check: does villa's optimized_inference + the public
#        canonical 2 um model reproduce the PUBLISHED prediction on PHerc0139
#        w035 (2.399 um surface volume)?  Gate: Pearson r >= 0.90 on the joint
#        support (a bounded window sweep if the first window misses).
#   w059 Track F arm B, full coverage: the same model on the second acquisition
#        (1.129 um scan, L1 = 2.258 um/px) of w059, forward AND reverse, plus
#        the same on w035's L1 volume as the modality control. The published
#        arm-B map covered a 23 mm strip; the full-coverage volume exists.
#   Prereg: trackD/bench/w059_c0/PREREG.md (sha logged below before any data).
#
# Everything is public: surface volumes on S3 (read anonymously -- the container
# code's anon=False is patched to anon=True), the model on HF, the reference
# prediction on S3. The battery (PROTOCOL_V2) runs LOCALLY on the ds16 maps.
#
# Env knobs: PORT BATCH_OI=8 SWEEP=1 (try windows 12/34 if C0 misses) DRY LINGER_EXIT
# =============================================================================
set -Eeuo pipefail
ROOT=${ROOT:-/workspace/w059}
PORT=${PORT:-8000}
FORCE=${FORCE:-0}
DRY=${DRY:-0}
DRY_FAIL_STAGE=${DRY_FAIL_STAGE:-}
LINGER_EXIT=${LINGER_EXIT:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
BATCH_OI=${BATCH_OI:-8}
SWEEP=${SWEEP:-1}
VILLA_SHA=${VILLA_SHA:-main}
SEED=20260903
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}

OUT=$ROOT/out;  VAR=$ROOT/var;  DATA=$ROOT/data;  PREDS=$ROOT/preds
SCRIPTS=$ROOT/scripts;  RESULTS=$OUT/results;  STATUS=$OUT/status.txt
OI=$ROOT/oi;  OIVENV=$ROOT/oivenv
export ROOT OUT VAR DATA PREDS SCRIPTS RESULTS STATUS SEED BATCH WORKERS OI OIVENV BATCH_OI

# =================================================================== L1 ======
# The very first actions: make the served dir and write the BOOT line.
mkdir -p "$OUT" "$VAR" "$DATA" "$PREDS" "$SCRIPTS" "$RESULTS" "$OUT/previews" "$DATA/tmp"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(now) $*" >> "$STATUS"; echo "$(now) $*"; }
say "BOOT pod_w059_c0 pid=$$ host=${HOSTNAME:-unknown} root=$ROOT -- status live; server next"

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
  "version": "w059_c0 v1 (2026-09-03): C0 contract check of the public canonical 2um model on w035 + Track F arm B at full coverage on w059's 1.129um-L1 surface volume, with w035's L1 volume as the modality control",
  "seed": 20260903,
  "model": {"repo": "scrollprize/ink_canonical_2um", "file": "r152_3ddec_v2_l5_epoch13.ckpt", "type": "resnet3d-152-3d-decoder", "tile": 256, "stride": 128, "window": "62 layers centred on the surface plane"},
  "code": "ScrollPrize/villa main ink-detection/optimized_inference (sha recorded in results); processing.py anon=False -> anon=True for the public bucket; no other modification",
  "inputs": {
    "w035_A": {"sv": "PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr", "shape": [109, 22640, 20400], "window": [23, 85], "reference": "ink-detection/PHerc0139-20260317000000-2.399um-...-new_canon_autoresearch_recipe-tile256-stride128.tif (40782909 B; canvas 22640x20400)"},
    "w035_B": {"sv": "PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/1.129um-0.22m-59keV-volume-20260413113053-L1.zarr", "shape": [116, 24080, 21700], "window": [27, 89]},
    "w059_B": {"sv": "PHerc0139/segments/20250223000000-w059_2025022312/surface-volumes/1.129um-0.22m-59keV-volume-20260413113053-L1.zarr", "shape": [116, 29860, 41440], "window": [27, 89], "directions": ["forward", "reverse"]}
  },
  "c0_gate": "Pearson r between our w035_A map and the published map, computed on the joint support (both > 0) at ds4; PASS iff r >= 0.90 with window [23,85). If it misses and SWEEP=1, windows [12,74) then [34,96) are tried once each (bounded: 3 passes max); the best window is recorded and used for every other run. If no window reaches 0.90 the contract is NOT reproduced: the run still completes w035_B and w059 but every downstream number is flagged CONTRACT_UNVERIFIED and Bet B's C1/C2a do not proceed.",
  "w059_measurements": "forward and reverse maps at full coverage; fwd/rev Pearson r on the joint support at ds4 (gate 5 input); ds16 (36.1 um/px) block-mean maps of every output for the local PROTOCOL_V2 battery; no verdict is computed on the pod",
  "prestated_readings": {"C0 pass, w035_B 4/4 map-internal gates locally, w059_B 4/4 + |fwd/rev r| < 0.20": "w059 clears the two-scanner rule -> escalation (>= 1000-perm rerun, human eyes, Discord); never letter language", "C0 pass, w059_B fails significance at full coverage with the control passing": "the A-arm ruling signal is not confirmed on the second scanner; Track F lead downgraded and closed", "C0 fail": "the 2um contract is not reproduced; nothing downstream is interpretable until it is"},
  "cost_cap": "$4; guard deadline 5 h"
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

pyrun() { "$OIVENV/bin/python" "$@"; }

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

cat > "$SCRIPTS/oi_score.py" <<'PY_OISCORE'
"""Scoring for the w059/C0 pod.
  python oi_score.py c0 <our.png> <published.tif>      -> results/c0_<tag>.json (Pearson r on joint support at ds4)
  python oi_score.py ds <name> <map.png> <px_um>        -> ds4/ds16 npys (block means) + stats
  python oi_score.py fwdrev <name> <fwd.png> <rev.png>  -> fwd/rev Pearson r at ds4 on joint support
Maps are uint8 PNGs from optimized_inference (cv2.imwrite); the published reference is a uint8 TIFF."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl


def load_map(path):
    import cv2
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        import tifffile
        a = tifffile.imread(path)
    if a.ndim == 3:
        a = a[..., 0]
    return a


def block_mean(a, f):
    H, W = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:H, :W].astype(np.float32).reshape(H // f, f, W // f, f).mean(axis=(1, 3))


def pearson_joint(a4, b4):
    m = (a4 > 0) & (b4 > 0)
    if m.sum() < 1000:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(a4[m], b4[m])[0, 1]), int(m.sum())


def c0(ours, ref, tag):
    a = load_map(ours); b = load_map(ref)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a4, b4 = block_mean(a[:h, :w], 4), block_mean(b[:h, :w], 4)
    r, n = pearson_joint(a4, b4)
    res = dict(tag=tag, ours_shape=list(a.shape), ref_shape=list(b.shape), r_ds4_joint=r, n_joint_ds4=n,
               ours_nonzero=float((a > 0).mean()), ref_nonzero=float((b > 0).mean()),
               ours_p99=float(np.percentile(a[a > 0], 99)) if (a > 0).any() else None,
               ref_p99=float(np.percentile(b[b > 0], 99)) if (b > 0).any() else None, gate=0.90, passed=bool(r >= 0.90))
    json.dump(res, open(os.path.join(cl.RESULTS, f"c0_{tag}.json"), "w"), indent=1)
    np.save(os.path.join(cl.OUT, "maps", f"c0_{tag}_ours_ds4.npy"), np.clip(np.rint(a4), 0, 255).astype(np.uint8))
    if not os.path.exists(os.path.join(cl.OUT, "maps", "c0_reference_ds4.npy")):
        np.save(os.path.join(cl.OUT, "maps", "c0_reference_ds4.npy"), np.clip(np.rint(b4), 0, 255).astype(np.uint8))
    cl.say(f"C0 {tag}: r_ds4(joint)={r:.4f} on {n} px; ours nonzero {res['ours_nonzero']:.3f} vs ref {res['ref_nonzero']:.3f}; "
           f"shapes {a.shape} vs {b.shape} -> {'PASS' if res['passed'] else 'below 0.90'}")
    sys.exit(0 if res["passed"] else 21)


def ds(name, path, px_um):
    a = load_map(path)
    a4 = np.clip(np.rint(block_mean(a, 4)), 0, 255).astype(np.uint8)
    a16 = np.clip(np.rint(block_mean(a, 16)), 0, 255).astype(np.uint8)
    np.save(os.path.join(cl.OUT, "maps", f"{name}_ds4.npy"), a4)
    np.save(os.path.join(cl.OUT, "maps", f"{name}_ds16.npy"), a16)
    nz = a > 0
    st = dict(name=name, shape=list(a.shape), px_um=float(px_um), ds16_px_um=float(px_um) * 16, nonzero_frac=float(nz.mean()),
              p50=float(np.percentile(a[nz], 50)) if nz.any() else None, p99=float(np.percentile(a[nz], 99)) if nz.any() else None)
    p = os.path.join(cl.RESULTS, "maps.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[name] = st; json.dump(d, open(p, "w"), indent=1)
    cl.say(f"DS {name}: {a.shape} nonzero {st['nonzero_frac']:.3f} p99 {st['p99']} -> ds4/ds16 npys ({st['ds16_px_um']:.1f} um/px at ds16)")


def fwdrev(name, f, r_):
    a4 = block_mean(load_map(f), 4); b4 = block_mean(load_map(r_), 4)
    h, w = min(a4.shape[0], b4.shape[0]), min(a4.shape[1], b4.shape[1])
    r, n = pearson_joint(a4[:h, :w], b4[:h, :w])
    p = os.path.join(cl.RESULTS, "maps.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[f"{name}_fwdrev"] = dict(r_ds4_joint=r, n_joint=n); json.dump(d, open(p, "w"), indent=1)
    cl.say(f"FWDREV {name}: r={r:.4f} on {n} ds4 px (control 0.094; corpus min 0.22; gate 5 requires < 0.20)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "c0":
        c0(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "ds":
        ds(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "fwdrev":
        fwdrev(sys.argv[2], sys.argv[3], sys.argv[4])
PY_OISCORE

}
write_scripts
say "scripts written to $SCRIPTS"

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
