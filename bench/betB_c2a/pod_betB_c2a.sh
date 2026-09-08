#!/bin/bash
# =============================================================================
# pod_betB_c2a.sh -- Bet B step C2a: the canonical 2 um model on the in-band  v1
#   PHerc1203 surface, rendered by us from the 2.403 um volume.
#   ctl   Positive control first: OUR renderer + villa optimized_inference +
#         scrollprize/ink_canonical_2um on PHerc0139 w035 (the published
#         9.362 um mesh transformed to the 2.399 um volume with the catalogue
#         matrix, rendered 78 slices) must reproduce the PUBLISHED w035 map
#         (r >= 0.90 on the joint support, best of three 62-layer windows) and
#         read the human letters (fwd AUC >= 0.95, reverse <= 0.80).
#   c2a   The same pipeline on PHerc1203: the published 9.362 um auto-grown
#         meshes transformed into the 2.403 um band with OUR derived transform,
#         restricted to the C1-selected tiles -- set H (w035-like along-normal
#         contrast >= 0.24) and a matched set L (the lowest-contrast tiles, the
#         merged-stack control) -- forward and reverse.
#   Readout (pre-registered, trackD/bench/betB_c2a/PREREG_C2A.md): per-tile
#   tripwire (value > w035 blank p99, area >= 0.876 mm^2, width >= 0.28 mm),
#   fraction above blank p99, fwd/rev Pearson r. No battery: 1.7 mm tiles cannot
#   hold ruling cycles. Escalation only; no letter language on the pod or off it.
#
# Everything is public: volumes and meshes on S3 (anonymous), the model on HF.
# Env knobs: PORT BATCH_OI=8 SLICES=78 TILE_RENDER=256 DRY LINGER_EXIT
# =============================================================================
set -Eeuo pipefail
ROOT=${ROOT:-/workspace/c2a}
PORT=${PORT:-8000}
FORCE=${FORCE:-0}
DRY=${DRY:-0}
DRY_FAIL_STAGE=${DRY_FAIL_STAGE:-}
LINGER_EXIT=${LINGER_EXIT:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
BATCH_OI=${BATCH_OI:-8}
SLICES=${SLICES:-78}
TILE_RENDER=${TILE_RENDER:-256}
SEED=20260908
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}
export AWS_NO_SIGN_REQUEST=YES AWS_DEFAULT_REGION=us-east-1

OUT=$ROOT/out;  VAR=$ROOT/var;  DATA=$ROOT/data;  PREDS=$ROOT/out/preds   # served on :8000 so the guard can fetch the TIFFs
SCRIPTS=$ROOT/scripts;  RESULTS=$OUT/results;  STATUS=$OUT/status.txt
OI=$ROOT/oi;  OIVENV=$ROOT/oivenv;  MESHES=$ROOT/meshes;  SV=$ROOT/sv
export ROOT OUT VAR DATA PREDS SCRIPTS RESULTS STATUS SEED BATCH WORKERS OI OIVENV BATCH_OI SLICES TILE_RENDER MESHES SV

# =================================================================== L1 ======
# The very first actions: make the served dir and write the BOOT line.
mkdir -p "$OUT" "$VAR" "$DATA" "$PREDS" "$SCRIPTS" "$RESULTS" "$OUT/previews" "$DATA/tmp"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(now) $*" >> "$STATUS"; echo "$(now) $*"; }
say "BOOT pod_betB_c2a pid=$$ host=${HOSTNAME:-unknown} root=$ROOT -- status live; server next"

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
 "name": "Bet B C2a -- canonical 2um model on the in-band PHerc1203 surface, rendered by us from the 2.403um band",
 "locked_before": "any provisioning, download, or data contact (sha logged on the pod before stage provision)",
 "date": "2026-09-08",
 "plan": "SEPTEMBER_PLAN.md 2.2 step 3 (C2a); C0 PASSED 2026-09-03 (r 0.9999997); C1 DONE 2026-09-08 (out/bet_b/c1_rank.md)",
 "model": {"repo": "scrollprize/ink_canonical_2um", "type": "resnet3d-152-3d-decoder", "tile": 256, "stride": 128, "window_layers": 62, "window_start_candidates": [8, 0, 16], "directions": ["forward", "reverse"]},
 "render": {"renderer": "runpod/render_tifxyz_sv.py (validated at 9.362um r=0.813; validated at 2.4um by the ctl of this run)", "slices": 78, "slice_step_vox": 1.0, "mesh_transform": "hunt/tifxyz_transform.py (published on-2.399 w035 mesh reproduced to median 1 px along z / 7 px in-plane)"},
 "ctl": {"mesh": "PHerc0139 w035 published 9.362um mesh -> 2.399um volume 20260102150214 via the catalogue matrix (inverse)", "region": "9.362 canvas rows 1024:2432, cols 896:2560 (middle 1408x1664 of the harness CTL_CROP)", "reference": "published w035 new_canon_autoresearch_recipe tile256/stride128 map", "labels": "curvelib.load_ctl_labels() sub-crop",
         "gate_C": {"r_ds4_joint_min": 0.90, "auc_fwd_min": 0.95, "auc_rev_max": 0.80, "on_fail": "CONTRACT_UNVERIFIED: 1203 numbers reported, not read"},
         "gate_T": {"tripwire_on_ink_min": 1, "on_fail": "tripwire readout INCONCLUSIVE; fraction statistics still reported"}},
 "c2a": {"volume": "PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr", "transform": "hunt/pherc1203_2403um_to_9362um.json (derived; translation + nominal scale; residual 22/16/10 um)",
         "set_H": "all C1 tiles with along-normal contrast >= 0.24 (w035 p10), cap 100 -> 48 tiles, 1.36 cm2", "set_L": "the same number of lowest-contrast tiles (merged-stack control) -> 48 tiles", "tile": "9x9 stored cells = 1.685 mm, +1 cell margin",
         "tripwire": {"threshold": "blank_p99 of the ctl forward map on labelled blank", "area_mm2_min": 0.876, "width_mm_min": 0.28, "origin": "0358 screen rule 1e4 px / 30 px at 9.362um"},
         "per_tile": ["p50/p99 fwd+rev", "frac > blank_p99 fwd+rev", "tripwire components fwd+rev", "fwd/rev Pearson r ds4 joint"],
         "escalation_candidate": "trip_fwd > 0 AND trip_rev == 0 AND frac_fwd > 3 * frac_rev (covered > 0.5)",
         "ESCALATE_iff": ["(a) >= 1 escalation candidate in H", "(b) L has 0 tripwire tiles, or one-sided Fisher exact H vs L p < 0.05", "(c) median per-tile fwd/rev r over H < 0.20"],
         "ESCALATE_action": "human look at previews (Ben); robustness re-render at window +-8 before any mention outside the repo; rules question first",
         "NULL": "otherwise: no evidence at the ctl-bounded sensitivity; Bet B 2.4um route closed for September; C1 ranking + validated renderer + null ship",
         "not_run": "PROTOCOL_V2 periodicity battery (1.7 mm tiles cannot hold ruling cycles)"},
 "language": "escalation only; no letter language on the pod, in the results, or in any post",
 "cost": {"cap_usd": 3, "guard_hours": 5, "balance_before_usd": 23},
 "seed": 20260908
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

cat > "$SCRIPTS/render_tifxyz_sv.py" <<'PY_RENDER'
#!/usr/bin/env python
"""Render a centered N-slice uint8 surface volume from a tifxyz mesh + volume zarr."""
import argparse
import math
import numpy as np
from numcodecs import Blosc
from scipy.ndimage import map_coordinates

from vesuvius.tifxyz import read_tifxyz
from vesuvius.ink_detection.volume_io import open_volume, read_bbox_with_padding
from vesuvius.label_zarr import open_v2_group, create_v2_array


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("tifxyz_dir")
    ap.add_argument("volume", help="scroll volume (local path or s3:// OME-Zarr); level 0 is read")
    ap.add_argument("output_zarr")
    ap.add_argument("--num-slices", type=int, default=21)
    ap.add_argument("--slice-step", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--max-crop-voxels", type=float, default=1.5e9)
    ap.add_argument("--cache-dir", default=None, help="zarr-3 chunk cache dir (optional)")
    ap.add_argument("--cache-max-gb", type=float, default=100.0)
    return ap.parse_args()


def main():
    args = parse_args()
    surf = read_tifxyz(args.tifxyz_dir, load_mask=False, validate=False)
    hs, ws = surf.shape                      # stored grid
    sy, sx = surf.get_scale_tuple()
    surf.use_full_resolution()
    Hr, Wr = surf.shape                      # villa: int(h/scale) -- 1 px SHORT when the
                                             # JSON scale is a float32-rounded value
                                             # (0.05000000074505806 -> 152/scale = 3039.99995)
    H, W = int(round(hs / sy)), int(round(ws / sx))   # canonical canvas (published SVs use this)
    if (Hr, Wr) != (H, W):
        print(f"WARNING: villa full-res shape {Hr}x{Wr} != canonical {H}x{W} "
              f"(int() truncation in vesuvius.tifxyz.types.Tifxyz.shape); rendering the "
              f"{Hr}x{Wr} region into a {H}x{W} canvas, extra row/col left zero")
    n = int(args.num_slices)
    offsets = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * float(args.slice_step)
    pad = int(math.ceil(np.abs(offsets).max())) + int(args.margin)
    print(f"full-res grid: {H} x {W} (rendered {Hr} x {Wr}), {n} slices, offsets {offsets[0]}..{offsets[-1]}")

    kwargs = {}
    if args.cache_dir:
        kwargs.update(cache_dir=args.cache_dir, cache_max_gb=args.cache_max_gb)
    try:
        vol = open_volume(args.volume, 0, **kwargs)
    except NotImplementedError:  # disk cache requires zarr 3
        print("zarr<3: continuing without chunk cache")
        vol = open_volume(args.volume, 0)

    group = open_v2_group(args.output_zarr)
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    out = create_v2_array(group, "0", shape=(n, H, W), chunks=(n, 256, 256),
                          dtype=np.uint8, compressor=comp, fill_value=0)

    def render_tile(r0, r1, c0, c1):
        x, y, z, valid = surf[r0:r1, c0:c1]
        if not np.any(valid):
            return
        nx, ny, nz = surf.get_normals(r0, r1, c0, c1)
        ok = valid & np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)
        if not np.any(ok):
            return
        z0 = int(np.floor(z[ok].min())) - pad; z1 = int(np.ceil(z[ok].max())) + pad + 1
        y0 = int(np.floor(y[ok].min())) - pad; y1 = int(np.ceil(y[ok].max())) + pad + 1
        x0 = int(np.floor(x[ok].min())) - pad; x1 = int(np.ceil(x[ok].max())) + pad + 1
        nvox = float(z1 - z0) * (y1 - y0) * (x1 - x0)
        if nvox > args.max_crop_voxels and (r1 - r0 > 64 or c1 - c0 > 64):
            rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
            for (a, b, c, d) in ((r0, rm, c0, cm), (r0, rm, cm, c1),
                                 (rm, r1, c0, cm), (rm, r1, cm, c1)):
                if a < b and c < d:
                    render_tile(a, b, c, d)
            return
        crop, _ = read_bbox_with_padding(vol, (z0, y0, x0, z1, y1, x1), fill_value=0)
        crop = crop.astype(np.float32, copy=False)
        th, tw = r1 - r0, c1 - c0
        tile_out = np.zeros((n, th, tw), dtype=np.uint8)
        zi, yi, xi = z[ok] - z0, y[ok] - y0, x[ok] - x0
        nzo, nyo, nxo = nz[ok], ny[ok], nx[ok]
        for si, off in enumerate(offsets):
            coords = np.stack([zi + off * nzo, yi + off * nyo, xi + off * nxo])
            vals = map_coordinates(crop, coords, order=1, mode="constant", cval=0.0)
            plane = np.zeros((th, tw), dtype=np.float32)
            plane[ok] = vals
            tile_out[si] = np.clip(np.rint(plane), 0, 255).astype(np.uint8)
        out[:, r0:r1, c0:c1] = tile_out

    t = int(args.tile)
    rows = list(range(0, Hr, t))
    for i, r0 in enumerate(rows):
        for c0 in range(0, Wr, t):
            render_tile(r0, min(Hr, r0 + t), c0, min(Wr, c0 + t))
        print(f"row band {i + 1}/{len(rows)} done")

    # occupancy level "3" (YX max-pool by 8) so infer.py can skip empty tiles
    p = 8
    occ = create_v2_array(group, "3", shape=(n, (H + p - 1) // p, (W + p - 1) // p),
                          chunks=(n, 256, 256), dtype=np.uint8, compressor=comp,
                          fill_value=0)
    band = 4096  # multiple of p
    for r0 in range(0, H, band):
        r1 = min(H, r0 + band)
        block = np.asarray(out[:, r0:r1, :])
        h = block.shape[1]
        ph, pw = (-h) % p, (-W) % p
        if ph or pw:
            block = np.pad(block, ((0, 0), (0, ph), (0, pw)))
        pooled = block.reshape(n, (h + ph) // p, p, (W + pw) // p, p).max(axis=(2, 4))
        occ[:, r0 // p : r0 // p + pooled.shape[1], :] = pooled
    print("done:", args.output_zarr)


if __name__ == "__main__":
    main()
PY_RENDER

cat > "$SCRIPTS/tifxyz_transform.py" <<'PY_TRANSFORM'
"""Re-express a tifxyz mesh in another volume's coordinates (the open-data convention):
apply a 3x4 affine (rows x, y, z; p_out = M . [p_in, 1]) to every stored cell and resample the
stored grid by the transform's linear scale so one stored cell stays 1/scale = 20 output voxels
(PHerc0139 w035: the published 9.362 um mesh is 291 x 262 stored cells; the published
"on-20260102150214-2.399um" mesh is 1132 x 1020 = round(291 x 3.8925), scale still 0.05).

  python tifxyz_transform.py <in_dir> <out_dir> --matrix M.json [--inverse] [--crop r0 r1 c0 c1]
                             [--keep-tiles tiles.json --tile 9 --margin 1] [--factor F]

--matrix     JSON with "transformation_matrix" (3 rows of 4). With --inverse the matrix maps
             out -> in (the catalogue publishes 2.4um -> 9.362um; we go 9.362 -> 2.4).
--crop       keep only stored rows r0:r1, cols c0:c1 of the INPUT grid (others invalid).
--keep-tiles JSON list of {"seg":..., "ij":[i,j]} (c1_tiles.jsonl rows) -- keep the union of
             tile x tile cells at those input-grid origins (+margin cells); others invalid.
Validity follows villa (z > 0 and finite); invalid output cells are written as 0.
"""
import argparse
import json
import os

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates


def load(d):
    X = tifffile.imread(os.path.join(d, "x.tif")).astype(np.float64)
    Y = tifffile.imread(os.path.join(d, "y.tif")).astype(np.float64)
    Z = tifffile.imread(os.path.join(d, "z.tif")).astype(np.float64)
    meta = json.load(open(os.path.join(d, "meta.json")))
    valid = (Z > 0) & np.isfinite(Z) & np.isfinite(X) & np.isfinite(Y)
    return X, Y, Z, valid, meta


def affine_3x4(m, inverse):
    A = np.array(m, dtype=np.float64)
    assert A.shape == (3, 4), A.shape
    if inverse:
        R, t = A[:, :3], A[:, 3]
        Ri = np.linalg.inv(R)
        A = np.concatenate([Ri, (-Ri @ t)[:, None]], axis=1)
    return A


def resample_grid(F, valid, factor):
    """Bilinear resample of a coordinate raster by `factor` (area-aligned: out cell I sits at
    in position (I + 0.5) / factor - 0.5). Cells whose 2x2 support is not fully valid are invalid."""
    h, w = F.shape
    H, W = int(round(h * factor)), int(round(w * factor))
    ii = (np.arange(H) + 0.5) / factor - 0.5
    jj = (np.arange(W) + 0.5) / factor - 0.5
    gi, gj = np.meshgrid(ii, jj, indexing="ij")
    coords = np.stack([gi.ravel(), gj.ravel()])
    out = map_coordinates(F, coords, order=1, mode="nearest").reshape(H, W)
    vmin = map_coordinates(valid.astype(np.float64), coords, order=1, mode="constant", cval=0.0).reshape(H, W)
    return out, vmin > 0.999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--inverse", action="store_true")
    ap.add_argument("--crop", nargs=4, type=int, default=None, metavar=("R0", "R1", "C0", "C1"))
    ap.add_argument("--keep-tiles", default=None)
    ap.add_argument("--tile", type=int, default=9)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--factor", type=float, default=0.0, help="stored-grid resample factor (default: transform scale)")
    a = ap.parse_args()

    X, Y, Z, valid, meta = load(a.in_dir)
    h, w = X.shape
    M = affine_3x4(json.load(open(a.matrix))["transformation_matrix"], a.inverse)
    scale = float(np.cbrt(abs(np.linalg.det(M[:, :3]))))
    factor = a.factor or scale
    keep = np.ones_like(valid)
    if a.crop:
        r0, r1, c0, c1 = a.crop
        keep[:] = False
        keep[max(0, r0):min(h, r1), max(0, c0):min(w, c1)] = True
    if a.keep_tiles:
        rows = json.load(open(a.keep_tiles)) if a.keep_tiles.endswith(".json") else [json.loads(l) for l in open(a.keep_tiles) if l.strip()]
        keep[:] = False
        for r in rows:
            i, j = r["ij"]
            keep[max(0, i - a.margin):min(h, i + a.tile + a.margin), max(0, j - a.margin):min(w, j + a.tile + a.margin)] = True
    valid = valid & keep
    n_in = int(valid.sum())

    Xr, vX = resample_grid(X, valid, factor)
    Yr, vY = resample_grid(Y, valid, factor)
    Zr, vZ = resample_grid(Z, valid, factor)
    v = vX & vY & vZ
    P = np.stack([Xr.ravel(), Yr.ravel(), Zr.ravel(), np.ones(Xr.size)])
    Q = M @ P
    Xo, Yo, Zo = (Q[k].reshape(Xr.shape) for k in range(3))
    v &= (Zo > 0) & np.isfinite(Zo)
    for A_ in (Xo, Yo, Zo):
        A_[~v] = 0.0
    os.makedirs(a.out_dir, exist_ok=True)
    for name, A_ in (("x", Xo), ("y", Yo), ("z", Zo)):
        tifffile.imwrite(os.path.join(a.out_dir, f"{name}.tif"), A_.astype(np.float32))
    bbox = [[float(Xo[v].min()), float(Yo[v].min()), float(Zo[v].min())], [float(Xo[v].max()), float(Yo[v].max()), float(Zo[v].max())]] if v.any() else None
    m2 = dict(meta)
    m2.update(bbox=bbox, area_vx2=float(meta.get("area_vx2", 0.0)) * (float(v.sum()) / max(1, n_in)) * (1.0 / scale) ** 2 * factor ** 2 if meta.get("area_vx2") else None,
              transform_note=f"tifxyz_transform.py: {a.matrix} inverse={a.inverse} factor={factor:.5f} scale={scale:.5f} in_valid={n_in} out_valid={int(v.sum())}")
    if m2.get("area_vx2") is None:
        m2.pop("area_vx2", None)
    json.dump(m2, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=4)
    print(f"{a.in_dir} {h}x{w} ({n_in} kept valid cells) -> {a.out_dir} {Xo.shape[0]}x{Xo.shape[1]} ({int(v.sum())} valid); "
          f"factor {factor:.5f}, transform scale {scale:.5f}, bbox {bbox}")


if __name__ == "__main__":
    main()
PY_TRANSFORM

cat > "$SCRIPTS/c2a_meshes.py" <<'PY_MESHES'
"""Bet B C2a, stage `meshes`: fetch the published 9.362 um tifxyz meshes and re-express them in the
2.4 um volumes (tifxyz_transform.py), keeping only the ctl sub-crop / the C1-selected tiles.

  python c2a_meshes.py <selection.json> <meshes dir>

Writes <meshes dir>/ctl_w035/ and <meshes dir>/seg_<name>/ (x/y/z.tif + meta.json + geom.json) and
<meshes dir>/meshes.json. geom.json carries, per kept tile, its pixel box on the 2.4 um canvas:
input stored cell i <-> output stored cell (i + 0.5) * factor - 0.5; one output cell = 20 px.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import requests

HTTPB = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
SCRIPTS = os.environ.get("SCRIPTS", os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


def say(msg):
    st = os.environ.get("STATUS")
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg
    print(line, flush=True)
    if st:
        with open(st, "a") as f:
            f.write(line + "\n")


def fetch(seg_dir, dest):
    os.makedirs(dest, exist_ok=True)
    for f in ("x.tif", "y.tif", "z.tif", "meta.json"):
        p = os.path.join(dest, f)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        url = f"{HTTPB}/{seg_dir.rstrip('/')}/{f}"
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=300)
                r.raise_for_status()
                open(p, "wb").write(r.content)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
                print("retry", url, e, flush=True)


def transform(in_dir, out_dir, matrix_path, inverse, crop=None, keep_tiles=None, tile=9, margin=1):
    cmd = [PYTHON, os.path.join(SCRIPTS, "tifxyz_transform.py"), in_dir, out_dir, "--matrix", matrix_path]
    if inverse:
        cmd.append("--inverse")
    if crop:
        cmd += ["--crop"] + [str(int(v)) for v in crop]
    if keep_tiles:
        cmd += ["--keep-tiles", keep_tiles, "--tile", str(tile), "--margin", str(margin)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout, out.stderr, flush=True)
    if out.returncode != 0:
        raise RuntimeError(f"tifxyz_transform failed: {out.stderr[-500:]}")
    note = json.load(open(os.path.join(out_dir, "meta.json")))["transform_note"]
    factor = float(note.split("factor=")[1].split()[0])
    return factor


def cell_box_px(i0, i1, j0, j1, factor, cell_px=20):
    """Input stored cells [i0,i1) x [j0,j1) -> output pixel box [R0,R1) x [C0,C1) on the 2.4 um canvas."""
    r = lambda c: int(round(((c + 0.5) * factor - 0.5) * cell_px))  # noqa: E731
    return [r(i0), r(i1), r(j0), r(j1)]


def main():
    sel = json.load(open(sys.argv[1]))
    root = sys.argv[2]
    raw = os.path.join(root, "raw")
    os.makedirs(raw, exist_ok=True)
    tile, margin = int(sel.get("tile", 9)), int(sel.get("margin", 1))
    mx0139 = os.path.join(root, "matrix_0139.json"); json.dump(sel["ctl"]["matrix"], open(mx0139, "w"))
    mx1203 = os.path.join(root, "matrix_1203.json"); json.dump(sel["matrix_1203"], open(mx1203, "w"))
    summary = dict(ctl=None, segments={}, tile=tile, margin=margin)

    # ---- ctl: w035 9.362 mesh -> 2.399 volume, sub-crop
    c = sel["ctl"]
    src = os.path.join(raw, "w035_9362")
    fetch(c["mesh"], src)
    outd = os.path.join(root, "ctl_w035")
    f = transform(src, outd, mx0139, bool(c.get("inverse", True)), crop=c["crop_cells"])
    r0, r1, c0, c1 = c["crop_cells"]
    y0, y1, x0, x1 = c["crop_px_9362"]                  # the label sub-crop, exact (fractional cells)
    geom = dict(kind="ctl", px_um=float(c["px_um"]), factor=f, crop_cells=c["crop_cells"], crop_px_9362=c["crop_px_9362"],
                label_crop=c["label_crop"], box_px=cell_box_px(y0 / 20.0, y1 / 20.0, x0 / 20.0, x1 / 20.0, f),
                mesh_box_px=cell_box_px(r0, r1, c0, c1, f), source_mesh=c["mesh"], matrix=sel["ctl"]["matrix"])
    json.dump(geom, open(os.path.join(outd, "geom.json"), "w"), indent=1)
    summary["ctl"] = geom
    say(f"MESH ctl_w035: cells [{r0},{r1})x[{c0},{c1}) of the 9.362 mesh -> box_px {geom['box_px']} at 2.399 um (factor {f:.5f})")

    # ---- 1203 segments
    for name, s in sel["segments"].items():
        src = os.path.join(raw, name)
        fetch(s["seg_dir"], src)
        outd = os.path.join(root, f"seg_{name}")
        kt = os.path.join(root, f"keep_{name}.json")
        json.dump([dict(ij=t["ij"]) for t in s["tiles"]], open(kt, "w"))
        f = transform(src, outd, mx1203, True, keep_tiles=kt, tile=tile, margin=margin)
        tiles = []
        for t in s["tiles"]:
            i, j = t["ij"]
            tiles.append(dict(ij=[i, j], set=t["set"], contrast=t["contrast"], box_px=cell_box_px(i, i + tile, j, j + tile, f)))
        geom = dict(kind="seg", name=name, px_um=float(sel["px_um_1203"]), factor=f, tile=tile, tiles=tiles, source_mesh=s["seg_dir"])
        json.dump(geom, open(os.path.join(outd, "geom.json"), "w"), indent=1)
        summary["segments"][name] = dict(n_tiles=len(tiles), n_H=sum(t["set"] == "H" for t in tiles), n_L=sum(t["set"] == "L" for t in tiles), factor=f)
        say(f"MESH seg_{name}: {len(tiles)} tiles (H {summary['segments'][name]['n_H']}, L {summary['segments'][name]['n_L']}) -> 2.403 um (factor {f:.5f})")
    json.dump(summary, open(os.path.join(root, "meshes.json"), "w"), indent=1)
    say(f"MESH done: ctl + {len(summary['segments'])} segments, {sum(v['n_tiles'] for v in summary['segments'].values())} tiles")


if __name__ == "__main__":
    main()
PY_MESHES

cat > "$SCRIPTS/c2a_score.py" <<'PY_SCORE'
"""Scoring for the Bet B C2a pod (all maps are uint8 tiled TIFFs from optimized_inference).

  python c2a_score.py ctl    <ours.tif> <published.tif> <mesh dir> <tag>
      -> results/ctl_<tag>.json: Pearson r (ds4, joint support) inside the rendered box against the
         published w035 map; forward pixel AUC on the human labels of the sub-crop; blank p99 (the
         tripwire reference); tripwire components and how many touch ink (the tripwire's own control).
         exit 0 iff r >= 0.90, else 21 (the stage sweeps the window on 21).
  python c2a_score.py ctlrev <fwd.tif> <rev.tif> <mesh dir> <tag>
      -> results/ctl_<tag>_rev.json: reverse-order AUC + fwd/rev r.
  python c2a_score.py tiles  <fwd.tif> <rev.tif> <mesh dir> <blank_p99> <name>
      -> results/tiles_<name>.json: per-tile statistics for sets H and L.
Tripwire (pre-registered, physical units from the 0358 screen at 9.362 um: 1e4 px / 30 px):
value > blank_p99, component area >= 0.876 mm^2, bbox width >= 0.28 mm.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl  # noqa: E402

TRIP_AREA_MM2 = 0.876
TRIP_WIDTH_MM = 0.28


def load_map(path):
    os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 40))
    import tifffile
    a = tifffile.imread(path)
    if a.ndim == 3:
        a = a[..., 0]
    return a


def block_mean(a, f):
    H, W = a.shape[0] // f * f, a.shape[1] // f * f
    out = np.empty((H // f, W // f), np.float32)
    rows = max(f, (4096 // f) * f)
    for y in range(0, H, rows):
        s = a[y:min(y + rows, H), :W]
        h = s.shape[0] // f * f
        acc = np.uint32 if np.issubdtype(s.dtype, np.integer) else np.float64
        blk = s[:h].reshape(h // f, f, W // f, f).astype(acc).sum(axis=(1, 3))
        out[y // f:(y + h) // f] = blk / float(f * f)
    return out


def pearson_joint(a4, b4):
    m = (a4 > 0) & (b4 > 0)
    if m.sum() < 1000:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(a4[m], b4[m])[0, 1]), int(m.sum())


def region(a, box):
    R0, R1, C0, C1 = box
    R0, C0 = max(0, R0), max(0, C0)
    return a[R0:min(a.shape[0], R1), C0:min(a.shape[1], C1)]


def tripwire(m, thr, px_um, ink_mask=None):
    """Components of (m > thr) meeting the physical area/width thresholds."""
    hot = m > thr
    if not hot.any():
        return dict(n=0, n_on_ink=0, components=[])
    lab, n = ndimage.label(hot)
    if n == 0:
        return dict(n=0, n_on_ink=0, components=[])
    area_min = TRIP_AREA_MM2 * 1e6 / (px_um ** 2)
    width_min = TRIP_WIDTH_MM * 1e3 / px_um
    sizes = ndimage.sum(hot, lab, index=np.arange(1, n + 1))
    objs = ndimage.find_objects(lab)
    comps = []
    for k, (sz, sl) in enumerate(zip(sizes, objs), start=1):
        if sz < area_min or sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) < width_min:
            continue
        on_ink = bool(ink_mask is not None and ink_mask[sl][lab[sl] == k].any())
        comps.append(dict(area_px=int(sz), area_mm2=float(sz * px_um ** 2 / 1e6), bbox=[int(sl[0].start), int(sl[0].stop), int(sl[1].start), int(sl[1].stop)],
                          max=int(m[sl][lab[sl] == k].max()), on_ink=on_ink))
    comps.sort(key=lambda c: -c["area_px"])
    return dict(n=len(comps), n_on_ink=int(sum(c["on_ink"] for c in comps)), components=comps[:50])


def ctl_labels_sub(geom):
    """pos/neg masks of the rendered sub-crop on the 9.362 um label grid."""
    pos, neg = cl.load_ctl_labels()                     # CTL_CROP grid (2432 x 2688)
    y0, y1, x0, x1 = geom["label_crop"]                  # sub-crop offsets inside CTL_CROP
    return pos[y0:y1, x0:x1], neg[y0:y1, x0:x1]


def auc_on_labels(map_region, pos, neg):
    q = cl.quantize_map(cl.resample_pred(map_region.astype(np.float32), pos.shape))
    return cl.hist_auc(cl.masked_hist(q, pos), cl.masked_hist(q, neg))


def nearest_mask(mask, shape):
    zi = np.minimum((np.arange(shape[0]) * mask.shape[0] / shape[0]).astype(int), mask.shape[0] - 1)
    zj = np.minimum((np.arange(shape[1]) * mask.shape[1] / shape[1]).astype(int), mask.shape[1] - 1)
    return mask[zi][:, zj]


def ctl(ours_p, ref_p, mesh_dir, tag):
    geom = json.load(open(os.path.join(mesh_dir, "geom.json")))
    a, b = load_map(ours_p), load_map(ref_p)
    ra, rb = region(a, geom["box_px"]), region(b, geom["box_px"])
    h, w = min(ra.shape[0], rb.shape[0]), min(ra.shape[1], rb.shape[1])
    ra, rb = ra[:h, :w], rb[:h, :w]
    r, n = pearson_joint(block_mean(ra, 4), block_mean(rb, 4))
    pos, neg = ctl_labels_sub(geom)
    auc = auc_on_labels(ra, pos, neg)
    auc_ref = auc_on_labels(rb, pos, neg)
    negf, posf = nearest_mask(neg, ra.shape), nearest_mask(pos, ra.shape)
    blank_vals = ra[negf & (ra > 0)]
    blank_p99 = float(np.percentile(blank_vals, 99)) if blank_vals.size else None
    tw = tripwire(ra, blank_p99, geom["px_um"], posf) if blank_p99 is not None else dict(n=0, n_on_ink=0, components=[])
    res = dict(tag=tag, box_px=geom["box_px"], ours_shape=list(a.shape), ref_shape=list(b.shape), r_ds4_joint=r, n_joint_ds4=n,
               ours_nonzero_in_box=float((ra > 0).mean()), ref_nonzero_in_box=float((rb > 0).mean()),
               auc_fwd=auc, auc_ref_published=auc_ref, n_pos=int(pos.sum()), n_neg=int(neg.sum()),
               blank_p99=blank_p99, ink_p50=float(np.percentile(ra[posf & (ra > 0)], 50)) if (posf & (ra > 0)).any() else None,
               tripwire_components=tw["n"], tripwire_on_ink=tw["n_on_ink"], tripwire=tw, gate_r=0.90, passed=bool(r >= 0.90))
    json.dump(res, open(os.path.join(cl.RESULTS, f"ctl_{tag}.json"), "w"), indent=1)
    np.save(os.path.join(cl.OUT, "maps", f"{tag}_box_ds4.npy"), np.clip(np.rint(block_mean(ra, 4)), 0, 255).astype(np.uint8))
    if not os.path.exists(os.path.join(cl.OUT, "maps", "ctl_reference_box_ds4.npy")):
        np.save(os.path.join(cl.OUT, "maps", "ctl_reference_box_ds4.npy"), np.clip(np.rint(block_mean(rb, 4)), 0, 255).astype(np.uint8))
        cl.save_preview(rb, os.path.join(cl.OUT, "previews", "ctl_reference_box.png"), ds=8)
    cl.save_preview(ra, os.path.join(cl.OUT, "previews", f"{tag}_box.png"), ds=8)
    cl.say(f"CTL {tag}: r_ds4(joint)={r:.4f} on {n} px; AUC fwd {auc:.4f} (published map on the same labels {auc_ref:.4f}); "
           f"blank p99 {blank_p99}; tripwire {tw['n']} components, {tw['n_on_ink']} on ink -> {'PASS' if res['passed'] else 'below 0.90'}")
    sys.exit(0 if res["passed"] else 21)


def ctlrev(fwd_p, rev_p, mesh_dir, tag):
    geom = json.load(open(os.path.join(mesh_dir, "geom.json")))
    a, b = load_map(fwd_p), load_map(rev_p)
    ra, rb = region(a, geom["box_px"]), region(b, geom["box_px"])
    pos, neg = ctl_labels_sub(geom)
    auc_rev = auc_on_labels(rb, pos, neg)
    r, n = pearson_joint(block_mean(ra, 4), block_mean(rb, 4))
    negf = nearest_mask(neg, rb.shape)
    bv = rb[negf & (rb > 0)]
    bp = float(np.percentile(bv, 99)) if bv.size else None
    tw = tripwire(rb, bp, geom["px_um"], nearest_mask(pos, rb.shape)) if bp is not None else dict(n=0, n_on_ink=0)
    res = dict(tag=tag, auc_rev=auc_rev, fwdrev_r_ds4=r, n_joint=n, rev_blank_p99=bp, rev_tripwire_components=tw["n"], rev_tripwire_on_ink=tw["n_on_ink"],
               depth_order_gate=dict(max_reverse=0.80, passed=bool(auc_rev <= 0.80)))
    json.dump(res, open(os.path.join(cl.RESULTS, f"ctl_{tag}_rev.json"), "w"), indent=1)
    np.save(os.path.join(cl.OUT, "maps", f"{tag}_rev_box_ds4.npy"), np.clip(np.rint(block_mean(rb, 4)), 0, 255).astype(np.uint8))
    cl.save_preview(rb, os.path.join(cl.OUT, "previews", f"{tag}_rev_box.png"), ds=8)
    cl.say(f"CTL {tag} reverse: AUC rev {auc_rev:.4f} (gate <= 0.80 {'passed' if auc_rev <= 0.80 else 'FAILED'}); fwd/rev r {r:.4f}; rev tripwire {tw['n']}")


def tiles(fwd_p, rev_p, mesh_dir, blank_p99, name):
    geom = json.load(open(os.path.join(mesh_dir, "geom.json")))
    bp = float(blank_p99)
    a, b = load_map(fwd_p), load_map(rev_p)
    rows = []
    for t in geom["tiles"]:
        ra, rb = region(a, t["box_px"]), region(b, t["box_px"])
        h, w = min(ra.shape[0], rb.shape[0]), min(ra.shape[1], rb.shape[1])
        ra, rb = ra[:h, :w], rb[:h, :w]
        va, vb = ra[ra > 0], rb[rb > 0]
        twa, twb = tripwire(ra, bp, geom["px_um"]), tripwire(rb, bp, geom["px_um"])
        r, n = pearson_joint(block_mean(ra, 4), block_mean(rb, 4)) if h >= 8 and w >= 8 else (float("nan"), 0)
        rows.append(dict(ij=t["ij"], set=t["set"], contrast=t["contrast"], box_px=t["box_px"], covered=float((ra > 0).mean()) if ra.size else 0.0,
                         fwd_p50=float(np.percentile(va, 50)) if va.size else None, fwd_p99=float(np.percentile(va, 99)) if va.size else None,
                         rev_p50=float(np.percentile(vb, 50)) if vb.size else None, rev_p99=float(np.percentile(vb, 99)) if vb.size else None,
                         frac_fwd=float((va > bp).mean()) if va.size else None, frac_rev=float((vb > bp).mean()) if vb.size else None,
                         trip_fwd=twa["n"], trip_rev=twb["n"], trip_fwd_max_mm2=(twa["components"][0]["area_mm2"] if twa["n"] else 0.0),
                         fwdrev_r=r, n_joint=n))
    def summ(rows_, s):
        rr = [x for x in rows_ if x["set"] == s and x["covered"] > 0.5]
        if not rr:
            return dict(n=0)
        ff = np.array([x["frac_fwd"] for x in rr], float); fr = np.array([x["frac_rev"] for x in rr], float)
        return dict(n=len(rr), trip_fwd_tiles=int(sum(x["trip_fwd"] > 0 for x in rr)), trip_rev_tiles=int(sum(x["trip_rev"] > 0 for x in rr)),
                    frac_fwd_mean=float(ff.mean()), frac_rev_mean=float(fr.mean()), frac_fwd_max=float(ff.max()),
                    fwd_p99_median=float(np.median([x["fwd_p99"] for x in rr])), rev_p99_median=float(np.median([x["rev_p99"] for x in rr])),
                    fwdrev_r_median=float(np.nanmedian([x["fwdrev_r"] for x in rr])),
                    escalation_candidates=[x["ij"] for x in rr if x["trip_fwd"] > 0 and x["trip_rev"] == 0 and (x["frac_rev"] or 0) * 3 < (x["frac_fwd"] or 0)])
    res = dict(name=name, px_um=geom["px_um"], blank_p99=bp, tiles=rows, summary=dict(H=summ(rows, "H"), L=summ(rows, "L")))
    json.dump(res, open(os.path.join(cl.RESULTS, f"tiles_{name}.json"), "w"), indent=1)
    np.save(os.path.join(cl.OUT, "maps", f"{name}_fwd_ds4.npy"), np.clip(np.rint(block_mean(a, 4)), 0, 255).astype(np.uint8))
    np.save(os.path.join(cl.OUT, "maps", f"{name}_rev_ds4.npy"), np.clip(np.rint(block_mean(b, 4)), 0, 255).astype(np.uint8))
    top = sorted([x for x in rows if x["covered"] > 0.5], key=lambda x: -(x["frac_fwd"] or 0))[:3]
    for x in top:
        cl.save_preview(region(a, x["box_px"]), os.path.join(cl.OUT, "previews", f"{name}_{x['set']}_{x['ij'][0]}_{x['ij'][1]}_fwd.png"), ds=2)
        cl.save_preview(region(b, x["box_px"]), os.path.join(cl.OUT, "previews", f"{name}_{x['set']}_{x['ij'][0]}_{x['ij'][1]}_rev.png"), ds=2)
    s = res["summary"]
    cl.say(f"TILES {name}: H n={s['H'].get('n')} trip fwd/rev {s['H'].get('trip_fwd_tiles')}/{s['H'].get('trip_rev_tiles')} frac fwd/rev {s['H'].get('frac_fwd_mean')}/{s['H'].get('frac_rev_mean')} "
           f"| L n={s['L'].get('n')} trip {s['L'].get('trip_fwd_tiles')}/{s['L'].get('trip_rev_tiles')} frac {s['L'].get('frac_fwd_mean')}/{s['L'].get('frac_rev_mean')} "
           f"| escalation candidates H {s['H'].get('escalation_candidates')}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ctl":
        ctl(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "ctlrev":
        ctlrev(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "tiles":
        tiles(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        raise SystemExit(f"unknown command {cmd}")
PY_SCORE

cat > "$SCRIPTS/selection.json" <<'SELECTION_JSON'
{
 "made_from": "out/bet_b/c1_rank.json (hunt/c1_rank_inband.py, threshold 0.24 fixed before scoring)",
 "threshold": 0.24,
 "n_H": 48,
 "n_L": 48,
 "tile": 9,
 "margin": 1,
 "px_um_1203": 2.403,
 "ctl": {
  "mesh": "PHerc0139/segments/20260317000000-w035_2026031718/mesh/20260317000000-on-20250728140407-9.362um.tifxyz",
  "matrix": {
   "transformation_matrix": [
    [
     -0.2569036650583705,
     -0.009715391643719536,
     0.002790733227688403,
     6677.364709056732
    ],
    [
     0.009850885045974334,
     -0.25680016838938274,
     -0.0014215065457306453,
     6689.120376657908
    ],
    [
     0.0029113360690184954,
     -0.0014844301309079428,
     0.257134581099402,
     624.1652625002474
    ]
   ]
  },
  "inverse": true,
  "px_um": 2.399,
  "crop_cells": [
   50,
   123,
   43,
   129
  ],
  "crop_px_9362": [
   1024,
   2432,
   896,
   2560
  ],
  "label_crop": [
   512,
   1920,
   512,
   2176
  ]
 },
 "matrix_1203": {
  "transformation_matrix": [
   [
    0.256676,
    0.0,
    0.0,
    -18.8
   ],
   [
    0.0,
    0.256676,
    0.0,
    19.2
   ],
   [
    0.0,
    0.0,
    0.256676,
    7940.1
   ]
  ],
  "_comment": "DERIVED (not published by the Vesuvius open-data catalogue). Maps PHerc1203 volume 20260319130212 (2.403um/77keV, 15137x26493x26493) into volume 20250820131727 (9.362um/113keV, 18977x6844x6844). Row o"
 },
 "segments": {
  "auto_grown_20250930104534929": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20250930104534929/",
   "tiles": [
    {
     "ij": [
      108,
      189
     ],
     "set": "H",
     "contrast": 0.7029,
     "z9": 8202.7
    },
    {
     "ij": [
      108,
      180
     ],
     "set": "H",
     "contrast": 0.5071,
     "z9": 8100.3
    },
    {
     "ij": [
      99,
      189
     ],
     "set": "H",
     "contrast": 0.4228,
     "z9": 8341.3
    },
    {
     "ij": [
      99,
      180
     ],
     "set": "H",
     "contrast": 0.4205,
     "z9": 8236.7
    },
    {
     "ij": [
      27,
      108
     ],
     "set": "H",
     "contrast": 0.2664,
     "z9": 8358.7
    },
    {
     "ij": [
      90,
      162
     ],
     "set": "L",
     "contrast": 0.0374,
     "z9": 8155.3
    },
    {
     "ij": [
      45,
      126
     ],
     "set": "L",
     "contrast": 0.0405,
     "z9": 8338.7
    },
    {
     "ij": [
      45,
      135
     ],
     "set": "L",
     "contrast": 0.0469,
     "z9": 8432.1
    },
    {
     "ij": [
      81,
      162
     ],
     "set": "L",
     "contrast": 0.0502,
     "z9": 8273.8
    }
   ]
  },
  "auto_grown_20251005230830031": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20251005230830031/",
   "tiles": [
    {
     "ij": [
      171,
      117
     ],
     "set": "H",
     "contrast": 0.5333,
     "z9": 8512.0
    },
    {
     "ij": [
      171,
      108
     ],
     "set": "H",
     "contrast": 0.3979,
     "z9": 8684.1
    },
    {
     "ij": [
      180,
      117
     ],
     "set": "H",
     "contrast": 0.3926,
     "z9": 8567.4
    },
    {
     "ij": [
      180,
      108
     ],
     "set": "H",
     "contrast": 0.3669,
     "z9": 8740.2
    },
    {
     "ij": [
      180,
      90
     ],
     "set": "H",
     "contrast": 0.363,
     "z9": 9033.6
    },
    {
     "ij": [
      162,
      117
     ],
     "set": "H",
     "contrast": 0.3075,
     "z9": 8444.5
    },
    {
     "ij": [
      90,
      72
     ],
     "set": "L",
     "contrast": 0.0335,
     "z9": 8681.3
    },
    {
     "ij": [
      54,
      72
     ],
     "set": "L",
     "contrast": 0.0432,
     "z9": 8293.4
    },
    {
     "ij": [
      99,
      45
     ],
     "set": "L",
     "contrast": 0.0462,
     "z9": 9333.2
    },
    {
     "ij": [
      117,
      18
     ],
     "set": "L",
     "contrast": 0.0467,
     "z9": 9977.6
    },
    {
     "ij": [
      72,
      72
     ],
     "set": "L",
     "contrast": 0.0471,
     "z9": 8728.3
    },
    {
     "ij": [
      9,
      144
     ],
     "set": "L",
     "contrast": 0.0474,
     "z9": 8254.4
    },
    {
     "ij": [
      54,
      126
     ],
     "set": "L",
     "contrast": 0.0502,
     "z9": 8152.3
    },
    {
     "ij": [
      81,
      45
     ],
     "set": "L",
     "contrast": 0.052,
     "z9": 9019.6
    },
    {
     "ij": [
      63,
      54
     ],
     "set": "L",
     "contrast": 0.0524,
     "z9": 8760.7
    },
    {
     "ij": [
      126,
      63
     ],
     "set": "L",
     "contrast": 0.0524,
     "z9": 9234.2
    },
    {
     "ij": [
      81,
      72
     ],
     "set": "L",
     "contrast": 0.0532,
     "z9": 8587.8
    },
    {
     "ij": [
      72,
      45
     ],
     "set": "L",
     "contrast": 0.0536,
     "z9": 8954.3
    },
    {
     "ij": [
      198,
      117
     ],
     "set": "L",
     "contrast": 0.0539,
     "z9": 8649.2
    },
    {
     "ij": [
      153,
      126
     ],
     "set": "L",
     "contrast": 0.0557,
     "z9": 8226.2
    },
    {
     "ij": [
      99,
      63
     ],
     "set": "L",
     "contrast": 0.0561,
     "z9": 9003.6
    },
    {
     "ij": [
      135,
      99
     ],
     "set": "L",
     "contrast": 0.0563,
     "z9": 8658.0
    },
    {
     "ij": [
      144,
      117
     ],
     "set": "L",
     "contrast": 0.0563,
     "z9": 8347.4
    },
    {
     "ij": [
      18,
      144
     ],
     "set": "L",
     "contrast": 0.0564,
     "z9": 8127.7
    }
   ]
  },
  "auto_grown_20250925223153537": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20250925223153537/",
   "tiles": [
    {
     "ij": [
      63,
      18
     ],
     "set": "H",
     "contrast": 0.4511,
     "z9": 8752.3
    },
    {
     "ij": [
      72,
      18
     ],
     "set": "H",
     "contrast": 0.3436,
     "z9": 8597.3
    },
    {
     "ij": [
      54,
      45
     ],
     "set": "H",
     "contrast": 0.3429,
     "z9": 8710.7
    },
    {
     "ij": [
      54,
      18
     ],
     "set": "H",
     "contrast": 0.3311,
     "z9": 8929.0
    },
    {
     "ij": [
      63,
      9
     ],
     "set": "H",
     "contrast": 0.3299,
     "z9": 8800.8
    },
    {
     "ij": [
      45,
      45
     ],
     "set": "H",
     "contrast": 0.3285,
     "z9": 8865.4
    },
    {
     "ij": [
      45,
      54
     ],
     "set": "H",
     "contrast": 0.3211,
     "z9": 8779.2
    },
    {
     "ij": [
      63,
      27
     ],
     "set": "H",
     "contrast": 0.3015,
     "z9": 8705.9
    },
    {
     "ij": [
      54,
      36
     ],
     "set": "H",
     "contrast": 0.2886,
     "z9": 8796.7
    },
    {
     "ij": [
      36,
      81
     ],
     "set": "H",
     "contrast": 0.2853,
     "z9": 8677.2
    },
    {
     "ij": [
      72,
      54
     ],
     "set": "L",
     "contrast": 0.0436,
     "z9": 8320.8
    }
   ]
  },
  "auto_grown_20251005230118636": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20251005230118636/",
   "tiles": [
    {
     "ij": [
      99,
      180
     ],
     "set": "H",
     "contrast": 0.3547,
     "z9": 8237.2
    },
    {
     "ij": [
      108,
      162
     ],
     "set": "H",
     "contrast": 0.2966,
     "z9": 8242.5
    },
    {
     "ij": [
      99,
      162
     ],
     "set": "H",
     "contrast": 0.2816,
     "z9": 8120.0
    },
    {
     "ij": [
      135,
      144
     ],
     "set": "H",
     "contrast": 0.2553,
     "z9": 8350.5
    },
    {
     "ij": [
      108,
      171
     ],
     "set": "H",
     "contrast": 0.2463,
     "z9": 8301.6
    },
    {
     "ij": [
      99,
      171
     ],
     "set": "H",
     "contrast": 0.2414,
     "z9": 8160.5
    }
   ]
  },
  "auto_grown_20250925165041633": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20250925165041633/",
   "tiles": [
    {
     "ij": [
      117,
      117
     ],
     "set": "H",
     "contrast": 0.3375,
     "z9": 8445.3
    },
    {
     "ij": [
      108,
      117
     ],
     "set": "H",
     "contrast": 0.3233,
     "z9": 8351.5
    },
    {
     "ij": [
      99,
      108
     ],
     "set": "H",
     "contrast": 0.3105,
     "z9": 8115.6
    },
    {
     "ij": [
      117,
      99
     ],
     "set": "H",
     "contrast": 0.2676,
     "z9": 8176.9
    },
    {
     "ij": [
      108,
      99
     ],
     "set": "H",
     "contrast": 0.2459,
     "z9": 8073.8
    }
   ]
  },
  "auto_grown_20251005231446965": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20251005231446965/",
   "tiles": [
    {
     "ij": [
      153,
      189
     ],
     "set": "H",
     "contrast": 0.3308,
     "z9": 8131.6
    },
    {
     "ij": [
      153,
      144
     ],
     "set": "H",
     "contrast": 0.2914,
     "z9": 8063.1
    },
    {
     "ij": [
      153,
      207
     ],
     "set": "H",
     "contrast": 0.2487,
     "z9": 8160.8
    },
    {
     "ij": [
      216,
      135
     ],
     "set": "L",
     "contrast": 0.0235,
     "z9": 9077.7
    },
    {
     "ij": [
      225,
      117
     ],
     "set": "L",
     "contrast": 0.0287,
     "z9": 8932.8
    },
    {
     "ij": [
      216,
      117
     ],
     "set": "L",
     "contrast": 0.0289,
     "z9": 8844.6
    },
    {
     "ij": [
      225,
      144
     ],
     "set": "L",
     "contrast": 0.0289,
     "z9": 9100.0
    },
    {
     "ij": [
      180,
      99
     ],
     "set": "L",
     "contrast": 0.0331,
     "z9": 8349.6
    },
    {
     "ij": [
      171,
      135
     ],
     "set": "L",
     "contrast": 0.0366,
     "z9": 8348.8
    },
    {
     "ij": [
      225,
      135
     ],
     "set": "L",
     "contrast": 0.0382,
     "z9": 9194.6
    },
    {
     "ij": [
      225,
      108
     ],
     "set": "L",
     "contrast": 0.0383,
     "z9": 8763.7
    },
    {
     "ij": [
      189,
      153
     ],
     "set": "L",
     "contrast": 0.0385,
     "z9": 8735.9
    },
    {
     "ij": [
      180,
      117
     ],
     "set": "L",
     "contrast": 0.0386,
     "z9": 8361.1
    },
    {
     "ij": [
      207,
      153
     ],
     "set": "L",
     "contrast": 0.0417,
     "z9": 8990.7
    },
    {
     "ij": [
      189,
      135
     ],
     "set": "L",
     "contrast": 0.0445,
     "z9": 8680.5
    },
    {
     "ij": [
      189,
      144
     ],
     "set": "L",
     "contrast": 0.0445,
     "z9": 8686.4
    },
    {
     "ij": [
      189,
      81
     ],
     "set": "L",
     "contrast": 0.0465,
     "z9": 8473.9
    },
    {
     "ij": [
      207,
      126
     ],
     "set": "L",
     "contrast": 0.0466,
     "z9": 8910.9
    },
    {
     "ij": [
      216,
      108
     ],
     "set": "L",
     "contrast": 0.0479,
     "z9": 8679.5
    },
    {
     "ij": [
      207,
      117
     ],
     "set": "L",
     "contrast": 0.0483,
     "z9": 8772.2
    },
    {
     "ij": [
      189,
      90
     ],
     "set": "L",
     "contrast": 0.0493,
     "z9": 8391.9
    },
    {
     "ij": [
      207,
      90
     ],
     "set": "L",
     "contrast": 0.0518,
     "z9": 8482.8
    },
    {
     "ij": [
      198,
      135
     ],
     "set": "L",
     "contrast": 0.052,
     "z9": 8831.5
    },
    {
     "ij": [
      180,
      108
     ],
     "set": "L",
     "contrast": 0.0524,
     "z9": 8343.4
    }
   ]
  },
  "auto_grown_20251005221856743": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20251005221856743/",
   "tiles": [
    {
     "ij": [
      180,
      153
     ],
     "set": "H",
     "contrast": 0.32,
     "z9": 8678.9
    },
    {
     "ij": [
      162,
      153
     ],
     "set": "H",
     "contrast": 0.2928,
     "z9": 8459.4
    },
    {
     "ij": [
      153,
      153
     ],
     "set": "H",
     "contrast": 0.2726,
     "z9": 8278.4
    },
    {
     "ij": [
      162,
      144
     ],
     "set": "H",
     "contrast": 0.2636,
     "z9": 8449.0
    },
    {
     "ij": [
      144,
      117
     ],
     "set": "H",
     "contrast": 0.2635,
     "z9": 8051.7
    },
    {
     "ij": [
      207,
      90
     ],
     "set": "H",
     "contrast": 0.2579,
     "z9": 8313.1
    },
    {
     "ij": [
      153,
      162
     ],
     "set": "H",
     "contrast": 0.2555,
     "z9": 8203.7
    },
    {
     "ij": [
      144,
      144
     ],
     "set": "H",
     "contrast": 0.2553,
     "z9": 8119.7
    },
    {
     "ij": [
      153,
      90
     ],
     "set": "H",
     "contrast": 0.2533,
     "z9": 8142.3
    },
    {
     "ij": [
      153,
      180
     ],
     "set": "H",
     "contrast": 0.2457,
     "z9": 8106.7
    }
   ]
  },
  "auto_grown_20250923164615417": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20250923164615417/",
   "tiles": [
    {
     "ij": [
      9,
      72
     ],
     "set": "H",
     "contrast": 0.2764,
     "z9": 8070.9
    }
   ]
  },
  "auto_grown_20250923164608661": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20250923164608661/",
   "tiles": [
    {
     "ij": [
      135,
      72
     ],
     "set": "H",
     "contrast": 0.2711,
     "z9": 8184.7
    }
   ]
  },
  "auto_grown_20251005231446963": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20251005231446963/",
   "tiles": [
    {
     "ij": [
      90,
      108
     ],
     "set": "H",
     "contrast": 0.2602,
     "z9": 8074.6
    },
    {
     "ij": [
      117,
      72
     ],
     "set": "L",
     "contrast": 0.035,
     "z9": 8326.0
    },
    {
     "ij": [
      126,
      63
     ],
     "set": "L",
     "contrast": 0.0406,
     "z9": 8347.2
    }
   ]
  },
  "auto_grown_20251005230830030": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20251005230830030/",
   "tiles": [
    {
     "ij": [
      144,
      36
     ],
     "set": "L",
     "contrast": 0.0486,
     "z9": 8470.9
    }
   ]
  },
  "auto_grown_20250930000321811": {
   "seg_dir": "PHerc1203/segments/raw/auto_grown_20250930000321811/",
   "tiles": [
    {
     "ij": [
      126,
      45
     ],
     "set": "L",
     "contrast": 0.0547,
     "z9": 8105.5
    }
   ]
  }
 },
 "H_contrast": [
  0.703,
  0.533,
  0.507,
  0.451,
  0.423,
  0.421,
  0.398,
  0.393,
  0.367,
  0.363,
  0.355,
  0.344,
  0.343,
  0.338,
  0.331,
  0.331,
  0.33,
  0.328,
  0.323,
  0.321,
  0.32,
  0.31,
  0.307,
  0.301,
  0.297,
  0.293,
  0.291,
  0.289,
  0.285,
  0.282,
  0.276,
  0.273,
  0.271,
  0.268,
  0.266,
  0.264,
  0.264,
  0.26,
  0.258,
  0.255,
  0.255,
  0.255,
  0.253,
  0.249,
  0.246,
  0.246,
  0.246,
  0.241
 ],
 "L_contrast": [
  0.023,
  0.029,
  0.029,
  0.029,
  0.033,
  0.033,
  0.035,
  0.037,
  0.037,
  0.038,
  0.038,
  0.038,
  0.039,
  0.04,
  0.041,
  0.042,
  0.043,
  0.044,
  0.045,
  0.045,
  0.046,
  0.047,
  0.047,
  0.047,
  0.047,
  0.047,
  0.047,
  0.048,
  0.048,
  0.049,
  0.049,
  0.05,
  0.05,
  0.052,
  0.052,
  0.052,
  0.052,
  0.052,
  0.052,
  0.053,
  0.054,
  0.054,
  0.055,
  0.056,
  0.056,
  0.056,
  0.056,
  0.056
 ]
}
SELECTION_JSON

}
write_scripts
say "scripts written to $SCRIPTS"

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
