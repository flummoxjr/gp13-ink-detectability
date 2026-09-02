#!/bin/bash
# =============================================================================
# tAB_0358_screen.sh -- FIRST-EVER ink screen of PHerc0358's first-ever
# surfaces. v1, authored 2026-08-25 AFTER PREREG_0358.md (sha256 below).
#
#   For each of the 8 gate-PASS patches grown 2026-08-25 (commit 0d902ef,
#   hunt/pherc0358_first_surfaces/paths_0358/): render a 21-slice surface
#   volume from the masked 9.362um CT, infer ink_9um seed42 FORWARD and
#   REVERSE, save full-res maps + strided ds4 npys (survey-verbatim battery
#   inputs) + ds8 previews and survey stats.
#   MEASUREMENT ONLY: no battery statistic, no gate, no verdict is computed
#   on the pod. The v2 5-gate battery runs LOCALLY after fetch, per
#   PREREG_0358.md (flag rule >=4/5 computable gates -> "region worth human
#   inspection", never letter-language).
#
# THE HONESTY FRAME (pre-registered, sha-logged as status line 3 before any
# provisioning or data): ink_9um measured CHANCE (pixel AUC 0.5382 fwd /
# 0.5055 rev, out/transfer_ladder/p2a_win1_baseline.json) on a clean foreign
# scroll. PHerc0358 is a foreign scroll. EXPECTED OUTCOME: NULL with BOUNDED
# SENSITIVITY -- a blank screen is weak evidence about ink. The run's value:
# (a) first-ever look at this scroll's surfaces, (b) the maps + meshes as
# community artifacts, (c) any local battery flag would be extraordinary and
# goes to human eyes + the escalation path, never to announcement.
#
# MESHES (identity frozen; per-file sha256 embedded, verified on-pod):
#   preferred : /workspace/tAB_0358_meshes.tgz  (scp it beside this script;
#               sha256 a18408e8b217a84a0b14bcd65837bc63db104c662557ca778101822114cfc662,
#               1760903 bytes)
#   fallback  : raw.githubusercontent.com/flummoxjr/measure-before-you-hunt/
#               master/hunt/pherc0358_first_surfaces/paths_0358/<patch>/<file>
#               NOTE (verified 2026-08-25): meta.json resolves TODAY, but the
#               x/y/z.tif payload is NOT in the repo yet -- .gitignore's
#               '*.tif' silently dropped it from commit 0d902ef. Before a
#               fetch-from-repo launch, push the fix:
#                 echo '!hunt/pherc0358_first_surfaces/**/*.tif' >> .gitignore
#                 git add .gitignore hunt/pherc0358_first_surfaces/paths_0358
#                 git commit && git push
#               Either path ends in the same 32-file sha256 gate.
#
# -----------------------------------------------------------------------------
# OBSERVABILITY FIRST (the v1/v2 pod_curve_audit contract, proven live by v12):
#   THE FIRST THREE STATUS LINES (t0 = container start):
#     L1  t ~= 0-1 s  "<utc> BOOT tAB_0358_screen v1 pid=... host=... root=..."
#     L2  t ~= 1-3 s  "<utc> SERVE http.server pid=... 0.0.0.0:8000 dir=...
#                     external=https://<POD_ID>-8000.proxy.runpod.net/status.txt
#                     local_probe=200"
#     L3  t ~= 2-4 s  "<utc> PREREG locked prereg.json sha256=... doc=PREREG_0358.md
#                     sha256=1a870223... expected=NULL-with-bounded-sensitivity"
#   HEARTBEAT every 60 s (stage, uptime, free disk) + http-server watchdog.
#   Stages open/close with === STAGE lines. "ALL DONE" is printed ONLY by
#   finalize after re-verifying the full inventory (8/8 patches healthy);
#   every failure path exits through die() into a FAILED linger loop.
#
# LAUNCH (pod: 1x RTX 5090 or >=16GB CUDA, runpod/pytorch image, volume
# >= 40 GB at /workspace, HTTP PORT 8000 EXPOSED):
#   scp tAB_0358_screen.sh tAB_0358_meshes.tgz root@POD:/workspace/
#   ssh root@POD 'sed -i "s/\r$//" /workspace/tAB_0358_screen.sh;
#                 nohup bash /workspace/tAB_0358_screen.sh </dev/null \
#                   >/workspace/screen0358_nohup.out 2>&1 & disown; echo KICKED'
#   curl -s https://<POD_ID>-8000.proxy.runpod.net/status.txt | tail -30
#   # on ALL DONE, fetch BOTH trees (v12 lesson: unfetched maps die with the pod):
#   scp -r root@POD:/workspace/screen0358/out   ./screen0358_out
#   scp -r root@POD:/workspace/screen0358/preds ./screen0358_preds
#   # THEN TERMINATE THE POD (it lingers serving results and bills until you do).
#
# TIMING / COST (honest; RTX 5090 $0.69-0.89/hr surveyed):
#   provision 2-15 min (126 s measured on a warm image) ; ckpt 138MB ~1-2 min ;
#   meshes + volume probe <2 min ; per patch 4-11 min (render 3040^2 x21 from
#   S3 2-6, infer fwd+rev ~1.5-4, stats <1) x8 serial = 32-88 min ;
#   finalize <1 min.  TOTAL ~0.7-1.75 h  => ~$0.50-$1.55.
#   (GPU-active time itself ~15-40 min; serial S3 render latency is the wide
#   variable. 8 renders + 16 inference passes, model loaded per pass -- the
#   exact 80-segment-proven survey invocation, nothing clever.)
#
# Env knobs: PORT=8000 BATCH=16 WORKERS=8 FORCE=1 (rebuild) DRY=1 (machinery
#   only, no network/GPU; DRY_FAIL_STAGE=<stage> injects a failure)
#   LINGER_EXIT=1 (exit instead of linger loops; local validation)
#   MESH_TGZ=/workspace/tAB_0358_meshes.tgz  MESH_TGZ_URL= (optional http src)
#   AUTOSTOP=0 (1 = runpodctl stop 30 min after ALL DONE, if available)
# This script never calls the RunPod API except the optional AUTOSTOP.
# House rules honored: status+server FIRST via PID file; NO pkill anywhere;
# per-op timeouts on every network/GPU op; logs inside the served dir; prereg
# sha-logged before data; ALL DONE unreachable on failure.
#
# ADVERSARIAL REVIEW 2026-08-25 (reviewer, not author) -- v1.1 changes:
#  R1 battery comparability: the pod now ALSO saves strided ds4 npys
#     (map[::4,::4] uint8) -- the EXACT survey_segments.py decimation the
#     71-segment corpus + w035_CONTROL_strided control used. PREREG had
#     wrongly said "block-mean ds4 = control convention"; corrected there too.
#  R2 finalize false-refusal: infer writes LZW-compressed tifs, so the old
#     ">1MB file size" floor could refuse a HEALTHY sparse map (the expected
#     near-null regime!). Replaced with a real tifffile shape==3040x3040 check.
#  R3 per-patch health failure (stats exit 3) now records INFRA_INCOMPLETE
#     and CONTINUES to the remaining patches (a patch-local failure must not
#     cost the other seven their first-ever measurement); finalize still
#     refuses ALL DONE unless 8/8 healthy. Other stats exits still die.
#  R4 DRY stage markers no longer poison a later wet run in the same ROOT.
#  R5 stale out/FAILED marker now cleared when a resumed run reaches ALL DONE.
#  R6 volprobe/stats/finalize wrapped in per-op timeouts (pyrun_t).
#  R7 PREREG_0358.md sha updated after its review amendments (base-rate
#     conflation fixed: >=4/5 fired 0/71 on GP corpus, not "2 vs 3.55";
#     control fwd/rev r is 0.094 survey-convention, not 0.076).
# =============================================================================

set -Eeuo pipefail

# ---------------------------------------------------------------- config ----
ROOT=${ROOT:-/workspace/screen0358}
PORT=${PORT:-8000}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}
FORCE=${FORCE:-0}
DRY=${DRY:-0}
DRY_FAIL_STAGE=${DRY_FAIL_STAGE:-}
LINGER_EXIT=${LINGER_EXIT:-0}
AUTOSTOP=${AUTOSTOP:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
MESH_TGZ=${MESH_TGZ:-/workspace/tAB_0358_meshes.tgz}
MESH_TGZ_URL=${MESH_TGZ_URL:-}

SCROLL=PHerc0358
VOLNAME="20250821151737-9.362um-1.2m-113keV-masked.zarr"
VOLURL="s3://vesuvius-challenge-open-data/$SCROLL/volumes/$VOLNAME"
VOLHTTP="https://vesuvius-challenge-open-data.s3.amazonaws.com/$SCROLL/volumes/$VOLNAME"
RAWBASE="https://raw.githubusercontent.com/flummoxjr/measure-before-you-hunt/master/hunt/pherc0358_first_surfaces/paths_0358"
MESH_TGZ_SHA=a18408e8b217a84a0b14bcd65837bc63db104c662557ca778101822114cfc662
PREREG_MD_SHA=1a87022385584881b25be20b511ac63457c7b05154a80a55590499d807f5814d
CKPT=/workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth

PATCHES=(
  auto_grown_20260825155611879
  auto_grown_20260825155613379
  auto_grown_20260825155615680
  auto_grown_20260825155619178
  auto_grown_20260825155619482
  auto_grown_20260825155621418
  auto_grown_20260825155624780
  auto_grown_20260825155625980
)

OUT=$ROOT/out;  VAR=$ROOT/var;  DATA=$ROOT/data;  PREDS=$ROOT/preds
SCRIPTS=$ROOT/scripts;  RESULTS=$OUT/results;  STATUS=$OUT/status.txt
export ROOT OUT VAR DATA PREDS SCRIPTS RESULTS STATUS BATCH WORKERS VOLURL VOLHTTP CKPT

# =================================================================== L1 ======
mkdir -p "$OUT" "$VAR" "$DATA" "$PREDS" "$SCRIPTS" "$RESULTS" "$OUT/maps" "$OUT/previews"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(now) $*" >> "$STATUS"; echo "$(now) $*"; }
say "BOOT tAB_0358_screen v1 pid=$$ host=${HOSTNAME:-unknown} root=$ROOT -- status live; server next"

# =================================================================== L2 ======
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

( while :; do
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
) >/dev/null 2>&1 &
HEART_PID=$!
echo boot > "$VAR/stage"

cleanup() {
  kill "$HEART_PID" 2>/dev/null || true
  SP=$(cat "$VAR/server.pid" 2>/dev/null || true)
  [ -n "$SP" ] && kill "$SP" 2>/dev/null || true
}
trap cleanup EXIT

# =================================================================== L3 ======
# Decision rules + honesty frame locked BEFORE provisioning or any data.
cat > "$OUT/prereg.json" <<'PREREG_JSON'
{
  "locked_before": "any provisioning, download, or data contact on this pod; full document PREREG_0358.md (sha256 1a87022385584881b25be20b511ac63457c7b05154a80a55590499d807f5814d) written 2026-08-25 and amended by adversarial review the same day, before launch and before any PHerc0358 data existed",
  "experiment": "first-ever ink screen of PHerc0358's first-ever surfaces: 8 gate-PASS patches (commit 0d902ef), render 21-slice SV from the masked 9.362um CT, infer ink_9um seed42 forward+reverse, save maps+stats. MEASUREMENT ONLY -- no battery statistic, gate, or verdict is computed on this pod.",
  "expected_outcome": "NULL with BOUNDED SENSITIVITY. ink_9um scored pixel AUC 0.5382 fwd / 0.5055 rev (CHANCE) on the clean foreign-scroll anchor 500p2a win1 (out/transfer_ladder/p2a_win1_baseline.json, 2026-08-25). PHerc0358 is likewise foreign to ink_9um. A blank screen is weak evidence about ink and is NOT evidence these patches carry no text.",
  "value_if_null": ["first-ever look at any recoverable surface of this scroll", "maps + correctly-oriented meshes as community artifacts for anyone with a better model", "a matched null for the local battery"],
  "local_battery": "PROTOCOL_V2 (analyze_survey_corpus_v2.py) 5 gates, run LOCALLY after fetch on the pod's strided ds4 npys (map[::4,::4] uint8, 37.448 um/px -- the exact survey_segments.py decimation the 71-segment corpus and the w035_CONTROL_strided control used; ds8 npys are preview-only) of the FORWARD maps; reverse maps descriptive only; gate 5 from the pod's full-res fwd/rev Pearson r; w035_CONTROL_strided must reproduce 5/5 before any patch is scored",
  "flag_rule": "a patch is 'flagged: region worth human inspection' iff its forward map passes >= 4 of 5 computable gates (non-computable = not passed); identical to Track F. NEVER letter-language.",
  "multiplicity": "8 flag-bearing tests; gate_significance raw p<=0.05 per v2 with Holm-8 reported beside; measured base rates (corpus_analysis_v2.json + null_scaling/ns4): >=4/5 flag rule fired 0/71 on the GP corpus (gate_fwd_rev failed all 71, min |r|=0.2215); gate_significance 4/71 raw (3.55 expected by chance), 2/71 under corrected nulls; foreign-scroll fwd/rev bleed is unmeasured -- if gate_fwd_rev passes here a flag needs >=3 of the other 4 gates, an 11/71 (15.5%) base rate, so expect 0 to ~1.2 chance flags among 8; up to 2 flags fully consistent with chance",
  "escalation": "any flag -> >=1000-perm gate-1 rerun + human eyes on map and CT BEFORE any public language; at most 'region worth human inspection'; report which gates carried the flag",
  "infra_vs_science": "data-gate mismatch or render/infer hard failure kills the run (infra); canvas != 3040x3040 or forward nonzero_frac < 0.05 = INFRA_INCOMPLETE recorded for that patch and the run CONTINUES to the remaining patches; either way ALL DONE requires 8/8 healthy; a null verdict may only come from a healthy map",
  "volume": {"name": "20250821151737-9.362um-1.2m-113keV-masked.zarr", "level0": {"shape": [14744, 7783, 7783], "chunks": [128, 128, 128], "dtype": "|u1", "dimension_separator": "/", "compressor": null, "fill_value": 0}},
  "model": {"repo": "scrollprize/ink_9um", "file": "hybrid_3d2d-seed42/step-075000.pth", "bytes": 138360039, "mode": "flat", "crop": [17, 128, 128], "norm": "robust_mad"},
  "render": {"num_slices": 21, "slice_step": 1.0, "renderer": "render_tifxyz_sv.py (r=0.813 validated; 80-segment survey harness)"},
  "meshes": {"n": 8, "grid": [152, 152], "scale": 0.05, "canvas": [3040, 3040], "px_um": 9.362, "sha256": "per-file manifest embedded in this script; tgz a18408e8b217a84a0b14bcd65837bc63db104c662557ca778101822114cfc662"},
  "patches": ["auto_grown_20260825155611879", "auto_grown_20260825155613379", "auto_grown_20260825155615680", "auto_grown_20260825155619178", "auto_grown_20260825155619482", "auto_grown_20260825155621418", "auto_grown_20260825155624780", "auto_grown_20260825155625980"]
}
PREREG_JSON
PRSHA=$(sha256sum "$OUT/prereg.json" | cut -c1-12)
say "PREREG locked prereg.json sha256=$PRSHA doc=PREREG_0358.md sha256=${PREREG_MD_SHA:0:12} expected=NULL-with-bounded-sensitivity (transfer anchor 0.5382) -- rules recorded before any provisioning, download, or data"

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

retry() { # retry <tries> <cmd...>  backoff 10/30/90/180
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
# same, with a wall-clock cap (house rule: per-op timeouts on data-path ops)
pyrun_t() { local T=$1; shift
  (cd /workspace/villa/vesuvius && timeout "$T" uv run --no-sync --extra models python "$@"); }

# render with per-attempt wall clock; log lives in the served dir
run_render() { # run_render <mesh_dir> <sv_zarr>
  (cd /workspace/villa/vesuvius && timeout 4000 uv run --no-sync --extra models \
     python "$SCRIPTS/render_tifxyz_sv.py" "$1" "$VOLURL" "$2" --num-slices 21) \
     >> "$OUT/render.log" 2>&1
}

# single-direction inference, the exact 80-segment-proven survey invocation,
# with per-attempt timeout + one retry (template hygiene: tmp then mv).
run_infer_dir() { # run_infer_dir <sv_zarr> <out_tif> <forward|reverse>
  local zarr=$1 out=$2 direction=$3
  if [ -s "$out" ] && [ "$FORCE" != 1 ]; then
    say "infer skip (exists): $(basename "$out") [$direction]"; return 0
  fi
  local tmp=$PREDS/tmp_$(basename "$out")
  rm -f "$tmp"
  say "infer OPEN $(basename "$zarr") -> $(basename "$out") [$direction]"
  local t0=$SECONDS
  if ! (cd /workspace/villa/vesuvius && timeout 3600 uv run --no-sync --extra models \
        python -m vesuvius.ink_detection.inference.infer \
        "$zarr" "$CKPT" "$tmp" --direction "$direction" \
        --batch-size "$BATCH" --num-workers "$WORKERS" --gpus 0) >> "$OUT/infer.log" 2>&1; then
    rm -f "$tmp"
    say "infer FIRST ATTEMPT FAILED $(basename "$out") -- retrying once in 20s (tail of infer.log on :$PORT)"
    sleep 20
    (cd /workspace/villa/vesuvius && timeout 3600 uv run --no-sync --extra models \
        python -m vesuvius.ink_detection.inference.infer \
        "$zarr" "$CKPT" "$tmp" --direction "$direction" \
        --batch-size "$BATCH" --num-workers "$WORKERS" --gpus 0) >> "$OUT/infer.log" 2>&1 \
      || { rm -f "$tmp"; die "inference failed twice: $(basename "$out") [$direction]"; }
  fi
  [ -s "$tmp" ] || die "inference produced no output: $(basename "$out") [$direction]"
  mv -f "$tmp" "$out"
  say "infer DONE $(basename "$out") [$direction] ($((SECONDS - t0))s)"
}

# ============================================================================
# Embedded programs + data files -- written before any stage so the exact
# code and the frozen mesh identity are on disk (and served) up front.
# ============================================================================
write_scripts() {

# ---- mesh identity: 32-file sha256 manifest (computed 2026-08-25 locally) --
cat > "$VAR/mesh_manifest.txt" <<'MANIFEST'
0cae74577c8b50bc23a2158b1c5b72dcad504a7de83fa4ab1a77267d94734b00  paths_0358/auto_grown_20260825155611879/meta.json
84cdd2413927350191f4be76efc6b5487b213a869430e1a7fb8997e4d4bedd84  paths_0358/auto_grown_20260825155611879/x.tif
0e22f7078863db26a26f8a81049b1a7c265891919e847df3b4d33d5c5b3ac5c5  paths_0358/auto_grown_20260825155611879/y.tif
19250908353b73448006f4c48554135e919bf08bea4e326850686039ff9c6088  paths_0358/auto_grown_20260825155611879/z.tif
16a5d617f0e5bbd234b0fdab50fbbe8fc82c6b98644526b9d17a443973d63059  paths_0358/auto_grown_20260825155613379/meta.json
d4ed9fcbc449f610826d5283caac04ad5ed936b6c822d91a084700af203707d6  paths_0358/auto_grown_20260825155613379/x.tif
90294d2dcb5e8268230b6cf2b69a3bd0758b13ef05322543897fa035c6a56bed  paths_0358/auto_grown_20260825155613379/y.tif
652e548b1793f41099cb549f9aa099b31510b7dea18e3cd92b0920e23617cd3f  paths_0358/auto_grown_20260825155613379/z.tif
2518f0030612563699a1742cffc8b4e24db335c018e2abd59bc0e761d3262c15  paths_0358/auto_grown_20260825155615680/meta.json
0a4550dcd0c54522a9a506f7d864dcb42ea78e52b7b4f9baf9d1bd2da54ebe51  paths_0358/auto_grown_20260825155615680/x.tif
d933d4d64639fa93e361ca702a7a3f339ec94630105115586cb44447f5002e40  paths_0358/auto_grown_20260825155615680/y.tif
420f38d39f27af098c5eedca7082c8fee60effa56ab467b3e53b86ddc33a6723  paths_0358/auto_grown_20260825155615680/z.tif
56b67caac471aba0827f27869adc383de8807a69c02ad3a113f783f6dd64f084  paths_0358/auto_grown_20260825155619178/meta.json
00a3323b9114a95de09f333a21d3c15dc2affe6181a3f8f69bc732f7548c30d8  paths_0358/auto_grown_20260825155619178/x.tif
c445c94fba3c78ab3f38976a3025489b459fed8f5b0b3fbfef1e6bc5a857dba9  paths_0358/auto_grown_20260825155619178/y.tif
1d71e862d5e7b4568364ec20ca28f17b87fb2a24372099efd593243203888f0d  paths_0358/auto_grown_20260825155619178/z.tif
9f9626a891937ff42b9480c4e1161e3b29ab943cf506fef8db78fcae38b10a8b  paths_0358/auto_grown_20260825155619482/meta.json
d317f206160ceaeea15eb45c9bc44dbaa43f1a75c9f1ca1c08c74d813b7ccdd3  paths_0358/auto_grown_20260825155619482/x.tif
8e0bb9eb3c3e9ba777f54a11bd84c8b177b64b143e448ffb076a529f3f1e5d93  paths_0358/auto_grown_20260825155619482/y.tif
77c926c3f94398d15386e5a8558295275bb6d4f102244d9eb8bd5bb37b907c80  paths_0358/auto_grown_20260825155619482/z.tif
eeeeb0473f37e4d77d6bdd87a25cc575a068d5b7bd72d5254b59237acc883004  paths_0358/auto_grown_20260825155621418/meta.json
47850473675737f76cf3ed7ce9364f43c16b30747abdb0635360195c7682d13e  paths_0358/auto_grown_20260825155621418/x.tif
a48597f6c53f79666077502f1b411cf713f8e7f52f2711207f5507611942a896  paths_0358/auto_grown_20260825155621418/y.tif
895218b274a67efda5cdc56091afc0d6b28fe4abfdb32c139f4419ec9d38949f  paths_0358/auto_grown_20260825155621418/z.tif
466e52433410850e20a49ed340a3cd76e6ee0d8036cde2ab2298152fc3039f94  paths_0358/auto_grown_20260825155624780/meta.json
07f648664ebe77d7269ce029c3cbcec46a2e1fa732cb0fa7a265cb9cc4433a5b  paths_0358/auto_grown_20260825155624780/x.tif
b55efe26d35f7b19fc42b261d0b12d360223e53ea0deed403724892ca37701ee  paths_0358/auto_grown_20260825155624780/y.tif
40d512cb2b5be0b92feb7620397b5f9abb13caeef66842058b7c5c7b1ad4c00a  paths_0358/auto_grown_20260825155624780/z.tif
37f866a575fc5324884c3017fd906a0cb4f15358281f573940cbf5539093b256  paths_0358/auto_grown_20260825155625980/meta.json
b48768aaead205877a9dee27f7ad93d3436e526598adb6bcba8d98552d151642  paths_0358/auto_grown_20260825155625980/x.tif
5dae27639db2dcfbc5c3f4292a78dfdc5489eadc6d5bd06420edac223063605b  paths_0358/auto_grown_20260825155625980/y.tif
2356cd19f45b741ea4e65110ee548a5f7be100dfacc887af2939d7bf0a9d7791  paths_0358/auto_grown_20260825155625980/z.tif
MANIFEST

# ---- patch metadata (from alignment_gate.json, quoted for the artifacts) ---
cat > "$VAR/patch_meta.json" <<'PATCHMETA'
{
 "auto_grown_20260825155611879": {"seed_xyz": [2640, 4797, 11072], "separability": 0.823, "angle_local_deg": 4.8, "area_cm2": 8.693},
 "auto_grown_20260825155613379": {"seed_xyz": [5328, 6189, 7264], "separability": 0.791, "angle_local_deg": 5.4, "area_cm2": 8.804},
 "auto_grown_20260825155615680": {"seed_xyz": [1861, 2224, 5360], "separability": 0.785, "angle_local_deg": 5.0, "area_cm2": 8.454},
 "auto_grown_20260825155619178": {"seed_xyz": [1908, 4368, 11248], "separability": 0.778, "angle_local_deg": 5.0, "area_cm2": 8.768},
 "auto_grown_20260825155619482": {"seed_xyz": [4704, 5865, 7162], "separability": 0.778, "angle_local_deg": 6.2, "area_cm2": 8.561},
 "auto_grown_20260825155621418": {"seed_xyz": [3856, 4512, 10469], "separability": 0.764, "angle_local_deg": 3.6, "area_cm2": 8.406},
 "auto_grown_20260825155624780": {"seed_xyz": [2160, 2564, 9406], "separability": 0.762, "angle_local_deg": 13.1, "area_cm2": 8.412},
 "auto_grown_20260825155625980": {"seed_xyz": [1728, 2528, 3451], "separability": 0.743, "angle_local_deg": 18.6, "area_cm2": 9.119}
}
PATCHMETA

# ---- renderer: byte-for-byte copy of trackD/runpod/render_tifxyz_sv.py ----
# (r=0.813 validated; proven on-pod across the 80-segment survey)
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
    surf.use_full_resolution()
    H, W = surf.shape
    n = int(args.num_slices)
    offsets = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * float(args.slice_step)
    pad = int(math.ceil(np.abs(offsets).max())) + int(args.margin)
    print(f"full-res grid: {H} x {W}, {n} slices, offsets {offsets[0]}..{offsets[-1]}")

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
    rows = list(range(0, H, t))
    for i, r0 in enumerate(rows):
        for c0 in range(0, W, t):
            render_tile(r0, min(H, r0 + t), c0, min(W, c0 + t))
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

# ---- volume + mesh data gates (dies via nonzero exit on any mismatch) ------
cat > "$SCRIPTS/volprobe.py" <<'PY_VOLPROBE'
"""Data gates before any rendering: the live volume must be byte-identical in
identity to what was verified during authoring (2026-08-25), and every mesh
must be the frozen 152x152/scale-0.05 grid whose bbox fits inside the volume."""
import json
import os
import sys
import urllib.request

import numpy as np
import tifffile

DATA = os.environ["DATA"]
VOLHTTP = os.environ["VOLHTTP"]
EXPECT_ZARRAY = {"shape": [14744, 7783, 7783], "chunks": [128, 128, 128],
                 "dtype": "|u1", "dimension_separator": "/", "compressor": None,
                 "fill_value": 0, "order": "C", "zarr_format": 2, "filters": None}
PATCHES = json.load(open(os.path.join(os.environ["VAR"], "patch_meta.json")))

def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())

za = fetch(VOLHTTP + "/0/.zarray")
for k, v in EXPECT_ZARRAY.items():
    assert za.get(k) == v, f"VOLGATE {k}: live={za.get(k)!r} expected={v!r}"
zat = fetch(VOLHTTP + "/.zattrs")
assert "multiscales" in zat, "VOLGATE .zattrs has no multiscales"
print(f"VOLPROBE volume OK: level0 {za['shape']} {za['dtype']} chunks {za['chunks']}")

Z, Y, X = za["shape"]
for name in sorted(PATCHES):
    d = os.path.join(DATA, "paths_0358", name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["format"] == "tifxyz", (name, meta.get("format"))
    assert abs(meta["vc_gsfs_params"]["voxelsize"] - 9.362) < 1e-6, (name, "voxelsize")
    assert abs(meta["scale"][0] - 0.05) < 1e-6, (name, "scale")
    (x0, y0, z0), (x1, y1, z1) = meta["bbox"]
    assert 0 <= x0 and x1 < X and 0 <= y0 and y1 < Y and 0 <= z0 and z1 < Z, \
        f"MESHGATE {name} bbox {meta['bbox']} outside volume {za['shape']}"
    for f in ("x.tif", "y.tif", "z.tif"):
        a = tifffile.imread(os.path.join(d, f))
        assert a.shape == (152, 152) and a.dtype == np.float32, (name, f, a.shape, a.dtype)
    print(f"VOLPROBE mesh OK: {name} grid 152x152 bbox inside volume")
print("VOLPROBE ALL GATES PASS")
PY_VOLPROBE

# ---- per-patch stats + ds8 + previews (survey conventions, verbatim stats) -
cat > "$SCRIPTS/patch_stats.py" <<'PY_STATS'
"""Stats + artifacts for one patch's fwd/rev maps. Survey stats() verbatim;
adds: full-res fwd/rev Pearson r (battery gate-5 input), STRIDED ds4 npy maps
(map[::4,::4] uint8 -- the exact survey_segments.py decimation; these are the
local battery's inputs), block-mean ds8 npy maps + PNG previews (redundancy),
health gates (canvas exactly 3040x3040, forward nonzero_frac >= 0.05).
Appends one JSONL record; 'error' key = unhealthy -> exit 3 (recorded; the
run continues but finalize will refuse ALL DONE)."""
import json
import os
import sys
import time

import numpy as np
import tifffile
from PIL import Image

name = sys.argv[1]
PREDS, OUT, VAR = os.environ["PREDS"], os.environ["OUT"], os.environ["VAR"]
BLANK_P99 = 195  # control blank-papyrus p99 (DN) -- survey tripwire value gate
EXPECT_CANVAS = (3040, 3040)
META = json.load(open(os.path.join(VAR, "patch_meta.json")))[name]
t0 = time.time()


def stats(pred):
    """Compact battery-relevant stats on a prediction map (survey verbatim)."""
    m = pred > 0
    if m.sum() < 1000:
        return None
    v = pred[m]
    out = {
        "canvas": list(pred.shape), "nonzero_frac": round(float(m.mean()), 4),
        "max": int(pred.max()), "p50": float(np.percentile(v, 50)),
        "p99": float(np.percentile(v, 99)),
        "frac_gt_half": round(float((pred > 0.5 * max(pred.max(), 1)).mean()), 4),
        "frac_gt_blankp99": round(float((pred > BLANK_P99).mean()), 6),
    }
    try:
        from scipy import ndimage
        lab, n = ndimage.label(pred > BLANK_P99)
        trips = []
        if n:
            sizes = ndimage.sum_labels(np.ones_like(lab, np.float32), lab, range(1, n + 1))
            big = [int(i) + 1 for i in np.argsort(sizes)[::-1][:12] if sizes[int(i)] >= 1e4]
            for cid in big:
                sel = lab == cid
                dist = ndimage.distance_transform_edt(sel)
                width = float(dist.max() * 2)
                if width >= 30:
                    ys, xs = np.nonzero(sel)
                    trips.append({"area": int(sel.sum()), "width": round(width, 1),
                                  "centroid": [int(ys.mean()), int(xs.mean())]})
        out["tripwire_hits"] = trips
        out["n_hot_components"] = int(n)
    except Exception as e:
        out["tripwire_error"] = str(e)[:100]
    return out


def ds8(a):
    """Anti-aliased 8x block mean, uint8."""
    H, W = a.shape
    p = 8
    ph, pw = (-H) % p, (-W) % p
    b = np.pad(a, ((0, ph), (0, pw)))
    m = b.reshape((H + ph) // p, p, (W + pw) // p, p).mean(axis=(1, 3))
    return np.clip(np.rint(m), 0, 255).astype(np.uint8)


rec = {"name": name, "scroll": "PHerc0358",
       "volume": "20250821151737-9.362um-1.2m-113keV-masked.zarr",
       "model": "ink_9um hybrid_3d2d-seed42 step-075000", "px_um": 9.362,
       "meta": META, "ds4_method": "strided_4_uint8_survey_verbatim",
       "ds8_method": "block_mean_8_round_uint8"}
maps = {}
for tag in ("fwd", "rev"):
    p = os.path.join(PREDS, f"{name}_{tag}.tif")
    if not os.path.exists(p):
        rec["error"] = f"missing_{tag}_map"
        break
    a = tifffile.imread(p).astype(np.float32)
    maps[tag] = a
    if tuple(a.shape) != EXPECT_CANVAS:
        rec["error"] = f"canvas_{tag}_{a.shape}_expected_{EXPECT_CANVAS}"
        break
    rec[tag] = stats(a)
    # battery input: survey_segments.py-verbatim strided decimation, uint8
    np.save(os.path.join(OUT, "maps", f"{name}_{tag}_ds4.npy"),
            a[::4, ::4].astype(np.uint8))
    d8 = ds8(a)
    np.save(os.path.join(OUT, "maps", f"{name}_{tag}_ds8.npy"), d8)
    Image.fromarray(d8).save(os.path.join(OUT, "previews", f"{name}_{tag}_ds8.png"))

if "error" not in rec and len(maps) == 2:
    a, b = maps["fwd"], maps["rev"]
    mm = (a > 0) & (b > 0)
    if mm.sum() > 1000:
        rec["fwd_rev_r"] = round(float(np.corrcoef(a[mm], b[mm])[0, 1]), 4)
    else:
        rec["error"] = "no_common_support_for_fwd_rev_r"
if "error" not in rec:
    if rec.get("fwd") is None or rec["fwd"]["nonzero_frac"] < 0.05:
        rec["error"] = "forward_map_empty_or_below_health_floor_0.05"

rec["secs"] = round(time.time() - t0, 1)
with open(os.path.join(OUT, "screen_0358.jsonl"), "a") as f:
    f.write(json.dumps(rec) + "\n")
trips = (rec.get("fwd") or {}).get("tripwire_hits") or []
print(f"{name}: fwd_rev_r={rec.get('fwd_rev_r')} "
      f"fwd_p99={(rec.get('fwd') or {}).get('p99')} TRIPWIRE={len(trips)} "
      f"{'ERR ' + rec['error'] if 'error' in rec else 'healthy'}")
if "error" in rec:
    sys.exit(3)
PY_STATS

# ---- finalize: full inventory re-verification; the ONLY path to ALL DONE ---
cat > "$SCRIPTS/finalize.py" <<'PY_FINALIZE'
"""Refuses (nonzero exit) unless all 8 patches are complete and healthy:
16 full-res maps that OPEN and have the exact 3040x3040 canvas (a byte-size
floor would false-refuse healthy sparse maps -- infer writes LZW-compressed
tifs), 16 strided ds4 npys (battery inputs), 16 ds8 npys, 16 previews,
8 error-free JSONL records. Writes results/results.json and the SUMMARY
lines. Prints no verdicts -- verdicts belong to the LOCAL battery."""
import json
import os
import sys

import numpy as np
import tifffile

OUT, PREDS, VAR = os.environ["OUT"], os.environ["PREDS"], os.environ["VAR"]
EXPECT_CANVAS = (3040, 3040)
PATCHES = sorted(json.load(open(os.path.join(VAR, "patch_meta.json"))))
missing = []
recs = {}
jl = os.path.join(OUT, "screen_0358.jsonl")
if os.path.exists(jl):
    for line in open(jl):
        try:
            r = json.loads(line)
            recs[r["name"]] = r  # later records supersede (FORCE reruns)
        except Exception:
            pass

for name in PATCHES:
    for tag in ("fwd", "rev"):
        p = os.path.join(PREDS, f"{name}_{tag}.tif")
        if not os.path.exists(p):
            missing.append(f"pred {name}_{tag}.tif")
        else:
            try:
                a = tifffile.imread(p)
                if tuple(a.shape) != EXPECT_CANVAS:
                    missing.append(f"pred canvas {name}_{tag} {a.shape}")
            except Exception as e:
                missing.append(f"pred unreadable {name}_{tag}: {str(e)[:80]}")
        for suff, shape in (("ds4", (760, 760)), ("ds8", (380, 380))):
            d = os.path.join(OUT, "maps", f"{name}_{tag}_{suff}.npy")
            if not os.path.exists(d):
                missing.append(f"{suff} {name}_{tag}")
            else:
                a = np.load(d)
                if a.shape != shape:
                    missing.append(f"{suff} shape {name}_{tag} {a.shape}")
        if not os.path.exists(os.path.join(OUT, "previews", f"{name}_{tag}_ds8.png")):
            missing.append(f"preview {name}_{tag}")
    r = recs.get(name)
    if r is None:
        missing.append(f"jsonl record {name}")
    elif "error" in r:
        missing.append(f"unhealthy {name}: {r['error']}")
    elif r.get("fwd_rev_r") is None:
        missing.append(f"fwd_rev_r missing {name}")

if missing:
    print("FINALIZE REFUSED -- INFRA_INCOMPLETE:")
    for m in missing:
        print("  MISSING/UNHEALTHY:", m)
    print(f"completed_healthy={sum(1 for n in PATCHES if n in recs and 'error' not in recs[n])}/8")
    sys.exit(4)

res = {"experiment": "tAB_0358_screen v1", "n_patches": 8,
       "prereg_present": os.path.exists(os.path.join(OUT, "prereg.json")),
       "pod_scope": "measurement only -- battery verdicts are computed LOCALLY per PREREG_0358.md",
       "patches": []}
for name in PATCHES:
    r = recs[name]
    res["patches"].append({
        "name": name, "meta": r.get("meta"), "fwd_rev_r": r.get("fwd_rev_r"),
        "fwd": {k: r["fwd"].get(k) for k in
                ("nonzero_frac", "p50", "p99", "frac_gt_blankp99", "n_hot_components")},
        "n_tripwire_fwd": len((r.get("fwd") or {}).get("tripwire_hits") or []),
        "rev_p99": (r.get("rev") or {}).get("p99")})
with open(os.path.join(OUT, "results", "results.json"), "w") as f:
    json.dump(res, f, indent=1)
for p in res["patches"]:
    print(f"SUMMARY {p['name']} fwd_rev_r={p['fwd_rev_r']} "
          f"fwd_p99={p['fwd']['p99']} frac_gt_blankp99={p['fwd']['frac_gt_blankp99']} "
          f"tripwires={p['n_tripwire_fwd']}")
print("SUMMARY inventory complete: 8/8 patches healthy; maps+stats ready for the LOCAL battery")
PY_FINALIZE

say "embedded scripts + mesh manifest + patch metadata written to $SCRIPTS and $VAR"
}
write_scripts

# ============================================================================
# DRY mode: machinery + embedded-code compile check only (no network/GPU).
# ============================================================================
if [ "$DRY" = 1 ]; then
  touch "$VAR/dry_run"   # sentinel: a later WET run in this ROOT must not trust DRY stage markers
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m py_compile "$SCRIPTS/render_tifxyz_sv.py" "$SCRIPTS/volprobe.py" \
      "$SCRIPTS/patch_stats.py" "$SCRIPTS/finalize.py" || die "DRY: embedded python failed py_compile"
    printf 'import json, sys\nfor p in sys.argv[1:]:\n    json.load(open(p))\nprint("DRY: embedded JSON parses OK")\n' > "$VAR/jsoncheck.py"
    "$PYTHON_BIN" "$VAR/jsoncheck.py" "$OUT/prereg.json" "$VAR/patch_meta.json" \
      || die "DRY: embedded JSON invalid"
    say "DRY: 4 embedded python programs py_compile OK; embedded JSON parses"
  else
    say "DRY: $PYTHON_BIN unavailable -- compile check skipped (machinery walk only)"
  fi
  DRY_STAGES=(provision ckpt meshes volprobe)
  for i in $(seq 1 8); do DRY_STAGES+=("screen_$i"); done
  for st in "${DRY_STAGES[@]}"; do
    if stage_done "$st"; then say "=== STAGE $st already done, skipping ==="; continue; fi
    stage_open "$st"
    sleep 0.2
    stage_close "$st"
  done
  stage_open finalize
  say "DRY finalize: machinery only, skipping inventory (no real outputs)"
  stage_close finalize
  rm -f "$OUT/FAILED"   # a DRY that reaches here is no longer failed
  say "ALL DONE (DRY)"
  echo IDLE > "$VAR/stage"
  if [ "$LINGER_EXIT" = 1 ]; then exit 0; fi
  while :; do sleep 300; say "IDLE (DRY) -- terminate when done"; done
fi

# ============================================================================
# WET RUN from here on. If this ROOT previously hosted a DRY walk, its stage
# markers describe nothing real -- clear them so no wet stage is skipped.
# ============================================================================
if [ -f "$VAR/dry_run" ]; then
  rm -f "$VAR"/done_* "$VAR/dry_run"
  say "wet run: cleared DRY-walk stage markers from $VAR (DRY markers must not skip real stages)"
fi

# ============================================================================
# STAGE provision -- villa-pin + uv env. VERBATIM the pod_curve_audit_v2
# recipe (v12 measured 126 s warm; every network op has retry + timeout).
# ============================================================================
export PATH="$HOME/.local/bin:$PATH"
if stage_done provision; then
  say "=== STAGE provision already done, skipping ==="
else
  stage_open provision
  cd /workspace
  timeout 300 bash -c 'apt-get update -qq && apt-get install -y -qq git' >/dev/null 2>&1 || true
  command -v git >/dev/null || die "git unavailable after apt"
  if [ ! -d villa ]; then
    retry 3 timeout 300 git clone --depth 1 https://github.com/flummoxjr/villa-pin-37e300d3.git villa >> "$OUT/provision.log" 2>&1 || die "villa-pin clone failed - see provision.log"
  fi
  VSHA=$(cd villa && git rev-parse --short HEAD)
  say "provision: villa @ $VSHA"
  cd /workspace/villa/vesuvius
  command -v uv >/dev/null \
    || retry 3 timeout 180 bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' >> "$OUT/provision.log" 2>&1 \
    || die "uv installer failed - see provision.log"
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || die "uv missing after install"
  say "provision: uv sync starting (full log at /provision.log on :$PORT)"
  retry 3 timeout 1200 uv sync --extra models >> "$OUT/provision.log" 2>&1 || die "uv sync failed - see provision.log"
  retry 2 timeout 1200 uv pip install "torch==2.11.0" torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 >> "$OUT/provision.log" 2>&1 || die "torch pin install failed - see provision.log"
  retry 2 timeout 1200 uv pip install tqdm scipy scikit-image pandas einops opencv-python-headless \
    tifffile aiohttp numba monai timm accelerate pytorch-lightning \
    pytorch-optimizer huggingface-hub dynamic-network-architectures nnunetv2 \
    batchgenerators fft-conv-pytorch fvcore connected-components-3d tensorstore \
    typed-argument-parser psutil nest-asyncio blosc2 lxml imagecodecs pynrrd \
    cachetools edt wandb s3fs pillow >> "$OUT/provision.log" 2>&1 || die "dep install failed - see provision.log"
  pyrun -c "import torch,scipy,tifffile,numcodecs,PIL,zarr,s3fs; \
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
# STAGE ckpt -- ink_9um seed42/step-075000 (public; expected 138360039 bytes;
# config gate mode=flat crop=(17,128,128) norm=robust_mad). Template verbatim.
# ============================================================================
if stage_done ckpt; then
  say "=== STAGE ckpt already done, skipping ==="
else
  stage_open ckpt
  if [ ! -s "$CKPT" ]; then
    fetch_ckpt() {
      (cd /workspace/villa/vesuvius && timeout 1200 uv run --no-sync --extra models python - <<'PY_CKPT'
from huggingface_hub import hf_hub_download
p = hf_hub_download("scrollprize/ink_9um", "hybrid_3d2d-seed42/step-075000.pth",
                    local_dir="/workspace/ckpts/ink_9um")
print("CKPT", p)
PY_CKPT
      )
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
# STAGE meshes -- the frozen 8-patch tifxyz bank. Preferred: local tarball
# (scp'd at launch). Fallback: raw URLs from the public repo (requires the
# *.tif gitignore fix pushed -- see header). BOTH paths end in the same
# 32-file sha256 gate; any mismatch is an infra KILL, never a result.
# ============================================================================
fetch_raw_mesh() { # fetch_raw_mesh <patch> <file>
  curl -fsSL --max-time 120 -o "$DATA/paths_0358/$1/$2" "$RAWBASE/$1/$2"
}
if stage_done meshes; then
  say "=== STAGE meshes already done, skipping ==="
else
  stage_open meshes
  if [ ! -s "$MESH_TGZ" ] && [ -n "$MESH_TGZ_URL" ]; then
    say "meshes: fetching tarball from MESH_TGZ_URL"
    retry 3 curl -fsSL --max-time 300 -o "$MESH_TGZ" "$MESH_TGZ_URL" || say "meshes: MESH_TGZ_URL fetch failed -- will try raw-URL fallback"
  fi
  if [ -s "$MESH_TGZ" ]; then
    TGZSHA=$(sha256sum "$MESH_TGZ" | cut -d' ' -f1)
    [ "$TGZSHA" = "$MESH_TGZ_SHA" ] || die "mesh tarball sha256 $TGZSHA != expected $MESH_TGZ_SHA"
    tar xzf "$MESH_TGZ" -C "$DATA" || die "mesh tarball extract failed"
    say "meshes: tarball verified (sha256 exact) and extracted"
  else
    say "meshes: no tarball at $MESH_TGZ -- falling back to public-repo raw URLs"
    for P in "${PATCHES[@]}"; do
      mkdir -p "$DATA/paths_0358/$P"
      for F in x.tif y.tif z.tif meta.json; do
        retry 3 fetch_raw_mesh "$P" "$F" || die "raw fetch failed: $P/$F (if x/y/z.tif 404: the *.tif gitignore fix has not been pushed -- see script header; or scp the tarball)"
      done
    done
    say "meshes: 32 files fetched from raw URLs"
  fi
  (cd "$DATA" && sha256sum --quiet -c "$VAR/mesh_manifest.txt") \
    || die "mesh sha256 manifest FAILED -- the on-pod meshes are not the frozen bank"
  NMESH=$(find "$DATA/paths_0358" -type f \( -name '*.tif' -o -name meta.json \) | wc -l)
  [ "$NMESH" -ge 32 ] || die "mesh file count $NMESH < 32"
  say "meshes: all 32 files sha256-verified against the frozen manifest"
  stage_close meshes
fi

# ============================================================================
# STAGE volprobe -- live volume identity + per-mesh geometry gates.
# ============================================================================
if stage_done volprobe; then
  say "=== STAGE volprobe already done, skipping ==="
else
  stage_open volprobe
  retry 3 pyrun_t 900 "$SCRIPTS/volprobe.py" >> "$OUT/volprobe.log" 2>&1 \
    || die "volume/mesh data gates FAILED (see /volprobe.log on :$PORT)"
  tail -3 "$OUT/volprobe.log" | while read -r L; do say "volprobe: $L"; done
  stage_close volprobe
fi

# ============================================================================
# STAGE screen_1..screen_8 -- per patch: render -> infer fwd -> infer rev ->
# stats/ds8/previews -> delete the SV zarr. Per-patch stage markers make a
# crashed run resumable mid-corpus (the survey's JSONL lesson).
# ============================================================================
IDX=0
for NAME in "${PATCHES[@]}"; do
  IDX=$((IDX + 1))
  ST="screen_$IDX"
  if stage_done "$ST"; then
    say "=== STAGE $ST ($NAME) already done, skipping ==="
    continue
  fi
  stage_open "$ST"
  say "$ST = $NAME"
  SV=$DATA/${NAME}.sv.zarr
  FWD=$PREDS/${NAME}_fwd.tif
  REV=$PREDS/${NAME}_rev.tif
  if [ ! -s "$FWD" ] || [ ! -s "$REV" ] || [ "$FORCE" = 1 ]; then
    if [ ! -d "$SV/0" ] || [ "$FORCE" = 1 ]; then
      rm -rf "$SV"
      T0=$SECONDS
      retry 2 run_render "$DATA/paths_0358/$NAME" "$SV" || die "render failed twice: $NAME (tail of render.log on :$PORT)"
      [ -d "$SV/0" ] || die "render produced no zarr: $NAME"
      say "render DONE $NAME ($((SECONDS - T0))s)"
    else
      say "render skip (exists): $NAME"
    fi
    run_infer_dir "$SV" "$FWD" forward
    run_infer_dir "$SV" "$REV" reverse
  else
    say "infer skip (both maps exist): $NAME"
  fi
  SRC=0
  pyrun_t 1800 "$SCRIPTS/patch_stats.py" "$NAME" >> "$OUT/stats.log" 2>&1 || SRC=$?
  tail -1 "$OUT/stats.log" | while read -r L; do say "stats: $L"; done
  rm -rf "$SV"
  if [ "$SRC" = 0 ]; then
    stage_close "$ST"
  elif [ "$SRC" = 3 ]; then
    # health gate failed: recorded in screen_0358.jsonl as INFRA_INCOMPLETE.
    # Continue -- a patch-local failure must not cost the other patches their
    # first-ever measurement. No stage marker, so a resume retries this patch.
    # finalize will refuse ALL DONE.
    touch "$VAR/unhealthy_$NAME"
    say "PATCH UNHEALTHY $NAME (INFRA_INCOMPLETE recorded in screen_0358.jsonl) -- continuing with remaining patches; ALL DONE will be refused"
  else
    die "patch_stats crashed (exit $SRC) on $NAME -- not a recorded health verdict; tail of stats.log on :$PORT"
  fi
done

# ============================================================================
# STAGE finalize -- full inventory re-verification. ALL DONE is printed here
# and nowhere else; finalize.py exits nonzero on ANY missing or unhealthy
# artifact, and every infra failure died before reaching this line.
# ALL DONE = the MEASUREMENT completed; the science verdicts are LOCAL.
# ============================================================================
stage_open finalize
pyrun_t 900 "$SCRIPTS/finalize.py" >> "$OUT/finalize.log" 2>&1 \
  || die "finalize inventory check refused -- missing/unhealthy artifacts listed in /finalize.log on :$PORT"
grep -E "^SUMMARY" "$OUT/finalize.log" | tail -9 | while read -r L; do say "$L"; done
rm -f "$OUT/FAILED"   # a resumed run that reaches here is no longer failed
cp -f "$STATUS" "$OUT/status_at_done.txt" 2>/dev/null || true
stage_close finalize
say "ALL DONE -- measurement complete; per prereg the EXPECTED outcome is a bounded-sensitivity null and NOTHING here is a verdict. Fetch BOTH trees: scp -r root@POD:$OUT ./screen0358_out ; scp -r root@POD:$PREDS ./screen0358_preds ; then TERMINATE THE POD. The v2 5-gate battery runs locally on the fetched maps."
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
