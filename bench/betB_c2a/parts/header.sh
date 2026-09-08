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
