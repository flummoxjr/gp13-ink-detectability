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
