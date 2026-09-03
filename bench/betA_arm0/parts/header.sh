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
SEEDS=${SEEDS:-"42 43"}
SMOKE_STEPS=${SMOKE_STEPS:-2000}
FULL_STEPS=${FULL_STEPS:-78125}
FETCH_THREADS=${FETCH_THREADS:-32}
POOL_PAR=${POOL_PAR:-3}
RUN_P2A=${RUN_P2A:-0}
SEED=20260902

OUT=$ROOT/out;  VAR=$ROOT/var;  DATA=$ROOT/data;  PREDS=$ROOT/preds
SCRIPTS=$ROOT/scripts;  RESULTS=$OUT/results;  STATUS=$OUT/status.txt
LABELS=$DATA/labels;  VOLS=$DATA/volumes;  NATIVE=$DATA/native;  RUNS=$OUT/runs
export ROOT OUT VAR DATA PREDS SCRIPTS RESULTS STATUS SEED BATCH WORKERS
export LABELS VOLS NATIVE RUNS SMOKE_ONLY SMOKE_STEPS FULL_STEPS FETCH_THREADS
