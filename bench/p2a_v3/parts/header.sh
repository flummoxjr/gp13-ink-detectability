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
