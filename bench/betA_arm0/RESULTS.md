# Bet A arm 0 (LOSO baseline) — smoke results (2026-09-03)

Pre-registration: `parts/prereg.json` (sha df6492905823, locked in the pod script before any data). Pod that
produced the numbers: `bzjjhx87k6hns9` (RTX A6000 48 GB, villa-pin a3f2c29, script v1f sha 660cacb91d678199),
SMOKE_ONLY=1 (seed 42, 2,000 steps), ALL DONE 11:42 UTC. Harvest: `experiments/betA0smoke8/` (results.json, bundle
226 MB, ckpts_keep 513 MB, 17 logs); results + status mirrored to `out/betA_arm0/smoke8/`.

## What the smoke validated

| stage | outcome |
|---|---|
| provision + trainer contract | villa-pin `vesuvius.ink_detection.training.train` runs 30 synthetic iterations on real 0814 labels |
| labels (15 kept + 5 native eval stores) | all synced with exact sha256 (84 min on this host; HF rate-limits at 32 threads, retried at 4) |
| level-2 sparse fetch | 19,682 chunk columns ≈ 35.0 GB in 11 min; absent-on-server ≤ 3.7 % per store (zero-filled) |
| pooling (2.4 µm level 2 → 21-slice 9.6 µm) | 15 volumes, 5.0 GB, **167 s** (POOL_PAR 4); pooled-volume gate 15/15 |
| LOSO config (seed 42) | 15 kept, 14 held out (all PHerc0139), quotas {1667: 40, Paris4: 20, 0814: 4}, batch 64, patch (17,128,128) |
| control gates (released ckpt) | native fwd AUC 0.9991 / rev 0.5118; scale fault ×1.95 → 0.7488 (FAULT_REPRODUCED) |
| training | 2,000 steps in 10:40, 3.12 it/s (A6000, batch 64), checkpoints at 1k and 2k, val loss 0.6185 |

## Eval (native-5 = w035/w039/w040/w041/w044 native 9.36 µm crops, human labels)

| checkpoint | native-5 mean best F1 | margin over floor | mean AUC fwd |
|---|---|---|---|
| s42_1000 | 0.5410 | +0.000 | 0.5343 |
| s42_2000 | 0.5411 | +0.001 | 0.5370 |
| ref_released | 0.9799 | +0.439 | 0.9988 |

The floor (0.540) equals khj1222's published floor (0.541) to rounding, so the eval harness agrees with the
anchor's convention. At 2,000 steps (2.5 % of the schedule, warm-up just finished) the model is at the floor, as
expected; the released reference scores 0.98 on its own home scroll. **No gate verdict in SMOKE_ONLY.**

## Sizing the full run

3.12 it/s on an A6000 ⇒ 78,125 steps ≈ 7 h per seed (+ ~2.5 h setup + ~1 h of checkpoint evals).
A 5090 is ~2.3× faster but tonight's 5090 hosts were either unbootable or without egress. Full run = two pods in
parallel (SEEDS=42 / SEEDS=43, SMOKE_ONLY=0), guard 13 h each, verdict combined locally with the pre-registered rule
(best-of-both ≥ 0.603 AND mean margin ≥ +0.06 AND peak at 10–30k with 75k below).


# Bet A arm 0 (LOSO baseline) — FULL RUN results (2026-09-03)

Two pods, one seed each, identical script v1f (sha 660cacb91d678199; prereg sha df6492905823 locked in-script):
seed 42 on `rxar0avvtanprd` (community 5090, 3.25 it/s, train 6 h 48 min), seed 43 on `nxcv6ufppr8t6m` (community
5090, 4.6 it/s, train 4 h 58 min). 78,125 steps, batch 64, patch (17,128,128), checkpoints every 5,000 steps, each
evaluated on the native-5 held-out PHerc0139 crops (human labels) exactly as the smoke. Seed 43's pod finalize
refused on a stage-list bug after all evals were done; its `results/eval.json`, `ctl.json` and best checkpoint were
pulled by hand before termination and assembled into `out/betA_arm0/s43/results_s43.json` (numbers unchanged).
Seed 42's pod finalized normally (`out/betA_arm0/s42/results_s42.json`; bundle 654 MB and 3.8 GB of checkpoints in
`experiments/betA0_s42/`).

## Native-5 mean best F1 by checkpoint (floor 0.540; anchor khj1222 0.653)

| step | seed 42 | seed 43 |
|---|---|---|
| 5k | 0.5857 | 0.5630 |
| 10k | 0.6202 | 0.6013 |
| 15k | 0.6269 | 0.6251 |
| 20k | 0.6270 | 0.6237 |
| 25k | 0.6120 | 0.6212 |
| 30k | 0.6217 | 0.6307 |
| 35k | 0.6042 | 0.5967 |
| 40k | 0.6103 | 0.5952 |
| 45k | 0.6011 | 0.5956 |
| 50k | 0.5960 | 0.6032 |
| 55k | 0.5977 | 0.6125 |
| 60k | 0.5877 | 0.5966 |
| 65k | 0.5847 | 0.6175 |
| 70k | 0.5860 | 0.6039 |
| 75k | 0.5845 | 0.6030 |

Best: seed 42 **0.6270 @ 20k** (margin +0.087, AUC fwd 0.7459); seed 43 **0.6307 @ 30k**
(margin +0.090, AUC fwd 0.7566). Reverse-direction means at the best checkpoints (depth-order
asymmetry, the instrument's signature): seed 42 {}; seed 43 {}.
Controls on both pods: released ink_9um on the native crops fwd 0.9991 / rev 0.5118, scale fault ×1.95 → 0.7489
(FAULT_REPRODUCED), released reference native-5 F1 0.9799.

## Pre-registered anchor gate (prereg §4, corrected 2026-09-02)

| clause | value | threshold | |
|---|---|---|---|
| best-of-both native-5 F1 | **0.6307** | ≥ 0.603 | ✓ |
| mean margin over the floor | **+0.088** | ≥ +0.06 | ✓ |
| peak at 10–30k with 75k below | seed 42 peak 20k (75k 0.585); seed 43 peak 30k (75k 0.603) | both | ✓ |

**Verdict: PASS** (`out/betA_arm0/verdict_arm0.json`, computed by `combine_verdict.py` with the same rule finalize.py
applies). The LOSO baseline is reproduced to within the seed spread of the anchor (0.63 vs 0.65; khj1222's own
seed spread ≈ 0.03). Arm 0 is therefore a valid comparator for arms 1 and 2 (native-noise-matched training),
which is the actual Bet A question; nothing here is a transfer result on a foreign scroll and no letter language
applies.

## Cost and timing

Smoke launches 1–8 ≈ $6 (host/egress/pipeline faults, all fixed in the script and launcher); full run ≈ $4.7 (seed 42)
+ $3.4 (seed 43). Wall-clock from first smoke to verdict: 20 h. For arms 1/2: same script, ~$4–5 per seed on a
community 5090 host from the good-host list; budget setup 2 h + train 5–7 h + evals 0.6 h; guard 13–15 h.
