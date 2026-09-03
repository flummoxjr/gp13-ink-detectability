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
