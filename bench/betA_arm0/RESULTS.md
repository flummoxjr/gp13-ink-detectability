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

# Bet A arms 1 and 2 (input-noise-matched training) — FULL RUN results and verdict (2026-09-04)

Pre-registration: `trackD/PREREG_BET_A.md` (final v1, committed 9afc544 before launch; §2 calibration amendment
committed before any arm-1 training). Code: fork `flummoxjr/villa-pin-37e300d3` branch `betA-arms`
(arm-1 s42/s43 and arm-2 s43 at 4516bedd; the arm-2 s42 relaunch at 45d5e03 = 4516bedd + the trainer skips a
non-finite-loss step instead of raising; the arm transforms are byte-identical across the two shas). Pod script v2e
(`gist_raw_url.txt`, sha fa2c3fc09ae17cdc). Four pods, one seed each: arm-1 s42 `s3mtrgwu2fwnsq` (5090 community),
arm-1 s43 `3zjbtmtn27dc9u` (5090 community, slow host), arm-2 s42 `8r68lpxrgcev1n` (A6000 secure), arm-2 s43
`2kfrgrqpgxv2om` (A6000 secure). Results: `out/betA_arm0/arm{1,2}_s{42,43}/results_*.json` (+ pod status logs);
verdict `out/betA_arm0/verdict_arms.json` from `arm_verdict.py`.

## Verdict: KILLED (frozen rule, prereg §5)

| arm | seed 42 best | seed 43 best | two-seed mean fwd AUC | gain vs arm 0 (0.7513) | threshold | |
|---|---|---|---|---|---|---|
| 0 (baseline) | 0.627 / 0.746 @ 20k | 0.631 / 0.757 @ 30k | 0.7513 | — | — | comparator |
| 1 (per-crop degradation, calibrated) | 0.627 / 0.751 @ 10k | 0.641 / 0.768 @ 30k | **0.7592** | **+0.008** | ≥ +0.05 (AUC ≥ 0.8013) | ✗ |
| 2 (per-volume PSD whitening, train + test) | 0.592 / 0.711 @ 20k | 0.587 / 0.698 @ 25k | **0.7045** | **−0.047** | ≥ +0.05 | ✗ |

(cells: native-5 mean best-F1 / mean forward pixel AUC at the best-of-grid checkpoint; best-of-grid chosen by
forward AUC, as registered.) Neither arm reaches the primary clause, so the 500p2a secondary clause is not
triggered. Reverse-direction AUC at every best checkpoint stays at chance (0.50–0.54), so no depth-order flag.
Arm 1's +0.008 is inside the arm-0 seed spread (0.011) and comes entirely from seed 43 (+0.011 over arm-0 s43; seed 42
is −0.001). Arm 2 is worse than the baseline on both seeds by 0.035–0.059 AUC and never leaves the 0.56–0.59 F1 band.

Positive controls on all four pods, unchanged from arm 0: released `ink_9um` on the native crops forward 0.9991 /
reverse 0.512, ×1.95 scale fault 0.749 (FAULT_REPRODUCED), released reference native-5 F1 0.980 / AUC 0.9988.
The harness is the same one that produced the arm-0 anchor PASS, so a +0.05 effect would have been seen.

## Trajectories (native-5 mean best-F1 / mean forward AUC per checkpoint)

| step | arm 0 s42 | arm 0 s43 | arm 1 s42 | arm 1 s43 | arm 2 s42 | arm 2 s43 |
|---|---|---|---|---|---|---|
| 5k | 0.586 / 0.700 | 0.563 / 0.651 | 0.588 / 0.698 | 0.553 / 0.625 | 0.584 / 0.697 | 0.541 / 0.568 |
| 10k | 0.620 / 0.740 | 0.601 / 0.722 | **0.627 / 0.751** | 0.592 / 0.709 | 0.577 / 0.689 | 0.556 / 0.656 |
| 15k | 0.627 / 0.751 | 0.625 / 0.751 | 0.616 / 0.737 | 0.623 / 0.747 | 0.583 / 0.692 | 0.561 / 0.666 |
| 20k | **0.627 / 0.746** | 0.624 / 0.750 | 0.612 / 0.733 | 0.624 / 0.748 | **0.592 / 0.711** | 0.580 / 0.686 |
| 25k | 0.612 / 0.738 | 0.621 / 0.746 | 0.604 / 0.723 | 0.630 / 0.756 | 0.591 / 0.709 | **0.587 / 0.698** |
| 30k | 0.622 / 0.747 | **0.631 / 0.757** | 0.605 / 0.731 | **0.641 / 0.768** | 0.574 / 0.684 | 0.576 / 0.678 |
| 35k | 0.604 / 0.727 | 0.597 / 0.715 | 0.606 / 0.721 | 0.604 / 0.721 | 0.583 / 0.691 | 0.582 / 0.688 |
| 40k | 0.610 / 0.723 | 0.595 / 0.710 | 0.607 / 0.722 | 0.614 / 0.737 | 0.581 / 0.698 | 0.557 / 0.659 |
| 45k | 0.601 / 0.719 | 0.596 / 0.712 | 0.582 / 0.694 | 0.594 / 0.716 | 0.565 / 0.673 | 0.585 / 0.691 |
| 50k | 0.596 / 0.708 | 0.603 / 0.712 | 0.583 / 0.688 | 0.607 / 0.724 | 0.567 / 0.671 | 0.579 / 0.682 |
| 55k | 0.598 / 0.720 | 0.613 / 0.731 | 0.581 / 0.696 | 0.602 / 0.721 | 0.577 / 0.687 | 0.579 / 0.674 |
| 60k | 0.588 / 0.697 | 0.597 / 0.706 | 0.597 / 0.713 | 0.614 / 0.735 | 0.556 / 0.652 | 0.576 / 0.652 |
| 65k | 0.585 / 0.685 | 0.617 / 0.737 | 0.579 / 0.695 | 0.601 / 0.718 | 0.560 / 0.656 | 0.574 / 0.658 |
| 70k | 0.586 / 0.693 | 0.604 / 0.718 | 0.585 / 0.704 | 0.603 / 0.721 | 0.562 / 0.662 | 0.567 / 0.644 |
| 75k | 0.585 / 0.691 | 0.603 / 0.717 | 0.588 / 0.704 | 0.601 / 0.717 | 0.562 / 0.663 | 0.565 / 0.645 |

All six runs peak in 10–30k and decay by 75k (the arm-0 anchor shape). Arm 1 tracks arm 0 checkpoint for
checkpoint; arm 2 runs ≈ 0.04–0.06 AUC below it from 10k onward on both seeds.

## Per-segment at the best checkpoint (best-F1 / forward AUC)

| pod (best ckpt) | w035 | w039 | w040 | w041 | w044 | mean fwd AUC | mean rev AUC |
|---|---|---|---|---|---|---|---|
| arm 0 s42 (20k) | 0.664 / 0.833 | 0.564 / 0.719 | 0.637 / 0.692 | 0.661 / 0.756 | 0.609 / 0.730 | 0.746 | 0.499 |
| arm 0 s43 (30k) | 0.666 / 0.827 | 0.553 / 0.708 | 0.641 / 0.715 | 0.668 / 0.775 | 0.625 / 0.757 | 0.757 | 0.532 |
| arm 1 s42 (10k) | 0.649 / 0.817 | 0.542 / 0.708 | 0.636 / 0.694 | 0.682 / 0.787 | 0.625 / 0.746 | 0.751 | 0.541 |
| arm 1 s43 (30k) | 0.667 / 0.836 | 0.615 / 0.780 | 0.623 / 0.691 | 0.666 / 0.774 | 0.633 / 0.758 | 0.768 | 0.521 |
| arm 2 s42 (20k) | 0.622 / 0.781 | 0.522 / 0.688 | 0.617 / 0.670 | 0.605 / 0.703 | 0.594 / 0.711 | 0.711 | 0.520 |
| arm 2 s43 (25k) | 0.599 / 0.760 | 0.501 / 0.641 | 0.618 / 0.664 | 0.619 / 0.712 | 0.598 / 0.716 | 0.698 | 0.500 |

Arm 1 seed 43's whole gain is w039 (0.708/0.719 → 0.780); the other four segments are within ±0.02 of arm 0.
Arm 2 loses on every segment, most on w041 (−0.05 to −0.07) and w039 (s43: −0.07).

## The input-statistics gate (prereg §2) — the premise is inverted

The `measure` stage on every pod (`parts/measure_inputs.py`, 64 random 128² in-plane windows per store, the same
2-D estimator arm 1 degrades with) gives, deterministically across the four pods:

| store class | snr_q025 (median over stores; per-store medians) | bandwidth (cyc/px) | DN headroom (p99.5 − p0.5) |
|---|---|---|---|
| pooled 2.4 → 9.6 µm training volumes (15 stores: PHerc1667 ×6, PHercParis4 ×8, PHerc0814 ×1) | **6.3** (5.4–7.9) | 0.34 (0.335–0.352) | 116–175 (median 132) |
| native 9.36 µm PHerc0139 crops (w035/w039/w040/w041/w044) | **≈ 30** (21–39) | 0.36–0.37 | 148–162 |
| k2b index targets (14 scrolls, 55 ROIs, 3-D estimator) | 74.5 | 0.496 | — |

By the estimator arm 1 uses, the pooled training inputs are **noisier and narrower-band than the native
target class**, not cleaner: the claim under test (prereg §1) is wrong in the other direction. The 3-D index and
the 2-D per-crop estimator disagree by ≈ 2–4× on the same native crops (index PHerc0139 snr 115.5 vs 2-D ≈ 30–51;
bandwidth 0.386 vs 0.36), which is why targets were calibrated (`target_scale` = (bw 0.934, snr 0.265, headroom
1.020), recorded in each arm-1 config). With calibrated targets the arm-1 transform could act as follows
(recomputed locally from `parts/k2b_index.json` and the pods' `input_stats.json`, draw = uniform over scrolls then
ROIs, as `draw_target` does):

- **noise step**: active on **24 % of draws** (targets from PHerc0257/0268/0800/0826/1218/1447 fall below the
  pooled crop SNR; calibrated target median 19.7, range 2.0–43.3, vs pooled 6.3) — the pre-registered gate
  (uncalibrated) reads 0 %;
- **blur step**: **0 %** (every calibrated bandwidth target is above the pooled 0.34);
- **headroom step**: active on essentially every draw, and in the *amplifying* direction (calibrated target
  median 221 DN vs pooled 132) — i.e. arm 1 in practice was mostly a ×1.5–1.7 contrast rescale plus occasional
  added white noise. That such a transform neither helps nor hurts (+0.008) is consistent with the recipe's
  existing intensity augmentation absorbing it.

Caveats that keep this an estimator statement, not a physical one: the residual floor is estimated from the
0.35–0.48 cyc/px band, so genuine high-frequency papyrus texture counts as "noise"; the pooled volumes are
resampled (isotropic 21-slice from 2.4 µm) while the native crops are raw 9.36 µm reconstructions; the
comparison is like-for-like across the two classes but is not a measurement of detector noise.

## Reading (prereg §7)

- **PASS** — no.
- **KILLED with the degradation active** — partially: noise on 24 % of draws, blur never, headroom always.
- **KILLED with the premise wrong in the other direction** — **this is the outcome.** Pooled 2.4 → 9.6 µm inputs
  are not cleaner than native 9 µm by the arm's own estimator; matching them to the index makes no difference,
  and canonicalising every volume by its own spectrum (arm 2) costs ≈ 0.05 AUC. Input noise, as defined and
  measured here, is not the 9 µm transfer gap. Bet C (max-corpus native generalist) is October's only model bet,
  as the plan already routed after the 500p2a finding.

Arm 2's harm is informative on its own: with q_ref 0.02 and the gain clipped at 8×, the whitener boosts the
0.3–0.5 cyc/px band of every volume by up to 8× before the network sees it, amplifying exactly the component the
residual floor calls noise. Softer variants (lower clip, q_ref at the structural band, whitening only at test
time) were not registered and are not run; the two seeds agree closely enough (0.711 vs 0.698) that a rerun is
not warranted on this budget.

## What ships

- `degradation.py` (2-D k2b estimator, `degrade_crop`, `PSDWhitener`, `sample_inplane_windows`, `draw_target`),
  the `input_degradation` / `input_whitening` config keys, dataset and inference hooks — fork branch `betA-arms`
  @ 45d5e03, absent keys byte-identical to villa a3f2c29;
- the pod harness (`bench/betA_arm0/`: measure stage, calibration, LOSO config generation, five-segment native
  evaluation with positive controls) and the four `results.json` with per-checkpoint, per-segment, both-direction
  numbers, the input statistics and the frozen prereg embedded;
- this null, with the input-statistics table, as the record that the "pooled inputs are cleaner" premise fails.

## Cost and timing

Arms wall-clock: first launch 21:42 UTC Sept 3 → last harvest 12:50 UTC Sept 4 (15 h). Pod time on the four
runs that finished: arm-1 s42 8.3 h ($5.7; labels 84 min, train 5.8 h at 3.8 it/s), arm-1 s43 13.0 h ($9.0; the
…644115cd host ran at 1.7–3 it/s, train 10.4 h), arm-2 s42 9.6 h ($5.1; train 6.7 h at 3.3 it/s), arm-2 s43 9.4 h
($5.0; train 6.5 h at 3.4 it/s). Lost to faults before that: four pods dead in the first `measure` stage (≈ $12),
two trainer_check deaths, four no-CUDA hosts, one non-finite-loss abort (≈ $2 together). Balance $55 before the
arm launches → $23.4 after (≈ $32 for arms 1/2 all-in, against the prereg's ≈ $20 clean estimate). Evals: 15
checkpoints × 5 segments × 2 directions in 33–50 min per pod.
