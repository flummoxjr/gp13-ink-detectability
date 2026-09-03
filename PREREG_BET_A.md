# PREREG — Bet A: input-noise-matched training for 9 µm ink transfer (FINAL v1, 2026-09-03)

Supersedes `PREREG_BET_A_DRAFT.md` (v0, 2026-09-01; §1b prior weakened 2026-09-02; §4 anchor gate corrected
2026-09-02). This file is committed before any arm-1/2 pod is launched. The arm-0 anchor gate (§4) has already
been run and PASSED (`bench/betA_arm0/RESULTS.md`, `out/betA_arm0/verdict_arm0.json`); its numbers below were
known when §5 was frozen and are used only to set the baseline the arms are compared against.

## 1. Claim under test (unchanged from the draft)

The 9 µm transfer gap is (at least partly) an input-statistics mismatch: `ink_9um`'s pooled 2.4 µm → 9.6 µm
training inputs are cleaner and wider-band than any native 9 µm scan of an eligible scroll. If true, training on
inputs degraded per batch to the *measured* noise statistics of the eligible scrolls (arm 1), or canonicalising
every volume by its own noise spectrum at train and test (arm 2), raises held-out native transfer. The prior was
weakened on 2026-09-02 (§1b of the draft): a pooled-class foreign input does not transfer better than native, so
the gap is probably not mostly input noise. This experiment decides it.

## 2. Design

Leave-PHerc0139-out on the published ink_9um recipe, exactly as arm 0 (`bench/betA_arm0/`: 15 kept
representations, quotas {1667: 40, Paris4: 20, 0814: 4}, batch 64, patch (17,128,128), 78,125 steps, checkpoints
every 5,000, seeds 42 and 43, one pod per seed). The five native 0139 segments (w035/w039/w040/w041/w044) are the
held-out ground truth in every arm.

| arm | training input | code |
|---|---|---|
| 0 | recipe unchanged | villa-pin `master` @ a3f2c29 |
| 1 | `input_degradation`: every training flat crop degraded, per crop, to a target drawn from the 14-scroll k2b index — blur to the target bandwidth, white noise to the target SNR at q = 0.25 cyc/px, contrast scaled to the target DN headroom (definitions in `degradation.py`; 2-D transposition of `k2b_detectability_index.py`) | branch `betA-arms` @ 67bb8c77 |
| 2 | `input_whitening`: every crop of every volume filtered by g(q) = √(PSD(q_ref)/PSD(q)) from that volume's own in-plane radial PSD (64 windows, q_ref 0.02, gain clipped to [1/8, 8]) at train, validation and inference | branch `betA-arms` @ 67bb8c77 |

Everything else is byte-identical: absent config keys leave the arm-0 code path untouched.

**Input-statistics gate (new, pre-registered here).** Each pod runs a `measure` stage before training that
measures the pooled sources and the native crops with the arm-1 estimator and reports the fraction of pooled
stores whose SNR (bandwidth) exceeds the index-target median. That fraction is the fraction of crops where the
noise (blur) step can act; if both are zero, arm 1 is only the headroom match and is read as such. The number is
reported whatever it is.

## 3. Evaluation

As arm 0: native-5 best-F1 (threshold sweep on the supervision mask) and pixel AUC in both depth orders at every
checkpoint; the released `ink_9um` checkpoint as the in-domain reference and the p2a_v3 positive controls (native
fwd/rev, ×1.95 scale fault, ×0.5) on every pod. Secondary (after the pods, on the saved best checkpoints):
500p2a win1/2/3 at 2.215 µm with `bench/p2a_v3`.

## 4. Anchor gate — PASSED 2026-09-03

best-of-both native-5 F1 0.6307 ≥ 0.603; mean margin +0.088 ≥ +0.06; peaks at 20k (s42) and 30k (s43) with 75k
below. Arm-0 AUC baseline: mean forward AUC at the best checkpoint 0.7459 (s42) / 0.7566 (s43), **mean 0.7513,
seed spread 0.011**; reverse 0.47–0.56 (chance) on every segment.

## 5. Decision rule (frozen)

- **PASS** iff arm 1 *or* arm 2 has a two-seed mean of best-of-grid forward native-5 AUC ≥ **0.8013**
  (= 0.7513 + 0.05; the +0.05 also exceeds 2 × the arm-0 seed spread), *and* the passing arm's best checkpoint
  reaches **500p2a win1 (iso) ≥ 0.65** (A₃ = 0.5211, so max(0.65, A₃ + 0.05) = 0.65) when run afterwards.
- **KILLED** otherwise. Code, configs, input statistics and held-out numbers ship either way.
- Reverse-direction AUC is reported; a passing arm whose reverse AUC leaves chance is flagged for a depth-order
  check before anything is claimed.
- Verdict computed locally from the four pods' `results.json` with `bench/betA_arm0/arm_verdict.py`.

## 6. Cost

Four pods (arm 1 s42/s43, arm 2 s42/s43), ≤ 15 h guard each, ≈ $5 per pod on a community 5090 (arm-0 timing:
setup 2 h, training 5–7 h, 15 checkpoint evals 35 min). Balance before launch $55.

## 7. What each outcome means

- PASS → the September instrument exists; screen the 47 gate-passing Tier-1 patches with it; Bet E prep.
- KILLED with the input-statistics gate showing the degradation was active → input noise is not the gap; Bet C
  (max-corpus native generalist) is October's only model bet; publish the null with the harness.
- KILLED with the degradation inactive (pooled sources already at or below the targets) → the premise itself was
  wrong in the other direction (pooled inputs are not cleaner than native by this estimator); publish that.
