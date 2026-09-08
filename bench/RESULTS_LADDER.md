# The 9 µm transfer ladder — results on the benchmark harness (2026-09-08)

_Companion to `BENCHMARK.md` (the harness) — this file is the numbers. Every cell below was produced by
the harness described there (fixed scorer = `p2a_v3` curvelib: exact tie-corrected pixel AUC on the native
label grid, both depth orders, positive controls in the same run). Model throughout: the released
`scrollprize/ink_9um` checkpoint `hybrid_3d2d-seed42/step-075000.pth` unless a row says otherwise.
Sources are the run records named in the last column; nothing here is quoted from a commit message._

## 1. The ladder

| rung | surface | pitch → model pitch | forward AUC | reverse AUC | run |
|---|---|---|---|---|---|
| 0 · home segment (trained on) | PHerc0139 w035 native crop rows 512:2944, cols 384:3072 (334,035 ink / 737,086 blank px) | 9.362 native | **0.9991** | 0.5118 | `out/p2a_v3/results.json` ctl; reproduced to 4 decimals on all six Bet A pods |
| 0 · home scroll, five segments | PHerc0139 w035/w039/w040/w041/w044 native crops, mean | 9.362 native | **0.9988** (mean best-F1 0.980) | 0.45–0.51 | `out/betA_arm0/*/results_*.json` `ref_released` |
| 0f · home segment, in-plane level fault | w035 crop upsampled ×1.9504 (the Aug-25 error: 4.80 µm presented as 9.36) | 4.80 → 9.362 | **0.7489** | 0.5381 | `p2a_v3` ctl_scalefault → FAULT_REPRODUCED |
| 0f · home segment, coarse level fault | w035 crop downsampled ×0.5 | 18.7 → 9.362 | **0.5227** | 0.4844 | `p2a_v3` ctl_half |
| 1 · detached fragment, photographic labels | Frag1 (PHercParis2Fr47), 3.24 µm surface volume pooled to model pitch | 3.24 → 9.362 | **0.6925** | 0.4477 | `out/curve_audit/expA_baseline.json` (gate 0.85 failed → curve not run) |
| 2 · clean foreign scroll, human labels | PHerc0500P2 `500p2a` win1 (4.39 M ink / 8.58 M blank px) at its **correct 2.215 µm** pitch, iso depth | 2.215 → 9.362 | **0.5211** | 0.5106 | `bench/p2a_v3/RESULTS.md`; fit17 depth 0.5301 / 0.5099; win2 0.4339, win3 0.4914 |
| 2 · the same, hallucination null | 40 rigid translations of the label shapes inside the blank | — | null median 0.504, max 0.750 | — | `p2a_v3` expB: the 0.52 read is inside the null |
| 3 · native held-out scroll, model retrained without it | LOSO on the ink_9um recipe (khj1222 `make_holdout_config.py`; 15 kept representations), PHerc0139 held out; five native crops, mean at the best-of-grid checkpoint, two seeds | 9.362 native | **0.746 / 0.757** (best-F1 0.627 / 0.631; floor 0.540, khj1222 0.653) | 0.499 / 0.532 | `bench/betA_arm0/RESULTS.md`, `out/betA_arm0/verdict_arm0.json` |
| 3 · same, training inputs degraded to the k2b index (Bet A arm 1) | as above, per-crop blur/noise/headroom to calibrated index targets | 9.362 native | 0.751 / 0.768 (+0.008 mean) | 0.541 / 0.521 | `out/betA_arm0/verdict_arms.json` — KILLED (gate +0.05) |
| 3 · same, per-volume PSD whitening at train and test (Bet A arm 2) | as above | 9.362 native | 0.711 / 0.698 (−0.047 mean) | 0.520 / 0.500 | `out/betA_arm0/verdict_arms.json` — KILLED |

## 2. What the ladder says

1. **The released model reads its home scroll and nothing else.** 0.999 at home; 0.69 on a detached
   fragment with photographic ground truth; 0.52 on a clean foreign scroll — a number that sits inside its
   own hallucination null (median 0.50, max 0.75), i.e. indistinguishable from reading nothing.
2. **A native scroll the model never saw is read at 0.75, not 0.99, even when the whole rest of the
   recipe is retrained for it.** That is the leave-one-scroll-out ceiling of the current recipe (two seeds,
   spread 0.011), and it is the number every "9 µm transfer" claim has to beat. Two pre-registered input
   interventions did not move it (+0.008, −0.047).
3. **Input noise is not the gap.** Measured with the same estimator that would have matched them, the pooled
   2.4 → 9.6 µm training inputs are *noisier* (structural SNR ≈ 6) than the native 9.36 µm target
   (≈ 30). The "cleaner training inputs" premise was inverted, not merely unconfirmed
   (`bench/betA_arm0/RESULTS.md` § input statistics).
4. **Depth-order reversal collapses every read to chance** — 0.999 → 0.512 at home, and 0.50–0.54 on every
   held-out checkpoint of every arm. The forward/reverse asymmetry is therefore a usable *in-domain* ink
   signature (Track D gate 5) and, equally, a fault that mimics transfer failure if a pipeline reverses
   its depth axis silently.
5. **Pipeline level faults mimic transfer failure too**: ×1.95 too fine costs 0.25 AUC (0.999 → 0.749),
   ×2 too coarse takes the model to chance (0.523). A foreign-scroll null reported without these controls
   in the same run cannot be told from a broken input — which is why the Aug-25 anchor (0.538, measured
   through exactly the ×1.95 fault) was voided and re-measured.
6. **Depth mode does not matter at the foreign rung** (iso vs fit17 within 0.01) — the failure is not a
   resampling choice.

## 3. Community numbers the ladder is consistent with

- khj1222 (#1608): LOSO retrains gain +0.06–0.17 F1 over the "everything is ink" floor on the held-out
  scroll; fine-tuning on one annotated target segment closes 82 % of the gap on Paris4, 24 % on 1667.
- nerln (#1582): pooled 2.4 µm input beats native 9.362 µm by +0.03–0.07 F1 on the same physical segments;
  a domain-match arm did not close it. Read together with rung 3 and finding 3: the pooled input's advantage
  is not that it is cleaner by a spectral-noise estimator.
- Nieuwlaar / DomRusso2: dense pseudo-labels from a higher-resolution acquisition of the *same* scroll lift
  native PHerc0139 held-out AUC 0.81 → 0.95 — the only demonstrated route past rung 3 so far.
- Eight groups in August (PHerc1447 ×3, 0826, 1203, 0343P, 1218) report the released model at chance on
  scrolls it was not trained on — rung 2 reproduces that, with controls.

## 4. How to add a row

Run the model through `bench/p2a_v3` (anchor + controls) or `bench/betA_arm0` (native tier + controls),
and report the four control cells from the same run beside the held-out cell. A row without its controls
is not a ladder row. Results JSON schema: `BENCHMARK.md` §5.
