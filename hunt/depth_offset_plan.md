# Investigation B — Depth-offset sweep: measurement, verdict, and pod design

_2026-08-17. All numbers below are measured, not estimated, unless marked ESTIMATE._

## Verdict up front

**The depth-offset hypothesis is refuted for offsets within ±6 voxels, which is the regime
it was proposed to explain.** Two independent measurements kill it:

1. On the control (w035), displacing the model's 17-slice window by ±3 voxels costs
   **≤5% of the detector's excess-over-chance**; at ±6 voxels it still retains **84%**
   (AUC 0.966 → 0.889). Letters do not vanish at ±3. They barely flinch.
2. Displacing the control window **does not reproduce the corpus symptom**. The
   forward/reverse correlation on w035 stays in **0.035–0.118 at every offset from −6 to +5**.
   The corpus range is 0.222–1.000. No amount of small misalignment turns the control into
   a corpus segment.

Separately, the corpus symmetry statistic turns out to be **largely a render-sparsity
artifact**, not a geometric signal (§1). And the single best-placed mesh in the corpus —
sitting 1.25 voxels off its sheet, geometry equal to the control's — is still null (§3).

What survives is a narrow residual: **offsets larger than ±6 voxels** (a mesh a whole sheet
off the recto), which the 28-slice published surface volume cannot test. §4 designs that
experiment. §5 costs it at **~$2–4 and ~2–3 h wall-clock**. §6 gives it **p ≈ 4%**.

The recommended next action is **not** to spend that money yet: a **$0 local prerequisite**
(§4.0) decides whether the residual hypothesis is even alive.

---

## 1. Part 1 — the fwd/rev symmetry distribution, and what it actually measures

Source: `out/survey/survey_all.json` (80 rows), `out/hunt/symmetry_stats.json`.
Script: `hunt/analyze_symmetry.py`.

### 1.1 Distribution

| stat | value |
|---|---|
| n rows | 80 (but only **66 unique meshes** — 14 were surveyed twice under bare + date-prefixed names) |
| p0 / p25 / p50 / p75 / p100 | 0.222 / 0.459 / 0.631 / 0.756 / 1.000 |
| mean ± sd | 0.616 ± 0.173 |
| **control (w035, ink-bearing)** | **0.076** |
| fraction of corpus below control+0.1 | **0.000** |

By scroll: PHerc0800 n=6 median 0.441 · PHerc1203 n=22 median 0.509 · PHerc1447 n=52 median 0.740.

The statistic is **reproducible**: across the 14 meshes surveyed twice, mean |Δr| = 0.009
(max 0.061). So r is measuring something systematic, not run noise. The question is *what*.

### 1.2 It is dominated by render sparsity — a confound

| covariate | Spearman ρ vs r (n=80) | partial ρ given nonzero_frac |
|---|---|---|
| **nonzero_frac** | **−0.650** | — |
| mask_frac | −0.692 | **−0.023** (vanishes) |
| p99 of prediction | +0.587 | +0.434 |
| n_hot_components | +0.374 | +0.236 |
| frac_gt_half | −0.190 | +0.232 |
| area_px | +0.129 | −0.132 |
| **ruling_z** | **+0.011** | **+0.027** |

Within PHerc1447 alone (n=52, removes the scroll confound) the effect is stronger:
ρ(r, nonzero_frac) = **−0.747**, ρ(r, mask_frac) = −0.720.

Split the corpus at nonzero_frac = 0.55:

- **sparse** (nz < 0.55, n=37): median r = **0.763**
- **well-filled** (nz ≥ 0.55, n=43): median r = **0.502**, and within this subset r is
  essentially uncorrelated with nz (ρ = +0.107)

The four highest-r segments have nonzero_frac 0.00, 0.05, 0.10, 0.20. The r=1.000 segment
(`z_dbg_gen_00325_inp_hr`) has a **0.00** nonzero fraction — its "perfect symmetry" is two
near-empty canvases agreeing about nothing.

**mask_frac's apparent ρ=−0.692 is pure collinearity with sparsity** — it goes to −0.023 once
conditioned. **ruling_z is unrelated to r** (ρ=+0.011, partial +0.027), so the periodicity
control and the symmetry statistic are independent measurements.

### 1.3 It is also a sheet-coherence signal (the part that isn't artifact)

`hunt/depth_profile.py` samples the CT along each mesh normal at −16..+16 voxels and reports
how sheet-like the profile is. For the 8 corpus meshes profiled (`hunt/out/depth_profiles.json`),
against their survey r:

| mesh | fwd/rev r | sheet modulation | mesh→sheet peak offset | frac tiles peaked within 2 vox |
|---|---|---|---|---|
| 1447 `auto_grown_20250702235910292` | 0.222 | **0.370** | −1.25 vox | **0.61** |
| 0800 `…20251028213516907` | 0.329 | 0.265 | +2.50 | 0.33 |
| 1203 `auto_grown_20250923164713356` | 0.399 | 0.204 | −1.75 | 0.50 |
| 1203 `auto_grown_20250923163217042` | 0.460 | 0.119 | 0.00 | 0.22 |
| 0800 `…20251028220955…` | 0.522 | 0.324 | +2.50 | 0.33 |
| 1447 `…20250502182142…` | 0.623 | **0.048** | **−10.00** | 0.10 |
| 1203 `auto_grown_20251005230830031` | 0.747 | 0.108 | +0.50 | 0.44 |
| 1447 `auto_grown_20250502160708188` | 0.914 | **0.043** | −3.50 | **0.00** |
| — **control w035** | **0.076** | **0.357** | −2.00 | 0.39 |
| — control w032 | — | 0.495 | −1.00 | 0.44 |

**Spearman ρ(r, modulation) = −0.833 (n=8).** High symmetry = the mesh is not on a coherent
sheet at all. The r=0.914 and r=0.623 meshes have modulation 0.043–0.048, an **order of
magnitude below the control's 0.357** — those meshes are in mush, and both faces of "mush"
predict the same artifact.

### 1.4 Reading

`fwd_rev_r` is **not** a geometry meter and does not mean "ink lives on both faces". It is a
*signal-dominance* meter: it is low when ink variance dominates the prediction, high when
shared non-ink structure does. It rises for two independent reasons — (a) the canvas is mostly
empty, (b) the mesh is not on a coherent sheet. Neither is "the window is a few voxels off".

**Consequence: the original motivation for Investigation B is substantially weaker than the
r = 0.22–0.91 vs 0.076 comparison suggested.**

---

## 2. Part 2 — the control tolerance budget (MEASURED, with the real model)

### 2.0 A better method than the brief assumed

The brief assumed we could not re-render w035 without a pod and proposed offline
depth-contrast statistics as a substitute. **Both halves of that turned out to be wrong, in
opposite directions:**

- **The offline substitute does not work.** `hunt/depth_offset_v2.py` searched 28 offline
  detectors (mean / d′-matched / std / peak-to-peak projections × 7 band-pass scales) on the
  w035 surface volume. Best aligned per-tile AUC: **0.5677**. The real model on the same
  pixels: per-tile median AUC **0.9993**, global **0.9991**. Confirmed by K1's independent
  result (raw z-mean AUC on w035 = 0.4956). **The letters are invisible to every hand-built
  statistic we can compute from the surface volume**, so any sweep of those statistics —
  including the v1 attempt in `hunt/depth_offset_control.py`, which returned a flat null at
  *every* offset including the aligned one — measures noise, not detectability.
- **A pod was never needed.** `infer.py` exposes `--layer-start` / `--layer-end`
  (`select_layer_indices`, infer.py:217-244), which selects a depth sub-range *before* the
  17-slice centre crop. So the model window can be slid inside an already-rendered surface
  volume with no re-render. The whole sweep ran **locally on the 4090 laptop for $0**
  (~65 s per inference on the 6.10 Mpx crop = **10.7 s/Mpx** at batch 8 / 0 workers /
  no-compile; 24 inferences ≈ 27 min).

Validation that the mechanism is exact: `--layer-start 6 --layer-end 23` produced a file
byte-size-identical to the default centre crop (3,230,160 B both).

Assets: `D:\vesuvius-data\trackD\w035_crop.sv.zarr` (28×2346×2602 uint8 = 6.10 Mpx),
`w035_crop_labels.npz`, checkpoint `D:\vesuvius-data\models\ink9um\step-075000_seed42.pth`.
Runner `hunt/sweep_local.sh`, scorer `hunt/score_crop.py`,
results `out/hunt/depth_offset_control_REAL.json`.

### 2.1 The measured curve

17-slice window centred at slice c; the model's native alignment is c=14 (the label plane).

| centre | offset (vox) | **fwd AUC** | rev AUC | **fwd/rev r** | fired frac | excess retained |
|---|---|---|---|---|---|---|
| 8 | −6 | 0.8894 | 0.5771 | 0.1180 | 0.293 | **83.6%** |
| 9 | −5 | 0.9242 | 0.5740 | 0.0789 | 0.305 | 91.1% |
| 10 | −4 | 0.9388 | 0.5673 | 0.0685 | 0.310 | 94.2% |
| 11 | −3 | 0.9546 | 0.5573 | 0.0625 | 0.310 | **97.6%** |
| 12 | −2 | 0.9635 | 0.5626 | 0.0411 | 0.308 | 99.5% |
| 13 | −1 | 0.9668 | 0.5696 | 0.0347 | 0.303 | 100.3% |
| **14** | **0** | **0.9656** | 0.5509 | **0.0381** | 0.308 | 100.0% |
| 15 | +1 | 0.9637 | 0.5590 | 0.0606 | 0.308 | 99.6% |
| 16 | +2 | 0.9583 | 0.5621 | 0.0655 | 0.298 | 98.4% |
| 17 | +3 | 0.9438 | 0.5703 | 0.0580 | 0.287 | **95.3%** |
| 18 | +4 | 0.9134 | 0.5641 | 0.0365 | 0.259 | 88.8% |
| 19 | +5 | 0.8709 | 0.5641 | 0.0976 | 0.239 | 79.7% |

(AUC here is on the labelled crop only, where background is all on-sheet papyrus. The
whole-segment figure of 0.9991 includes easy off-sheet background; 0.9656 is the harder,
fairer number and is the right control anchor.)

### 2.2 The tolerance budget

- **±3 voxels: ≥95% of excess-over-chance retained.** Letters are essentially unaffected.
- **±5 voxels: ~80–91% retained.** Still overwhelming detection (AUC 0.87–0.92).
- **±6 voxels: 84% retained** (only −6 was reachable).
- The curve is **flat-topped over ±2 and near-symmetric**; the peak is at −1, i.e. the
  published mesh sits ~1 voxel off the ink plane and nobody noticed because it does not matter.
- **The half-power point is not reached anywhere in ±6.** Linear extrapolation of the
  −4→−6 and +3→+5 slopes (~5 pts of retained excess per voxel) puts 50% retention at roughly
  **|offset| ≈ 9–11 voxels** — i.e. only when the ink plane is leaving the 17-slice
  (±8 voxel) window entirely. **Truncation, not depth prior, is what eventually kills it.**

**This directly answers the brief's framing question.** The brief proposed: *"if letters vanish
at ±3 voxels, a mesh misplaced by 5 voxels explains our whole corpus null."* Measured: letters
retain 97.6% at ±3 and 80% at ±5. **A 5-voxel misplacement explains nothing.**

### 2.3 The second, stronger refutation

The last column that matters is **fwd/rev r**, which is the exact statistic the corpus null was
built on. Across the whole ±6 sweep it stays in **0.035–0.118**, with no trend that approaches
corpus values. The corpus **minimum** over 66 unique meshes is 0.222 — still 1.9× the control's
worst offset.

**A depth-misaligned window on real ink still looks nothing like a corpus segment.** The
corpus's symmetry cannot be manufactured by sliding the window.

---

## 3. What this leaves of the hypothesis

Dead:
- "the 21-slice render window straddles or misses the ink layer by a few voxels" — refuted;
  ±6 costs 16% of excess and does not raise fwd/rev r.
- "high fwd/rev r indicates a misplaced mesh" — refuted; r tracks render sparsity
  (ρ=−0.65) and sheet incoherence (ρ=−0.83), not offset.

Also damaging, and independent of the sweep: **`auto_grown_20250702235910292` (PHerc1447)
has control-quality mesh placement** — peak offset −1.25 vox, 61% of tiles peaked within
2 voxels, modulation 0.370 vs the control's 0.357 — the *best* geometry in the profiled
corpus, the *lowest* fwd/rev r (0.222), a well-filled canvas (nz 0.67) — **and it is null.**
For that segment the depth-offset explanation is excluded by direct measurement and the
null still stands.

Alive, narrowly:
- **|offset| > 6 voxels**, untestable from a 28-slice volume. This is not hypothetical:
  `hunt/depth_profile.py` already measured a mesh sitting **−10.00 voxels** off its sheet
  (1447 `…20250502182142…`). At −10 the ink plane is at the very edge of, or outside, the
  17-slice window. Roughly 30–40% of the corpus has low modulation and unmeasured placement.

---

## 4. The pod experiment, redesigned

### 4.0 PREREQUISITE — spend $0 before spending $4 (do this first)

`hunt/depth_profile.py` already exists, runs locally, streams CT from S3, needs no GPU, and
costs **$0**. It has profiled 10 meshes; the other **56 unique meshes** are ~1–2 min each
(the 10 done landed inside a ~11-minute window), so **~1–2 h local, unattended**.

It measures the exact quantity the residual hypothesis needs: `tile_peak_off_med_vox` — how
far the mesh sits from its sheet.

**Decision gate:**
- If **no** corpus mesh has |peak offset| > 6 vox → the residual hypothesis is dead too.
  **Do not run the pod experiment. Save the $4 and redirect (§7).**
- If a meaningful set (say ≥5 meshes) sits >6 vox off → run §4.1 on exactly those.

This gate is the highest-value action in the whole investigation because it is free and it
can close the question outright.

### 4.1 E1 — deep-window sweep (only if the gate opens)

**Change vs the survey:** render **61 slices** (`--num-slices 61 --slice-step 1.0`, i.e.
±30 voxels) instead of 21, then slide the 17-slice model window across the full depth with
`--layer-start/--layer-end` — **one render, many inferences**. The survey's own tooling
already does the render (`runpod/render_tifxyz_sv.py`); the sweep is the §2.0 mechanism,
now validated end-to-end.

**Offsets:** centres at −22, −18, −14, −10, −6, −3, 0, +3, +6, +10, +14, +18, +22 → **13
positions**, forward only for 11 of them, forward+reverse at the **best** offset and at 0.
Step 4 voxels in the far field (the control shows the response is smooth on that scale) and
3 voxels near zero. Range ±22 covers two sheet thicknesses either way (papyrus at 9.36 µm
≈ 11–21 voxels/sheet).

**Segments (9 + 2 controls), justified:**

| # | segment | scroll | Mpx | why |
|---|---|---|---|---|
| C+ | **w035** (re-rendered 61-slice) | PHerc0139 | 30.5 | **positive control — mandatory.** Reproduces §2.1 and extends it to ±22. Without it a corpus null at ±22 is uninterpretable. |
| C− | `auto_grown_20250502160708188` | 1447 | 7.7 | **negative control**: modulation 0.043, r=0.914, 0% of tiles peaked within 2 vox. The "mesh in mush" case. If the sweep fires here, the sweep is broken. |
| 1 | `auto_grown_20250702235910292` | 1447 | 10.5 | lowest r (0.222), best corpus geometry, nz 0.67. Most control-like segment we own. |
| 2 | `auto_grown_20250703034159599` | 1447 | 20.2 | r 0.303, nz 0.65, large area |
| 3 | `auto_grown_20250703025628283` | 1447 | 17.4 | r 0.314, nz 0.67 |
| 4 | `…20251028213516907` | 0800 | 4.7 | r 0.329, profiled (mod 0.265, peak **+2.5**) — a measured non-zero offset |
| 5 | `z_dbg_gen_00320` | 1447 | 17.0 | r 0.370, nz 0.67, **ruling_z 3.36** (near corpus max 5.94) — the only segment with an independent periodicity hint |
| 6 | `…20251028220042762` | 0800 | 3.5 | r 0.387, **nz 0.75** — densest canvas in 0800 |
| 7 | `auto_grown_20250923164713356` | 1203 | 9.2 | r 0.399, profiled (mod 0.204, peak −1.75), best 1203 |
| 8 | `auto_grown_20250925223153537` | 1203 | 7.0 | r 0.427, nz 0.59 |
| 9 | `…20251028220955…` | 0800 | 9.7 | r 0.522 but **nz 0.88** — highest-coverage canvas in the corpus |

Selection rule: **well-filled canvases (nz ≥ 0.55) sorted by ascending fwd/rev r**, spread
across all three scrolls, with both profiled extremes included as controls. Deliberately
*not* selected on ruling_z, which §1.2 shows is independent of r.
Total corpus area ≈ 99 Mpx; with both controls ≈ 137 Mpx.

**Note:** the 9 corpus segments should be **replaced by the >6-vox meshes** if §4.0's gate
identifies a different set — the gate's measurement outranks the r-ordering above.

### 4.2 Acceptance criterion — what counts as a hit

Calibrated against the control, all four must hold at the same offset:

1. **Asymmetry gate — `fwd_rev_r < 0.20`.** The control never exceeded **0.118** at any of
   12 offsets; the corpus minimum over 66 meshes is **0.222**. The gap between those two
   numbers is the cleanest discriminator we have measured. A corpus segment that crosses
   below 0.20 at some offset has entered control territory for the first time.
2. **Depth-tuning gate — the response must PEAK.** Detection statistic vs offset must show a
   maximum with a control-like shape: a plateau within ±3 and a fall of **≥15% of
   excess-over-chance by ±6**. This is the design's real contribution: **real ink is
   depth-tuned; render artifacts, beam hardening and mesh-edge effects are not.** A flat or
   monotone response across ±22 is an artifact regardless of how strong it looks. The survey
   could not apply this test at all — it had one offset.
3. **Structure gate** — at the winning offset, the map must pass the existing 4-test text
   battery + tripwire (`salvage/verdict_*.py`), which all 66 meshes failed.
4. **Seed agreement** — the same offset must win for **both** released checkpoints
   (seed42 and seed43). The control gives 0.9991 / 0.9982, so genuine signal is
   seed-stable. **Caveat: only `step-075000_seed42.pth` is on disk locally**
   (`D:\vesuvius-data\models\ink9um\`); seed43 must be re-fetched from
   `scrollprize/ink_9um` on the pod (it was used in the original control run, so the
   pull path is known). Apply this gate only at the winning offset — 13 offsets × 2 seeds
   doubles the bill for no extra discrimination.

**Kill criterion (equally important):** if all 9 segments show a flat depth response with
`fwd_rev_r > 0.20` at every offset, the depth-offset explanation is dead for this corpus and
Track D should stop re-screening 1203/1447/0800 and redirect (§7).

---

## 5. Cost

Measured pod throughput from the 80-segment survey (`survey_all.json`, n=80, R²=0.356):

```
secs  =  159  +  8.06 × Mpx        # mesh download + 21-slice render + 2 inferences
```

R² = 0.356 — area explains only about a third of the variance (S3 download jitter and mesh
size dominate the rest), so treat this as an order-of-magnitude fit, not a precise model.
That is why the contingency below is 1.5×.

ESTIMATE for the deep sweep, splitting that marginal term (render scales with slice count,
inference does not): render61 ≈ (61/21) × render21, and 13 inferences instead of 2. Under
both plausible splits (render21 = 4 or 6 s/Mpx) the total lands at **30–36 s/Mpx**; use
**35 s/Mpx ± 20%**.

| item | quantity | time |
|---|---|---|
| fixed overhead | 11 segments × 159 s | 0.49 h |
| deep render + 13 infers | 137 Mpx × 35 s/Mpx | 1.33 h |
| **subtotal** | | **1.8 pod-h** |
| ×1.5 contingency (retries, setup, download) | | **2.7 pod-h** |

**Cost: 2.7 pod-h × $0.69/h ≈ $1.90 → budget $4** (one spare pod-hour). Even at a 3×
overrun it stays under $6, well inside the ~$45 remaining.

**Wall-clock:** ~2.5–3 h on one pod; ~1.5 h on two pods sharded (the survey already shards,
`survey_*.jsonl`).

**What must be built** (small — the pipeline exists):
1. `runpod/sweep_depth.py` — thin variant of `runpod/survey_segments.py`: render once at
   `--num-slices 61`, then loop `--layer-start/--layer-end` over the 13 centres, saving
   ds4 prediction maps + per-offset stats. **~80 lines**; both halves are already validated
   (render in the survey, layer-shift in §2.0).
2. Extend the analyser to emit the §4.2 depth-tuning curve per segment. **~40 lines.**
3. `hunt/depth_profile.py` needs **no change** for §4.0 — just run it over the remaining 56.

No new models, no training, no new data access.

---

## 6. Calibrated probability

**p(this route surfaces real letters) ≈ 4%.**

For a hit, both of these must be true:
- **(i) a mesh is >6 voxels off the true recto** — plausible; one measured case at −10 vox,
  and 30–40% of the corpus has low sheet modulation. Call it 35%.
- **(ii) that scroll/region actually carries ink detectable by ink_9um at 9 µm** — this is
  the binding constraint, and the evidence against it is heavy. 66/66 unique meshes null,
  zero tripwire hits. The corpus is PHerc1447 (SNR **8.5**, worst in our own k2b index),
  PHerc0800 (SNR 20, degraded) and PHerc1203 (SNR 87, mid). Most damningly, **the
  best-placed mesh in the corpus is null** (§3) — where geometry is demonstrably not the
  problem, there is still nothing. Call it ~10–12%.

0.35 × 0.11 ≈ 0.04. I would not go above 5%.

The honest summary: **§2 and §3 moved this from "promising explanation of the corpus null"
to "narrow residual worth one cheap check".** Its main value is now **negative**: it closes
a live alternative explanation cleanly, at low cost, so the corpus null can be reported as a
statement about the *scrolls* rather than about our *meshes*. That has real publication value
for the Aug 31 deliverable, independent of finding letters.

---

## 7. Where the money should actually go

§4.0's gate will probably close. The corpus we screened is not where signal is, and this
investigation has now shown the null is not a geometry artifact.

Our own scan-quality index (`out/k2b_index/`, `report/sections/01_index.md`) ranks
**PHerc0813 (SNR 159.6), 0125 (114.2), 1545 (112.2), 0211 (106.6), 0191 (99.6),
0358 (91.8)** as the best-scanning GP scrolls. **None has a published segment or mesh.**
We screened 1447 (SNR 8.5) and 0800 (SNR 20) because meshes existed there — we surveyed
where meshes were, not where signal is, and Investigation B has now removed the main excuse
for the resulting null.

The high-value question is therefore **"can we produce our own surface on PHerc0813?"**, not
"were our windows a few voxels off on PHerc1447?". That should be costed next.

---

## Appendix — artefacts produced

| path | what |
|---|---|
| `out/hunt/depth_offset_control_REAL.json` | the §2.1 sweep — 12 offsets × (fwd AUC, rev AUC, fwd/rev r, fired) |
| `out/hunt/preds/c{8..19}_{forward,reverse}.tif` | 24 real ink_9um predictions on the w035 crop |
| `out/hunt/depth_offset_v2.json` | the 28-detector offline search that failed the gate (best AUC 0.5677 vs model 0.9993) |
| `out/hunt/symmetry_stats.json` | §1 distribution + covariance |
| `hunt/out/depth_profiles.json` | mesh→sheet geometry for 10 meshes |
| `hunt/depth_offset_v2.py` | offline-detector search (new) |
| `hunt/sweep_local.sh`, `hunt/score_crop.py` | the local sweep runner + scorer |

**Superseded:** `hunt/depth_offset_control.py` / `out/hunt/depth_offset_control.json` — the
offline-statistic v1. Its flat null is *correct but uninformative*: §2.0 shows no offline
statistic sees these letters at any offset, so its `excess_retained` column (values of 4.25,
−3.98 from dividing by a negative aligned excess) is meaningless and must not be quoted.
