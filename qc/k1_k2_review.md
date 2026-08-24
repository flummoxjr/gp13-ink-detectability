# QC review — K1 (intensity-AUC kill test) and K2 (spectral information ceiling)

Reviewer: adversarial audit, 2026-08-16.
Scope: `trackD/k1_intensity_auc.py` + `out/k1_w035_auc.{json,png}`; `trackD/k2_spectral_ceiling.py` + `out/k2_spectra.{json,png}`; cached data on `D:\vesuvius-data\trackD\` (w035_{surf,ink,sup}.npy, k2_roi_*.npy).
Method: code read + full numeric reproduction + new discriminating experiments (per-slice AUC sweep, depth-contrast + within-patch controls, texture AUCs, synthetic PSD/floor validation, air-ROI noise references streamed for all three scrolls, 2 extra papyrus ROIs on PHerc0813).
Reproduction scripts: `trackD/qc/qc_k1.py`, `qc_k1b_depth.py`, `qc_k2_local.py`, `qc_k2_stream.py`, `qc_k2_air0139.py`; results in `qc_k1_results.json`, `qc_k1b_results.json`, `qc_k2_local.json`, `qc_k2_stream.json`, `qc_k2_air0139.json`; figure `qc_k2_air.png`. New ROI caches: `D:\vesuvius-data\trackD\qc_*.npy`.

---

## BLOCKERS

### B1. K1's kill conclusion is OVERTURNED: absolute intensity DOES carry linearly-recoverable ink signal at 9.36 µm — the registered statistic was structurally guaranteed to miss it

The arithmetic is fine (all four AUCs reproduce, O1) and the pre-registered rule `|AUC−0.5| < 0.02` is genuinely met **by the 17-slice z-mean statistic** (max |dev| = 0.0059). But a per-slice sweep (all 28 slices, average-rank AUC, same mask) shows the channel is alive:

- Per-slice raw-intensity AUC rises to **0.553 at z=12** (0.543–0.553 for z=11–13) and falls to **0.455–0.457 at z=5–7** — a sign flip with depth. Per-slice detrended (σ=50): 0.537 at z=12/13, so it survives illumination/thickness detrending.
- Mean-intensity depth profile (ink minus background, DN): **+8.6 at z=12, −6.3 at z=5–6**, +3.6 at z=23–24. Summed over the registered window z=5..21 this antisymmetric profile integrates to **+0.3 DN** — the z-mean cancels it *by construction*. The kill statistic did not measure "no signal"; it measured "the depth profile of the signal integrates to ≈0 over these 17 slices."
- A trivial linear depth-contrast feature, `mean(z10–14) − mean(z4–8)`, gives pooled **AUC 0.5715** (detrended σ=50: 0.5473); single slice z=12 alone: 0.5532. |AUC−0.5| = 0.072 — **3.6× the kill threshold**, on the same segment, mask, and labels, with a statistic that is linear in absolute intensity (so "linearly-recoverable" is the correct description of what was found).
- Within-patch control (7 connected supervision components — rules out patch-level illumination confounds): depth-contrast AUC per patch = 0.628, 0.562, 0.501, 0.637, 0.483, 0.559, 0.504 → **mean 0.553 ± 0.023 (SE over patches), 2.3σ above 0.5**; 4/7 patches individually ≥ 0.56, one weakly reversed. The signal exists *inside* individual patches.

Physical reading: ink shifts/reshapes the density profile along the surface normal (brighter at/just above the annotation plane — labels sit at z=14 by corpus construction — darker ~5 slices below) rather than shifting the depth-integrated mean. That is exactly what a 3D model can key on and what a z-mean deletes.

**Required fix before submission.** Do not publish "absolute intensity carries no linearly-recoverable ink signal at 9 µm / channel dead." Defensible statements: (a) **depth-integrated** (z-mean) intensity is dead (|AUC−0.5| ≤ 0.006, robust to detrending scale and to masked detrending); (b) **depth-resolved** intensity is measurably alive (per-slice AUC to 0.55, depth-contrast 0.57, ~15 DN peak-to-trough profile difference). Because the depth-contrast bands were chosen after seeing this segment's sweep (W4), freeze the statistic and validate once on the sibling native-9 µm labeled segments (w039/w040/w041/w044 — same S3/HF layout, ~1 GB each, CPU-only) before headlining the positive claim.

### B2. K2's JSON ships physically impossible headline numbers (`smallest_recoverable_halfperiod_um` = 6.6–6.9 µm, below the Nyquist half-period of 9.362 µm)

`k2_spectra.json` reports `crossover_cyc_per_px` 0.67–0.71 and `smallest_recoverable_halfperiod_um` 6.62–6.94 for all three scrolls. A half-period smaller than one voxel cannot be "recoverable" at any SNR; those crossovers live at radial |q| > 0.5 — the anisotropic corner region of the 3D FFT cube, where only near-diagonal orientations exist and the radial average is not a per-axis statement. Claim (c) of the writeup already disowns them, but the numbers remain in the shipped artifact under a name that asserts the wrong interpretation. Independent recompute: **zero crossings of the 2×-floor criterion anywhere in the per-axis band (q ≤ 0.5) for any scroll** (min data/floor at q=0.5: 9.5× / 10.0× / 23.3× for 0813/1203/0139).
**Required fix:** delete both keys (or rename to `corner_artifact_crossover_DO_NOT_USE`) and restrict every reported statistic to q ≤ 0.5. Regenerate anything downstream that consumed them.

### B3. K2's "cliff = the real information ceiling" (claim b) is OVERTURNED by an in-scan noise reference: real structure persists above scan noise to ~0.43–0.50 cyc/px, i.e. essentially to Nyquist

The steep PSD falloff is real, but a falloff proves nothing about *information* — red spectra are what real objects look like, and the falloff region is also where the Paganin low-pass bites. The discriminator is an air ROI from the same volume through the same PSD path: papyrus-PSD excess above air-PSD = structure above the measurement noise actually present in the released volume. Streamed air ROIs (selection: fully inside the release mask, fill > 0.999, minimal mean/variance; verified mean ≈ 46–54 DN ≈ the window's air level):

| scroll | papyrus/air @0.15 | @0.25 | @0.35 | @0.5 | pap < 2× air at | half-period |
|---|---|---|---|---|---|---|
| PHerc0813 | 115× | 30× | 6.8× | 1.17 | q = 0.433 | **10.8 µm** |
| PHerc1203 | 89× | 29× | 8.1× | 1.45 | q = 0.444 | **10.5 µm** |
| PHerc0139 (clean 128³ air, see W9) | 127× | 39× | 13.5× | 1.97 | q = 0.499 | **≈9.4 µm (Nyquist)** |

At the claimed "ceiling" of 0.25–0.35 cyc/px, papyrus still carries **7–39× the noise power**. The measured SNR≈1 point sits at 0.43–0.50 cyc/px: the released 9.36 µm volumes are **sampling/resolution-limited, not mid-band information-limited**. The "plunge" at 0.25–0.40 (steepest log-slope −11 to −13 at q = 0.36–0.40, O11) is the papyrus autocorrelation × transfer-function shape — descriptively true, but "the real information ceiling ~15–19 µm half-period" is not supported and materially understates what downstream methods can recover (by 0.3–1.0 octave).
**Required fix:** replace claim (b) with the air-referenced statement, and report cliff/crossover numbers only against the air reference (with the caveats in W8/W9). Note the constructive corollary for the report: PHerc0139 — the letters-bearing scroll — has the cleanest scan of the three (air hp-noise 2.6 DN vs 3.7/3.8) and structure above noise all the way to Nyquist.

---

## WARNINGS

### W1 (K1) — Tie handling in `auc_rank` is biased, but negligibly here
`argsort(argsort(·, stable))` assigns ordinal ranks; positives are concatenated first, so every tied pos–neg pair counts as a loss instead of a half-win, biasing AUC low by 0.5·P(tie). Measured: raw_zmean P(tie) = 9.4e−4 → bias ≈ 4.7e−4 (0.49541 ordinal vs 0.49587 average-rank); detrended stats P(tie) ≈ 6e−8. Harmless at the 0.02 threshold — but raw single-slice uint8 data has far more ties, so any per-slice AUC (as in B1) must use average ranks (`scipy.stats.rankdata`; done in QC re-runs). The 2M subsample cap never triggers here (334k pos / 737k neg), so the rng path is dead code on this segment.

### W2 (K1) — Detrending leaks mask-boundary artifacts; use masked detrending in follow-ups
`gaussian_filter(zmean, σ)` mixes the zero-valued outside-mask region into the background estimate near patch borders, inflating detrended values there. Masked detrending (Gaussian of masked data ÷ Gaussian of mask) moves σ=25/50 AUCs from 0.5059/0.5016 to 0.5067/0.5060 — no conclusion change for the z-mean statistic, but the artifact grows with σ and small patches; report the masked variant.

### W3 (K1) — Negative set is adequate
AUC with negatives restricted to distance bands from labeled ink (raw / detrended-σ50): 0–5 px 0.489/0.488; 5–15 px 0.508/0.511; 15–40 px 0.494/0.502; >40 px 0.495/0.501. No monotonic trend → no evidence of unlabeled ink in the background or of near-stroke contamination large enough to move the null. (Validates the z-mean null; does not affect B1.)

### W4 (K1) — Selection caveat on the positive finding
The depth-contrast bands (z10–14 vs z4–8) were chosen after seeing this segment's per-slice sweep; the patch-level SE (±0.023) covers sampling noise, not selection. The within-patch replication and the physically sensible profile shape argue it is real, but the clean protocol is: freeze the statistic now, evaluate once on w039/w040/w041/w044.

### W5 (K1) — "Strongest-known 9 µm ink segment" premise is untested here
The kill logic leans on w035 being the best case. That is a corpus-level claim not verified in this audit; the sibling-segment validation in B1/W4 resolves it for free.

### W6 (K2) — Quantization floor model is correct for rounding, but window clipping is a second, unmodeled release-format error
Synthetic check (red-spectrum field at matched DN scale): pure rounding error reproduces the uniform(−step/2, step/2) floor within ~13% across 0.05–0.5 cyc/px, with or without dither — the empirical floor methodology is sound (O7). But the release window **clips**: cached ROIs have 1e−4 to 8e−4 of voxels pinned at DN 0/255, and clip errors are unbounded; at a 1e−3 clip fraction, synthetic clip-error PSD reached ~2× the rounding floor at some frequencies. Immaterial to claim (a) (data ≥ 9.5× floor in-band; scan noise alone is 12–27× the floor at q = 0.4–0.5 in all three scrolls), but report per-scroll clip fractions as a release-format health metric.

### W7 (K2) — Transfer-function overlay: shape-only, minor modeling caveats
Formulas verified exactly (O8). Caveats: (i) the overlay is amplitude-anchored to the first PSD bin, which contains the leaked DC/mean term (0.76× flat level in the white-noise test) — anchor at q ≈ 0.02–0.05 instead; (ii) if the unsharp ran on 2D slices, the sphere-averaged |H|² is 0.72–0.85× the isotropic-3D model at q = 0.1–0.4 — invisible on a 7-decade log plot, but disqualifying for quantitative |H|²-division ("unfiltered spectrum" estimates); (iii) the metadata path (`tomo.processing.preprocessing.phase`) says the unsharp ran at the projection/phase-retrieval stage, and a radially symmetric detector-plane filter maps through FBP to an approximately isotropic 3D filter — so the isotropic model is the better-motivated one; state the stage instead of guessing.

### W8 (K2) — One 256³ ROI per scroll is not enough for per-scroll claims: site-to-site PSD spread reaches ~25× at the frequencies that matter
Two additional PHerc0813 papyrus ROIs (different z-thirds; origins (6784,3168,4608) and (11936,5088,4608)) vs the cached ROI: PSD max/min spread = 1.8× at q=0.05, 2.4× at 0.15, **4.9× at 0.25, 11.2× at 0.35, 24.6× at 0.5**. Any single-ROI crossover, cliff, or floor-margin number carries up to ~1 decade of site variance exactly where the claims live; the between-scroll differences in the current JSON are within single-site variance and cannot rank scrolls. Part of the spread is selection-criterion sensitivity (the K2 picker maximizes fill; one extra ROI has L5 mean 177 DN — likely containing denser material — and it is the high-PSD outlier), which is itself evidence that "papyrus-rich ROI" under-specifies the measurement. **Fix:** ≥5–10 ROIs per scroll with a fixed, documented selection rule; report median ± IQR of each statistic.

### W9 (K2) — Air-ROI selection needs variance-awareness and a cleanliness gate (first 0139 attempt was contaminated; 0813/1203 airs have mask-bleed)
Selecting air by minimum block mean alone gave a contaminated "air" ROI on PHerc0139 (hp-std 13.2 DN, PSD ≈ papyrus PSD — it is low-density *structure*, not noise); a variance-aware 128³ selection found clean air (mean 50.0, std 3.1, hp-std 2.6 DN, zero-frac 0). The 0813/1203 256³ airs contain 4.4%/6.5% exact-zero (masked) voxels, which add mask-edge structure to the air reference — this *overstates* the noise floor, so the B3 ceilings are conservative (the true ceilings can only be higher). Gate any air ROI on: fill = 1 at L0 (zero-frac ≈ 0), mean ≈ the window's air DN, minimal hp-std; the n-independence of the PSD normalization (O6) makes 128³ fallbacks legitimate.

---

## OK

- **O1 (K1) — Headline numbers reproduce.** Verbatim reimplementation on cached arrays: raw 0.49541 (json 0.4956; the 2e−4 delta is float32-vs-float64 z-mean tie noise on the heavily quantized raw statistic), detrend s25/s50/s100 = 0.50589/0.50164/0.49860 vs json 0.5059/0.5016/0.4986. Mask counts match exactly (valid 1,071,082; ink 334,002).
- **O2 (K1) — Windowing is AUC-irrelevant.** The recorded-window inversion is monotonic affine; rank AUC is invariant. No off-by-one in the z window (slices 5..21 inclusive = 17 of 28).
- **O3 (K1) — Max-projection of labels is trivially lossless.** The "3D" ink/supervision zarrs are nonzero only at z=14 of 28 (verified; corpus doc confirms annotation at Z=14 by construction). There is no label z-structure to wash out — the washing-out happens in the *intensity* aggregation (B1), not the label projection.
- **O4 (K1) — Sampling path clean.** Subsample cap not hit; seed irrelevant; pos/neg extraction verified identical.
- **O5 (K1) — POSITIVE FINDING TO REPORT: texture statistics carry clear signal at 9.36 µm.** On a 3-px-eroded valid mask (border-leak-safe), average-rank AUCs: per-pixel z-std over the central 17 slices **0.537** (σ=8-smoothed: **0.564**); in-plane 3×3 local std 0.470 and gradient energy 0.474, dropping when σ=8-smoothed to **0.423** (|AUC−0.5| = 0.077; ink is locally *smoother* in-plane and more variable in depth). Two independent texture channels clear the 0.55-equivalent bar. Together with B1 this explains why the released models work at 9 µm: the signal is depth-profile + texture, not depth-integrated brightness — worth a headline line in the report.
- **O6 (K2) — `radial_psd` is numerically sound.** White-noise test: flat to ±12% across 0.02–0.5; level matches the convention's expectation to 0.1%; bin/center alignment correct. The ÷ΣW normalization is unconventional but internally consistent, cancels in every ratio the claims use, and is n-independent for stationary fields (so 128³ and 256³ ROIs are directly comparable). Only defect: the DC/mean term leaks into the first reported bin (q ≈ 0.0055), which passes the q > 0.005 filter — harmless (see W7-i).
- **O7 (K2) — The floor comparison is not double-counting.** PSD(quantized) = PSD(clean) + floor verified to 0.4% on synthetic data. The data PSD already contains the floor, so "data ≫ floor ⇒ format not binding" and "data ≈ floor ⇒ only quantization left" are the correct inference directions.
- **O8 (K2) — Transfer-function formulas verified.** λ = hc/E = 10.972 pm at 113 keV; Paganin 1/(1+πλz(δ/β)q²) with q in cycles/m is the standard TIE-Hom filter (μ = 4πβ/λ derivation checked; z = 1.2 m, δ/β = 1000 from metadata); unsharp 1+c(1−exp(−2π²σ²q²)) is the correct multiplicative form of `in + c·(in − G_σ∗in)` with σ in px. Script values match independent recomputation exactly (H(0.25) = 0.1418, H(0.5) = 0.0420).
- **O9 (K2) — Claim (a) confirmed, with sharpened numbers.** Papyrus/floor across the per-axis band: ~3×10⁶ (6.5 decades) at q=0.05; 1.1–1.5×10⁴ at 0.25; 443–828 at 0.35; **9.5×/10.0×/23.3×** at q=0.5 (0813/1203/0139). "1–6 decades" is accurate (band edge ≈ 1.0–1.4 decades). Independently: in-scan noise itself sits 12–27× above the floor at q=0.4–0.5, and air hp-noise (2.6–3.8 DN) is 9–13× the quantization σ (0.289 DN) — uint8 rounding is nowhere the binding constraint. Cross-checked directly against B3: since air noise ≫ floor, the format argument no longer rests on the papyrus PSD at all.
- **O10 (K2) — Claim (c) confirmed.** The 0.67–0.71 crossovers are corner artifacts; nothing crosses 2× floor at q ≤ 0.5 (cleanup required per B2).
- **O11 (K2) — Cliff location, descriptively.** Steepest log-log slope: −12.6 at q=0.378 (0813), −12.3 at 0.356 (1203), −11.2 at 0.400 (0139); steepening begins ≈0.2–0.25. The claimed "0.25–0.35" is the onset, not the steepest point — and per B3 it is an object-spectrum × transfer-function feature, not a ceiling.
- **O12 (K2) — DN usage is healthy.** Papyrus ROIs: mean 99–110 DN, std 35–42 DN, clip fractions ≤ 8e−4 — the recorded window is well matched to the data; no evidence of dynamic-range waste that would change any conclusion.

---

## Deliverable shaping — the per-scroll "detectability index" numbers that survive this review

Report **three numbers per scroll**, each as median ± IQR over ≥5 papyrus ROIs with a fixed documented selection rule (W8), against a vetted air ROI (W9):

1. **Air-referenced structural bandwidth** `q_c`: the largest q ≤ 0.5 with papyrus-PSD ≥ 2× air-PSD, quoted as half-period in µm. This is the honest "what detail exists in the released volume above its own noise" number (this audit: 10.8 / 10.5 / ≈9.4 µm for 0813 / 1203 / 0139 — single-ROI values, spread pending).
2. **Mid-band structural SNR** at fixed q = 0.25 cyc/px (18.7 µm half-period, stroke-relevant scale): (papyrus−air)/air in power. A graded quality number that can actually rank scrolls (this audit: ≈29 / 28 / 38) instead of a binary crossover.
3. **Noise & format headroom**: air high-pass noise in DN (scan-noise floor; 3.7 / 3.8 / 2.6 DN), its ratio to the quantization σ = 0.289 DN, plus DN 0/255 clip fractions. This pins "uint8 not binding" quantitatively and flags any badly windowed future release.

Deprecate from the deliverable: 2×-quant-floor crossovers (measure nothing once scan noise ≫ floor), all |q| > 0.5 statistics (B2), and slope-based "cliff frequency" (object property, not a ceiling). The Paganin×unsharp overlay stays as a labeled shape annotation only (W7).

---

## Verdicts

- **K1 as pre-registered:** the kill rule fired correctly *for the statistic it was registered on* (17-slice z-mean, |AUC−0.5| ≤ 0.006). **The generalized conclusion "absolute intensity carries no ink signal at 9 µm / channel dead" is overturned** (B1): depth-resolved intensity reaches AUC 0.553 per-slice and 0.5715 with a two-band linear contrast (patch-level 0.553 ± 0.023), because the ink signature is an antisymmetric depth profile (+8.6/−6.3 DN) that the z-mean integrates to ≈0.3 DN. Additionally, texture channels are clearly alive (z-std AUC 0.564 smoothed; in-plane smoothness 0.423) — an important positive finding for the report (O5). Validate the frozen statistics on w039/w040/w041/w044 before publishing the positive form.
- **K2:** claims (a) and (c) confirmed (O9, O10) — uint8 is not the binding constraint, and the 0.67–0.71 crossovers are corner artifacts (but must be deleted from the JSON, B2). **Claim (b) is overturned** (B3): with an in-scan air-noise reference, real structure persists to q = 0.43–0.50 cyc/px (half-period ~9.4–10.8 µm) in all three scrolls — the release volumes are sampling-limited, not cliff-limited at 15–19 µm. Single-ROI-per-scroll statistics have up to ~25× site variance at the relevant frequencies (W8) and cannot support per-scroll rankings yet.
