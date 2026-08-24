# K3 — Lead-detection sensitivity bound for the Paris 4 dual-energy screen

**Scope.** Converts the measured noise floor of the K3 stage-2 screen (PHerc.Paris.4, BM18 45.532 µm
74/110 keV pair, level-1 = 91.064 µm voxels, 3³ aggregation, detrended ratio σ = 0.1259, 4σ LOW-channel
threshold, 2 singleton hits in ~2.7×10⁸ tested aggregates) into an explicit statement of what Pb loading
the screen could and could not have detected. Companion numbers file: `k3_sensitivity_bound.json`.

**Verdict up front (honest).** The LOW-channel null is *clean but nearly vacuous as a lead exclusion*.
At the 4σ threshold used, **no physically realizable Pb-ink configuration can fire the screen**: the
required ratio shift (0.504) exceeds the largest shift lead can produce given (a) the total available
papyrus→Pb ratio contrast from the measured baseline and (b) saturation of the uint8 export windows
(maximum achievable response ≈ 2.8–3.4σ for a fully ink-filled aggregation neighborhood, ≈ 1.1–1.7σ for
realistic single-surface stroke geometry). The Pb areal densities actually measured in Herculaneum ink
by Tack et al. (2016) and Brun et al. (2016) — 84 ± 5 and 16 ± 5 µg/cm² — would have produced ~1σ and
~0.3σ responses respectively. The null therefore *constrains only Pb loadings ≈ 2.4–13× the published
values, and only under implausible full-neighborhood coverage*; it is best read as a validation of the
noise model (2 hits vs ~8,500 expected for a Gaussian 4σ tail), not as evidence of absence of metallic
ink. This section supersedes the linearized stage-1 QC estimate ("realistic Pb ink > 5σ"), which
neglected the ratio's saturation toward the pure-Pb asymptote and the window clipping analyzed here.

---

## 1. Attenuation physics

Mass attenuation coefficients (µ/ρ, cm²/g), log-log interpolated from NIST/XCOM tabulations
(monochromatic; good to a few %, re-verify against XCOM before external publication):

| Material | µ/ρ @ 74 keV | µ/ρ @ 110 keV | **µ₇₄/µ₁₁₀** | Note |
|---|---|---|---|---|
| Carbon (carbonized papyrus proxy) | 0.165 | 0.147 | **1.12** | Compton-dominated |
| Cellulose C₆H₁₀O₅ | 0.179 | 0.158 | **1.13** | Compton-dominated |
| **Pb** | **2.95** | **4.37** | **0.674** | K-edge **88.005 keV** sits *between* the beams |
| — measured papyrus (this screen) | 0.063 cm⁻¹ | 0.0505 cm⁻¹ | **1.248** (1.216 offset-corr.) | per-voxel median ratio; µ₁₁₀ back-solved from it — see caveat C4 |

The K-edge is the whole story: at 74 keV Pb is *below* its K-edge (µ/ρ ≈ 2.95), at 110 keV *above* it
(µ/ρ ≈ 4.37). **Lead therefore LOWERS the 74/110 ratio** — the counterintuitive sign. Sign check: adding
to a papyrus voxel (ratio ≈ 1.25) any material whose own ratio (0.674) is below the baseline pulls the
mixture ratio monotonically *down* toward 0.674; it can never cross below it. Every classic heavy-metal
ink candidate behaves the same way (Au 80.7, Hg 83.1, Pb 88.0, Bi 90.5 keV K-edges all lie between the
beams), while every mid-Z element (Ca, Fe, …) *raises* the ratio. The LOW channel is the heavy-metal
channel; the HIGH channel (51,343 hits, the known Ca/Fe-like incrustation) is mid-Z.

## 2. Signal model

Let a level-1 voxel (path length L = 91.064 µm) contain papyrus (measured linear attenuation
µ₇₄ᵖ, µ₁₁₀ᵖ) plus ink occupying volume fraction f, the ink carrying Pb mass fraction w at bulk density
ρ_ink. Define the **effective Pb partial density** x = f·w·ρ_ink (g Pb per cm³ of voxel), equivalently
the **Pb areal density** σ_Pb = x·L (g/cm²) through the voxel. To first order the ink's organic matrix
is ratio-neutral (≈ papyrus), so:

    R(x) = ( µ₇₄ᵖ + (µ/ρ)₇₄^Pb · x ) / ( µ₁₁₀ᵖ + (µ/ρ)₁₁₀^Pb · x )
         = ( µ₇₄ᵖ + 2.95·x ) / ( µ₁₁₀ᵖ + 4.37·x )        → 0.674 as x → ∞

Linearized slope at x = 0: dR/dx = (2.95 − R₀·4.37)/µ₁₁₀ᵖ ≈ **−50 per g/cm³** (raw frame,
R₀ = 1.248, µ₁₁₀ᵖ = 0.0505) or **−31** (offset-corrected frame, R₀ = 1.216, µ₁₁₀ᵖ = 0.0773).
The linearization is *only* valid for |ΔR| ≪ R₀ − 0.674; the 4σ criterion is far outside that regime,
so all bounds below are computed from the exact rational form.

Two additional transfer factors sit between x and the screen statistic:

- **Aggregation dilution d** = (ink-bearing papyrus voxels)/(papyrus voxels) in the 3³ neighborhood
  (den ≥ 14). A 10–20 µm ink layer is confined to the one-voxel surface layer of a sheet: a 273 µm
  cube centered on it holds ≈ 9 inked and 14–27 total papyrus voxels → **d ≈ 0.33–0.5** for realistic
  stroke geometry (0.67 for the rare both-facing-surfaces case); d = 1 only if ink pervades the volume.
- **uint8 window clipping.** The exports clip at µ₇₄ = 0.270, µ₁₁₀ = 0.200 cm⁻¹. The 110 keV channel
  (larger Pb coefficient, smaller headroom) clips first, at x_sat ≈ 0.0343 g/cm³ ⇔ **σ_Pb ≈ 311 µg/cm²
  ≈ 0.27 µm of metallic-Pb-equivalent per voxel column**. Beyond x_sat the ratio *rises again* (µ₇₄
  keeps growing over a frozen denominator), re-crossing the papyrus baseline near σ_Pb ≈ 560–580 µg/cm²
  and pegging at 0.27/0.20 = 1.35 once both channels clip (σ_Pb ≳ 640 µg/cm²) — i.e., **heavily loaded
  Pb ink exits the low tail entirely and masquerades as an unremarkable slightly-high-ratio bright
  voxel, flagged by neither channel.**

## 3. Detection floor at 4σ — and why it is unreachable

Required aggregated shift: ΔR = 4σ = 4 × 0.1259 = **0.504**.

| Frame (baseline calibration) | R₀ | x for 4σ (no clip) | σ_Pb for 4σ | Max drop before 110-clip | Max response |
|---|---|---|---|---|---|
| Raw (as exported, stage-1 medians) | 1.248 | 0.084 g/cm³ | 761 µg/cm² | 0.429 | **3.4σ** |
| Offset-corrected (as-implemented stage 2) | 1.216 | 0.233 g/cm³ | 2126 µg/cm² | 0.357 | **2.8σ** |
| Perfectly calibrated Compton frame | 1.13 | — (target 0.626 < 0.674 asymptote) | unreachable | 0.456 (no clip needed) | **3.6σ** |

In *every* calibration frame the 4σ requirement lies at or beyond the maximum ratio depression lead can
produce — from the physical Compton baseline it exceeds the *total* papyrus→pure-Pb contrast even before
clipping, and the pink beam only shrinks that contrast further (expected pink-beam Pb ratio 0.7–0.95
rather than 0.674). **`reachable_4sigma: false` in all frames.** With realistic stroke dilution
(d ≈ 0.33–0.5) the ceiling drops to **1.1–1.7σ**; clipping at the native L0 resolution (45.5 µm, where a
surface layer clips at ≈ 156 µg/cm²) tightens the d = 1 ceiling from 3.4σ to ≈ 2.7σ. Separately, the
co-brightness gate (both energies above papyrus p75) itself excludes ink below ≈ 70–170 µg/cm² on
median-to-PV-dimmed papyrus, independent of the ratio statistic.

**Scenario table** (raw frame; per-aggregate significance at neighborhood ink fractions d):

| Configuration | σ_Pb (µg/cm²) | per-voxel ΔR | d = 1 | d = 0.5 | d = 0.33 |
|---|---|---|---|---|---|
| Tack/Brun large fragment (measured letters) | 84 | 0.255 | 2.0σ | 1.0σ | 0.7σ |
| Tack/Brun small fragment | 16 | 0.076 | 0.6σ | 0.3σ | 0.2σ |
| Dense hypothetical ink: 20 µm layer, 10 wt% Pb, ρ 1.5 | 300 | 0.425 | 3.4σ | 1.7σ | 1.1σ |
| Saturation optimum | 311 | 0.429 (max) | 3.4σ | 1.7σ | 1.1σ |
| Extreme loading (both windows clip) | 1000 | **−0.10 (ratio ABOVE baseline)** | −0.8σ | −0.4σ | −0.3σ |

## 4. The bound, in words

- **As-run (4σ): the screen would have detected no physically realizable Pb ink.** There is no
  (Pb mass fraction, areal coverage) pair that reaches threshold; the clean null (2 singletons /
  2.68×10⁸, vs ~8,500 expected for Gaussian tails — the detrended statistic is strongly sub-Gaussian)
  confirms the false-positive calibration and nothing else about lead.
- **Best defensible re-reading of the same data** (3σ, raw frame, before pink-beam dilution): the screen
  statistic could have flagged ink only inside the narrow window **σ_Pb ≈ 203–343 µg/cm² per voxel
  column** — e.g. a 15–20 µm ink layer with **≥ 7–9 wt% Pb** (ρ_ink 1.5) — and only if it covered
  **≥ 88% of the papyrus voxels of the entire 273 µm aggregation neighborhood** (not just one voxel
  face; single-face coverage never reaches 3σ). Loadings above ~343 µg/cm² clip back out of the 3σ
  window (above ~420 µg/cm² even the 2σ window closes).
- **Against the literature values:** Tack et al. (Sci. Rep. 6:20763, 2016) and Brun et al. (PNAS
  113:3751, 2016) measured, on two unrolled Herculaneum fragments from the Institut de France
  collection (ESRF ID21 XRF + Monte Carlo quantification), Pb areal densities in letters of
  **84 ± 5 µg/cm²** (large fragment) and **16 ± 5 µg/cm²** (small), i.e. Pb weight fractions of
  ≈ 0.78 wt% and 0.15 wt% *in the probed papyrus+ink matrix* (Tack's Monte Carlo model: a homogeneous
  300 µm papyrus layer at ρ ≈ 0.36 g/cm³ — not an ink-layer concentration); they argued the lead was
  intentional (pigment or drier; speciation ambiguous — Pb-L3 XANES closest to a lead(II)-acetate-like
  carboxylate, galena disfavored by S-K XANES). **Both loadings sit 2.4× and 13× below even the most
  favorable 3σ window of this screen and produce ≤ 1σ under realistic stroke geometry.**
  A "PHerc.172-like" ink — dense enough to be directly visible in single-energy full-resolution CT —
  needs only a few-to-tens of µg/cm² of Pb-equivalent excess at µm voxel scale, i.e. it is *also* below
  this screen's floor; the right instrument for that regime is full-resolution intensity/texture
  channels (K1-style), not a 91 µm ratio screen.

## 5. Comparison with the PHerc.0332 dual-energy null (axiosdevs, 53/70 keV)

The 53/70 keV pair lies entirely *below* the Pb K-edge, so it probes lead through the photoelectric E⁻³
slope, not the edge jump: µ₅₃/µ₇₀(Pb) ≈ 6.92/3.40 = **2.04** vs papyrus ≈ 1.09–1.12 — Pb appears in the
**high** tail there (opposite sign to our design), with larger fractional headroom (+0.9 vs −0.57) and
no edge-related pink-beam dilution. Its structural weakness is degeneracy: Fe/Ca minerals also sit high
(Fe 53/70 ≈ 2.0, Ca ≈ 1.8 — Fe is essentially degenerate with Pb's 2.04 there), so a 53/70 high-tail
hit cannot be signed as heavy metal by ratio alone,
whereas the 74/110 K-edge straddle uniquely signs heavy metals by direction. The two nulls are therefore
complementary in mechanism, but neither null means "no Pb" until an equivalent saturation-and-noise
sensitivity floor is published for the 0332 screen; we do not have axiosdevs' σ, thresholds, or window
parameters and make no quantitative claim about their sensitivity.

## 6. Caveats (all apply to the numbers above)

1. **Pink beam.** BM18 runs filtered white beam; "74/110 keV" are nominal effective energies of broad,
   inconsistently documented spectra (the two scans used different filtrations; the 110 keV scan was
   acquired as "109keV_B"). All monochromatic predictions carry ±10% systematic; the part of the 74 keV
   spectrum above 88 keV and of the 110 keV spectrum below it dilutes the Pb contrast (expected
   pink-beam Pb ratio 0.7–0.95), making every σ-level in this memo an *upper* bound on real sensitivity.
   The sign of the Pb shift is robust to any plausible spectrum.
2. **Residual phase contrast.** Both reconstructions used Paganin delta_beta = 10 (far below the 500–1000
   used elsewhere for this scroll): strong fringes remain, and the 74 keV volume is ~22% more blurred
   (filter width ∝ √λ). Fringe undershoot is the likely origin of the *negative* "air offsets"
   (−0.031/−0.027) used for zero calibration, so the offset correction itself is uncertain — hence the
   two bracketing frames above.
3. **uint8 windows.** Clipping at µ₇₄ = 0.270 / µ₁₁₀ = 0.200 cm⁻¹ caps the ratio response (Section 2)
   and, at L0, bites at ≈ 156 µg/cm² for surface-layer ink. Worse, heavily loaded Pb saturates *both*
   channels at ratio ≈ 1.35 and is flagged by neither channel. A saturation channel (co-occurring
   u8 = 255) was not implemented; counting saturated voxels is a cheap, necessary follow-up before any
   "no dense metallic ink" statement.
4. **The 1.25-vs-1.13 discrepancy.** Measured papyrus ratio (1.248 raw; 1.216 after the stage-2 offset
   correction) still exceeds the Compton-only expectation (1.12–1.14). The stage-2 air-offset
   calibration moved it in the *right* direction but closed only ~25% of the gap; a ~8–10%
   multiplicative or offset systematic remains unexplained (mixture of zero-offset error,
   effective-energy shift, mineral content). The bound is therefore quoted per calibration frame rather
   than as one number.
5. **Bound is for the LOW channel as implemented.** 4σ on the detrended 3³-aggregated ratio, den ≥ 14,
   plus per-voxel co-brightness above papyrus p75 — the gate alone floors sensitivity at ≈ 70–170 µg/cm²
   and specifically disfavors the PV-dimmed sheet-surface voxels where ink actually lives. Alignment is
   validated to sub-voxel only at the tested level; per-voxel ratio integrity at sheet boundaries
   inherits that residual.
6. **Dilution geometry is modeled, not measured**: d ≈ 0.33–0.5 assumes ≥ 273 µm stroke width, one inked
   face, 1–2-voxel sheet thickness at L1. Thicker sheets or interleaved inked faces shift d modestly;
   no realistic geometry approaches the d ≈ 0.9 the 3σ window requires.

## 7. What a real Pb bound would take

The gap to Tack/Brun-level sensitivity is ≈ 5× in σ at stroke scale: σ_ratio ≈ 0.02–0.03 per matched
aggregate. Concretely: (i) re-screen at L0/L1 with a *sheet-surface-conformal* matched filter (aggregate
along the writing surface only — removes the d dilution and averages ~10²–10³ voxels per stroke);
(ii) recalibrate zero offsets on resolved air and re-derive σ; (iii) add the saturation channel;
(iv) drop the p75 co-brightness gate in favor of a surface-band mask; (v) treat 2–2.5σ candidates as a
ranked shortlist for L0 visual review rather than using a hard 4σ cut. Until then, the defensible public
sentence is: **"the Paris 4 74/110 keV screen excludes only lead loadings an order of magnitude beyond
those ever measured in Herculaneum ink; it does not test the Brun/Tack metallic-ink hypothesis."**

---
*Method note: all derived quantities in `k3_sensitivity_bound.json`; computation script archived at
`trackD/k3_sensitivity_calc.py` (pure closed-form algebra on the model above). Measured inputs
verified against `out/k3_s2_stats.json`, `qc/k3_stage1_review.md`, and `k3_stage2_screen.py`.*

Sources: [Tack et al. 2016, Sci. Rep. 6:20763](https://www.nature.com/articles/srep20763)
([PMC4745103](https://pmc.ncbi.nlm.nih.gov/articles/PMC4745103/)) · [Brun et al. 2016, PNAS 113(14):3751](https://www.pnas.org/doi/10.1073/pnas.1519958113)
([PMC4833268](https://pmc.ncbi.nlm.nih.gov/articles/PMC4833268/)) · NIST XCOM mass-attenuation tabulations.

---

## Referee notes (hostile pass, 2026-08-16)

**Verified correct (recomputed independently, no change needed):**

- **Sign of the Pb shift.** µ/ρ(Pb) = 2.948 at 74 keV (below the 88.005 keV K-edge, NIST log-log
  60→80 keV) and 4.373 at 110 keV (above it, 100→150 keV); µ₁₁₀/µ₇₄ = 1.48 > 1, so the 74/110 ratio
  of Pb is 0.674 < papyrus ≈ 1.25 and lead **lowers** the ratio. The memo's sign, arithmetic, and
  monotonic-approach-to-asymptote argument are all correct.
- Carbon/cellulose µ/ρ values are NIST-consistent (74/110 ratios 1.12/1.13); Pb 53/70 = 6.92/3.40 =
  2.04 also reproduces.
- Mixture model: additive *linear* attenuation with Pb partial density x (g Pb per cm³ voxel) is the
  correct algebra; no volume-/mass-fraction confusion. σ_Pb = x·L correctly makes thin-layer areal
  density invariant to voxel size (partial volume handled at voxel level, dilution d only at
  aggregation — no double count).
- σ propagation: 0.1259 is measured *on the 3³-aggregated detrended statistic* (verified in
  `k3_stage2_screen.py`); the memo compares d·ΔR against it directly and does **not** divide by √27
  again. Correct.
- All raw-frame numbers (x_sat 0.0342 / 311 µg/cm² / ratio 0.819 / max 3.4σ; 4σ at 761 µg/cm²
  unclipped; 3σ window 203–343 µg/cm², d ≥ 0.88; L0 clip 156 µg/cm² → 2.72σ; gate floor 71/173
  µg/cm²; scenario shifts; Gaussian 4σ expectation 8,488 vs 2 observed singletons) reproduce by hand.
- Literature: both papers do report **84 ± 5 (large) / 16 ± 5 (small) µg/cm²** for the letters
  (verified against PMC4745103 §results and PMC4833268); PNAS 113(14):3751 and Sci. Rep. 6:20763 are
  the right citations; Institut-de-France fragments, ESRF ID21, intentional-lead conclusion, and the
  carboxylate/galena speciation statement all check out.

**Errors found and fixed (md + json + `k3_sensitivity_calc.py` regenerated):**

1. **Post-clip re-cross was wrong by ~1.7×**: prose claimed the ratio re-crosses the papyrus baseline
   "near 900–1000 µg/cm²"; the correct value from their own model is **576 µg/cm² (raw frame) /
   561 (corrected)**, with the 1.35 plateau from ≈ 639 µg/cm² (both channels clipped). Fixed; the
   json now carries `areal_recross_baseline_ug_cm2` and `areal_both_clip_ug_cm2` per frame.
2. **Corrected-frame baseline was internally inconsistent**: the raw frame used µ₁₁₀ᵖ = 0.0505
   (pinned by the measured per-voxel ratio median 1.248), but the corrected frame added the air
   offset to 0.050, giving µ₁₁₀ᵖ = 0.0768 and R₀ = 1.224. Made consistent (0.0505 + 0.0268 =
   0.0773, R₀ = 1.216): corrected-frame numbers moved from slope −31.3 → −30.7, 4σ areal
   1749 → 2126 µg/cm², max drop 0.364 → 0.357, ceiling 2.89σ → 2.84σ. No conclusion changes
   (still `reachable_4sigma: false`, still no 3σ window in this frame).
3. **Tack et al. quantification model misdescribed**: the 0.78 / 0.15 wt% Pb values are for a
   **homogeneous 300 µm papyrus+ink layer at ρ = 0.36 g/cm³** (0.0078 × 0.36 × 300 µm = 84 µg/cm²,
   self-consistent), *not* "their assumed ~50 µm writing layer" — the paper assumes no ink-layer
   thickness at all. Fixed in §4 and in the json `literature` block.
4. **3σ-window exit bound**: "loadings above ~420 µg/cm² clip back out of the window" mixed up the
   2σ and 3σ windows; the 3σ window closes at ~343 µg/cm². Fixed.
5. **Caveat 4 self-contradiction**: the offset correction moves the papyrus ratio 1.248 → 1.216,
   *toward* the 1.12–1.14 Compton expectation — the memo said "the wrong direction". Fixed (right
   direction, ~25% of the gap).
6. **Fe 53/70 understated the §5 degeneracy**: Fe is ≈ 2.03, essentially identical to Pb's 2.04
   (not "≈ 2.3-like"); ratio-only signing at 53/70 is fully degenerate, which *strengthens* the
   stated complementarity argument. Fixed.
7. Cosmetic: §1 table now quotes µ₁₁₀ᵖ = 0.0505 and per-voxel median ratios (1.248 / 1.216) so the
   quotient matches the stated R₀; verdict "≥ 2–20×" tightened to the computed "≈ 2.4–13×"
   (203/84 and 203/16 against the most favorable 3σ window floor).

**Remains uncertain (flagged, not fixable from the archived numbers):**

- The stage-1 medians are known only to ~2 s.f. (µ₇₄ ≈ 0.063, µ₁₁₀ ≈ 0.050 cm⁻¹ in the QC memo);
  the raw-frame µ₁₁₀ᵖ = 0.0505 is back-solved from the ratio median 1.2484. Corrected-frame R₀ is
  therefore uncertain at the 1.216–1.224 level; all corrected-frame areal numbers inherit ~±5–20%.
- The scenario table is quoted in the *raw* frame (the more favorable one). In the as-implemented
  corrected frame the Tack-84 response is ≈ 1.5σ at d = 1 (≈ 0.7σ at d = 0.5), i.e. ~25% lower than
  tabled — the memo's "≤ 1σ realistic" claim survives either way.
- `n_tested ≈ 2.68×10⁸` aggregates is consistent with the L3 papyrus count (4.79×10⁶ × 64, minus
  erosion/den≥14) but was not re-derived from the L1 arrays here.
- The co-brightness gate floor assumes the ink brightens a papyrus voxel by its Pb content alone
  (no organic-matrix term) against corrected-frame p75 thresholds; the ~70–170 µg/cm² floor is a
  model estimate, not a measurement.
- Monochromatic NIST values throughout; the pink-beam ±10% systematic and the 0.7–0.95 effective
  Pb-ratio dilution (caveat 1) are estimates, not measured spectra. The sign conclusion is robust;
  every σ ceiling quoted is an upper bound on true sensitivity, which only strengthens the
  "nearly vacuous as a lead exclusion" verdict.
