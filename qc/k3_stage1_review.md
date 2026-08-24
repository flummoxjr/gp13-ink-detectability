# QC review — K3 stage 1 (Paris 4 45.5 µm dual-energy 74/110 keV ratio screen)

Reviewer: adversarial audit, 2026-08-16.
Scope: `trackD/k3_dualenergy_stage1.py`, `trackD/out/k3_stage1_stats.json`, `trackD/out/k3_stage1_overview.png`, volume metadata in `trackD/meta/PHercParis4.json` (nodes 20260310170716 / 20260310173927 and their scans 20260310152857 / 20260310160232).
Method: code read + metadata cross-check + independent re-analysis of the cached L3 volumes (stats reproduced exactly; erosion, distance-to-edge, selection-bias controls, clustering, offset and trend checks run on `D:\vesuvius-data\trackD\paris4_45um_L3_{74,110}keV.npy`).

---

## BLOCKERS (must fix before scaling)

### B1. The screen looks in the wrong tail for its stated target (Pb/heavy-metal ink)
The K-edges of every classic heavy-metal ink candidate sit **between** the two beam energies: Au 80.7 keV, Hg 83.1 keV, **Pb 88.0 keV**, Bi 90.5 keV. Below its K-edge (at 74 keV) Pb attenuates *less* than above it (at 110 keV): µ/ρ(Pb) ≈ 2.9 cm²/g at 74 keV vs ≈ 4.4 cm²/g at 110 keV, so **µ74/µ110(Pb) ≈ 0.67** — far *below* the papyrus baseline of 1.25, not above it. A Pb-bearing voxel shows up as a **low-ratio, co-bright** anomaly. Stage 1 flags only the *high*-ratio/bright-in-74 tail, which selects mid-Z material (Ca, Fe — K-edges 4.0 / 7.1 keV, ratio 1.4–2.3) and can **never find lead ink**. The polychromatic pink beam softens but cannot flip this sign (see Physics Reference).
**Fix:** add a second channel — voxels bright in *both* energies with ratio *below* ~1.0 — as the Pb/Au/Hg/Bi screen. Keep the high tail as a separate mid-Z (mineral / Fe-ink) channel and label it as such.

### B2. The dense-voxel statistic (1.705) is inflated by rim contamination and numerator selection bias
Empirical re-analysis of the cached L3 data:
- **36% of dense voxels lie within 2 voxels of the mask edge** (papyrus overall: 1.0% — a 37× over-representation). This is the red rim visible in the overview PNG. 76% of dense voxels are spatial singletons.
- Eroding the mask by 3 voxels drops the dense median from **1.705 → 1.56** (re-quantiled: 1.575).
- Selection-bias control: selecting the top 0.05% of **f110** (the denominator) instead gives ratio median **1.46**; a symmetric (standardized-sum) selection gives **1.62**. Selecting on the numerator alone biases the ratio high; the 1.46–1.71 spread brackets ~0.1–0.25 of pure selection/edge artifact.
- The rim itself is expected physics, not material: with equal delta_beta the Paganin filter width still scales as sqrt(lambda), so the 74 keV volume is ~22% more blurred than the 110 keV volume; at papyrus/air interfaces this systematically inflates f74/f110.
**Fix:** erode the both-mask by ≥3 L3 voxels (or use a distance-transform cut), require co-brightness (both energies above their own quantiles), and require spatial clustering (drop singleton components) before quoting any dense-material ratio.
**Note:** after these fixes a *real* signal remains — see OK section, O9. The separation is real but it is mid-Z, not heavy metal.

### B3. Absolute ratio scale is uncalibrated (unknown per-volume zero offsets)
The ratio f74/f110 equals µ74/µ110 only if both reconstructions have zero offset (air → 0). Check on interior gap voxels (both-mask, >3 voxels from edge, below papyrus threshold in both; n = 561k): median **f74 = +0.0025 but f110 = +0.0080**, where pure air should read ≈ 0 in both and the more-blurred 74 keV volume should if anything read *higher* in gaps, not lower. The interior histograms show no resolved air mode at L3 (gaps are sub-voxel), so this is PV-contaminated — but an f110 offset of just +0.005 shifts the papyrus ratio by ~10%. This plausibly accounts for the whole gap between observed papyrus median 1.248 and the Compton-only expectation ≈ 1.12–1.14 — i.e., the absolute number currently carries a ~10% unexplained systematic.
**Fix:** calibrate offsets before any quantitative Z inference — measure air at full resolution in resolved gaps/lumen, or from the unmasked reconstructions outside the scroll; alternatively work only with *relative* spatial anomalies and say so explicitly.

### B4. Alignment is validated only at level 3 — insufficient for any finer-scale ratio work
Phase correlation at L3 (8× binned, 364 µm voxels) with shift (0,0,0) constrains residual translation only to ±0.5 L3 voxel = **±4 full-resolution voxels**. The two scans are separate reconstructions with different rotation-axis positions (1131.5 vs 1131.625 px), different projection counts (4000 vs 3600), and z shapes 4071 vs 4066 (the L3 arrays happen to both round to 509, so the script's z-crop was a no-op — the L0 crop-at-start assumption is still unverified). Papyrus sheets at full res are only a few voxels thick; a 1–2 voxel misregistration destroys per-voxel ratios at sheet boundaries.
**Fix:** before scaling to finer levels, do sub-voxel registration (greyscale, not binary mask; allow small rotation) at the target level and report residuals; validate the z-crop convention at L0.

---

## WARNINGS (fix before publishing)

- **W1 — Latent `np.roll` wraparound bug.** Harmless at shift 0, but with any nonzero shift, voxels (and mask fill) wrap from the opposite face, corrupting both the shifted volume and the post-shift IoU. Use `scipy.ndimage.shift(..., cval=0)` or slice-based shifting.
- **W2 — u8=0 collision: fill vs data.** 0 is both the mask fill and the code for f ≤ window-min. The interior histograms show a smooth continuum at u8 = 1,2,3 (16–18k voxels each), so in-mask voxels quantized to 0 certainly exist and are silently absorbed into the "mask". Consequence: **IoU 0.9816 is partly a value-agreement metric, not a pure alignment metric** — don't quote it as mask overlap alone. (Papyrus stats are unaffected since selection requires f > 0.02.)
- **W3 — `np.maximum(f110, 1e-4)` clamp.** Did not trigger on this run (0% of dense voxels clamped; none below the 0.02 papyrus threshold either), but it can silently manufacture ratios up to ~2700 instead of failing. Replace with an explicit denominator floor plus a reported excluded fraction. Also make the dense-selection conditioning symmetric with the papyrus stats (currently dense voxels need only m110, not f110 > 0.02).
- **W4 — Systematic ratio trends not removed.** Measured on L3: papyrus ratio median drifts **1.236 → 1.287 (+4%) from z-bottom to z-top** (consistent with spectral/flux drift — machine current ramped mid-scan in both acquisitions: 73.8→81.7 mA and 79.9→89.6 mA, 16-bunch top-ups) and **+2% center → edge radially** (differential beam hardening). Both are comparable to plausible mid-Z ink signals. Detrend (z and radius) before screening at scale; for the Pb channel the expected signal (ratio 0.7–1.0) dwarfs these, but false-positive maps will still be spatially structured.
- **W5 — Papyrus threshold f > 0.02 is arbitrary and asymmetric.** It corresponds to u8 ≥ 61 (74 keV) vs ≥ 64 (110 keV) and ~30–40% of each papyrus median; it admits many partial-volume boundary voxels. The quoted spread is overstated: p95 drops from 1.501 to 1.466 after 3-voxel erosion. Erode first, then threshold; justify the level per volume (e.g., fraction of papyrus mode).
- **W6 — Everything at L3 is a partial-volume mixture.** 364 µm voxels exceed sheet thickness; no L3 voxel is pure papyrus, air, or ink. Stage-1 ratios are mixture ratios; do not present them as material constants.
- **W7 — Beam/spectrum metadata is internally inconsistent.** The recorded attenuator tables are identical for both scans, but the free-text comments differ ("wba25 = G-carbon 45mm rods" for 74 keV vs "wba11 = Mo 1.30mm" for 110 keV) and the motor positions confirm different filter selections. The 110 keV volume was acquired as scanRadix "…_109keV_B" but is labeled 110.0 keV. Nominal energies are design values of broad filtered-white (pink) BM18 spectra; treat all theoretical ratio predictions as ±10% and do not attempt spectrum reconstruction from this metadata.
- **W8 — Figure histograms are dominated by quantization artifacts.** The comb structure in the f74/f110/ratio histograms is u8 quantization aliased by the bin count, plus a genuine odd/even sawtooth from the u16→u8 requantization (visible in raw counts). Use quantization-aligned bins (multiples of 1/255 of each window) or state the cause; as published, the figure invites "what is that spike at 1.37?" questions.
- **W9 — Phase-correlation robustness.** Peak 0.163 is accepted without any threshold or failure check; correlation is run on binary masks only (cannot sense rotation, scale, or interior misalignment), on the central subvolume only. Add a peak-quality gate and a greyscale-interior cross-check.
- **W10 — Unverifiable encode convention.** The /255 inversion assumes the export mapped [lo, hi] → [0, 255] linearly with rounding. If the exporter reserved 0 for fill and mapped data to [1, 255], all f values are biased by ≤1 LSB (0.0013 / 0.0009). Negligible for ratios but state the assumption; the u16 windows in the metadata pin the endpoints but not the u8 rounding rule.

---

## OK (verified sound)

- **O1 — Window inversion math.** `u8/255*(hi-lo)+lo` is the correct linear inverse for an endpoint mapping (0→lo, 255→hi); /256 would be wrong. No off-by-one.
- **O2 — Windows verified against metadata, twice over.** Script windows match the export metadata exactly, and the export's u16 windows are self-consistent with the reconstruction's recorded 32-bit conversion ranges: 74 keV u16 [22812, 52540] over recon range [-0.30969, 0.41337] → [-0.0580, 0.2700]; 110 keV u16 [18040, 57516] over [-0.14968, 0.24875] → [-0.0400, 0.2000]. Agreement < 1e-4 in both.
- **O3 — Mask ordering correct.** Masks are computed on u8 before inversion; fill values (which invert to f = lo) never leak into any statistic or percentile (all are mask- or papyrus-restricted).
- **O4 — Phase-correlation implementation correct** (rfftn normalization, spectral whitening, wraparound disambiguation; the `p <= n//2` even-length edge case is conventional). Shift (0,0,0) is independently plausible: same mount, identical helical geometry (z_start −355.0, z_end −177.4252, z_step 5.919 in both), scans ~30–90 min apart.
- **O5 — Paganin parameters are equal: delta_beta = 10.0 in both scans** (confirmed in `preprocessing.phase` for scans 20260310152857 and 20260310160232; both nabu/GHBP, same detector, same 11.00 m distance). The gross delta_beta confound is excluded — only the λ-dependent filter-width difference remains (see B2). Note delta_beta = 10 is far below the 500–1000 used for the other volumes of this scroll, so both volumes retain strong phase fringes.
- **O6 — Stats reproduce exactly.** Independent re-run on the cached L3 arrays reproduces every JSON value (papyrus median 1.2484, p05 0.9939, p95 1.5010, dense 1.7051, n = 4,790,408 / 3,196).
- **O7 — Values are physically sensible as µ in cm⁻¹.** Papyrus core medians µ74 ≈ 0.063, µ110 ≈ 0.050 cm⁻¹ correspond to carbonized papyrus at ρ ≈ 0.35–0.4 g/cm³ — the right ballpark.
- **O8 — Quantization is negligible for the ratio.** Steps: q74 = 0.328/255 = 1.286e-3, q110 = 0.24/255 = 9.41e-4. Per-voxel σ = q/√12 → 3.7e-4 / 2.7e-4. At papyrus values (f74 ≈ 0.07, f110 ≈ 0.05, r ≈ 1.25–1.4): σ_r(quant) ≈ **0.011**, vs observed papyrus robust σ ≈ 0.118 (p05–p95 0.507). Quantization is <1% of the ratio variance; the spread is real. Moreover the f74/f110 fluctuations are ~90% correlated between energies (z-difference σ per volume ≈ 22%/18% of median, yet ratio σ only ≈ 9.5%), i.e., most of the per-volume "noise" is shared real density structure.
- **O9 — A genuine interior dense population exists.** After removing rim and singletons, spatially coherent components remain: the six largest (263, 111, 44, 28, 27, 20 voxels; largest ≈ 12.7 mm³) all cluster in one region (L3 z ≈ 85–155, y ≈ 153–191, x ≈ 185–200), sit 6–8 voxels (~2.5–3 mm) inside the surface, are co-bright in both energies (f74 ≈ 0.15–0.21, f110 ≈ 0.10–0.13), with ratio 1.48–1.58. This is a real dense phase — but at ratio ~1.5 it is **mid-Z (Ca/Fe-bearing mineral incrustation or debris), not heavy metal** (Pb would sit below 1; see B1), and its localized clump geometry does not look like ink.

---

## PHYSICS REFERENCE (expected µ74/µ110)

Monochromatic NIST mass-attenuation values (cm²/g), log-log interpolated to 74 and 110 keV. Verify against XCOM before publishing; values here are good to a few %.

| Material | µ/ρ @74 keV | µ/ρ @110 keV | **µ74/µ110** | K-edge | Note |
|---|---|---|---|---|---|
| Carbon (graphite) | 0.165 | 0.147 | **1.12** | 0.28 keV | Compton-dominated |
| Water / cellulose-like | 0.189 | 0.166 | **1.14** | — | cellulose ≈ 1.12–1.14 |
| Ca / calcite (approx) | ~0.25 | ~0.18 | **~1.4** | 4.0 keV | mid-Z, photoelectric tail |
| Fe | 0.72 | 0.32 | **2.25** | 7.1 keV | mid-Z, strong |
| Au | — | — | **< 1** | 80.7 keV | edge between energies |
| Hg | — | — | **< 1** | 83.1 keV | edge between energies |
| **Pb** | 2.95 | 4.37 | **0.67** | **88.0 keV** | **ratio inverts** |
| Bi | — | — | **< 1** | 90.5 keV | edge between energies |

Key structural fact: **every mid-Z element (K-edge < 74 keV: Ca, Fe, Cu, Zn, Sr, Sn, Ba…) raises the ratio above papyrus; every classic heavy metal (Au, Hg, Pb, Bi — K-edges 80.7–90.5 keV) lowers it below papyrus.** The two tails of the ratio distribution are two different chemistry channels.

**Polychromatic (pink beam) bias.** BM18 runs filtered white beam; "74" and "110 keV" are nominal effective energies of broad spectra (filtration differs between the scans and is inconsistently recorded — W7). Effects: (i) predicted ratios uncertain at the ±10% level; (ii) beam hardening through the scroll → the observed +2% radial and (with source drift) +4% z trends; (iii) the Pb contrast is diluted — the part of the "74 keV" spectrum above 88 keV sees post-edge attenuation — expected Pb ratio under pink beams ≈ 0.7–0.95, but it cannot rise above the papyrus value for any plausible spectrum, so the sign of B1 is robust; (iv) observed papyrus 1.248 vs Compton 1.13 is jointly explained by offset miscalibration (B3), effective-energy shifts, and mineral content — do not interpret the absolute value until B3 is fixed.

**Sensitivity at the measured noise floor** (per-L3-voxel ratio σ ≈ 0.12; papyrus µ74 = 0.063, µ110 = 0.050 cm⁻¹; µPb = 33 / 50 cm⁻¹ at 74/110 keV):
- Pb: d(ratio)/d(volume fraction) ≈ −600 → 3σ single-voxel detection at **~0.06 vol% Pb**, i.e. ~0.2 µm Pb-equivalent thickness per 364 µm voxel path. A realistic Pb-ink stroke (few-µm layer, ≥10% Pb) drives the voxel ratio to ~0.7–1.0 — a >5σ single-voxel event; with N ≈ 25–50-voxel stroke aggregation, sensitivity reaches ~0.01 vol%. **Pb ink is easily detectable — in the low-ratio channel.**
- Fe: d(ratio)/dv ≈ +50 → 3σ at ~0.7 vol% Fe single-voxel; iron-based ink needs spatial aggregation.
- Carbon ink: ratio-invisible (≈ papyrus), as expected — this screen addresses only metallic/mineral inks.

---

## Verdict on the 1.705 vs 1.248 separation

Partly artifact, partly real, and — critically — **misattributed**. Roughly 0.1–0.25 of the elevation is mask-rim contamination plus numerator selection bias (honest co-bright interior value ≈ 1.5–1.6). What remains is a genuine, spatially coherent dense phase a few mm below the surface — but a ratio of ~1.5 is the mid-Z (Ca/Fe mineral) signature. It is **not** evidence of lead or any heavy-metal ink: those would appear at ratio < 1, in the tail the current screen ignores.
