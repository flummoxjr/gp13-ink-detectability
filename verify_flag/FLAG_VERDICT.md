# FLAG VERDICT — PHerc1447 / `z_dbg_gen_00166_inp_hr`

Date: 2026-08-17. Analyst: verification skeptic. Flag under test: the single
segment out of 80 that cleared the corpus periodicity screen
(`trackD\analyze_survey_corpus.py` → `out\survey\corpus_analysis.json`) with
**ruling_z = +5.94 at period 7.26 mm**, θ = 15°.

Scripts and outputs: `trackD\verify_flag\vf_*.py` / `vf_*.json`, figure
`trackD\verify_flag\flag_verification.png`.

---

## VERDICT: **REFUTED**

> The corpus screen's own null, run properly, does not support the flag. With
> 400 block permutations instead of 16 the score is **z = +2.43 (95% CI
> +2.03…+3.00), empirical p = 0.037**; at full ds4 resolution **z = +3.88 (CI
> +3.30…+4.65), p = 0.012**; and through the **validated battery protocol**
> that scored the human-verified control at z = +25.6…+34.1, it is
> **z = +0.54 (CI +0.34…+0.86), empirical p = 0.18 — 71 of 400 shuffles of this
> same map beat it.** Across 69–80 screened segments, the probability that the
> corpus maximum reaches +5.94 from the screen's own 16-permutation noise is
> **0.51**, and the expected number of segments at z ≥ 5 is **1.07** — exactly
> the one that was observed. The map also fails all three non-periodicity
> battery tests, most decisively fwd/rev r = 0.753 (control letters: 0.055).

**What specifically produced 5.94** — three multiplicative errors, in order of
size:

1. **The 16-draw null sd was 2.2× too small.** True sd (400 draws) = 8.43, not
   3.92. The null prominence distribution is heavy-tailed and its *maximum*
   (66.6) exceeds the observed value (35.4). On its own data the flag is +2.43.
2. **The screen scored the segment's inference-patch geometry, not its
   texture.** The segment is three disjoint 128-px-patch regions with different
   mean response, ringed by the model's boundary halo. Subtracting three
   constants (one per patch) removes 56% of the score; eroding 40 full-res px
   of patch rim drops the prominence from 47.3 to 13.3, at the null mean.
3. **No detrending + a 14-bin search band.** The θ=15° profile is 838 px long,
   so the 1.7–8.4 mm band contains only 14 integer Fourier bins, and the
   "period" is **k = 4 — the lowest bin in the band.** A four-cycle step
   function scores as "ruling".

---

## Check 1 — is z = 5.94 real or a small-null artifact?

The screen was reproduced exactly first (prom 35.42, θ=15°, 7.26 mm, null
12.12 ± 3.92 over 16 draws, z = +5.94), then re-run with 400 permutations.
`vf_perm.py` → `vf_perm.json`, `vf_validated.py` → `vf_validated.json`.

| procedure | obs prom | null mean ± sd | null max | **z** | 95% CI on z | **empirical p** |
|---|---|---|---|---|---|---|
| screen as run (16 perms, ds8) | 35.42 | 12.12 ± 3.92 | — | **+5.94** | — | — |
| same geometry, **400 perms** | 35.42 | 14.88 ± 8.43 | 66.64 | **+2.43** | +2.03 … +3.00 | **0.0374** (14/400) |
| **full ds4**, no extra downsample, 400 perms | 47.27 | 14.91 ± 8.34 | 62.87 | **+3.88** | +3.30 … +4.65 | **0.0125** (4/400) |
| **validated battery protocol**, 400 perms | 24.68 | 19.46 ± 9.69 | 130.95 | **+0.54** | +0.34 … +0.86 | **0.180** (71/400) |
| — control w035, same validated protocol | 124.3 † | — | — | **+25.6 … +34.1** ‡ | — | — |
| — PHerc1203, same validated protocol | — | — | — | −2.7 … +1.0 ‡ | — | — |

† recomputed here on the ds4 control (θ = 177°, 4.68 mm) — reproduces the prior
run's 4.68–4.73 mm lock to the digit. ‡ quoted from
`salvage\ink9um_1203_verdict.md`; those z values used 5 permutations, so they
are optimistic in the same direction as the screen's — and the control clears
by an order of magnitude anyway.

The empirical p is the honest statistic and it is 0.012–0.18 depending on
protocol — nowhere near a discovery even before multiplicity. The last row is
the one that matters: the validated protocol differs from the corpus screen
only in the four things the screen dropped for speed (40-px rim erosion,
σ=90 detrending, 3° instead of 15° orientation grid, joint map+mask
permutation), and under it the flag is **statistically invisible**.

Note the null's maximum: shuffled versions of this very map reach prominence
66.6 (screen protocol) and 131.0 (validated protocol), both far above the
observed. The statistic has a long right tail that 16 draws cannot see.

## Check 2 — multiple comparisons

`vf_perm.py` re-ran the 400-permutation procedure at full ds4 on **10 other
segments** (9 drawn at random from the 69 scored, plus the two next-highest
screen hits). Every one collapses:

| segment | scroll | screen z (16 perms) | **corrected z (400 perms, ds4)** | empirical p |
|---|---|---|---|---|
| `z_dbg_gen_00166_inp_hr` (**the flag**) | 1447 | **+5.94** | **+3.88** | 0.012 |
| `z_dbg_gen_00260` | 1447 | +3.52 | +2.46 | 0.022 |
| `z_dbg_gen_00320` | 1447 | +3.36 | +0.29 | 0.284 |
| `z_dbg_gen_00316_inp_hr` | 1447 | +1.26 | +0.71 | 0.172 |
| `20250703034159-auto_grown_…599` | 1447 | +1.26 | +0.70 | 0.140 |
| `z_dbg_gen_00169_inp_hr` | 1447 | — | −0.21 | 0.481 |
| `20251028220955-auto_grown_…262` | 0800 | — | −0.23 | 0.466 |
| `auto_grown_20250703025628283` | 1447 | — | −0.43 | 0.599 |
| `20250702235910-auto_grown_…292` | 1447 | — | −0.46 | 0.638 |
| `auto_grown_20250502164121265` | 1447 | — | −0.50 | 0.658 |
| `20250502185519-auto_grown_…733` | 1447 | — | −0.70 | 0.751 |

From those 10 segments' 4,000 null prominences, two null distributions of the
z statistic were built: a leave-one-out z (the honest large-N estimator) and a
resampled **16-permutation estimator — the one the screen actually used**.

| null z distribution | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| leave-one-out (large-N estimator), n = 4,000 | −0.24 | 1.13 | 1.90 | 3.91 | 8.12 | 10.55 |
| **screen's 16-perm estimator**, n = 400,000 | −0.26 | 1.56 | 2.60 | **5.64** | 12.26 | 42.10 |

**Familywise expectation under the screen's own null:**

| quantity | 69 tests | 80 tests |
|---|---|---|
| **P(max z ≥ 5.94)** | **0.456** | **0.506** |
| expected max z | 4.85 | 5.14 |
| P(max z ≥ 3.88), corrected-z value | 0.809 | 0.853 |
| P(max z ≥ 2.43), ds8 corrected value | 0.980 | 0.990 |
| **E[# segments with z ≥ 5]** | 0.92 | **1.07 — observed: 1** |

A corpus maximum of +5.94 across 80 segments is a **coin flip**. The screen
produced exactly the number of z ≥ 5 hits its own permutation noise predicts.
Even the corrected z = 3.88 has P(max of 80) = 0.57 under the large-N null.

## Check 3 — is 7.26 mm physically sensible? No: it is not a period at all.

`vf_look.py`, `vf_tiles.py`, `vf_mesh.py`, `vf_targeted.py`.

**It fails every test of being periodic.**

- **Autocorrelation.** A ruling comb repeats: ρ(1P), ρ(2P), ρ(3P) > 0. Here
  ρ(1P) = **−0.269**, ρ(2P) = +0.013, ρ(3P) = −0.025. The control at 4.68 mm
  oscillates strongly and positively (figure panel G).
- **Only 4 cycles exist.** Profile length 838 ds4 px = 29.0 mm; 29.0 / 7.24 =
  **exactly 4.00 cycles**. The 1.7–8.4 mm band spans Fourier bins
  k = 3.45…17.04, i.e. 14 integer bins, and the "peak" is **k = 4, the lowest
  bin in the band** — the band-edge/low-frequency leakage failure mode the
  PHerc1203 verdict already flagged ("the 'best' periods pile up at the band
  edge… exactly what the nulls produce"). Corpus-wide this is systematic:
  all 5 segments with screen z ≥ 3 have periods > 6 mm, and the null draws'
  own best periods sit at p50 = 5.45 mm with 34% above 6 mm.
- **It is not stable.** First half of the profile: prom 5.25. Second half:
  8.40. (Whole: 47.27.) Restricting to well-covered rows moves the "period" to
  5.01 / 5.43 / 7.91 mm. Eroding the mask moves it to 6.76 mm (10 px), then
  3.55 mm at θ=60° (20 px), then 3.23 mm at θ=60° (40 px).

**Where the structure actually comes from — mesh and tiling geometry:**

- Mesh fetched anonymously from `s3://vesuvius-challenge-open-data/PHerc1447/segments/raw/z_dbg_gen_00166_inp_hr/`
  (tifxyz, `scale` 0.05). Grid **216 × 186**, mesh step 19.9 / 20.3 voxels =
  **0.172 / 0.175 mm** — far below the search band, so the mesh lattice itself
  is not the flagged frequency. Implied render canvas 4320 × 3720, matching the
  survey's 4319 × 3719.
- The `_inp_hr` inpainting fills the grid, but **30.0% of the grid points are
  degenerate** (x = y = z ≤ 0, outside the meta.json bbox). Those parts of the
  surface never land in the scanned volume.
- The prediction covers only **44.6%** of the canvas, and it is a strict subset
  of the valid mesh support. **Every mask edge lies exactly on a 64-full-res-px
  lattice — gcd(edge coordinates) = 64 px for both rows and columns, residual
  0.0.** That is the ink_9um inference stride: the checkpoint's
  `patch_size [17,128,128]` (`trackD\runpod\ink9um_runbook.md:18`) with
  `DEFAULT_OVERLAP = 0.5`
  (`villa\vesuvius\src\vesuvius\ink_detection\inference\infer.py:50`,
  `resolve_patch_stride`) → stride 128 × (1 − 0.5) = **64 px**; and patches
  whose raw input is empty are dropped outright (`run_block_inference`, the
  `metadata[:, 4] > 0` occupancy gate), leaving those canvas regions at zero.
- Result: the segment is **3 disjoint connected regions** with means 93.0, 97.7
  and 117.9 DN, plus a model boundary response along every patch outline.
- The tile pitch itself (64 px = 0.553 mm; patch 128 px = 1.106 mm) is *below*
  the 1.7 mm band floor, so the flagged frequency is not the tile pitch. What
  the tiling does is carve the map into a few large blocks whose mean-response
  steps and edge halos make a 4-cycle undulation.

**The boundary halo, quantified** (`vf_rim.json`, `vf_rim2.json`; distance
transform from the patch outline):

| distance from patch boundary | mean DN | frac > 195 DN |
|---|---|---|
| 0 – 0.69 mm | 76 | ~0.001 |
| **0.69 – 1.73 mm** | **119** | **0.034** |
| > 1.73 mm (deep interior) | 99 | 0.0007 |

**46.5% of the valid area is within 1.73 mm of a patch boundary, and 90.8% of
every pixel above the control's blank-papyrus p99 lives in the 0.69–1.73 mm
halo ring** (25.6% of the area). The halo is present identically in the reverse
render (mean 111.3 rev vs 107.7 fwd) — it is geometry, not ink.

**As line spacing.** 7.26 mm is 1.55× the control's ruling (4.68–4.73 mm on
PHerc0139, label-derived ground truth 4.59 mm). Not impossible a priori for a
different scroll and hand, but irrelevant here: there is no period to interpret.

**Targeted nulls** (`vf_targeted.json`) confirm the decomposition:

- Patch-demeaning (three constants, one per connected region) removes **56%**
  of the prominence: 47.27 → 20.77.
- Against a **within-patch** null (blocks shuffled only inside each region, so
  the patch-mean steps survive), z = +4.24, p = 0.0100 — the residue is the
  halo ring, which the 40-px erosion then removes (47.3 → 13.3).

## Check 4 — does it look like text? No, on all three remaining tests.

`vf_battery.py` → `vf_battery.json`. Control = PHerc0139 w035, same checkpoint
(`ink_9um/hybrid_3d2d-seed42/step-075000`), block-mean ds4 for scale matching;
all numbers in physical units.

**Stroke morphology (p80 threshold, in-mask, 40-px eroded):**

| population | n comp | area p50 | width p50 |
|---|---|---|---|
| control **LETTERS** | 13 | **3.73 mm²** | **0.621 mm** |
| control blank papyrus | 604 | 0.302 mm² | 0.359 mm |
| **THE FLAG** | 179 | **0.119 mm²** | **0.216 mm** |

KS(flag vs LETTERS): **D = 0.899 on area, 0.922 on width** — complete
separation; the flag has essentially no components in the letter regime
(31× smaller in area, 2.9× thinner). KS(flag vs blank papyrus): D = 0.229
(p = 7.3e-7) on area, 0.377 on width — same family, slightly finer-grained.
At p60 the blank-papyrus match is near-perfect (D = 0.111, p = 0.20) while
separation from letters holds (D = 0.798 / 0.908).

**Map-scale fwd/rev z-symmetry:** flag **r = 0.7534** (survey record 0.7538);
control letters **r = 0.0552**; PHerc1203 texture 0.51–0.60. Ink sits on one
face of the sheet and is destroyed by reversing the render. This segment is
*more* z-symmetric than the PHerc1203 texture that the battery already rejected.
This test alone refutes the flag.

**Intensity:** flag p50/p90/p99 = 91/145/194, **frac > 195 DN = 0.0067**;
control letters frac > 195 = **0.694** (103× higher); control blank p50 = 68,
p99 = 195. Wasserstein-1: flag → blank papyrus **18.7**, flag → letters **86.5**.
No letter-strength population exists, and what little there is sits in the
patch-boundary halo (above). Tripwire hits: 0.

**Figure:** `trackD\verify_flag\flag_verification.png` — the map with its three
patch means, the 64-px-lattice mask, the control's letters and the flag at
matched mm scale (panels C/D are the visual verdict: Greek letterforms vs
amorphous fibre texture), the θ=15° profile with the claimed period marked, the
14-bin band spectrum, the autocorrelation vs the control's, both null
distributions, the familywise max-z null, and the battery summary.

---

## Residual ambiguity

None material. The largest honest caveat is that the corrected-p at full ds4
(0.012) is nominally below 0.05 — but that is (a) one of 69–80 tests, where
P(max of 80 ≥ 3.88) = 0.57–0.85, and (b) fully explained by patch geometry,
since the same map scores p = 0.18 once the rim is eroded and the profile
detrended. The +2.46 residue on `z_dbg_gen_00260` is the same family
(period 7.88 mm, i.e. the band edge again) and needs no separate treatment.

## Recommendations for the corpus screen

The screen is not wrong in design, it is under-powered and under-specified. Three
one-line fixes, in order of impact:

1. **Report empirical p, not z, and raise N_PERM to ≥ 200.** With 16 draws the
   sd of a heavy-tailed statistic is unusable; z ≥ 5 does not mean p ≤ 1e-6, it
   means p ≈ 0.01, and one such hit per 80 segments is the *expected* outcome.
2. **Adopt the validated battery's preprocessing**: erode 40 full-res px
   (removes the inference-patch halo, which is where 91% of hot pixels live)
   and detrend at σ = 90. Both were dropped in `analyze_survey_corpus.py`.
3. **Require the peak to be a period**: ≥ 6 cycles inside the profile, positive
   autocorrelation at 2P and 3P, and reject the band's lowest 1–2 Fourier bins.
   All four of the screen's z ≥ 3 hits die on the first of these.

A fourth, free filter: **fwd/rev r < 0.2 at map scale.** Every one of the top
five screen hits has r = 0.37–0.83; the control has 0.055.

## Files

All in `C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag\`:

- `vf_common.py` — shared scorer (faithful reimplementation of the screen's
  statistic with switches for downsample / detrend / erosion)
- `vf_diag.py` / `.json` — exact reproduction of z = +5.94; ds4 vs ds8
- `vf_geom.py` / `.json` — 2D wavevector, reverse-map check, mask-support spectrum
- `vf_look.py` / `.json` / `vf_look.png` — profile dissection, erosion sweep,
  half-split, autocorrelation
- `vf_perm.py` / `.json` / `.log` — **check 1 + 2**: 400-permutation flag test and
  the 10-segment familywise null
- `vf_family.json` — familywise probabilities
- `vf_validated.py` / `.json` / `.log` — the flag through the full validated protocol
- `vf_targeted.py` / `.json` — within-patch null, patch-demeaning, band-bin count
- `vf_tiles.py` / `.json` — 64-px lattice fit, connected components, patch means
- `vf_mesh.py` / `.json` + `mesh\` — tifxyz from S3, grid pitch, degenerate fraction
- `vf_battery.py` / `.json` — **check 4**: morphology / symmetry / intensity vs control
- `vf_rim.py` / `vf_rim.json` / `vf_rim2.json` — boundary-halo quantification
- `vf_nulls_save.py` / `vf_nulls.npz` — raw null prominence arrays
- `vf_figure.py` / **`flag_verification.png`** — the figure
