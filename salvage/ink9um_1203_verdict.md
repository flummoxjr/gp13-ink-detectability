# Verdict — ink_9um on PHerc1203 (Track D probe)

Date: 2026-08-17. Analyst: verdict subagent. Inputs: 8 PHerc1203 prediction maps
(2 auto-grown segments x 2 seeds x 2 z-directions, `out\ink9um_1203\`) judged
against the validated positive control `out\ink9um_w035\` (PHerc0139 segment
w035, letters proven, pixel AUC 0.964 vs human labels recomputed here).
All scripts `salvage\verdict_*.py`; figure `salvage\verdict_gallery.png`.

## VERDICT (read this first): STROKE-SCALE TEXTURE ONLY — no text-like organization on 1203

The model is healthy and the test battery is sharp: on the control it finds the
text ruling at 4.68 mm with z = 26–34 against a texture-preserving null, its
letter strokes are 0.55 mm wide and saturate the output scale. On PHerc1203,
**every one of the 8 maps fails every text signature simultaneously**:

1. **No ruling periodicity.** Best spectral peak z = −2.7 to +1.0 vs null
   (control: +25.6 to +34.1). Detected "best" orientations wander 14°–143.5°
   across seeds/z-directions of the same segment (control: stable to 4.5°).
2. **Morphology is the control's blank-papyrus response, not letters.**
   1203 components vs control letter components: KS D = 0.92–0.97 on area and
   width (p ≤ 1e-13). Same components vs control *off-letter* (blank) components:
   D = 0.06–0.23 — statistically near-identical. 1203 filaments are ~7–10x
   thinner and ~40–100x smaller than letter strokes.
3. **No letter-strength intensity population.** Only 0.06–0.11% of 1203 pixels
   exceed the control's blank-papyrus p99; 69.4% of true letter pixels do.
   Wasserstein distance of 1203's value distribution: 10.5–12.8 to blank
   papyrus, 93.6–95.6 to letters.
4. **The response is z-direction symmetric.** Control letters live on one
   surface: reversing the z-rendering destroys the map (r = 0.076). 1203's
   fwd-vs-reverse correlation (0.51–0.60) equals its seed-vs-seed correlation
   — the signature of volumetric fiber/texture response, not surface ink.

This is a *different* failure mode from the round-1 ink_3d blanket
(`qc_live\round1_verdict.md`): ink_9um is not blanket-painting — it produces a
structured, seed-consistent (r 0.51–0.60), quiet-region-bearing response and
proves it can draw letters on the control. On 1203 it outputs its
blank-papyrus/fiber-texture response everywhere. The two rendered segments
(~15.7 + ~12.8 cm^2) contain, per this model, no ink organization at all.

---

## Test 1 — line-ruling periodicity (verdict_periodicity.py → verdict_periodicity.json)

Method: block-mean ds4 (37.4 µm/px), valid-mask eroded 40 px (kills the bright
rim on 1203 patch borders). Orientation search θ ∈ [0°,180°) (2° coarse, 0.5°
fine): rotate map+mask, masked row-mean projection, detrend (Gaussian σ=90 ds-px),
Hann window, power spectrum. Score = max band power / band median ("prom") in
the ruling band 1.7–8.4 mm. Band calibrated from the human labels
(verdict_labelgeom.json): letter height p50 = 242 px (2.26 mm), row spacing from
glyph centroids ≈ 490 px (4.59 mm), stroke width 68 px (0.64 mm). Null = 5x
joint block-permutation of (map, mask) in 256-px tiles — preserves stroke-scale
texture, destroys long-range row alignment — through the identical search.

| map | θ best | peak period | prom | null mean±sd | **z** | quiet frac | row/col aniso |
|---|---|---|---|---|---|---|---|
| w035 seed42 | 176.5° | 500 px / 4.68 mm | 143.6 | 17.7±4.9 | **+25.6** | 0.31 | 1.75 |
| w035 seed43 | 180.5° | 505 px / 4.73 mm | 102.6 | 17.1±3.2 | **+26.5** | 0.25 | 2.20 |
| w035 rendered s42 | 176.0° | 500 px / 4.68 mm | 113.3 | 18.9±2.8 | **+34.1** | 0.29 | 1.47 |
| 1203A s42 | 14.0° | 6.85 mm | 19.6 | 30.7±11.6 | −1.0 | 0.30 | 1.10 |
| 1203A s42 rev | 79.5° | 3.33 mm | 29.9 | 29.5±9.9 | 0.0 | 0.28 | 0.97 |
| 1203A s43 | 34.5° | 6.92 mm | 19.2 | 29.2±8.9 | −1.1 | 0.29 | 0.92 |
| 1203A s43 rev | 137.5° | 6.00 mm | 22.3 | 39.4±25.4 | −0.7 | 0.32 | 1.34 |
| 1203B s42 | 16.0° | 7.74 mm | 34.4 | 27.3±7.1 | +1.0 | 0.23 | 1.19 |
| 1203B s42 rev | 143.5° | 7.42 mm | 31.3 | 29.9±10.0 | +0.1 | 0.18 | 1.13 |
| 1203B s43 | 18.0° | 7.78 mm | 36.8 | 33.7±10.3 | +0.3 | 0.19 | 0.93 |
| 1203B s43 rev | 78.5° | 7.75 mm | 26.9 | 36.4±3.6 | −2.7 | 0.35 | 0.95 |

- **Control positives:** all three variants lock onto the same ruling —
  4.68–4.73 mm at θ = 176–180.5° — matching the label-derived ground truth
  (4.59 mm) within 3%. Prominence is 6–8x the null and z ≥ 25.
- **1203 negatives:** no map beats its own shuffled-texture null (max z = 1.0;
  median ≈ 0). The "best" periods pile up at the band edge (7.4–7.8 mm on
  segment B) — residual 1/f leakage, exactly what the nulls produce. Orientation
  is incoherent between seeds and z-directions of the same physical segment;
  the control's is reproducible to <5°.
- Footnote: the sparse label map itself (12 glyphs, 1.3% fill) is underpowered
  for this test (z = 1.3) — ground-truth spacing was therefore taken from glyph
  centroids; the *prediction* maps carry the full-row signal (the model fires on
  unlabeled rows too, which is what makes the control such a strong template).

## Test 2 — stroke morphology (verdict_morph.py → verdict_morph.json)

Threshold each map at its own in-mask p60/p80; 8-connected components ≥20 px;
per-component area, elongation (major/minor), width (area/skeleton length),
wiggliness (skeleton length/major axis). Control components split by ≥50%
overlap with the dilated human labels. p80 numbers below (p60 agrees).

| population | n comp | area p50 (px) | width p50 (px / mm) | elong p50 | wiggle p50 |
|---|---|---|---|---|---|
| w035 LETTER (s42/s43) | 13 / 10 | 42,363 / 43,264 | 57.9 / 59.8 ≈ **0.55 mm** | 2.39 / 2.26 | 1.71 / 1.79 |
| w035 off-letter | 960 / 731 | 911 / 1,774 | 8.5 / 13.3 | 1.68 / 1.69 | 2.03 / 2.10 |
| 1203 (8 maps, range) | 462–722 | 365–1,032 | 5.0–9.4 ≈ **0.05–0.09 mm** | 1.52–1.77 | 1.71–2.34 |

KS two-sample tests (p80):

- **Control letters vs 1203:** D(log area) = 0.92–0.96, D(width) = 0.92–0.97,
  all p ≤ 1e-13. Complete separation: 1203 has essentially no components in the
  letter regime (multi-mm, 0.5-mm-wide, high-solidity strokes).
- **Control off-letter (blank response) vs 1203:** D(area) = 0.06–0.17,
  D(width) = 0.08–0.23, D(elong) = 0.04–0.19. The 1203 component population is
  the same family as the control's blank-papyrus false-positive texture —
  slightly wormier (wiggle p50 up to 2.34) and thinner, i.e. fiber-like
  filaments, not stroke-like marks.

## Test 3 — spatial organization + model consistency (verdict_periodicity.json, verdict_consistency.py → verdict_consistency.json)

- **Quiet bands:** quiet-row fraction alone does not discriminate (control
  0.25–0.31, 1203 0.18–0.35) — what discriminates is that the control's quiet
  bands are *oriented, periodic interline gaps*: row/col anisotropy 1.47–2.20
  and stable orientation, vs 1203's 0.92–1.34 (isotropic patchiness, no
  reproducible axis).
- **Seed-to-seed pixel correlation** (in-mask, full res):

| pair | control w035 | 1203A | 1203B |
|---|---|---|---|
| seed42 vs seed43 (fwd) | 0.668 | 0.508 | 0.598 |
| seed42 vs seed43 (rev) | 0.413 | 0.543 | 0.599 |
| **fwd vs reverse (s42)** | **0.076** | **0.506** | **0.598** |

  The model is moderately seed-consistent everywhere (r ≈ 0.5–0.67) — 1203's
  output is a real, reproducible response to the volume, not noise. But the
  *z-direction* contrast is diagnostic: on the control, reversing the rendering
  z-order destroys the map (r = 0.076 — ink sits on one surface); on 1203 the
  reverse maps are as correlated as different seeds (r = 0.51–0.60) — the
  response is symmetric through the sheet, as texture is and ink is not.
  (r at ds8 equals r at full res everywhere — the maps are smooth below ~75 µm,
  so this is regional agreement, not pixel speckle.)

## Test 4 — intensity calibration (verdict_intensity.py → verdict_intensity.json)

Reference distributions from the control (valid mask eroded 40 px): LETTER =
pixels under human labels (n = 396k); BLANK = pixels outside labels dilated
13 px. Caveat: BLANK still contains unlabeled text rows — which biases this
test *against* our conclusion, and it is decisive anyway.

| distribution | p50 | p90 | p99 | frac > blank-p99 (195) |
|---|---|---|---|---|
| w035 LETTER | 199 | 205 | 212 | **0.694** |
| w035 BLANK | 68 | 122 | 195 | 0.01 (def.) |
| w035 whole map | 68 | 128 | 200 | 0.0197 |
| 1203 (6 maps, range) | 79–83 | 132–137 | 179–183 | **0.0006–0.0011** |

Wasserstein-1 distances: every 1203 map is 10.5–12.8 from BLANK and 93.6–95.6
from LETTER. 1203 runs slightly *hotter* than control blank papyrus (median 81
vs 68 — consistent with its denser fiber texture) but its distribution has no
letter-strength component: the >195 tail (0.06–0.11% of pixels) is 60x weaker
than the control map's letter tail and spatially scattered (it is the thin
filaments of Test 2, not 2–4 mm marks).

## Contrast with the round-1 ink_3d failure signature

Unlike ink_3d_dino_guided on 1203 2.4 µm (blanket firing, zero silent tiles,
unrankable), ink_9um degrades gracefully: it keeps quiet interline-scale
regions, stays seed-consistent, and demonstrably draws letters on a scroll
segment (w035) with AUC 0.96. Its 1203 output is not garbage — it is a
well-behaved "no ink found, here is the fiber texture" response. That makes
this a much stronger negative for these two segments than round 1 was: the
instrument works; the reading is blank.

## What would settle residual ambiguity

The negative is strong for THESE two segments; it does not yet generalize to
the whole scroll:

1. **Coverage, not method, is now the limit.** Two auto-grown patches ≈ 28.5
   cm^2 of a large scroll; both could genuinely sit on uninscribed papyrus
   (verso, margin, interlinear-heavy region). Highest-value follow-up: render
   2–3 more segments from geometrically distinct regions (different winding
   radius / near the expected recto face) and push them through this exact
   battery — scripts are turnkey, ~5 min/segment CPU.
2. **z-offset sweep on the same segments.** If the surface localization is off
   by a sheet thickness, ink would be missed while texture survives. Re-render
   segment A at ±1, ±2 layer offsets; the Test-3 fwd/rev symmetry already hints
   the model sees mid-sheet material, so this is cheap insurance.
3. **Letter-strength tripwire for any future render:** flag any component with
   value > 195 (control blank p99), area > 10^4 px, and width > 30 px. Today's
   maps contain zero such components; a single one would justify a human look.
4. If 2–3 additional regions and a z-sweep still return this signature, bank
   the negative: at 9 µm with this checkpoint, PHerc1203's rendered surfaces
   show no detectable ink organization, with the w035 control proving the
   pipeline would have seen it at ~5 mm ruling scale.

## Files

All in `C:\Users\benbl\Desktop\Vsuvious\trackD\salvage\`:

- `verdict_common.py` — shared loaders/masks (valid mask = filled nonzero
  support, eroded; 1203 rim artifact excluded at 40 px)
- `verdict_prep.py` / `verdict_prep.json` — sanity: shapes, masks, control AUC
- `verdict_labelgeom.py` / `verdict_labelgeom.json` — ground-truth letter
  geometry from w035 labels (height 242 px, spacing 490 px, stroke 68 px)
- `verdict_periodicity.py` / `verdict_periodicity.json` + `rot_*.npz` — Test 1+3a
- `verdict_morph.py` / `verdict_morph.json` + `morph_*_p{60,80}.npz` — Test 2
- `verdict_consistency.py` / `verdict_consistency.json` — Test 3b
- `verdict_intensity.py` / `verdict_intensity.json` — Test 4
- `verdict_figure.py` / `verdict_gallery.png` — control vs 1203, rotated, ruling
  annotated, profiles + null-calibrated spectra + matched-scale crops
- `verdict_quicklook.py` / `verdict_quicklook.png` — raw side-by-side of all maps
