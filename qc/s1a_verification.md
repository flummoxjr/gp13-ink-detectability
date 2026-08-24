# S1a adversarial verification — PHerc1667 w032 letter-contrast claim

**Verdict: REFUTED.** The headline pooled AUC of 0.651 is an artifact of a
background definition that — provably, in every tile — contained **zero**
on-prediction pixels. With a corrected on-papyrus background the pooled AUC at
the surface is **0.427** (letters slightly *darker*, i.e. inverted), and no
depth offset exceeds 0.51. The best tile (7,5) drops from 0.917 to **0.646**
(k=-2), which lies inside that tile's own placebo-null range (0.43–0.70,
z = 1.4). Nothing in this experiment survives contact with a proper control.

Verification scripts: `trackD/qc/qc_s1a_extract.py`, `qc_s1a_analyze.py`,
`qc_s1a_overlay.py`. Full numbers: `trackD/qc/qc_s1a_results.json`.
Figures: `qc_s1a_overlay.png` (T6), `qc_s1a_fullres_ink.png`,
`qc_s1a_overview.png`. Extracted per-tile data: `D:\vesuvius-data\trackD\w032\qc\`.

## Reproduction (T0)

The QC pipeline re-fetched the volume and reproduced the original stats
**exactly** (pooled 0.651 at k=0; all per-tile AUCs and onpap fractions
identical). So the refutation below is not a pipeline discrepancy — it is the
original experiment measuring the wrong thing.

## The fatal flaw: the background never contained a single real prediction

The ink TIF's nonzero values have a hard floor at ~28 (P1 of nonzero = 30;
there are *no* pixels with value 1–19 anywhere near the tiles). The original
background `ink < 20` therefore consisted of **100.0% ink == 0 no-data pixels
in all six tiles** (verified per tile). Value 0 means "no prediction /
off-render", not "model says no ink".

Worse, the tile-selection criterion `bg_frac >= 0.30` (fraction of ink<20)
actually demanded ≥30% *no-data* per tile, which forced all six tiles onto the
prediction boundary along the big central tear of the segment (see
`qc_s1a_overview.png`). At that boundary the tifxyz surface passes through
junk: the ink==0 zones coincide precisely with the saturated-white regions of
the 7.91µm renders (shape-for-shape match, `qc_s1a_overlay.png`), and the
surviving ink==0 background pixels (after the raw0>5 gate) sample the dim seam
between the papyrus and the white zone (tile raw0 medians 38–65 uint8). The
original 0.651 measured "textured papyrus vs off-sheet seam", not "ink vs
papyrus".

## Confound tests (pooled AUC, letters vs background)

Corrected background = ink in [25,60] (predicted-but-low; the requested [3,20]
band is empty — see floor above), within 60 px of letters, ≥8 px from any
ink==0 zone.

| Test | k=-2 | k=-1 | k=0 | k=+1 |
|---|---|---|---|---|
| T0 original (bg = 100% no-data) | 0.610 | 0.643 | **0.651** | 0.639 |
| T1 corrected background | 0.483 | 0.454 | **0.427** | 0.411 |
| T1b + saturation-excluded (raw≥240, ±8px, both classes) | 0.485 | 0.456 | 0.427 | 0.410 |
| T1b' same with raw≥200 | 0.483 | 0.456 | 0.427 | 0.411 |
| T1c + per-tile z-scored pooling | 0.497 | 0.474 | 0.445 | 0.429 |
| T4 letters eroded ×3 | 0.493 | 0.463 | 0.432 | 0.414 |
| T3 ink11 letters (only tile 9,6 has coverage) | 0.526 | 0.513 | 0.477 | 0.444 |
| T3 ink24∩ink11 letters | 0.540 | 0.519 | 0.473 | 0.433 |

- **T1 (no-data contamination): the AUC does not merely shrink — it inverts.**
  0.651 → 0.427 at k=0. The contamination *created* the entire positive
  signal. Per-tile corrected values at k=-1: (3,5) 0.326, (4,5) 0.315,
  (6,5) 0.521, (7,5) 0.622, (8,5) 0.515, (9,6) 0.547 — sign flips across
  tiles, the signature of regional brightness gradients, not ink.
- **T2 (placebo):** with corrected masks, a 180°-rotated letter mask gives AUC
  0.401 (k=0) / 0.465 (k=-1); +64 px shifts give 0.449/0.475. Twelve random
  shifts give a null of mean 0.489, sd 0.063, range 0.358–0.580. The real
  corrected AUC (0.427) is ~1 sd from the null mean — indistinguishable from a
  randomly placed mask. For the best tile (7,5) alone: real 0.646 at k=-2 vs
  tile-null mean 0.572, sd 0.053, range 0.432–0.704 (z = 1.4). That tile has a
  regional gradient that rewards *any* mask placed where its letters are.
- **T3 (independent mask): impossible in 5 of 6 tiles.** The 1.129µm-model TIF
  has **zero prediction coverage** in tiles (3,5), (4,5), (6,5), (7,5), (8,5)
  (its coverage hole is exactly the analyzed column). In the one half-covered
  tile (9,6), the two models' letter masks agree poorly (dice 0.17; only 13%
  of ink24 letters on ink11-covered area are confirmed at >170), and the
  intersection-mask AUC is ~0.5 at the surface. Global registration of the two
  TIF frames was verified (best global shift ≈ -36,+32 ds4 px, small at tile
  scale), so this is genuine non-coverage/disagreement, not misalignment.
- **T4 (stroke cores):** eroding the letter mask changes nothing
  (0.427 → 0.432 at k=0). No core-concentrated signal.
- **T5 (saturation proximity):** the higher-AUC tiles are those whose letters
  sit closer to the saturated zones (Spearman rho = 0.58 between
  frac-letters-within-16px-of-saturation and corrected per-tile AUC; p = 0.23,
  n = 6 — the "bad sign" direction, though underpowered and moot after T1).
- **T6 (visual):** `qc_s1a_overlay.png`. No letterforms are traceable in the
  7.91µm renders under the letter contours — the contours sit on generic,
  heavily streaked papyrus texture. The renders' saturated-white zones
  duplicate the ink TIF's no-data zones outline-for-outline. The ink11 panels
  are solid black (no predictions).
- **T7 (depth profile):** the corrected profile has **no surface peak**: it is
  ~0.50 at k=-4..-3 and declines monotonically to 0.41 at k=+1..+2. That is
  the profile of a systematic geometry/brightness difference between letter
  and background positions, not of a surface ink layer (original claimed peak
  at k=-1..0 was inherited from the contaminated background).
- **Supplementary:** Spearman correlation between ink-TIF value and 7.91µm
  brightness at k=0 within the clean on-prediction domain is **negative in all
  six tiles** (-0.01 to -0.41). There is no monotone letters-brighter relation.

## The premise itself is broken

The tested coordinates are *not* "where the June-2026 read proves letters
exist". At full 2.4µm resolution (`qc_s1a_fullres_ink.png`) the "letters" in
these tiles are amorphous ~0.5–1 mm blobs with visible model-window
checkerboard artifacts — no glyph shapes — sitting on the edge of the
segment's central tear, in a region the 1.129µm model does not cover at all.
The segment does contain organized text lines elsewhere (overview figure), but
the selection criteria steered the experiment away from them and onto the
pathological boundary strip (candidate tiles off this strip were rejected by
the onpap<0.7 gate, which is itself a red flag: 30–60% of the mapped surface
was off-papyrus everywhere nearby).

## Code review (label-leak audit)

- `WEAK_BG = 20` is below the model's output floor (~28) → background = pure
  no-data. **This is the bug that produced the result.**
- `bg_frac >= 0.30` tile selection requires ≥30% no-data → selects tear-edge
  tiles (selection bias toward the artifact).
- Normals come only from grid derivatives — no ink-dependent quantity leaks
  into the feature. (Checked: the grid is fully valid across all six tiles, so
  the invalid-neighbor `np.gradient` contamination risk does not apply here.)
- `raw0 > 5` gate censors the dark tail of both classes equally — biases AUC
  toward 0.5, did not create the signal.
- Pooled AUC concatenates raw values across tiles, so between-tile brightness
  offsets leak in; per-tile and z-scored pooling (T1c) are the honest
  statistics (they agree with T1 here).
- Window mapping is linear/monotonic — no AUC effect. `tie_auc` subsampling is
  seeded and fine; reproduction was exact.

## Three most important caveats on this refutation

1. **This refutes the evidence, not the thesis.** The experiment says nothing
   either way about whether GP-resolution (8–9µm) ink signal exists — it was
   never actually tested here, because the tiles contain no verified letters
   and the control was broken. A redo on tiles with real letterforms (e.g.
   rows 8–15, cols 8–14, where both models predict and the read presumably
   lives), with on-prediction background from the start, is the correct next
   step.
2. The corrected "background" had to be redefined as the [25,60] prediction
   band (the specified [3,20] band does not exist). If the model's low outputs
   are themselves brightness-correlated for artifactual reasons, the corrected
   AUC could be biased either way — but the placebo null brackets the result
   regardless, and the slight inversion (letters darker) is also consistent
   with the negative ink-vs-brightness correlation.
3. Small-n effects: 6 tiles, spatially autocorrelated pixels, and placebo
   nulls estimated from 12–20 shifts. None of the corrected effects approach
   even these loose significance bars, but the T5 proximity correlation
   (n = 6) should not be over-interpreted in either direction.

**Bottom line for the prize effort:** do not cite S1a. The 0.651/0.917 numbers
measure the contrast between papyrus and the segmentation's failure zones. If
this had gone into a submission narrative as "ink signal exists at GP
resolution", an informed reviewer would have found the 100%-no-data background
in an afternoon.
