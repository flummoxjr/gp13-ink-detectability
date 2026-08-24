# S1a-v2 adversarial verification — PHerc1667 w032 letter-contrast, second pass

**Verdict: REFUTED — (a) pooled surface-peaked positive: REFUTED; (b) tile
(9,9) 0.902: REFUTED as ink evidence.** The v2 experiment fixed v1's literal
bug (no-data background) but reproduced the same failure one level up: the
strongest tile's "background" is again mostly **not papyrus** (72% of (9,9)'s
background pixels are depth-flat, dim, off-sheet positions at raw0 ≈ 40–50,
which sail through the `raw0 > 5` gate), the letter/background classes are
grossly non-exchangeable in geometry in every tile (distance-to-tear AUC up to
0.92, tilt AUC up to 0.78), and the "unfakeable" depth profile decomposes into
surface-fit-alignment artifacts. After restricting both classes to actually
on-sheet pixels and dropping (9,9), the pooled AUC at k=-1 is **0.5035**
(z = 0.34 against its own 40-shift null; 18/40 nulls exceed it). Nothing
submission-grade survives.

Verification scripts: `trackD/qc/qc_s1a_v2_extract.py`, `qc_s1a_v2_register.py`,
`qc_s1a_v2_nulls.py`, `qc_s1a_v2_final.py`, `qc_s1a_v2_overlay.py`.
Numbers: `trackD/qc/qc_s1a_v2_nulls.json`, `qc_s1a_v2_register.json`,
`qc_s1a_v2_final.json`, and `D:\vesuvius-data\trackD\w032\qc\qc_s1a_v2_repro.json`.
Figure: `trackD/qc/qc_s1a_v2_overlay.png`. Extracted per-tile stacks:
`D:\vesuvius-data\trackD\w032\qc\tile_v2_*.npz`.

## Reproduction (exact)

Re-extraction reproduced every reported per-tile AUC at every k to 4 decimals
(all 4 tiles, real + rot + shift nulls). The refutation below is not a pipeline
discrepancy.

## A1. Null distribution (40 random cyclic shifts per tile, |shift| ≥ 32 px)

Two variants (raw roll, exactly like the v2 fixed nulls; and "matched" = roll ∧
dist-ok ∧ on-prediction ∧ not-real-letters) agree closely; matched numbers:

| Target | real | null mean ± sd | null range | z | P(null ≥ real) |
|---|---|---|---|---|---|
| pooled k=-1 | 0.6134 | 0.5177 ± 0.0372 | [0.417, 0.594] | **2.57** | 0/40 |
| pooled k=0 | 0.6084 | 0.5199 ± 0.0397 | [0.415, 0.599] | 2.23 | 0/40 |
| (9,9) k=-1 | 0.9021 | 0.4965 ± 0.0698 | [0.417, 0.721] | **5.82** | 0/40 |
| (8,10) k=0 | 0.6848 | 0.5066 ± 0.0932 | [0.316, 0.675] | 1.91 | 0/40 |
| (8,10) k=-1 | 0.6552 | 0.5096 ± 0.1057 | [0.303, **0.732**] | 1.38 | 4/40 |
| (12,11) k=-1 | 0.6136 | 0.5084 ± 0.0630 | [0.358, **0.651**] | 1.67 | 2/40 |
| (11,12) k=-1 | 0.3657 | 0.5036 ± 0.0706 | [**0.316**, 0.672] | -1.95 | two-sided: 2/40 more extreme |

So yes: the pooled 0.614 and (9,9)'s 0.902 nominally exceed their 40-rep null
maxima. **That is the strongest thing the experiment has going for it, and it
is not enough**, for two reasons established below: (i) the null tests
location-exchangeability, but the confound is not random location — the ink
models read the same volume as the render, so their high-output blobs co-locate
with bright folds/crests by construction (circularity; trend-only AUCs of
0.70–0.86 in 3 of 4 tiles prove letters are predictable from the smooth
brightness field alone); (ii) the moment the broken ingredients are corrected
(on-sheet background, or dropping the broken tile), the pooled effect falls
*inside* the null band (§A4). Individually, (12,11) and (8,10)@k=-1 are within
their own null ranges even uncorrected; only (9,9) is far outside — and (9,9)
is the tile whose background is not papyrus.

## A2. Depth-profile specificity — "peaked at |k|≤1" is not rare under the null

Fraction of 40×4 = 160 per-tile null profiles with peak at |k| ≤ 1 AND range
> 0.08: **19/160 ≈ 12%** (per tile: 4–7/40). Allowing peak *or trough* at
|k| ≤ 1 (the reading under which the "coherently inverted" (11,12) counts as a
hit): **46/160 ≈ 29%**. Pooled null profiles: 4/40 peaked at |k| ≤ 1 with range
> 0.08. A surface-localized-looking profile is produced by randomly placed
masks about one time in eight per tile — with four tiles and post-hoc peak
picking, the claimed profile shapes carry little evidence. The reason: brightness
fields themselves vary along the normal wherever the surface fit is imperfect,
so location nulls routinely generate near-surface extrema.

## A3. Brightness/coverage confounds — the classes are not exchangeable

Per-tile letters vs background (`qc_s1a_v2_nulls.json → confounds`):

| Tile | dist-to-nodata AUC | tilt AUC | mean n_z L/B | trend-only AUC (k=0) | residual AUC (k=0) |
|---|---|---|---|---|---|
| (12,11) | **0.923** (452 vs 257 px) | 0.736 | -0.12 / +0.09 | **0.816** | 0.472 |
| (8,10) | 0.708 | 0.649 | +0.46 / +0.02 | **0.804** | 0.597 |
| (9,9) | 0.369 | **0.780** | +0.60 / +0.21 | 0.856¹ | 0.812¹ |
| (11,12) | 0.345 | 0.476 | -0.05 / -0.08 | **0.253** | 0.439 |

¹ (9,9)'s trend fit is unreliable — background is sparse near its letters; see A4.

- In (12,11) you can classify letters vs background at **AUC 0.92 from
  distance-to-tear alone**, and a smooth brightness trend fitted only on
  background pixels "predicts" letters better (0.816) than actual brightness
  does (0.574). Its residual (trend-removed) AUC at k=0 is 0.472.
- Letters sit on systematically different surface orientation in every tile
  (e.g. (9,9): mean normal (+0.35,-0.66,+0.60) vs background (+0.11,-0.86,+0.21);
  fraction n_z<0: 5% vs 31%). Letters and background live on different terrain.
- Residual profiles (letters vs local trend): (12,11) peak drifts to k=-2..-3
  at 0.59–0.60 with a 0.39 flank at +2 — not surface-peaked; (8,10) 0.597@0
  with 0.32@+4; (11,12) monotone 0.40→0.56 (inverted "peak" gone).

## A4. Tile (9,9) — the 0.902 is papyrus-vs-air plus fold-crest brightness

- **The background is mostly not on the sheet.** Median per-pixel std of the
  9 depth samples: letters 0.0217, background **0.0037** (other tiles: both
  classes 0.021–0.026). 83% of its background has raw0 in [20,60] — a flat,
  dim, structureless zone (the big dark region in the renders) where the
  surface has left the papyrus. Air in this masked volume reads ~40–50 uint8,
  so v2's `raw0 > 5` "on-papyrus" gate (onpap = 0.954) caught none of it. This
  is v1's failure mode wearing an on-prediction costume: the model outputs
  28–60 ("no ink seen") precisely where the render is off-sheet dim, so the
  background band *selects* the artifact.
- Restricting both classes to depth-structured pixels (std > 0.01) removes 73%
  of its background and drops the profile to 0.55/0.66/0.77/**0.81**/0.76/0.67/
  0.60/0.54/0.50; restricting to raw0 ≥ 60 gives peak **0.718** with both
  flanks elevated (0.58@-4, 0.56@+4) — a regional letters-brighter-everywhere
  signature. Against a struct-restricted 40-shift null the on-sheet 0.806@k=-1
  is still z = 3.18 (null max 0.747) — an on-sheet excess remains, but:
- **The local peak shape is manufactured by surface-fit misalignment.** Against
  its own on-sheet ring (background 4–48 px from letters, n = 1987):
  0.814@k=-1, falling to 0.43@+4 — looks ink-like. But the ring pixels'
  brightness argmax piles up at the +4 edge (28% at k=+4; mean ring profile
  monotone **rising** through the whole window: 0.007→0.022) — around the
  letters the surface sits *above* the sheet, while on the letters it is
  centered (letters argmax mode at k=-1). Comparing depth-alignment-invariant
  brightness (per-pixel max over k): **0.703**; aligned 3-sample mean:
  **0.740**. The "surface peak" of the local comparison is the difference in
  surface-fit centering between the two classes, not an ink layer.
- What remains — letters locally 0.70–0.74 brighter than their ring — sits on a
  fold crest (tilt AUC 0.78) whose margins are geometrically darker (tilt
  shading, partial volume with air at the fold edge). 90% of the letter mask is
  **one 5334-px blob** draped over that crest; the overlay
  (`qc_s1a_v2_overlay.png`, top row) shows no letterform, just a contour around
  the generic bright crumpled zone, while the rot-null contours land in the dark
  zone — which is why every null is low. Model-bright ∧ crest-bright ∧
  ring-dark: circular, not ink.

## A5. The inverted tile (11,12) — explainable, but as geometry, not ink

- Letters' mean brightness never reaches background's at *any* k in ±4
  (max 0.024 vs 0.033): they are in a locally dim/damaged area (trend-only AUC
  0.25). Their per-pixel brightness argmax skews deep (24% at k=+4; class mean
  profile peaks at k=+2 vs background k=0), i.e. the surface fit sits ~2 voxels
  above the sheet there. Ring comparison is monotone rising (0.39@-4 → 0.61@+4):
  a depth-offset signature, not an ink-on-the-other-side signature (an offset
  ink layer would still reach papyrus brightness at its own depth; these
  letters never do).
- Under trend removal the inverted surface peak disappears (residual profile
  monotone 0.40→0.56). Under corrected ink11 registration its background turns
  out to be 16.4% ink11 > 60 (see A6). Its "coherent inversion" is a
  surface-fit + locale artifact and does **not** corroborate the ink story;
  under the null it is itself only a z = -1.95 event (2/40 nulls more extreme).

## A6. Registration — the "dual-model confirmed" premise is broken

The v2 resample of ink11 into the ink24 frame (zoom 2.258/2.399, origin-aligned,
no shift term) is misregistered by **(dy,dx) ≈ (-36,+39) ds4 px ≈ 350 µm**:
global NCC rises 0.218 → 0.407 at the best shift; per-tile local fits agree
((-31,+44) NCC 0.85, (-40,+42) 0.69, (-32,+36) 0.84; (9,9) (-45,-8) 0.58, its
field dominated by a coverage edge). This is the same ~36-px offset I flagged
in the v1 review. Consequences (`qc_s1a_v2_register.json`):

- The offset is comparable to letter-blob size: under corrected registration
  the letter masks change with dice 0.51–0.73; **15–21% of the claimed
  dual-confirmed letter pixels in (8,10)/(9,9) lose dual confirmation**, and
  the corrected masks are 1.6–2.4× larger (the misregistration was mostly
  *deleting* legitimate confirmations and keeping accidental overlaps).
- Background contamination: 8.2% of (12,11) and 16.4% of (11,12) "no-ink"
  background is actually ink11 > 60 at the corrected position.
- Re-running the full pipeline with corrected masks moves per-tile AUCs by
  0.03–0.08 ((12,11) 0.614→0.570; (11,12) 0.366→0.439; (8,10) 0.655→0.671;
  (9,9) 0.902→0.847; pooled 0.613→0.605). No conclusion changes, but the
  published masks' identity is not robust and the "independent second model
  confirms these letters" claim is unsupported as executed.

## A7. Corrected pooled result and effective sample size

| Pooled variant | k=-1 | z vs own 40-null | nulls ≥ real |
|---|---|---|---|
| as published (4 tiles) | 0.6134 | 2.57 | 0/40 |
| excluding (9,9) | 0.5547 | 1.19 | 4/40 |
| on-sheet (struct>0.01), 4 tiles | 0.5414 | 1.35 | 3/40 |
| on-sheet, excluding (9,9) | **0.5035** | **0.34** | 18/40 |

The entire pooled positive is (9,9)'s papyrus-vs-air contrast plus weak trend
effects. Block bootstrap (64-px blocks) on the published pooled k=-1: sd =
0.030, 95% CI **[0.557, 0.667]**, vs naive iid sd 0.0025 → **ESS deflation
×146** (≈ 275 effective letter pixels across all four tiles — i.e. a handful
of independent blobs per tile; the honest unit of evidence is n = 4 tiles, of
which one is inverted).

## What would need to be true for a v3 to survive

1. Background must be proven on-sheet per pixel (depth-structure or brightness
   criterion — not `raw0 > 5`), and reported per tile.
2. Ink11 confirmation must use the measured (-36,+39) registration.
3. The comparison must be locale-matched and geometry-matched (same fold, same
   tilt band, same distance-to-tear), or explicitly regress these out; report
   trend-only and residual AUCs alongside.
4. Depth profiles must be alignment-controlled (per-pixel surface re-centering,
   or max-over-k statistics) so surface-fit accuracy differences cannot fake a
   surface peak.
5. Any positive must show letter-shaped residual structure (glyph overlay), not
   single-blob masks; and the null must include model-output-conditioned
   placebos (e.g. equally-bright non-letter blobs from elsewhere), not only
   location shifts, to break the model-reads-the-same-volume circularity.

**Bottom line for the prize effort: do not cite S1a-v2.** The pooled
0.614-peaked profile and the (9,9) 0.902 measure, respectively, one broken
tile's sheet-vs-air contrast and a fold crest's brightness under a misregistered
"confirmation" mask. The only genuinely unexplained remnant — (9,9)'s
alignment-corrected local excess of ~0.70–0.74 against 1987 ring pixels on one
crest — is exactly the kind of number that dies in a locale-matched design, and
nothing letter-shaped is visible under the contours.
