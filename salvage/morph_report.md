# Morphology analysis (H2 coplanarity, H3 component shapes) — PHerc1203 2.4µm ink screen samples

Date: 2026-08-17. Analyst: morphology subagent. Inputs: the ~60 sampled 4×-downsampled
probability maps pulled live from the round-1 fleet (ink_3d_dino_guided, Paris4-trained,
256³ L0 tiles → 64³ probL2), matched against streamed band-L2 CT and the released
surface prediction (surface-m7-L2-th0.2, native frame = band L2).

## TL;DR

- **H3 (component shapes): REFUTED as an ink signature.** The fired volume is not made
  of stroke-shaped marks. 79% of fired voxels sit in giant (≥2000 vox) connected
  components that are **thin sheet-conformal membranes/networks** — local thickness
  2.6 vox mean / 4.0 vox p90 (25–40µm), 21% of large components span the full tile.
  The model paints papyrus sheet surfaces wholesale; genuine through-sheet crack
  ribbons are rare (0.2% of fired voxels), revising round-1's "crack-follower" reading:
  it is a *sheet-surface follower*.
- **H2 (sheet coplanarity): SUPPORTED as geometry, but it does not rescue ink.**
  Component principal planes align strongly with local sheet normals (median angle
  17.5° vs 50.2°±2.0 across-tile shuffle null, 59.8°±2.2 isotropic null, p_emp < 1/200;
  within-tile shuffle null 21.2°±1.0, z ≈ −3.9). Everything the model fires at is
  sheet-conformal, everywhere, in every tile — conformality carries no discriminative
  power when the whole output has it.
- **No ink-plausible subpopulation.** A small "patch" class (flat, sheet-conformal,
  stroke-thickness) exists — 78 components, 6.3% of fired voxels — but it is the small-
  fragment tail of the same membrane mode, not a separate mode: same thickness range
  (median e3 4.2 vs membrane 2.6–4.0), same mean probability (0.76 vs 0.81), round in-plane
  footprint (aspect 1.54 median — flecks, not strokes), and present in 25/40 usable tiles
  rather than clustered in any candidate writing region. **Zero of 63 tiles are
  patch-dominant** (max patch fraction 0.33; median over non-edge tiles 0.03).

## Data and sample bias (read before trusting any rate below)

- 63 unique sampled tiles (round_1: 27 from w0/w4/w5; round_2: 54 from all six workers;
  18 duplicate pulls verified byte-identical). z-slabs 0–2816 (L0), i.e. mostly the low-z
  end of the band; w1–w3 appear only in round 2.
- **The sample is top-pmax biased and therefore EDGE-biased**: median sampled tile sits at
  the **98.8th percentile of fleet pmax** but only the **4.4th percentile of fleet f05**;
  29/63 tiles have fill < 0.2 (mask-edge fragments; fleet median fill = 1.00). 23/63 tiles
  are edge-halo-dominated in component terms. Usable interior tiles: 40 (component census)
  / 34 (sheet-position analysis). All population rates below are conditional on this
  biased draw of n≈60; treat them as a characterization of the sampled morphology, not
  unbiased scroll-wide rates. The full-fleet stats (35,148 scored tiles, median f05 0.065,
  zero silent) independently establish the blanket-firing context.

## Methods (scripts in trackD\salvage\)

- `m0_inventory.py` → `inventory.json` — dedupe/verify the two sample rounds.
- `m1_fetch.py` — streamed 64³ CT (band L2) + surface-pred (its L0 = band-L2 frame)
  subvolumes per tile from S3 (cache\ct_*.npy, cache\surf_*.npy; 126 files).
- `m2_components.py` → `components.json`, `tile_scores.json`, `coplanarity_null.json` —
  connected components of prob>0.5 (26-conn, ≥20 vox): PCA axes (elongation, planarity,
  minor-axis extent e3 = thickness), plate normal = minor PCA axis; local sheet normal =
  dominant structure-tensor eigenvector of the smoothed surface pred (CT fallback when
  surf coverage < 2%; only 83/352 components needed the fallback, and the two sources give
  the same answer: median angle 17.5° surf-based vs 16.0° CT-based). Classes: patch
  (flat: planarity ≥ 2, e3 ≤ 10; angle ≤ 30°), ribbon (flat, angle ≥ 60°), oblique
  (flat, 30–60°), filament (elongation ≥ 3, not flat), blob (rest), edge (>50% of voxels
  within 3 vox of the CT mask background).
- `m2b_deepdive.py` → `deepdive.json` — local thickness (2×EDT) of large components;
  patch-subpopulation profile.
- `m2c_position.py` → `position_vs_sheet.json` — signed distance of fired voxels to the
  sheet mask vs a volume-random null.
- `m3_gallery.py` → `morph_gallery.png`.

DISCIPLINE: every alignment/enrichment claim carries a ≥200-draw permutation or
resampling null; observed statistics are reported against the null mean±sd with
empirical p. Within-tile shuffles are included specifically because sheet normals are
spatially autocorrelated inside a tile.

## Results

### Component census (352 components ≥20 vox across 63 tiles)

| class | n | voxel share | size med | thickness e3 med | on-sheet med | angle med |
|---|---|---|---|---|---|---|
| blob (membrane networks) | 123 | **0.786** | 1367 | 11.4 (global PCA; see local thickness) | 0.73 | 20.6° |
| edge (mask-edge halos) | 117 | 0.103 | 172 | 5.7 | 0.04 | 23.9° |
| patch (flat, conformal) | 78 | 0.063 | 174 | 4.2 | 1.00 | 12.6° |
| filament | 15 | 0.040 | 297 | 7.9 | 0.85 | 15.1° |
| oblique | 12 | 0.004 | 134 | 3.9 | 0.76 | 37.7° |
| ribbon (through-sheet) | 6 | **0.002** | 100 | 4.1 | 0.35 | 62.2° |

95% of fired voxels are in components ≥500 vox; 79% in components ≥2000 vox.

### The dominant morphology is a thin, tile-spanning, sheet-conformal membrane

Global PCA calls the big components "blobs", but that is an artifact of percolation.
Local thickness (2×EDT) over the 139 components ≥500 vox: **mean thickness median 2.6
L2 vox (~25µm), within-component p90 median 4.0 vox (~38µm)**; max component 26,934 vox;
21% of large components span >90% of the tile. The largest components are thin sprawling
sheets, not compact masses. Combined with the coplanarity result and the position
result below: the model's false-positive mode is *painting the papyrus sheet surface
(and sheet-boundary interfaces) as a quasi-continuous thin blanket*. Genuinely
crack-like, through-sheet plates are nearly absent (6 components, 0.2% of voxels).

### H2 — coplanarity (`coplanarity_null.json`, 234 valid non-edge components, 200 perms)

| statistic | observed | across-tile null | within-tile null | isotropic null |
|---|---|---|---|---|
| median angle | **17.5°** | 50.2°±2.0 (z −16.8) | 21.2°±1.0 (z −3.9) | 59.8°±2.2 (z −19.5) |
| frac ≤30° | **0.705** | 0.230±0.027 (z +17.6) | 0.642±0.017 (z +3.8) | 0.129±0.021 |
| frac ≥60° | **0.107** | 0.367±0.025 | 0.128±0.014 (z −1.5, p 0.085) | — |

Alignment is overwhelming vs across-tile/isotropic nulls and survives (weakly) the
conservative within-tile shuffle. Ink predicts alignment — and alignment is observed —
but so does a sheet-surface-texture false positive, which is what the rest of the
evidence indicates. H2's directional prediction is met **by ~everything the model
fires at**, so it cannot separate ink from this FP mode.

### Fired voxels concentrate at the sheet surface (`position_vs_sheet.json`, 34 tiles)

The surface pred is a thin shell (16% of interior voxels). Fired voxels sit within ±2
vox of it at rate **0.506 vs 0.372±0.001** for volume-random placement (p_emp < 1/200);
inside-shell fraction 0.314 vs 0.216. A 1.36× surface enrichment — real, but mild:
half the fired volume still lies off-shell (inter-sheet gaps and sheet cores),
another way the output differs from ideal ink (which should be ≈100% at-surface).

### H3 — the "patch" (ink-candidate) subpopulation is not a separate mode

78 patch components (6.3% of fired voxels). Profile: thickness e3 median 4.2 vox
(p25–p75 3.0–6.2 ≈ 29–60µm) — inside the 4–8-vox stroke-thickness window; on-sheet
median 1.00; angle median 12.6°. So far ink-plausible. But:

- **In-plane footprint is round, not stroke-like**: e1/e2 aspect median 1.54
  (p25–p75 1.3–1.7); sizes median 174 vox; in-plane extents ~10–24 vox (96–230µm).
  At this scale a tile (614µm) is smaller than a letter, so a real stroke crossing a
  tile should appear as an elongated band or merge with tile-spanning structure — the
  compact flecks observed look like fragments of the membrane blanket, and their mean
  probability (0.76) and thickness match the membranes (0.81, 2.6–4.0 vox).
- **Ubiquitous, not localized**: ≥1 patch in 25/40 non-edge tiles; 54/78 pass the
  strict stroke-like gate (2≤e3≤10, e1≥8, on-sheet, away from mask edge) and those
  spread over 14 tiles in all sampled z-slabs. No tile or region stands out as a
  candidate writing zone.
- **Never dominant**: patch voxel fraction over non-edge tiles median 0.03, p90 0.13,
  max 0.33 (tile z=2560 y=11264 x=17664). Flag criterion (patch-dominant ≥0.5 +
  on-sheet + stroke thickness): **0 of 63 tiles**.

### Gallery (`morph_gallery.png`)

9 rows (6 highest patch-fraction tiles, 2 membrane exemplars, 1 mask-edge tile) ×
3 columns (CT L2 | P(ink) | overlay + sheet outline). Visual verdict matches the
numbers: firing traces sheet surfaces and sheet-edge/void boundaries as elongated
conformal bands and halos; the best "patch" tiles are damage/fragment zones, and no
panel contains anything resembling glyph strokes or letter fragments.

## Verdicts

- **H2 (sheet coplanarity of firing): SUPPORTED as a geometric fact** — median
  plate-vs-sheet angle 17.5° vs 50–60° nulls, p_emp < 1/200 — **but evidentially
  neutral for ink**, because conformality characterizes the entire output (the FP mode
  is itself sheet-conformal). Confidence: high (n=234 components, three nulls, two
  independent normal sources agreeing).
- **H3 (ink-like component shapes): REFUTED.** The output decomposes into (a)
  tile-spanning thin membranes = sheet-surface blanket (79% of voxels), (b) mask-edge
  halos (10%), (c) small conformal flecks continuous with (a) (6%), (d) rare true
  through-sheet ribbons (0.2%). No stroke-shaped, localized, patch-dominant
  subpopulation exists in the sample. Confidence: moderate-high for the sampled
  distribution; the sample is top-pmax/edge-biased and covers ~2% of scored tiles, so
  a rare localized ink region elsewhere cannot be excluded by morphology alone — but
  the fleet-wide stats (zero silent tiles, unimodal f05, anti-clustered tail) already
  make that prior very low.
- **Salvage implication**: consistent with the round-1 verdict — nothing here revives
  threshold/gating salvage. One refinement worth carrying forward: the failure mode is
  *sheet-surface painting*, not crack-following; any fine-tune should include
  hard-negative sheet-surface patches from 1203, and the on-sheet gating idea would
  NOT help (the FPs are already on-sheet).

## Files

All under `C:\Users\benbl\Desktop\Vsuvious\trackD\salvage\`:
`m0_inventory.py`, `m1_fetch.py`, `m2_components.py`, `m2b_deepdive.py`,
`m2c_position.py`, `m3_gallery.py`; outputs `inventory.json`, `components.json`,
`tile_scores.json`, `coplanarity_null.json`, `deepdive.json`,
`position_vs_sheet.json`, `morph_gallery.png`, this report; streamed subvolumes in
`cache\` (63×ct + 63×surf, 64³ uint8 npy, band-L2 frame).
