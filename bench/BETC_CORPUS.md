# Bet C corpus — what exists at ~8 µm with human ink labels (memo from `betC_corpus_manifest.json`, 2026-09-02)

_The manifest was assembled by a background agent from live listings/headers on 2026-09-02 (every URL
HEAD-checked; one legacy path 404). Areas were computed from the label rasters it downloaded to its own
scratchpad (ink = label ≥ 128 for Scroll 1's antialiased PNGs, > 0 for binary; surface = mask > 0). The agent
was cut off before writing this memo; the numbers below are its, the reading is mine._

## Totals

| source | pitch | segments | ink-labelled cm² | in ink_9um already? |
|---|---|---|---|---|
| Scroll 1 (PHerc. Paris 4) 2023 Grand Prize legacy stacks + `younader/Vesuvius-Grandprize-Winner` labels | 7.91 µm, 54 keV | 44 (65-layer uint16 TIFF, layer 32 = surface, `<seg>_mask.png`, `.ppm`, `.obj`, `area_cm2.txt`) | **153.7** | partial: 4 segments are ink_9um training segments via the 2.4 µm rescan pooled to 9.6 µm; 8 more are superseded-lineage siblings; the rest are clean by id |
| Scroll 1 outermost wrap recto/verso (Stephen Parsons community upload) | 7.91 µm | 1 wrap × 2 sides (65-layer stacks, mask PNG) | 4.4 | no |
| Scroll 5 (PHerc. 172, Bodleian) legacy paths + `Bodillium/Herculaneum-Scroll-Labels` | 7.91 µm, 53 keV | 23 legacy paths (21 with lossy 8-bit JPEG 65-layer stacks, 2 with uint16 TIFF); 13 labelled | **23.1** | no (0172 is not in the 29-representation manifest) |
| Frag5 PHerc1667Cr01Fr03 | 3.24 + 7.91 µm, 70 keV | 1 | 0.20 | no |
| Frag6 PHerc0051Cr04Fr08 | 3.24 (53/70/88 keV) + 7.91 µm (53 keV) | 1 | 0.45 | no |
| **total new native ~8 µm ink** | | | **181.2 cm²** (92.4 cm² clean vs ink_9um) | |
| labelled surface (mask > 0), all sources | | | 2,361.9 cm² | |

For scale: ink_9um's own 29 representations carry the PHerc0139 / 1667 / Paris4 / 0814 labels annotated at a
single Z plane; the community's strongest transfer datapoint is that lift tracks training-corpus size (ρ = 0.93).
This corpus roughly doubles the labelled ink area available at ~9 µm and adds two scrolls' worth of native
~8 µm texture (Paris 4 at 7.91 µm native rather than 2.4 µm pooled; Scroll 5 entirely new).

## Download and prep (from the manifest's size accounting)

- Raw: 606 GiB for everything. **139 GiB** if each stack is cropped to its label bbox and only layers 20–44
  are fetched (the 21-slice × 9.36 µm window = 197 µm = 24.9 legacy layers around layer 32) — do this on the
  pod, never through the laptop (1.5 MB/s).
- Pooling to the model pitch: in-plane 7.91 → 9.36 is a non-integer 1.1833 anti-aliased downsample; depth
  the same factor along the normal (legacy layers are 1 voxel = 7.91 µm apart). Fragments at 3.24 µm: 2.889×
  (65 × 3.24 µm = 211 µm ≈ 22.5 slices at 9.36 — the whole stack becomes the 21-slice window).
- Scroll 5's 21 JPEG stacks are lossy 8-bit: use them (the E2 control on Track F showed the ds8-JPEG chain
  costs nothing measurable at battery scale) but record the flag per representation; the two TIFF paths and
  the 53 modern S3 flatboi segments (33-slice uint8) are the clean alternative where labels overlap.

## Dedup and hold-out rule (from `ink9um_exclusion_and_dedup`)

villa keys physical segments as `<scroll>:<w>`; a 7.91 µm legacy stack whose id (or superseded lineage,
first 13 digits) matches an ink_9um Paris 4 row must share that physical key, so one surface gets one batch
budget even with two representations (2.4-pooled and 7.91-native). Exact matches: 20230702185753,
20230929220926, 20231106155351, 20231210121321. Siblings: 8 more (list in the manifest). Everything else on
Scroll 1, all of Scroll 5 and both fragments are clean, which is what the 92.4 cm² figure counts.

For the benchmark: Scroll 5 is the natural **second held-out native scroll** (Bet C's gate wants two in
rotation) — it is entirely absent from ink_9um, natively ~8 µm, and has 23 cm² of human labels.

## Three risks

1. **Label quality and format differ across sources.** Scroll 1 GP labels are antialiased PNGs on the 2023
   flattened canvases (some later re-segmented — the superseded lineages); Scroll 5's are a volunteer's
   CC BY-NC-SA set on legacy JPEG stacks; ink_9um's are single-plane zarrs. Train with per-source
   sampling quotas (the recipe's fixed scroll prior already does this) and hold each source's masks as its own
   validation, never mixed.
2. **Canvas and registration conventions.** 2023 layers are flattened-UV canvases with `.ppm` per-pixel
   geometry, not tifxyz; the modern S3 trees for Paris 4 / 0172 are tifxyz with 33-slice uint8 surface
   volumes. Mixing representations of one physical segment is the dedup problem above; the safe path is
   to treat the legacy canvas as the training canvas and never re-render it.
3. **Licensing.** Scrolls 1–4 and Fragments 1–6 are under the Vesuvius Challenge Data Agreement (no
   redistribution; no public disclosure of hidden text or ink-label images without written consent; EduceLab
   citation). Scroll 5 data is CC BY-NC 4.0; Bodill's labels CC BY-NC-SA 4.0 (credit the repo). Training a
   model and releasing weights is fine; redistributing the label rasters is not — the release ships the
   manifest and the prep scripts, not the labels. First Letters eligibility for a model trained on these is
   rules question (e) in the plan's appendix (assumed yes; ask on Discord).

## Recommended order on the pod (October, Bet C)

1. Scroll 1 legacy: 32 clean segments first (bbox-cropped, layers 20–44), then the 8 siblings keyed to their
   ink_9um physical segments, then the 4 exact duplicates only as the 7.91-native representation of a segment
   already in the budget.
2. Scroll 5: the 2 TIFF paths + the 13 labelled JPEG paths.
3. Fragments 5–6 at 7.91 µm (tiny area, but four energies on Frag6 make it the energy-augmentation source).
4. Pool everything to 9.36 µm with the anti-aliased resampler from `bench/p2a_v3` curvelib; fingerprint every
   input (sha256 of small files, shape/pitch of stacks) into the training manifest before the first step.
