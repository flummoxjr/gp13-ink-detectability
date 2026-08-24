# PHerc1667 ground truth for the "1667 instrument" experiment

Scout report, 2026-08-16. Source of record: `C:\Users\benbl\Desktop\Vsuvious\trackD\meta\PHerc1667.json`
(dump of official open-data `metadata.json`), cross-checked live against
`s3://vesuvius-challenge-open-data` and `data.aws.ash2txt.org` / `dl.ash2txt.org`.

**Verdict: the experiment can run.** A direct, landmark-derived affine
2.4µm (20251217075048) → 7.91µm (20231117161658) is published in the metadata (12 landmark pairs),
and — better — the challenge team already shipped per-segment ink-prediction images together with
segment coordinate grids *pre-transformed into the 7.91µm frame*, so letter coordinates can be read
out in old-volume voxel space without applying any matrix ourselves.

---

## 1. Volumes (all OME-Zarr v2, uint8, chunks 128³, `dimension_separator: /`, arrays indexed [z, y, x])

| volume id | long_id | scan_id | voxel | energy | shape [z,y,x] | uint8 window (f32) | uint8 window (u16) |
|---|---|---|---|---|---|---|---|
| 20231117161658 | `20231117161658-7.910um-53keV-masked.zarr` | 20231117161658 | 7.91 µm | 53 keV | [11173, 3340, 3440] | [-0.03, 0.145] | [12240, 59440] |
| 20231107190228 | `20231107190228-3.240um-88keV-masked.zarr` | 20231107190228 | 3.24 µm | 88 keV | [26132, 7960, 8120] | [-0.03, 0.145] | [12460, 61260] |
| 20251217075048 | `20251217075048-2.399um-0.2m-78keV-masked.zarr` | 20251204164204 | 2.399 µm | 78 keV | [37076, 15229, 15229] | [-0.03, 0.21] | [7396, 45012] |
| 20260323082859 | `20260323082859-1.129um-0.2m-59keV-masked.zarr` | 20251208002146 | 1.129 µm | 59 keV | [42209, 22122, 20276] | [-0.03, 0.145] | [1344, 29664] |

Notes:
- All four produced by `mask_layers_zarr_export`; windows above are the linear clip ranges used for
  the uint8 conversion (f32 window for phase-retrieved float data; u16 window for the 16-bit form).
  The 2.4µm volume uses a DIFFERENT window (max 0.21 vs 0.145) — normalize before cross-volume intensity comparisons.
- `z_direction_is_top_to_bottom`: 7.91µm = false, 2.4µm = true (the affine's negative z-coefficient absorbs this).
- Paper confirmation: arXiv 2606.29085 ("Complete virtual unwrapping and reading of a rolled
  Herculaneum papyrus") states the read was done on the BM18 2.4µm / 78 keV / 0.22 m scan
  = volume 20251217075048. No separate data-availability release; the open-data bucket IS the release.

## 2. Verified data URLs (checked 2026-08-16)

| URL | check | status |
|---|---|---|
| `https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr/0/.zarray` | GET | 200 (shape matches) |
| same volume, chunk `0/40/10/10`, ranged GET 0-1023 | GET | 206 |
| `https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231107190228-3.240um-88keV-masked.zarr/0/.zarray` | GET | 200 |
| `https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr/0/.zarray` | GET | 200 |
| same volume, chunk `0/140/50/50`, ranged GET 0-1023 | GET | 206 |
| w032 ink TIF (see §4) | HEAD | 200, 89,691,397 B |
| w032 `...on-20231117161658-7.91um.tifxyz/x.tif` | HEAD | 200 |
| merged-segment on-7.91µm tifxyz (x/y/z.tif + meta.json) | LIST | present (14.1/14.2/12.0 MB) |
| `https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/transforms/20231107190228-to-20231117161658.json` | GET | 200 (matrix identical to metadata) |

S3 bucket layout under `PHerc1667/`: `photos/`, `representations/`, `segments/`, `volumes/`.
`representations/predictions/` contains ONLY `lasagna/` (`predictions/ink/` exists but is EMPTY, KeyCount=0).
The dl.ash2txt.org volpkg (`full-scrolls/Scroll4/PHerc1667.volpkg/`) holds only legacy 2023–2024 paths
(`paths/` stops at 20240304161941); everything current lives in the open-data bucket.

## 3. Transform chain — THE key answer

**Convention** (verified numerically and against the volpkg 4x4 JSON): column vector
`p_target = A · p_source + t`, with `p = (x, y, z)` in level-0 voxel units of each volume
(zarr index order is [z,y,x] — swap when indexing). Matrices below are the 3x4 `[A | t]`.

### 3a. Direct 2.4µm → 7.91µm (what the instrument needs) — EXISTS, first-class

`20251217075048 → 20231117161658`, derivation `20251217075048-20231117161658`,
landmark-derived (12 landmark pairs stored in metadata alongside the matrix):

```
[ 0.04894527965062537, -0.30118843227994675,  0.02123063938481646,  3077.0443172423622 ]
[-0.30186393002822093, -0.04870841623005263,  0.00275999418787795,  4213.6945905053700 ]
[ 0.00095534945436288, -0.02118165313478389, -0.30381086764484706, 11550.6612526774620 ]
```

(≈90° in-plane rotation + z-flip + scale ≈0.3046–0.3058 ≈ 2.399/7.91; row norms 0.8% above nominal —
the landmark fit absorbs real voxel-size calibration.)

Inverse (7.91µm → 2.4µm), also published directly:

```
[ 0.52155979, -3.22816080,  0.00712073, 11915.37199665 ]
[-3.21948563, -0.52274583, -0.22973009, 14762.72566177 ]
[ 0.22610218,  0.02629462, -3.27548232, 37027.46281244 ]
```

**Numeric verification performed:** M(7.91→2.4)·M(2.4→7.91) = identity to machine precision;
composing 2.4→3.24→7.91 reproduces the direct matrix to ≤5e-5 voxel in translation; mapping the
merged segment's published 2.4µm bbox through M(2.4→7.91) lands consistently on its published
7.91µm bbox. Chain is self-consistent.

### 3b. Direct 2.4µm → 3.24µm (secondary target volume)

`20251217075048 → 20231107190228`, derivation `20251217075048-20231117161658 . 20231117161658-20231107190228⁻¹`:

```
[ 0.12330607, -0.72393424,  0.04948655,  7218.04222339 ]
[-0.72212095, -0.11546155,  0.01191409, 10014.49628932 ]
[ 0.00136020, -0.04758934, -0.73519126, 27768.26063180 ]
```

### 3c. Primitive transforms (per-volume `properties.transforms`, with landmark counts)

- 20231107190228 → 20231117161658 (5 landmarks) — also mirrored in volpkg `transforms/` dir:
```
[ 0.41559224,  0.00318313, -0.00085214,  69.06699838 ]
[ 0.00039936,  0.41809799,  0.00304823, -60.87277464 ]
[ 0.00211393, -0.00018336,  0.41337991,  58.39794227 ]
```
- 20251217075048 → 20231117161658 (12 landmarks) — §3a.
- 20260323082859 → 20251217075048 (6 landmarks):
```
[ 0.47215447, -0.00598911,  0.00167082,  2919.87392969 ]
[ 0.00048432,  0.47497578, -0.00274219,  2418.11874960 ]
[-0.00116501,  0.00956509,  0.46620250, 12502.42748364 ]
```

All 12 pairwise matrices (every from→to combination among the 4 volumes) are precomputed in
`sample.properties.volume_transforms` in the JSON — no composition needed at runtime; remaining ones
of interest:

- 20231117161658 → 20231107190228:
```
[ 2.40619639, -0.01831700,  0.00509519, -167.60131788 ]
[-0.00220867,  2.39179285, -0.01764141,  146.77783555 ]
[-0.01230574,  0.00115460,  2.41904835, -140.34724164 ]
```
- 20231107190228 → 20251217075048:
```
[ 0.21548204, -1.34802867, -0.00734104, 12148.31750701 ]
[-1.33868764, -0.22876490, -0.09381580, 14558.77067788 ]
[ 0.08705266,  0.01231405, -1.35413111, 36850.19695710 ]
```
- (1.129µm relations exist likewise; omitted from the instrument path.)

## 4. Letter-coordinate sources in the 2.4µm frame, ranked

### #1 (best): official per-segment ink-detection TIFFs + "on-7.91µm" tifxyz grids — transform-free

Every one of the 19 flat "window" segments (wNNN_flatboi, all `original_volume_id=20231117161658`)
under `s3://vesuvius-challenge-open-data/PHerc1667/segments/<id>-<suffix>/` carries:

- `ink-detection/PHerc1667-<id>-2.399um-0.22m-78keV-volume-20251217075048-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif`
  — ink probability image in the segment's flattened (u,v) frame, computed from the 2.4µm surface
  volume (i.e. the letters of the June-2026 read). ~90 MB each; `-ds8.jpg` preview in `downsampled/`.
- `mesh/<id>-on-20231117161658-7.91um.tifxyz/{x,y,z}.tif` — per-(u,v) 3D coordinates DIRECTLY in
  7.91µm voxel space (`meta.json`: `scale: [0.05, 0.05]`, i.e. the coordinate grid is 1/20 the
  flattened resolution — sample bilinearly: grid_px = uv_px * 0.05).
- Sibling `on-20231107190228-3.24um.tifxyz/` and `on-20251217075048-2.399um.tifxyz/` grids for the
  3.24µm target and for cross-checks; plus `surface-volumes/*.zarr` (the 2.4µm layers the ink model saw).

Recipe: threshold/pick letter strokes in the ink TIF at (u,v) → bilinear-sample x/y/z.tif at
(u·0.05, v·0.05) → (x,y,z) voxels in 20231117161658 → index zarr as [z,y,x]. Zero matrix math,
per-vertex accuracy (not just a global affine). Example segment (verified HEAD 200):
w032 = `20260105050000-w032_2026010505_flatboi` (10240×11720 uv). Segment ids:
20240304141531(w013), 20240304144031(w018), 20240304161941(w023), 20251206103305(w012),
20251208130119(w028), 20251212185248(w029), 20251220020000(w030), 20251223230000(w031),
20260105050000(w032), 20260110010000(w033), 20260115160000(w034), 20260116230000(w035),
20260119120000(w036), 20260123230000(w037), 20260128140000(w038), 20260130150000(w039),
20260203210000(w040), 20260205070000(w041), 20260108140509(w011-window).

### #2: merged full-read segment `20260612121456-w011_20260108140509268_merged_v4_flatboi_straightened_v4`

The canonical whole-scroll surface (184099×14179 uv, 99.2% coverage of the 7.91µm volume), traced in
the 20231117161658 frame, with on-7.91µm / on-3.24µm / on-2.4µm tifxyz grids (all verified on S3).
CAVEAT: **no ink-detection TIF was released for it** (22 objects total, meshes only) — to use it we
would render ink ourselves or register paper figures onto its uv space. Best used as the canonical
geometry backbone; letters come from #1.

### #3: 1.129µm-model ink detections (independent second reading)

Same 19 segments also carry
`ink-detection/PHerc1667-<id>-1.129um-0.22m-59keV-volume-20260323082859-L1-20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif`
— predictions from the 1.129µm scan on the same uv grids. Perfect for a "letters confirmed by two
independent scans/models" high-confidence mask (intersect with #1 before testing the instrument).

### #4: lasagna volumetric predictions (2.4µm frame)

`s3://vesuvius-challenge-open-data/PHerc1667/representations/predictions/lasagna/20251217075048-lasagna-20260419180421-L2/`
(`PHerc1667-20251217075048-lasagna-20260724_cos.ome.zarr` + `.lasagna.json`; model 20260419180421,
level 2). Volumetric, in 20251217075048 voxel space → map with the §3a affine. Surface/structure
oriented (cos channel), not a letter mask per se — supporting evidence, not primary ground truth.

### #5 (weak): community uploads

`dl.ash2txt.org/community-uploads/bruniss/scrolls/s4/3d_ink/s4_predictions.zarr/` exists but has a
nonstandard nested layout (`0.zarr/0/...`; `.zarray` probes 404) and unknown frame/vintage
(pre-dates the 2.4µm read). Also `manual_segmentations/` (2024-era). Not needed.

## 5. Gaps / risks

1. **No character-level annotations released**: no per-letter bounding boxes, transcription
   alignment, or vectorized glyphs in the bucket or with the paper (arXiv page has no
   data-availability section; HTML points only to scrollprize.org/data). Letter locations must be
   segmented out of the ink TIFFs (high contrast, so thresholding + connected components should do).
2. Merged segment lacks an ink render (see #2) — per-window segments overlap-tile the scroll, so
   coverage is still complete, but expect duplicate letters in overlap zones between windows.
3. tifxyz grids are 0.05× the uv resolution — interpolate; near mesh seams/foldbacks a 20px cell can
   straddle sheets. Sanity-filter mapped points against the 7.91µm scroll mask (fill_value 0 outside).
4. uint8 window differs for the 2.4µm volume ([-0.03,0.21] vs [-0.03,0.145]) — rescale if a detector
   trained on one volume's histogram is applied to another.
5. The global affine (§3a) is a 12-landmark rigid+scale fit — good to a few 7.91µm voxels globally,
   but the tifxyz route (#1) is per-vertex and strictly better where available; keep the affine for
   spot checks and for mapping anything volumetric (e.g. lasagna, #4).
6. `representations/predictions/ink/` (empty) may get populated later — worth re-listing before
   the run; a sample-level ink representation would supersede the per-segment TIFF stitching.
