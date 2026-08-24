# Investigation D — the 2.4 µm path on PHerc1203

_Status: 2026-08-17. All numbers below are measured today unless marked as an estimate._

## Bottom line

**The registration is not the killer. It is solved.** The PHerc1203 9.362 µm ↔ 2.403 µm
transform was derived from the data in about 40 minutes of laptop compute and $0, and it is
confirmed four independent ways (beamline stage coordinates, image cross-correlation at two
pyramid scales, a visual overlay, and agreement between the two volumes on where a sheet sits
relative to a transformed mesh). Residual is **22/15/11 µm RMS (z/y/x)** — 2.4 voxels at 9 µm,
6 voxels at 2.4 µm. The same stage arithmetic reproduces PHerc0139's *published* matrices to
49–129 µm, so the method is validated against ground truth.

Three things matter more than the transform:

1. **Only 24.1 cm² of the 130.2 cm² of published PHerc1203 segment surface lies inside the
   2.4 µm band** (the band is a 36 mm slab covering 20.5 % of the scroll's height). That is the
   whole prospecting target, and it is 18.5 % of what we already screened at 9 µm.
2. **A free, zero-registration positive control already exists.** Our validated control segment
   w035 (PHerc0139, human-verified Greek letters, pixel AUC 0.9991 at 9 µm) sits 98.2 % inside
   PHerc0139's own 2.403 µm band and 100 % inside its 2.399 µm volume — and the team has already
   published, for that segment, the transformed meshes, a rendered 2.399 µm surface volume, **and
   a reference ink prediction from the canonical 2 µm recipe**. We can validate our entire
   inference contract against a published ground-truth output before spending anything on 1203.
3. **The material, not the registration, is what should worry us.** Across 48 in-band mesh
   patches the along-normal intensity contrast is a median 0.138, against 0.397 for w035 under
   an identical export window (Mann-Whitney p = 2.6 × 10⁻¹¹). Re-probing the flattest patches in
   the 2.403 µm band raises mean-profile contrast only 0.073 → 0.102 — the 2.4 µm data is far
   sharper per pixel, but these sheets are merged, not merely blurred. Only ~15 % of the in-band
   patches look w035-like. **Render that subset first (§2.7).**

Separately, and independent of this whole investigation: **our August 9 µm survey screened only
41 % of PHerc1203's published segment area** (54.0 of 130.2 cm²) because the segment catalogue
picked `versions/0` — the first growth generation — for 9 of the 22 segments. See §5.

---

## 1. The two volumes: exact extents

| | 9.362 µm volume | 2.403 µm band |
|---|---|---|
| volume id | `20250820131727` | `20260319130212` |
| scan id | `20250720004030` | `20250721080303` |
| energy / detector distance | 113 keV / 1.2 m | 77 keV / 0.22 m |
| shape (z, y, x) | 18977 × 6844 × 6844 | 15137 × 26493 × 26493 |
| physical size | 177.66 × 64.07 × 64.07 mm | 36.37 × 63.66 × 63.66 mm |
| size at level 0 | 889 GB | 10.6 TB |
| stage z travel (`sz` motor) | −358.300 → −190.863 mm | −286.0552 → −255.3276 mm |
| detector vertical coverage | 1104 px (bin 2) = 10.336 mm | 2368 px = 5.690 mm |
| export window (f32 attenuation) | [−0.03, 0.145] | [−0.03, 0.19] |

Both are zarr v2, uint8, **128³ chunks with `compressor: null`** (raw bytes,
`dimension_separator: '/'`), 6 pyramid levels each with pure `scale` coordinate transformations
and **no translation**. Raw chunks mean HTTP range reads work — that is what made the
registration experiment cheap.

The lateral fields of view are nearly identical (63.66 vs 64.07 mm). The 2.4 µm scan is
therefore a **full-width, 36 mm-tall slab through the middle of the scroll**, not a small ROI.

Also present for the 2.4 µm band and worth knowing about:

- `20260319130212-surface-20260413222639-surface-m7-L2-th0.2.zarr`, shape 3785 × 6624 × 6624
  — surface predictions computed at pyramid level 2, i.e. **9.612 µm effective**;
- matching `.normal-grids/` (xy / xz / yz + preview images).

There is **no** published segment, mesh, surface volume or ink prediction on the 1203 2.4 µm
band. All 22 published 1203 segments live under `segments/raw/` (auto-grown, unreviewed) on the
9.362 µm volume.

---

## 2. There is no published transform — and one is derivable

### 2.1 Confirmed absent

- `sample.properties.volume_transforms` = `null` for PHerc1203.
- `properties.transforms` = `null` on **both** 1203 volumes.
- OME `.zattrs` for both carries scale-only `coordinateTransformations`.

This is a PHerc1203-specific gap, not a project-wide convention. PHerc0139, PHerc0814,
PHerc1667 and PHercParis4 all publish `transformation_matrix` + `from_landmarks`/`to_landmarks`
entries in the same catalogue. PHerc1203 simply has not been processed to that stage.

### 2.2 Analytic prior from beamline stage coordinates

Both scans are helical on BM18 with the sample z position recorded as the `sz` motor in the same
frame, and `srot` identical (−0.0) in both, so no azimuthal re-mount happened. Reconstructed
z extent = stage travel + detector vertical coverage, which checks out to 0.11 mm (9 µm) and
0.04 mm (2.4 µm). Mapping the 2.4 µm slab into 9 µm voxel z gives a predicted band of
**z ≈ 7965 … 11855**.

### 2.3 Measured by image cross-correlation

`trackD/hunt/derive_transform.py` — coarse pass at pyramid level 5 (299.6 µm/vox for the 9 µm
volume), a central 19.7 mm column of the 2.4 µm volume resampled onto the 9 µm grid, brute-force
NCC over every z offset and both z-orientations:

```
flip=False : best dz = 7936 (level-0 9 µm voxels)   NCC = 0.793
flip=True  : best dz = 6912                          NCC = 0.146   <- rejected
nearest rival peak far from the winner               NCC = 0.432
```

The z axes of the two volumes run the **same** way, and the measured offset agrees with the
stage-coordinate prediction to **29 voxels = 0.27 mm**.

`trackD/hunt/refine_transform.py` — refinement at 9 µm level 2 (37.45 µm/vox) against 2.4 µm
level 4 (38.45 µm/vox), 128³ blocks, phase correlation with 10× upsampling, at **15 probe
positions** spanning z2 = 2500…14000 and three lateral positions:

```
all 15 blocks registered, NCC 0.800 – 0.886
similarity fit      : scale 1/3.89502 (nominal 1/3.89596), t = (7939.6, 18.4, -19.8)
                      residual RMS 2.36 / 1.66 / 1.08 voxels at 9 µm = 22 / 16 / 10 µm
translation-only fit: t = (7940.1, 19.2, -18.8)
                      residual RMS 22 / 15 / 11 µm
```

The fitted scale differs from the nominal pixel-size ratio by 0.024 % — i.e. the published pixel
sizes are right and a **pure translation with the nominal scale is sufficient** at this
tolerance. Residuals show a faint systematic pattern (dy drifts with z, dx with lateral
position) consistent with a sub-0.1° residual rotation; fitting a full affine from more probe
blocks would shrink the residual further if ever needed.

### 2.4 The transform

In the same convention the open-data catalogue uses for the other scrolls (rows are **x, y, z**;
`p9 = M · [p2, 1]`), mapping 2.403 µm volume `20260319130212` → 9.362 µm volume `20250820131727`:

```json
{
  "transformation_matrix": [
    [0.256676, 0.0,      0.0,      -18.8],
    [0.0,      0.256676, 0.0,       19.2],
    [0.0,      0.0,      0.256676, 7940.1]
  ]
}
```

Inverse (9 µm → 2.4 µm) is scale 3.89596 with translation −(t · 3.89596).

**The 2.4 µm band occupies 9 µm voxel z = 7940 … 11825** (of 18977).

### 2.5 Visual QC

`trackD/hunt/qc_transform.png` — a 192³ block at the derived correspondence, shown as
9 µm / 2.4 µm / red-green overlay in both xy and xz. Volumetric NCC **0.829**; sheets are
coincident and the overlay is essentially yellow throughout.
`trackD/hunt/qc_native_detail.png` — the same 1.9 mm field at both native resolutions: the 9 µm
scan is a blur of laminae, the 2.4 µm scan resolves individual fibres and bright inclusions.

### 2.6 A transformed mesh lands on a real sheet — measured

`trackD/hunt/probe_mesh_on_band.py` takes a 1.7 mm patch of `auto_grown_20250925223153537`
(78.4 % in band) at 9 µm z ≈ 9361, maps it through the transform, and samples **both** volumes
along the surface normal (`trackD/hunt/probe_mesh_on_band.png`):

- 9.362 µm profile: peaks at **+19 µm**, sheet FWHM ≈ 100 µm, contrast 74 → 113.
- 2.403 µm profile: peaks at **+31 µm**, contrast 62 → 98.

Two things fall out. First, the two independent volumes agree on where the sheet is relative to
the mesh, which is a fourth confirmation of the transform (a wrong transform would put the
2.4 µm peak somewhere unrelated). Second — and this is the actionable number — **the auto-grown
mesh sits ~20–30 µm off the sheet centre, on the +normal side.** A 62-layer window at 2.4 µm
is ±74 µm, so the sheet stays inside it, but off-centre by ~13 voxels. The rendered faces at
both resolutions show the same physical structure, with the 2.4 µm one visibly sharper.

### 2.7 …but most in-band mesh patches sit in merged laminae, and that is the real finding

`trackD/hunt/mesh_bias_survey.py` repeated that probe on **48 patches** across the 8 highest
in-band segments, in the 9.362 µm volume. `trackD/hunt/mesh_bias_w035.py` ran the **identical**
estimator on 24 patches of the w035 control. Both volumes were exported with the **same**
intensity window (f32 [−0.03, 0.145], u16 [0, 65535]), so the comparison is not confounded.

| | median \|offset to profile peak\| | median along-normal contrast | range |
|---|---|---|---|
| w035 (letters readable at 9 µm) | 89 µm | **0.397** | 0.22 – 0.58 |
| PHerc1203, 48 in-band patches | 122 µm | **0.138** | 0.05 – 0.30 |

Mann-Whitney on contrast: **p = 2.6 × 10⁻¹¹**. Only 7 of 48 PHerc1203 patches reach even the
10th percentile of w035's contrast.

Two conclusions, and the first one retracts something I nearly claimed:

- **The offset statistic is not evidence of bad meshes.** It says the w035 mesh is a median
  89 µm off the brightest lamina — on a segment whose letters we read at AUC 0.9991. Argmax of
  mean intensity finds the brightest lamina in a stack, not the ink-bearing recto face. Mesh
  placement has to be judged by the layer-window sweep against ink output (C0 step 2), not by
  intensity geometry. Do not quote a "% of meshes off-surface" number from this.
- **The contrast difference is real and it is the finding.** On w035 the mesh sits on a
  well-separated lamina; on most PHerc1203 in-band patches the profile is nearly flat — the mesh
  is inside a densely packed, merged stack with no resolvable single sheet.

`trackD/hunt/contrast_2um_vs_9um.py` then asked whether that is a *resolution* limit or a
*material* one, by re-probing 4 of those patches (deliberately the 3 flattest plus one median —
a worst-case sample by construction) in the 2.403 µm band:

```
mean-profile along-normal contrast   9 µm 0.073  ->  2.4 µm 0.102
per-pixel along-normal rel. std      9 µm 0.227  ->  2.4 µm 0.301   (+33%)
```

`trackD/hunt/contrast_2um_vs_9um.png`: the rendered faces at 2.403 µm are dramatically sharper —
fibre texture, cracks and grain that the 9 µm scan smears — but the *mean* along-normal profile
stays flat. **The 2.4 µm band buys real detail; it does not, on these patches, turn a merged
stack into a separated sheet.** Compression is a material problem here, visible at both
resolutions.

**Plan consequence:** the along-normal contrast is a cheap 9 µm pre-filter (4 MiB per patch) for
deciding *where* to spend 2.4 µm rendering. Rank the in-band surface by it and render the
w035-like subset first (~7/48 of patches ≈ 3–4 cm²) rather than all 24.1 cm². That is both
cheaper and higher-yield than a uniform sweep.

### 2.8 The method itself is validated against a published transform

`trackD/hunt/validate_0139.py` runs the *stage-coordinate* arithmetic on PHerc0139, which
publishes its matrices, and compares:

| 0139 2.4 µm volume | published z translation | stage-derived | error |
|---|---|---|---|
| `20250820105138` (2.403 µm) | 3100.51 | 3095.22 | **+5.3 vox = 49 µm** |
| `20260319133050` / `20250822062710` (2.403 µm) | 8933.79 | 8947.60 | **−13.8 vox = −129 µm** |
| `20260102150214` (2.399 µm, 76953 slices) | 624.17 | 1295.39 | −671 vox = −6.3 mm |

The first two are single-scan bands like 1203's and land within 50–130 µm. The third is a
multi-sub-scan, full-height helical stack whose recorded `z_start` covers only the first
sub-scan — the arithmetic does not apply there, and that is a known limit of the prior, not of
the image-based refinement.

---

## 3. What is actually inside the band

Point-weighted fraction of each segment's mesh vertices with 9 µm z in [7940, 11825]
(`trackD/hunt/band_extents.json`):

| segment | area cm² | in band | cm² in band | screened at 9 µm in Aug |
|---|---|---|---|---|
| auto_grown_20251005230830031 | 16.10 | 42.9 % | **6.91** | 0.60 cm² only |
| auto_grown_20251005231446965 | 12.53 | 30.0 % | **3.76** | 0.58 cm² only |
| auto_grown_20251005221856743 | 11.79 | 31.7 % | **3.74** | 0.59 cm² only |
| auto_grown_20250925223153537 | 2.92 | **78.4 %** | 2.29 | full |
| auto_grown_20250930104534929 | 8.40 | 24.0 % | 2.01 | 0.58 cm² only |
| auto_grown_20251005230830030 | 6.19 | 23.1 % | 1.43 | 0.59 cm² only |
| auto_grown_20251005230118636 | 7.61 | 11.3 % | 0.86 | 0.60 cm² only |
| auto_grown_20251005231446963 | 3.74 | 21.5 % | 0.80 | 0.58 cm² only |
| _14 more_ | 60.9 | 0–13 % | 2.3 | mostly full |
| **total** | **130.2** | **18.5 %** | **24.1** | 54.0 cm² (41 %) |

Three segments (`…163217042`, `…163219195`, `…163230146`) do not reach the band at all. The
segment cloud spans 9 µm z 5403…10150, i.e. it sits mostly *below* the band and pokes into its
lower half.

---

## 4. `ink_canonical_2um` — the input contract

**Weights.** `scrollprize/ink_canonical_2um` → `r152_3ddec_v2_l5_epoch13.ckpt`, 1.55 GB,
PyTorch-Lightning 2.0.9, epoch 13 / global step 113,246. `config.json`:
`{"pred_shape": [4608, 13824], "size": 256, "enc": "r152", "with_norm": true, "total_steps": 8089}`.

**Runner.** `villa/ink-detection/optimized_inference`, `MODEL_TYPE=resnet3d-152-3d-decoder`
(ResNet3D-152 encoder → 3D U-Net decoder with depth-attention collapse and an aux head).

**Tensor contract.** `(B, 1, 62, 256, 256)`, i.e. `TILE_SIZE=256`, `START_LAYER=1`,
`END_LAYER=63` → `CFG.in_chans = 62`. This is the combination the villa README's own GPU smoke
test uses, and the published w035 prediction filename records `tile256-stride128`. Verified
against the weights: the stem is `backbone.conv1.weight` of shape `(64, 1, 7, 7, 7)` —
single-channel 3D — and the forward pass returns `(B, 1, 64, 64)`, i.e. 4× downsampled from the
tile, which the runner interpolates back up.

**Measured cost (this laptop, RTX 4090 Laptop 16 GB, fp16 autocast, no `torch.compile`):**

| batch | ms/step | tiles/s | peak VRAM |
|---|---|---|---|
| 1 | 1303 | 0.77 | 1.22 GiB |
| 2 | 2470 | 0.81 | 1.95 GiB |
| 4 | 5116 | 0.78 | 3.40 GiB |
| 8 | 10326 | 0.77 | 6.30 GiB |

**It is compute-bound, not memory-bound** — throughput is flat in batch size and VRAM is tiny.
Batch size is therefore a scheduling knob only; a 32 GB 5090 buys nothing through batching. Scale
comes from more GPUs (the runner's `NUM_PARTS`/`PART_ID` map-reduce) and from `torch.compile`,
which the runner enables by default and which I did not benchmark. Budget **~0.8 tiles/s per
4090-class GPU**, and assume 2–3 tiles/s on a 5090 with compile until measured.

**Preprocessing (this is the whole normalisation).**
1. Extract the tile from the surface-volume zarr as `(H, W, C)`.
2. `np.clip(tile, 0, CFG.max_clip_value)` with `max_clip_value = 200`.
3. `albumentations.ToFloat(max_value=200)` → `[0, 1]`, then `ToTensorV2`.
There is **no** mean/std normalisation and no per-tile standardisation. A `reverse` flag flips
the depth axis. Output tiles are blended with a normalised Hann window (`use_hann_window=True`).

**Output.** Full-resolution single-channel raster. The published w035 2.399 µm prediction is
22640 × 20400 uint8 — exactly the surface volume's lateral shape, no downsampling.

**Surface-volume shape.** Published w035 surface volumes are `(109, 22640, 20400)` at 2.399 µm
and `(28, 5820, 5240)` at 9.362 µm — both ≈ 262 µm of depth. So the team renders ~110 layers and
the model consumes a 62-layer window from within them; which 62 is not recorded in the filename
and must be recovered on the control (see §6, C0).

**A domain-shift detail worth pre-empting.** Because the only normalisation is "clip at 200,
divide by 200", the model is sensitive to the uint8 export window. The likely training volumes
(PHerc1667 `20251217075048` and Paris4 `20260411134726`) were exported with f32 windows
[−0.03, 0.21] and [−0.04, 0.22]; **PHerc1203's 2.4 µm band uses [−0.03, 0.19]**. Same offset,
~9 % different gain. The correction is one multiply on the rendered uint8
(0.22/0.24 ≈ 0.917) and should be run both ways. Note that the control (§6, C0) is a
[−0.03, 0.21] volume, so it validates the code path but **not** this correction.

---

## 5. Side finding: 59 % of PHerc1203's segment area was never screened at 9 µm

`trackD/runpod/list_segments.py::has_tifxyz` returns the first prefix in S3 listing order that
contains `x/y/z.tif`. For the 9 PHerc1203 segments that have a `versions/` subtree, that is
`versions/0` — the **first growth generation** — not the top-level final mesh. Concretely,
`auto_grown_20251005231446965` was screened at 0.58 cm² when its published mesh is 12.53 cm².

Totals: 54.0 of 130.2 cm² screened (41 %). The tripwire-negative result for PHerc1203 stands
only over that 41 %. Re-running the existing 9 µm survey on the top-level meshes is cheap
(~$1–2 on the existing fleet script, one-line fix) and should happen regardless of what we
decide about the 2.4 µm path.

---

## 6. End-to-end plan

### C0 — Reproduce a published 2 µm prediction (the control). Do this first.

Zero registration, zero rendering — everything already exists on S3:

```
PHerc0139/segments/20260317000000-w035_2026031718/
  surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr   (109, 22640, 20400) uint8
  ink-detection/PHerc0139-…-new_canon_autoresearch_recipe-tile256-stride128.tif  (22640, 20400)
  mesh/20260317000000-on-20260102150214-2.399um.tifxyz
```

w035 is 100 % inside that volume and 98.2 % inside 0139's 2.403 µm band `20250820105138`.

1. Sweep first, cheaply: a **1024 × 1024** window over a block of human-verified letters, 12
   values of `START_LAYER` across the 109 available layers at a fixed 62-layer width. That is
   12 × 64 = 768 tiles ≈ 5 min on one 5090 at the measured throughput (§4).
2. Take the best window and run one **4096 × 4096** pass (1024 tiles, ~7 min) forward and
   reversed.
3. **Gate:** pixel correlation with the published tif > 0.9 in the window and letters legible.
   If we cannot reproduce a published prediction from published inputs, stop — the contract is
   wrong and nothing downstream is interpretable.
4. Free rider: this also gives the first honest 9 µm-vs-2.4 µm comparison on *identical
   surface*, since we already have the 9 µm ink_9um prediction for w035 (AUC 0.9991), and it
   calibrates the layer-window sweep we will need in C2.

Cost: ~1–2 GB of streamed zarr + ~2 k tiles of inference. **≈ $0.50, well under an hour.**

### C1 — Transform the meshes

`volume-cartographer/apps/src/vc_transform_geom.cpp` already applies a 4×4 affine to a single
tifxyz **or a whole directory of them**, reading the matrix from JSON under the key
`transformation_matrix` — exactly the format in §2.4 and exactly what the team used to produce
w035's `mesh/…-on-…-2.403um.tifxyz`. Their convention (verified on w035): coordinates are
scaled, `scale` stays 0.05, `area_vx2` is rewritten. Grid pitch therefore stays 187 µm in
physical terms while the raster grows 3.896× per axis.

Apply to the 5 highest in-band segments (18.7 of the 24.1 cm²). Then **rank before rendering**:
run the 9 µm along-normal contrast probe (`mesh_bias_survey.py`, 4 MiB per patch) densely over
the in-band surface and keep the w035-like subset (contrast ≳ 0.24, ~7/48 of patches in the
pilot ≈ 3–4 cm²). Rendering that subset first is the difference between a 30-minute run and a
5-hour one, and §2.7 says the rest is mostly merged stack anyway.

### C2 — Render at 2.4 µm and run the model

`trackD/runpod/render_tifxyz_sv.py` already does tifxyz + volume → N-slice uint8 surface volume
and was validated end-to-end at 9 µm (r = 0.813 on w035). The changes are small: point it at the
transformed mesh and the 2.4 µm volume, `--num-slices 62` (render 128–192 and slide, see §7),
and add the optional gain correction from §4.

Then `optimized_inference` with `MODEL_TYPE=resnet3d-152-3d-decoder`, `TILE_SIZE=256`,
`STRIDE=128`, `START_LAYER/END_LAYER` from C0, forward **and** z-reversed passes.

### C3 — Adjudicate with the existing battery

`trackD/salvage/verdict_*.py` (4-test text battery + tripwire), the ruling-periodicity control,
the forward/reverse asymmetry statistic, plus a blank in-band negative control and the 9 µm
render of the same patch for contrast.

### Alternative that removes registration entirely

The team published **surface predictions and normal grids for the 2.4 µm band**
(level 2, 9.612 µm effective). `vc_grow_seg_from_seed` can therefore grow segments natively in
2.4 µm coordinates, no transform involved. That is the more robust path if the transformed
meshes turn out to sit off-surface (§7.1) — and it is also how new, larger in-band area would be
obtained beyond the 24.1 cm² the existing meshes give us. The transform is a shortcut for
reusing 22 existing meshes, not a hard dependency of the 2.4 µm path.

---

## 7. Risks, honestly

1. **Merged laminae are the binding constraint, and it is now measured (§2.7).** PHerc1203's
   in-band mesh patches have a median along-normal contrast of 0.138 against w035's 0.397
   (p = 2.6 × 10⁻¹¹, identical export windows), and going to 2.4 µm raises the mean-profile
   contrast only 0.073 → 0.102 on the flattest patches. Most of the 24.1 cm² is compressed
   stack, not separated sheet. This is a material property; no model and no resolution fixes it.
   **Mitigation: pre-filter by contrast and render the w035-like subset first.**
2. **Surface localisation within the window.** A 62-layer window at 2.4 µm is ±74 µm. The
   registration residual (22 µm) is comfortably inside it; the mesh-to-sheet placement is not
   measurable by intensity geometry (§2.7), and we already had the symptom at scale:
   forward-vs-z-reversed prediction correlation is 0.076 on the w035 control (ink on one face)
   but 0.22–0.91 on every 1203 segment (symmetric through the sheet). **Mitigation: render
   128–192 slices and slide the 62-layer window, scoring each offset** — the same sweep as
   C0 step 2.
3. **The 1203 meshes are `raw/` and unreviewed.** w035 is a curated segment; these are not.
   Expect self-intersections and sheet jumps.
4. **Only 24.1 cm² is in band** — a small fraction of a scroll whose total surface runs to
   thousands of cm². Even if 1203 has readable ink, the odds it intersects these particular
   auto-grown patches are not high.
5. **Intensity-window shift** vs the training volumes (§4). Cheap to correct, cannot be
   validated on the control.
6. **Silence from the team is ambiguous.** The 2.4 µm band was exported 2026-03-19 and surface
   inference ran 2026-05-13, yet nothing downstream is published. That is consistent with "not
   processed yet" (they have no 2.4 µm segments for it at all) — but also with "processed
   internally, nothing found". We cannot distinguish these.
7. **Our own track record**: this project has refuted three of its own false positives. Any
   positive here must clear the full battery plus the blank control before it is called anything.

---

## 8. Cost

Inference throughput is now measured, not guessed: **0.8 tiles/s on a 4090 Laptop**, flat in
batch size, 1.2 GiB VRAM (§4). Assume 2–3 tiles/s per 5090 with `torch.compile`. At stride 128 a
cm² of 2.403 µm surface is ~1 060 tiles, so **1 cm² ≈ 7 min of single-GPU inference per pass**.

| stage | compute | wall clock | $ |
|---|---|---|---|
| transform derivation + contrast triage | **done**, laptop, HTTP range reads | ~2 h | **$0** |
| C0 control (w035, existing surface volume) | 1 × 5090, ~1–2 GB stream, ~2 k tiles | ~30 min | ~$0.50 |
| C1 mesh transform + contrast ranking | CPU + ~1 GB of 9 µm reads | ~1 h | ~$0 |
| C2a render + infer the **3–4 cm² high-contrast subset**, fwd + reverse | ~20–40 GB streamed, ~8 k tiles | ~1.5 h | ~$1.5 |
| C2b (only if C2a is interesting) all 24.1 cm², fwd + reverse | 100–300 GB streamed, ~50 k tiles | 6–9 h on 1 GPU, ~2 h on 4 | ~$5 |
| C3 adjudication | CPU | ~1 h | ~$0 |
| **total, stopping after C2a** | | **~4 h** | **≈ $2** |
| **total, full sweep** | | **~1 day** | **≈ $7–10** |

The streamed-chunk range comes from two estimates that bracket it: chunk-counting over a
24.1 cm² sheet (128³ chunks are 307.6 µm on a side at 2.403 µm → ~25 k chunks minimum, ~64 k
with obliquity and depth inflation = 128 GB) and a pessimistic extrapolation of the actual
bounding-box read in the §2.6 probe (121 MiB for 0.010 cm² → 286 GB). The renderer reads
chunk-granular, so the low end is the realistic one; the disk cache in
`render_tifxyz_sv.py --cache-dir` already exists for this.

Against ~$45 remaining of the $80 cap, the C2a-and-stop path costs ~$2 and answers the question
on the surface most likely to carry signal. Note that this laptop's S3 throughput is ~1.5 MB/s
and its HF throughput ~1 MB/s (the 1.44 GiB checkpoint took 31 min) — **all heavy data movement
must happen on the pod**, as with the previous survey.

Things to build (none large):
- `--transform` support in the render path, or a `vc_transform_geom` invocation + JSON matrix;
- a layer-window sweep harness (shared by C0 and C2);
- the uint8 gain correction flag;
- the one-line `list_segments.py` fix from §5.

---

## 9. Calibrated probability

**P(this route surfaces genuine, defensible letters) ≈ 0.10.**

Decomposed:

| step | p | reasoning |
|---|---|---|
| pipeline runs and is calibrated | 0.85 | control exists with a published reference output; main risk is the layer window |
| PHerc1203 has ink the un-adapted canonical model can see anywhere in the band | 0.40 | it is the production model that read 1667 and Paris4, but 1667 needed six pseudo-label iterations, and the team's own docs say some scrolls "show little or no convincing ink" |
| that ink intersects surface we can actually localise | 0.40 | **revised down from 0.6 by §2.7**: the in-band surface has 2.9× less along-normal contrast than the w035 control (p = 2.6 × 10⁻¹¹) and 2.4 µm does not undo it; only ~15 % of patches look w035-like |
| result survives the tripwire + verdict battery and the blank control | 0.80 | three prior false positives refuted; the battery is strict by design |

Product ≈ 0.11. Call it **10 %**, and ~6–8 % for "enough letters to be
First-Letters-defensible". Today's work moved this *down* from the ~15 % I would have quoted
before §2.7: the registration turned out easier than expected and the material turned out
worse.

That is materially better than the 9 µm survey's prior, for one specific reason: this is the
only GP-eligible data above the resolution wall (**PHerc1203 is the only one of the 13
GP-eligible scrolls with any sub-5 µm volume — verified across all 13 catalogue entries**), and
we would be running the model that actually reads scrolls rather than one that has only ever
read letters at 9 µm once.

The expected value does not rest on the 10 %. A calibrated negative — "the production 2.4 µm
model, validated against a published reference output on a known-letters control, finds nothing
in 24 cm² of PHerc1203's 2.4 µm band" — plus a published, validated 9.362 µm ↔ 2.403 µm
transform that the open-data catalogue is missing, is itself a monthly-prize-shaped deliverable.

---

## 10. Artefacts produced today

| file | what |
|---|---|
| `trackD/hunt/band_extents.py` / `.json` | per-segment bboxes from S3 + in-band fractions |
| `trackD/hunt/zarr_http.py` | raw-chunk HTTP reader for the open-data zarrs |
| `trackD/hunt/derive_transform.py` / `derive_transform.json`, `coarse_ncc.npy` | coarse level-5 registration, flip test |
| `trackD/hunt/refine_transform.py` / `.json` | 15-block level-2 refinement, rigidity test, final matrix |
| `trackD/hunt/pherc1203_2403um_to_9362um.json` | the derived transform in the catalogue's own `transformation_matrix` format, ready for `vc_transform_geom` |
| `trackD/hunt/validate_0139.py` | stage-coordinate method validated against PHerc0139's published matrices; w035-in-band check |
| `trackD/hunt/qc_overlay.py`, `qc_transform.png`, `qc_native_detail.png` | visual QC of the transform |
| `trackD/hunt/probe_mesh_on_band.py` / `.json`, `probe_mesh_on_band.png` | along-normal sheet profile of a transformed mesh in both volumes |
| `trackD/hunt/mesh_bias_survey.py` / `.json` | 48-patch along-normal probe across the 8 top in-band segments |
| `trackD/hunt/mesh_bias_w035.py` / `.json` | same estimator on the w035 control — the comparison that retracts the "meshes are off-surface" reading and isolates the contrast finding |
| `trackD/hunt/contrast_2um_vs_9um.py` / `.json`, `.png` | 9 µm vs 2.4 µm along-normal structure on the flattest patches |
| `trackD/hunt/bench_canonical2um.py` | model contract + VRAM/throughput benchmark (run: 0.8 tiles/s, 1.2 GiB) |
