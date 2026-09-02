# PHerc0826 — GrowPatch surfaces, release v1 (2026-09-02)

21 of 24 grown patches pass the alignment gate (< 30° to the local lamellae at the seed): **172.96 cm²** of correctly-oriented surface on a scroll with no published segmentation.

- Volume: `20250821151701-9.362um-1.2m-113keV-masked.zarr` (9.362 µm); surface prediction: `20250821151701-surface-20260413222639-surface-m7-L0-th0.2` + its released normal grids.
- Tracer: vc_grow_seg_from_seed, villa main (release vc-tracer-de3c2494; built from e2442b7). Params: seed mode, 75 generations, min_area 0.3 cm², single-threaded per seed.
- Seeds: the scroll's 24 separability-ranked ROIs (structure-tensor planarity), each moved to the nearest m7 sheet voxel
  (support gate against the ~49 % phantom rate of m7 positives beyond CT support).
- Gate: mesh normal vs local sheet normal from a 256³ CT cube at the seed; published GP meshes sit at 13.1° median.
- Format: tifxyz (`x.tif`, `y.tif`, `z.tif`, `generations.tif`, `meta.json`), coordinates in the volume's voxel units.
- Known caveat: every tracer mesh stores `scale` as float32 0.05 → villa's `Tifxyz.shape` gives a canvas 1 px short
  (`issue_drafts/filing/tifxyz_fullres_shape_truncation.md`); render onto `round(h/scale)`.

| patch | area cm² | angle ° | abs n_z | separability | PASS |
|---|---|---|---|---|---|
| auto_grown_20260902140829487 | 7.749 | 4.3 | 0.466 | 0.758 | PASS |
| auto_grown_20260902140830497 | 8.915 | 11.8 | 0.098 | 0.743 | PASS |
| auto_grown_20260902140833266 | 9.345 | 31.7 | 0.2 | 0.735 | fail (ok) |
| auto_grown_20260902140834273 | 7.937 | 3.6 | 0.091 | 0.733 | PASS |
| auto_grown_20260902140837335 | 8.733 | 44.6 | 0.041 | 0.715 | fail (ok) |
| auto_grown_20260902140838462 | 8.194 | 25.1 | 0.116 | 0.695 | PASS |
| auto_grown_20260902140841364 | 7.588 | 2.2 | 0.183 | 0.693 | PASS |
| auto_grown_20260902140842483 | 8.258 | 10.5 | 0.048 | 0.686 | PASS |
| auto_grown_20260902140845521 | 8.771 | 5.0 | 0.009 | 0.67 | PASS |
| auto_grown_20260902140846561 | 7.86 | 0.9 | 0.228 | 0.667 | PASS |
| auto_grown_20260902140849408 | 7.725 | 2.2 | 0.338 | 0.654 | PASS |
| auto_grown_20260902140850514 | 7.596 | 2.3 | 0.238 | 0.643 | PASS |
| auto_grown_20260902140853399 | 8.498 | 9.9 | 0.17 | 0.638 | PASS |
| auto_grown_20260902140854546 | 8.587 | 10.1 | 0.123 | 0.614 | PASS |
| auto_grown_20260902140858608 | 9.214 | 4.8 | 0.007 | 0.577 | PASS |
| auto_grown_20260902140900360 | 7.614 | 4.2 | 0.205 | 0.576 | PASS |
| auto_grown_20260902140909085 | 8.718 | 17.3 | 0.472 | 0.548 | PASS |
| auto_grown_20260902140909884 | 7.591 | 4.0 | 0.25 | 0.532 | PASS |
| auto_grown_20260902140913191 | 8.368 | 10.3 | 0.505 | 0.53 | PASS |
| auto_grown_20260902140917784 | 8.507 | 8.1 | 0.279 | 0.511 | PASS |
| auto_grown_20260902140919788 | 9.581 | 52.1 | 0.325 | 0.478 | fail (ok) |
| auto_grown_20260902140921494 | 8.581 | 7.8 | 0.442 | 0.398 | PASS |
| auto_grown_20260902140924503 | 8.385 | 6.4 | 0.228 | 0.328 | PASS |
| auto_grown_20260902140928704 | 8.303 | 3.6 | 0.307 | 0.301 | PASS |
