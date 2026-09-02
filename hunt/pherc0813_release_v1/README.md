# PHerc0813 — GrowPatch surfaces, release v1 (2026-09-02)

18 of 24 grown patches pass the alignment gate (< 30° to the local lamellae at the seed): **154.63 cm²** of correctly-oriented surface on a scroll with no published segmentation.

- Volume: `20250821151723-9.362um-1.2m-113keV-masked.zarr` (9.362 µm); surface prediction: `20250821151723-surface-20260413222639-surface-m7-L0-th0.2` + its released normal grids.
- Tracer: vc_grow_seg_from_seed, villa main (release vc-tracer-de3c2494; built from e2442b7). Params: seed mode, 75 generations, min_area 0.3 cm², single-threaded per seed.
- Seeds: the scroll's 24 separability-ranked ROIs (structure-tensor planarity), each moved to the nearest m7 sheet voxel
  (support gate against the ~49 % phantom rate of m7 positives beyond CT support).
- Gate: mesh normal vs local sheet normal from a 256³ CT cube at the seed; published GP meshes sit at 13.1° median.
- Format: tifxyz (`x.tif`, `y.tif`, `z.tif`, `generations.tif`, `meta.json`), coordinates in the volume's voxel units.
- Known caveat: every tracer mesh stores `scale` as float32 0.05 → villa's `Tifxyz.shape` gives a canvas 1 px short
  (`issue_drafts/filing/tifxyz_fullres_shape_truncation.md`); render onto `round(h/scale)`.

| patch | area cm² | angle ° | abs n_z | separability | PASS |
|---|---|---|---|---|---|
| auto_grown_20260902135206580 | 8.978 | 77.1 | 0.335 | - | fail (ok) |
| auto_grown_20260902135207525 | 8.97 | 24.9 | 0.342 | - | PASS |
| auto_grown_20260902135210623 | 8.537 | 11.8 | 0.271 | - | PASS |
| auto_grown_20260902135211606 | 8.438 | 4.5 | 0.139 | - | PASS |
| auto_grown_20260902135214612 | 9.788 | 47.6 | 0.214 | - | fail (ok) |
| auto_grown_20260902135215584 | 8.133 | 3.6 | 0.156 | - | PASS |
| auto_grown_20260902135218636 | 8.333 | 2.9 | 0.267 | - | PASS |
| auto_grown_20260902135219602 | 8.568 | 4.5 | 0.11 | - | PASS |
| auto_grown_20260902135222556 | 9.541 | 32.6 | 0.313 | 0.75 | fail (ok) |
| auto_grown_20260902135223975 | 8.338 | 1.8 | 0.24 | 0.747 | PASS |
| auto_grown_20260902135226709 | 8.474 | 2.9 | 0.282 | 0.743 | PASS |
| auto_grown_20260902135229488 | 9.662 | 2.8 | 0.012 | 0.74 | PASS |
| auto_grown_20260902135234699 | 8.175 | 11.7 | 0.598 | 0.735 | PASS |
| auto_grown_20260902135241382 | 8.216 | 3.5 | 0.38 | 0.731 | PASS |
| auto_grown_20260902135242982 | 8.203 | 3.2 | 0.255 | 0.73 | PASS |
| auto_grown_20260902135246304 | 8.83 | 48.8 | 0.118 | 0.677 | fail (ok) |
| auto_grown_20260902135249482 | 8.895 | 24.4 | 0.224 | 0.672 | PASS |
| auto_grown_20260902135251304 | 8.721 | 51.5 | 0.172 | 0.671 | fail (ok) |
| auto_grown_20260902135253399 | 8.818 | 17.1 | 0.002 | 0.671 | PASS |
| auto_grown_20260902135257383 | 7.999 | 0.4 | 0.057 | 0.664 | PASS |
| auto_grown_20260902135258104 | 8.542 | 10.9 | 0.333 | 0.659 | PASS |
| auto_grown_20260902135302193 | 8.766 | 82.5 | 0.025 | 0.654 | fail (ok) |
| auto_grown_20260902135302282 | 9.374 | 15.2 | 0.241 | 0.654 | PASS |
| auto_grown_20260902135304797 | 8.954 | 16.3 | 0.079 | 0.615 | PASS |
