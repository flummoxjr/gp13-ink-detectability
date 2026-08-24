# atlas_1203 — ink3d response atlas of the PHerc1203 2.4 µm band

**Files:** `atlas_1203.npy` (float32, shape (60, 104, 104), axis order z,y,x) and
`atlas_1203.png` (montage of the 14 scored z-slabs).
One cell = one 256³ tile of the L0 CT grid = 0.615 mm; cell (i,j,k) covers L0
voxels [256i:256i+256, 256j:..., 256k:...] of
`PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr`.
Value = **f05**: the fraction of tile voxels that the Paris4-trained
`scrollprize/ink_3d_dino_guided` checkpoint (v3-78k-fullsup) scored P(ink) > 0.5.
NaN = not scored (the screening fleet was stopped at ~45% coverage; 29,748 scored
tiles in 14 z-slabs, all at z ≤ 5.9 mm of the 14.8 mm band).

## What this measures

The response of an **out-of-domain ink model** — i.e. how much each 0.6 mm cell
of PHerc1203 *texturally resembles Paris 4 ink* to this checkpoint. On this
volume that resemblance is a **material-condition read-out**, verified against
independent proxies with block-permutation nulls (n = 29,748, 200 perms,
p < 0.005 throughout):

- ρ(f05, mean CT density) = **+0.73** — the dominant driver
- ρ(f05, CT texture std) = **−0.40** (−0.54 interior-only) — *quieter* where cracked/damaged
- ρ(f05, m7 surface-mask density) = +0.52; ρ(f05, sheet recovery per material voxel) = +0.49

Dark cells = dense, homogeneous, well-preserved, well-segmentable papyrus.
Light cells inside the mask = sparse, cracked, damaged material or the scroll rim.

## What this does NOT measure

- **Not ink.** Zero of 29,748 tiles are silent (in-domain behavior is mostly-silent);
  firing is sheet-conformal membrane painting, not letter-shaped (see qc_live round-1
  verdict and morph_report.md). No threshold on this field ranks tiles for text.
- **Not damage density.** The sign is *inverted*: damage lowers f05. Anyone wanting a
  crack/damage atlas should use CT texture statistics directly.
- **Not a substitute for free CT statistics.** Mean CT computed from the public zarr
  pyramid in seconds predicts m7 segmentation holes *better* than f05 (AUC 0.892 vs
  0.855), and f05 adds zero incremental information once cheap CT covariates are
  controlled (partial ρ = −0.001 ± 0.010 null; leave-slab-out ΔR² = 0.0000).

Keep this file as (a) documentation of ink3d's cross-scroll failure mode and
(b) the fixed before/after baseline for any future 1203-fine-tuned ink model —
not as a condition-assessment tool.
