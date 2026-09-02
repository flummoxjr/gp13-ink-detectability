# alignment_gate

A standalone quality gate for scroll surface patches: **does this tifxyz mesh
actually follow the lamellae it sits in?**

It takes a tifxyz mesh directory and the CT volume it was traced in, measures
the angle between the mesh's surface normal and the local sheet (lamella)
normal of the material at the patch site, and returns a per-patch angle plus a
verdict. No dependency on any project layout — just `numpy`, `scipy`,
`tifffile` (and optionally `zarr`).

## Why this gate exists

A surface can look completely healthy by every ordinary check — right surface
area, ~100% valid vertices, non-zero intensity sampled everywhere — **while
tracking no lamella at all**. A mesh lying oblique to the papyrus sheets
samples *across* them, so its depth profile averages the lamella modulation
away, and everything downstream (flattening, ink detection) is built on a
surface that does not correspond to a physical sheet.

The case that motivated the tool: **PHerc0813, 8 auto-grown patches** (grown
without a direction field). All 8 passed the healthy-looking checks (validity
fill ~0.9999, plausible area, non-zero DN). Their mesh-vs-sheet angles:
median **68.1 deg**, **0/8 within 30 deg** — *worse than random* (a random
orientation has median 60 deg). All 8 rejected. Their depth-profile lamella
modulation was 0.037–0.073 against 0.443 for an aligned control — exactly the
signature of sampling across the sheets. This failure mode is invisible to
area, validity and intensity checks; only the angle exposes it.

## Calibrated reference

Measured on real data with this exact method (PHerc0813 / published GP meshes,
2026-08, sentinel-fixed):

| population | median angle | within 30 deg |
|---|---|---|
| published direction-field GP meshes | **13.1 deg** | 7/9 |
| random-orientation null | 60.0 deg | (analytic: P(a<30) = 1−cos30 ≈ 13%) |
| failed auto-grown patches | 68.1 deg | 0/8 |

**Gate: REJECT when the angle exceeds 30 deg.** That threshold sits far above
the good population's median (13.1) and far below both the random null (60)
and the failed population (68.1); it separates the two measured populations
with no overlap on the good side (the two published rejects at 47.5/59.2 deg
are genuine misses of that pipeline, not gate noise).

**What a REJECT means:** the surface is not tracking a lamella at its site.
Its depth profile will show little or no sheet modulation, and texture/ink
signal extracted along it is geometrically meaningless. Reject > 60 deg means
the mesh cuts across the sheets nearly perpendicular — the patch grew through
the sheet stack, not along a sheet.

**What the gate does NOT check:** gap spacing, coverage, self-intersection,
ink. It is one necessary condition, not a full validation.

## Method (changing it invalidates the calibration numbers)

- **Mesh normal** — per-vertex cross product of the tifxyz grid tangents
  (`np.gradient` along the two grid axes, in (z,y,x) voxel space), axially
  averaged as the leading eigenvector of the orientation tensor (a normal's
  sign is arbitrary, so arithmetic means are wrong).
  - **Sentinel fix (critical):** tifxyz marks invalid vertices with **-1, not
    0**. Validity is `(x>=0)&(y>=0)&(z>=0)` minus exact (0,0,0). A mask on
    zeros matches nothing, so -1 sentinels get treated as real coordinates and
    their boundary gradients (thousands of voxels long) skew the average. The
    published GP meshes are ~50% invalid vertices and ours ~4%, so the old bug
    biased the two populations by very different amounts — this fix is what
    produced the corrected 68.1/13.1 pair.
  - **One-vertex erosion:** a central difference at p needs p's neighbours, so
    the valid mask is eroded by one vertex (3x3) before any vertex may
    contribute; nothing adjacent to a sentinel enters the average.
- **Sheet normal** — structure tensor per 32^3 block over a 256^3 window of
  the volume centred on the patch site (gaussian sigma 1.0, `np.gradient`
  derivatives). Blocks with <98% non-zero voxels are dropped (air/padding);
  per-block leading eigenvectors are coherence-weighted
  ((λ2−λ1)/(λ2+λ1)) and axially averaged; at least 8 usable blocks required,
  else INDETERMINATE.
- **Angle** — `arccos(|dot|)` folded to [0, 90] deg.

This implementation reproduces the calibration artifacts **exactly**: on the
8 PHerc0813 meshes and the 9 published meshes (with their calibration cubes),
per-mesh angles match the recorded values to 0.0000 deg (medians 68.07 /
13.11). An independent re-derivation with different numerics (forward
differences + Sobel structure tensor) agrees with those medians to within
~1 deg (67.1 / 12.4).

## Usage

```
python alignment_gate.py --mesh PATH/TO/tifxyz_dir --volume PATH/TO/volume [options]
```

- `--mesh` — tifxyz directory containing `x.tif`, `y.tif`, `z.tif` (2D grids
  of voxel x=column, y=row, z=slice). Repeatable: gate many meshes in one run.
- `--volume` — the CT volume those coordinates index into, axis order (z,y,x):
  - `.npy` 3D array (memory-mapped, only the window is read),
  - a single 3D `.tif`/`.tiff` stack (only the needed pages are read),
  - a directory of 2D `.tif` slices, sorted filename order = z,
  - a `.zarr` array or group (`pip install zarr`; group member via
    `--zarr-subpath`, default `0`).
- `--cube 256` — window side per site (default matches the 256^3 calibration
  cubes; clamped to the volume, INDETERMINATE below 2 blocks).
- `--block 32`, `--sigma 1.0` — structure-tensor parameters (calibrated).
- `--reject-deg 30` — the gate threshold.
- `--split S` — tile the mesh grid SxS and gate every tile at its own site
  (per-tile angles + per-mesh median). Default 1 = one site per mesh, as
  calibrated. Use e.g. `--split 3` for large meshes that might be aligned in
  one region and drifting in another.
- `--out results.json` — full machine-readable output (per-tile angles,
  normals, sites, coherence, calibration constants, params).

Exit codes: **0** all ALIGNED, **1** at least one REJECT, **2** at least one
INDETERMINATE (and no reject), **3** bad input.

Example output:

```
mesh                                       angle        verdict  note
aligned                                      0.0        ALIGNED  mesh follows the sheets
oblique45                                   45.0         REJECT  oblique to the sheets
orthogonal                                  90.0         REJECT  mesh cuts ACROSS the sheets
```

## Self-test

```
python selftest.py
```

Builds a 200^3 synthetic volume with horizontal lamellae (period 12 voxels)
and four 64x64 meshes of known orientation, then runs the CLI end-to-end:

- `aligned` (z=const plane) → 0.0 deg, ALIGNED
- `aligned_sentinel` (same plane + a border ring and interior blob of -1
  sentinels) → 0.0 deg, ALIGNED — the sentinel-fix regression test; also
  unit-checks that the fixed mask excludes exactly the -1 vertices (the old
  zero-mask keeps all of them) and that erosion bars sentinel-adjacent
  vertices from contributing
- `oblique45` (45-deg tilted plane) → 45.0 deg, REJECT
- `orthogonal` (plane containing the z axis) → 90.0 deg, REJECT

Exit 0 = all assertions passed.

## Interpreting borderline results

- 13 deg is the *median* of good meshes; individual good meshes ranged 4.4 to
  23.8 deg (plus two genuine rejects). An angle of 25–30 deg is worth a look
  but passes.
- Low `mean_coherence` (< ~0.1) in the JSON means the local material has weak
  sheet structure; the sheet normal is then poorly defined and the angle noisy
  — prefer INDETERMINATE-style caution over trusting a single tile. Gate with
  `--split` to see whether the mesh is consistently misaligned or just locally
  ambiguous.
- The gate measures orientation *at the patch site(s)*. A mesh that is aligned
  at its centroid but wanders elsewhere needs `--split` > 1 to be caught.
