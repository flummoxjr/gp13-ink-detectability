# Investigation E — grow segments where the signal is

**Verdict: LIKELY→PROVEN-ON-PAPER. No blocker found.** Every input needed to grow
patches on PHerc0813 is already published on the open bucket, the tool that
produced all 80 segments we screened is a headless CPU-only CLI, a prebuilt
Linux image containing that binary is anonymously pullable, and the binary reads
volumes straight from `s3://` with no volpkg and no local copy. Measured growth
cost on the published corpus is **0.9–1.9 cm²/min on one CPU thread**.

Everything below was verified today against the live bucket and the `cycle2`
checkout at `C:\Users\benbl\Desktop\Vsuvious\villa`. Nothing here has been *run*
yet — see [§9 Honest status](#9-honest-status).

---

## 1. The gap, confirmed on S3

Our index (`trackD/report/sections/01_index.md`) ranks the 9.362 µm/113 keV GP
volumes by mid-band structural SNR. Cross-referencing that ranking against what
actually exists on `s3://vesuvius-challenge-open-data/`:

| # | Scroll | SNR @0.25 | `segments/` on S3? | Screened? |
|---|--------|-----------|--------------------|-----------|
| 1 | **PHerc0813** | **159.6** | **no** | **no** |
| 2 | PHerc0139 (calibrator) | 115.5 | yes | control, reads letters |
| 3 | **PHerc0125** | **114.2** | **no** | **no** |
| 4 | **PHerc1545** | **112.2** | **no** | **no** |
| 5 | **PHerc0211** | **106.6** | **no** | **no** |
| 6 | **PHerc0191** | **99.6** | **no** | **no** |
| 7 | **PHerc0358** | **91.8** | **no** | **no** |
| 8 | PHerc1203 | 87.2 | yes (22) | yes — 0 hits |
| 13 | PHerc0800 | 20.1 | yes (6) | yes — 0 hits |
| 14 | PHerc1447 | 8.5 | yes (52) | yes — 0 hits |

Verification: listing `<Scroll>/` with `delimiter=/` returns a `segments/`
prefix for **exactly** PHerc1203, PHerc1447, PHerc0800 and PHerc0139 — the three
we screened plus the control. The six top-ranked scrolls return only
`photos/ representations/ volumes/`. There is no mesh on any of them.

So the corpus we screened is rank 8, 13 and 14. **Ranks 1, 3, 4, 5, 6 and 7 have
never been looked at by anyone, because there is no geometry to render.**

PHerc0813 is also in the *same scan campaign as the calibrator* — 9.362 µm /
113 keV, same beamline (BM18), same reconstruction — so `ink_9um` faces no
domain shift there. It out-scores the calibrator on our own index.

---

## 2. What actually grows a patch

`volume-cartographer/apps/src/vc_grow_seg_from_seed.cpp` (1535 lines). This is
not a guess: every published segment's `meta.json` carries
`"source": "vc_grow_seg_from_seed"`, and the directory prefix `auto_grown_` on
S3 is literally `name_prefix` at line 359 of that file.

Per `docs/tracing.md`, it "optimize[s] a surface from a **thresholded surface
prediction**". Line 578 confirms the acceptance test is `v >= 128` — i.e. the
`--volume` argument is the **m7 surface prediction**, not the CT. The CT is only
needed later, at render time.

```
--volume,-v      OME-Zarr surface-prediction path  (local OR http(s):// OR s3://)
--target-dir,-t  output dir; writes <target>/auto_grown_<ts>/{x,y,z}.tif + meta.json
--params,-p      JSON parameter file
--seed,-s        x y z   (optional — omit and it finds its own)
--segment-name   write straight into target-dir instead of a timestamp subdir
```

Mode is derived, not declared: pass `--seed` → `explicit_seed`; omit it →
`random_seed`, where the tool rejection-samples the volume for an on-sheet point
and calls `check_existing_segments()` so parallel processes against one
`--target-dir` tile the scroll instead of colliding. That is how the 1203 corpus
was made.

Output is **tifxyz** — `x.tif`/`y.tif`/`z.tif` + `meta.json` — which is exactly
what our validated `trackD/runpod/render_tifxyz_sv.py` already consumes. The
chain closes with no new format work.

---

## 3. Every required input is already published

Checked for PHerc0813 (and the same layout exists for 0125/1545/0211/0191/0358,
all with the identical `-surface-20260413222639-surface-m7-L0-th0.2` suffix):

```
PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr/      <- CT, render only
PHerc0813/representations/predictions/surfaces/
    20250821151723-surface-…-m7-L0-th0.2.zarr/          <- TRACER INPUT
    20250821151723-surface-…-m7-L0-th0.2.normal-grids/  <- optional quality boost
```

- Surface prediction: `shape [16993, 7947, 7947]`, `uint8`, chunks `192³`,
  blosc/zstd. Strictly binary — `nonzero frac == (>=128) frac == 0.0968`.
  Interior chunks are ~0.4–0.7 MB compressed (~14× ratio).
- **Normal grids are already computed and released.** This is the one
  prerequisite the segmentation tutorial warns about ("REQUIRED!", normally
  built with `vc_gen_normalgrids`). The released tree is `xy/ xz/ yz/` +
  `metadata.json`, files named `%06d.grid` — an exact match for
  `NormalGridVolume::pimpl::plane_dirs` and `relative_grid_path()`. Nothing to
  compute.
- `normal_grid_path` is **optional** in the tracer (`GrowPatch.cpp:3432` guards
  it with `params.contains(...)`), and only 13/36 published segments used one.
- Structure-tensor `direction_fields` (used by 11/36 published segments) are
  **not** on the open bucket for any scroll, 1203 included — that was an
  internal `/volpkgs/...` asset. Also optional. We proceed without it.

### Sheet quality — the thing that actually decides whether tracing works

Pulled a real 192×768×768 L0 slab from each scroll's prediction at a
central, well-conditioned ROI and measured sheet thickness / gap along x:

| Scroll | sheet thick (vox) | gap (vox) | period | fetch |
|--------|-------------------|-----------|--------|-------|
| **PHerc0813** | 8 | 20 | **262 µm** | 3.3 s |
| PHerc0125 | 12 | 34 | 431 µm | 3.4 s |
| PHerc1545 | 14 | 32 | 431 µm | 2.7 s |
| PHerc0211 | 5 | 17.5 | 211 µm | 1.8 s |
| PHerc0191 | 8 | 12 | 187 µm | 3.7 s |
| PHerc0358 | 5 | 12 | 159 µm | 2.2 s |
| PHerc1203 *(traceable, 22 meshes)* | 10 | 35 | 421 µm | 1.4 s |

PHerc0813's sheets are cleanly separated (8 on, 20 off) and in the same class as
PHerc1203, which demonstrably traces. This is the strongest single piece of
evidence that growth will succeed there.

---

## 4. Build: **don't**. A prebuilt image exists.

`ghcr.io/scrollprize/villa/volume-cartographer:edge` — verified live today:

```
GET https://ghcr.io/v2/scrollprize/villa/volume-cartographer/tags/list
  -> ["main","edge","buildcache","builder-ubuntu-24.04",…,"builder-ubuntu-26.04-…"]
manifest edge -> HTTP 200 with an ANONYMOUS token (no GitHub login needed)
amd64: 41 layers, 2.76 GB compressed
```

The `Dockerfile`'s `runtime` stage runs
`cmake --install … --component vc_runtime`, and the install walk
(`CMakeLists.txt:775-808`) installs **every** `EXECUTABLE` target in `apps` to
`<prefix>/bin`. So `/usr/local/bin/vc_grow_seg_from_seed` is in that image,
alongside `vc_render_tifxyz`, `vc_tifxyz2obj`, etc., with internal `.so`s
resolved by `$ORIGIN/../lib`.

**Headless is fine.** `vc_grow_seg_from_seed` links only
`vc_core vc_tracer Boost::program_options`; `vc_core` links no Qt at all
(opencv, nlohmann, OpenMP, TIFF, OpenABF, libcurl). No X11, no `DISPLAY`, no
`QT_QPA_PLATFORM`. Those only matter for the `VC3D` GUI, which we never launch.

**CPU only.** `use_cuda` reaches `g_use_cuda`, whose only consumer in
`GrowPatch.cpp:2986-2996` is commented out. Production confirms it:
every published `meta.json` carries `"use_cuda": false`.

### If you must build from source (fallback)

`scripts/install_build_deps.sh` + `cmake --preset ci-release-gcc`. Two gotchas:
the dep list pins `flang-21` / `libclang-rt-21-dev`, which **only resolve on
Ubuntu 26.04** — a stock RunPod `ubuntu:22.04`/CUDA image will fail here; and
`apps/CMakeLists.txt` unconditionally does `add_subdirectory(VC3D)`, so
`qt6-base-dev` is needed to *build* even though it isn't needed to *run*.
Budget ~40–60 min on 16 cores. Use the prebuilt image instead.

---

## 5. No volpkg needed — it reads S3 directly

`vc_grow_seg_from_seed.cpp:34-38` detects `http://`, `https://`, `s3://` and
routes to `Volume::NewFromUrl`, which (`Volume.cpp:1338-1465`):

- opens the zarr pyramid over HTTP and **synthesizes** the metadata
  (`width/height/slices/format/uuid`) from `.zarray`, so no `meta.json` is
  required — the released prediction zarrs only ship `metadata.json`;
- explicitly **falls back to anonymous** when AWS creds are absent or stale
  (`auth = {}; // anonymous — no SigV4`), so the public bucket works with no
  credentials at all;
- takes `voxelsize` from the params JSON as an override
  (`hasExplicitVoxelSizeOverride_`) — which is why we must pass it.

So: no `.volpkg`, no `config.json`, no download of a 1 TB volume. The
`normal_grids` store streams too, via a `normal-grids-remote.json` marker file
(`NormalGridVolume.hpp:18`) containing `{"url": "..."}`, which the loader reads
and then lazily fetches + caches `%06d.grid` files behind 4 prefetch threads.

---

## 6. Measured cost

Pulled `meta.json` for all 80 catalogued segments; 36 carry timing:

| Scroll | n | area (med) | elapsed (med) | **cm²/min** |
|--------|---|-----------|---------------|-------------|
| PHerc1203 (SNR 87) | 22 | 3.89 cm² | 121 s | **1.93** |
| PHerc1447 (SNR 8.5) | 14 | 3.90 cm² | 701 s | **0.25** |
| corpus | 36 | — | 209 CPU-min total, 188 cm² | 0.90 |

Single-thread (`thread_limit: 1`), CPU, area range 2.1–16.1 cm², generations
65–200. **Tracer speed tracks scan quality 7.7×** — 1447 (worst scan in the
index) is 7.7× slower per cm² than 1203. PHerc0813 has better-separated sheets
than 1203, so ≥ 1.9 cm²/min is the reasonable expectation.

Do **not** just throw threads at it. `vc_grow_seg_from_seed.cpp:578-588`:
wall-time per cm² is lowest at **4 threads** on both machines the devs measured,
and unbounded costs **2.2×–3.5×** more. Set `thread_limit: 1` and run N
processes.

**Target 10–30 cm² on PHerc0813 = 3–8 patches ≈ 6–16 CPU-minutes single-threaded**,
or well under a minute wall-clock with 16 parallel processes. Growth is not the
cost centre. The cost centre is the downstream render + `ink_9um` inference we
already know how to price.

### Budget

| Stage | Pod | Time | $ |
|-------|-----|------|---|
| pull image (2.76 GB) | any | 3–6 min | — |
| grow ~8 patches, 0813 | CPU pod (16 vCPU) | ~10 min | ~$0.10 |
| render 21-slice SVs (S3 CT) | reuse GPU pod | ~20–40 min | ~$0.35 |
| `ink_9um` + verdict battery | 5090 @ $0.69/hr | ~30 min | ~$0.35 |
| **first full PHerc0813 pass** | | **~1.5 h** | **≈ $1** |

All six unscreened scrolls, ~10 patches each: **≈ $6–10** of the ~$45 left.
The 80-segment corpus pass cost ~$5, so this is the same order.

Our `trackD/runpod/rp.py:56` already parameterises `imageName`, so launching a
pod on the VC3D image is a one-argument change to `fleet_launch.py`.

---

## 7. Run recipe

**Stage A — grow (CPU pod, image `ghcr.io/scrollprize/villa/volume-cartographer:edge`)**

```bash
S=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0813/representations/predictions/surfaces
B=20250821151723-surface-20260413222639-surface-m7-L0-th0.2

mkdir -p /work/paths /work/cache /work/ngrid
# stream the released normal grids instead of downloading them
printf '{"url":"%s/%s.normal-grids"}' "$S" "$B" > /work/ngrid/normal-grids-remote.json

cat > /work/seed.json <<'JSON'
{
  "mode": "seed",
  "generations": 75,
  "step_size": 20,
  "min_area_cm": 0.3,
  "thread_limit": 1,
  "use_cuda": false,
  "voxelsize": 9.362,
  "cache_size": 4e9,
  "cache_root": "/work/cache",
  "normal_grid_path": "/work/ngrid"
}
JSON

# 8 parallel single-threaded random-seed workers; they de-conflict via
# check_existing_segments() against the shared --target-dir
for i in $(seq 8); do
  vc_grow_seg_from_seed \
    --volume "$S/$B.zarr" \
    --target-dir /work/paths \
    --params /work/seed.json \
    --skip-overlap-check &
done; wait
```

Drop `--skip-overlap-check` if you want the overlap bookkeeping. To force a
location instead of random sampling, add `--seed <x> <y> <z>`.

**Verified on-sheet seeds for PHerc0813** (computed today from the L0 prediction;
each sits on a sheet inside a 24³ neighbourhood whose sheet fraction is in the
healthy 0.18–0.42 band — full list in the scratchpad `seeds0813.json`):

```
--seed 3968 3970 14878   (sheet_frac 0.384)
--seed 5761 4513 11744   (0.419)
--seed 4512 4511 11744   (0.390)
--seed 1664 4519  4705   (0.361)
--seed 2494 4512  4705   (0.257)
--seed 5408 4480  4704   (0.196)
--seed 4160 1592 14973   (0.206)
--seed 3328 1594 14980   (0.180)
```

Sanity gates before spending anything downstream: the tool prints
`zarr dataset size for scale group 0 [16993, 7947, 7947]` (proves S3 + anonymous
auth work), `Loaded normal grid level …` (proves remote grid streaming works),
and `seed location … value is 255` (proves the seed is on a sheet). Then check
`area_cm2` and `elapsed_time_s` in each output `meta.json`.

**Stage B — render + score (existing validated pipeline, GPU pod)**

Copy the tifxyz dirs across — a whole segment is **~280 KB**
(`x/y/z.tif` ≈ 92 KB each), so this is a nothing transfer.

```bash
python render_tifxyz_sv.py /work/paths/auto_grown_<ts> \
  s3://vesuvius-challenge-open-data/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr \
  /out/sv_<ts>.zarr --num-slices 21
# then ink_9um + salvage/verdict_*.py exactly as in the 80-segment corpus pass
```

Note this is where the **CT** volume finally enters. Keep the two zarr paths
straight: prediction for Stage A, CT for Stage B.

---

## 8. Route (3): building a surface without VC3D

Asked whether the m7 field alone suffices to fit a renderable sheet in Python.
**Technically yes, and we have the pieces** — `vesuvius.tifxyz.write_tifxyz()`
exists, so we could threshold → connected-component from a seed → fit/parameterise
→ write `x/y/z.tif` → feed our existing renderer.

**But it is the wrong call, and I recommend against it.** The reasons are
concrete, not aesthetic:

- It reimplements the one component the villa team has iterated on for years.
  `GrowPatch.cpp` is ~4500 lines of Ceres losses whose entire job is refusing to
  leak between adjacent sheets where the prediction merges — with a 20-voxel gap
  and 8-voxel sheets, merges are exactly what a naive CC will do.
- A hand-fit patch has no guarantee of **metrically correct arclength
  parameterisation**. Our ink instrument was validated at AUC 0.9991 on a
  *tracer-produced* tifxyz; feeding it a distorted grid silently invalidates the
  calibration, and we would not be able to tell a bad parameterisation from
  absent ink. Given this project has already refuted three of its own false
  positives, adding an uncalibrated geometry step is the exact wrong risk.
- It buys nothing. The real tool is a 2.76 GB `docker pull` and costs ~$0.10 per
  scroll to run.

Keep it filed as a fallback **only** if the binary turns out not to run on a pod
— it does not currently look like it will come to that.

---

## 9. Honest status

**What is proven:** the assets exist and are complete (listed and fetched); the
predictions are binary and well-resolved (measured); the image exists and pulls
anonymously (HTTP 200); the binary is Qt-free, CUDA-free and S3-capable (read in
source); growth cost is 0.9–1.9 cm²/min (measured from 36 published `meta.json`).

**What is NOT yet demonstrated:** *we have not run it.* There is no Docker or
working WSL on this Windows box (`docker: command not found`; `wsl --status` →
`REGDB_E_CLASSNOTREG`), so the first actual execution must happen on a pod. The
residual risks, in order:

1. **Untested combination.** Remote-S3 volume + remote normal-grid marker +
   `random_seed` is a path each of whose parts is supported in source, but which
   this codebase's own agent-bridge notes flag as awkward for seeding
   (`SKILL.md` §9 warns `normal_grid_path` is "a static local path, with no
   awareness of remote/streaming normal-grid stores" *when driven through the
   VC3D GUI*). The CLI + marker-file route sidesteps that, but it is unproven.
   Mitigation: `normal_grid_path` is optional — drop it and grow anyway.
2. **Growth could stall or produce junk** on a scroll nobody has traced. The
   `min_area_cm` gate discards sub-0.3 cm² patches, so failure is likely to look
   like empty output rather than bad output.
3. **Finding geometry ≠ finding ink.** This route fixes *where we look*. It does
   nothing about the symmetry symptom (forward-vs-reversed r = 0.22–0.91 on
   every GP segment vs 0.076 on the control), which remains unexplained and is
   an independent reason the corpus screen may have been reading the wrong
   surface. Growing fresh patches is arguably a *test* of that hypothesis — if
   the auto-grown meshes are systematically off the recto, our own fresh ones
   will reproduce the same symmetry, which would be informative either way.

**Next action:** one CPU pod on the VC3D image, PHerc0813, 8 workers, ~10 min,
~$0.10. Success criterion: ≥ 3 patches with `area_cm2 ≥ 2`. Only then spend
anything on rendering.
