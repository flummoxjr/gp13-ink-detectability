# ink_9um labeled training corpus — verified download locations

Scouted 2026-08-16. All HTTP statuses below were actually observed with curl (anonymous, no auth).

## TL;DR

The corpus is split in two:

- **Labels** (inklabels / supervision / validation zarrs): Hugging Face **Bucket**
  `scrollprize/datasets`, prefix `ink_9um/` — public, anonymous, tiny (~0.7 MB per segment).
  Not a normal HF dataset repo; the scrollprize org page shows "no datasets" — the data is
  at `https://huggingface.co/buckets/scrollprize/datasets/tree/ink_9um`.
- **Surface volumes** (CT data): the public S3 bucket `vesuvius-challenge-open-data`
  (anonymous), per-segment under `<Scroll>/segments/<segment-id>/surface-volumes/`.
  The bucket README states explicitly: "Labels only — no CT data; surface volumes come
  from the open-data server."

For the **native 9.362 µm PHerc0139 segments (incl. w035)** the S3 surface volume is used
**as-is** and its array shape exactly matches the labels — no preprocessing needed.
For the **aligned ~9.6 µm corpus** you must locally prepare a 21-slice surface volume from
the public 2.399 µm zarr (`prepare_9um_isotropic_input`, level 2, 4x z-mean).

## Verified URLs (with observed HTTP status)

### w035 — native 9.362 µm (THE prime target)

| Item | URL | Status |
|---|---|---|
| Surface volume `.zgroup` | `https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/.zgroup` | 200 |
| Surface volume `.zattrs` (OME multiscale, scale 9.362 µm) | same base + `/.zattrs` | 200 |
| Surface volume `0/.zarray` | same base + `/0/.zarray` | 200 |
| Surface volume data chunk (ranged GET) | same base + `/0/0/20/20` (bytes 0–99) | **206** |
| Ink labels `0/.zarray` | `https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/w035/w035_inklabels.zarr/0/.zarray` | 200 (via 302 to `us.aws.cdn.hf.co`) |
| Ink labels data chunk | `.../w035_inklabels.zarr/0/0.20.20` | 200 |
| Supervision mask `0/.zarray` | `.../w035_supervision_mask.zarr/0/.zarray` | 200 |

Key metadata (fetched):

- Surface volume `0`: `shape [28, 5820, 5240]`, `chunks [28,128,128]`, dtype `|u1`,
  **compressor null** (raw chunks), `dimension_separator "/"` (chunk keys like `0/0/y/x`).
  OME-Zarr group with pyramid levels 0,1,... at 9.362 / 18.724 / ... µm.
  Total size at S3: **2223 objects, ~1016 MB** (whole zarr incl. pyramid).
- Labels `w035_inklabels.zarr`: zarr-2 group, level `0` array `shape [28, 5820, 5240]`,
  `chunks [28,128,128]`, `|u1`, blosc/zstd, `.`-separated chunk files inside `0/`.
  **Shapes match the surface volume exactly.** 28 slices deep, annotated only at Z=14.
  Whole w035 label folder: **5128 files, ~0.74 MB** (summed via bucket API).

### Other native 9.362 µm labeled segments (PHerc0139) — all verified 200 on `0/.zarray`

| Corpus name | S3 segment dir | 9.362 µm surface volume (all: `9.362um-1.2m-113keV-volume-20250728140407.zarr`) |
|---|---|---|
| w035 | `20260317000000-w035_2026031718` | 200 |
| w039 | `20260302000000-w039_2026030210` | 200 |
| w040 | `20250831000000-w040_2025083102` | 200 |
| w041 | `20260108000000-w041_2026010816` | 200 |
| w044 | `20260115000000-w044_2026011522` | 200 |

Labels for each: `https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/<seg>/<seg>_inklabels.zarr/...` (+ `_supervision_mask`).

### Aligned ~9.6 µm corpus (24 segments) — spot-verified 200

- `pherc0139-w035` inklabels `0/.zarray` — 200
  `https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-w035/pherc0139-w035_inklabels.zarr/0/.zarray`
- `pherc1667-w029` inklabels `0/.zarray` — 200; its `_validation_mask.zarr/.zgroup` — 200
- `pherc0814-46527` inklabels `0/.zarray` — 200
- w035's aligned source volume `2.399um-0.22m-78keV-volume-20260102150214.zarr/.zattrs` (same S3 segment dir) — 200

Aligned label arrays are 21 slices deep, annotated only at Z=10 (other planes zero).
Validation masks exist for exactly 3 segments: `pherc0139-w016`, `pherc0814-46527`, `pherc1667-w029`
(these are the online-validation cases the released ink_9um checkpoints report metrics on).

## Corpus manifest (29 representations, matches villa config `aligned21_hybrid_3d2d.json`)

- `aligned-scrollprizeorg-21slices/` (24): pherc0139-w016 w017 w028 w029 **w035** w039 w040 w041 w043;
  pherc0814-46527; pherc1667-w013 w018 w023 w028 w029 w031; phercparis4-w00 w01 w02 w03 w05 w06 w07 w09
- `native9-scrollprizeorg-21slices/` (5): **w035** w039 w040 w041 w044

The bucket README (`.../resolve/ink_9um/README.md`, 10.7 kB, saved to scratchpad) carries the
full segment → public-source-volume mapping table. Critical gotcha it states: **corpus segment
names do NOT track public w-numbering** — e.g. `pherc0139-w016` maps to public segment
`20250108000004-w029`. But `pherc0139-w035` and native `w035` both map to
`20260317000000-w035_2026031718` (w-number happens to agree for w035/w039/w040/w041).

Selected aligned-source mappings (2.399 µm volumes, PHerc0139 all use
`2.399um-0.22m-78keV-volume-20260102150214.zarr`; PHerc1667 uses
`2.399um-...-20251217075048.zarr`; Paris4 `2.4um-...-20260411134726.zarr`;
PHerc0814 `2.399um-...-20260309142202.zarr`):

- pherc0139-w035 → `PHerc0139/segments/20260317000000-w035_2026031718/`
- pherc1667-w029 → `PHerc1667/segments/20251212185248-w029_20251212185248662_flatboi/`
- pherc0814-46527 → `PHerc0814/segments/20260226000000-46527_2um_try2/`
- phercparis4-w00 → `PHercParis4/segments/20231016151002/`

(Full table: scratchpad copy of the bucket README, or re-fetch the resolve URL above.)

## How to download

### Labels (HF bucket)

- Bulk/CLI: `hf buckets sync hf://buckets/scrollprize/datasets/ink_9um/labels ./ink_9um/labels`
  (huggingface_hub CLI; `hf buckets cp` for single files). Per-segment sync works by
  appending the subpath. Labels are tiny — the whole 29-segment label set is likely well
  under 100 MB (w035 native folder measured at 0.74 MB).
- Raw HTTP (verified anonymous): `https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/<path>` → 302 → CDN 200.
- Enumeration API (verified): `https://huggingface.co/api/buckets/scrollprize/datasets/tree/<path>`
  — recursive, JSON, paginated at 1000 entries with a `Link: ...cursor=...; rel="next"` header.
  Rate limit observed: 500 requests / 300 s window ("api" policy); resolvers 3000/300 s.

### Surface volumes (S3, anonymous)

- `aws s3 cp --recursive --no-sign-request s3://vesuvius-challenge-open-data/PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/ ./w035-surface-volume.zarr/`
  (~1.0 GB for w035; similar order for the other native segments) — or rclone, or plain
  HTTPS (`https://vesuvius-challenge-open-data.s3.amazonaws.com/<key>`; ranged GETs work, 206 verified).
- Streaming without download also works — see loader notes below.
- villa also ships `vesuvius.ink_detection.preprocessing.download_required_zarr_chunks`
  (records source + chunk plan in root attrs, resumable) for pulling only needed chunks
  of the big 2.399 µm volumes.

## How villa's ink_detection loader expects data on disk

From `C:\Users\benbl\Desktop\Vsuvious\villa\vesuvius\src\vesuvius\ink_detection\data\segment.py`,
`data\dataset.py`, and `volume_io.py`:

- `gather_segments` **iterates `segments_path` as a local directory** (`Path.iterdir`), so the
  label folders must exist locally, one directory per segment:
  `<segments_path>/<segment_name>/<segment_name>_inklabels.zarr` (+ `_supervision_mask.zarr`,
  optional `_validation_mask.zarr`). Extensions `.zarr` (training) — `.tif/.tiff` recognized by
  the parser too. Versioning: `_v<N>` before the extension; unversioned = v1; highest version
  wins unless `label_version` pins one. A dir named `unused` is skipped.
- The **image/surface volume can be remote**: `surface_volume_paths` values starting with
  `s3://`, `http://`, `https://` skip the local-existence check and are opened via
  fsspec/zarr. `s3://` URLs containing `vesuvius-challenge-open-data` automatically get
  `anon=True` (`volume_io.py:_PUBLIC_S3_VOLUME_SUBSTRING`). HTTPS needs no auth for this
  bucket. Optional compressed-chunk disk cache requires Zarr 3 (`cache_dir`/`cache_max_gb`).
- If neither `surface_volume_paths` nor `surface_volume_path` is given, it falls back to
  `<segment_dir>/<segment_dir_name>.zarr` (note: NOT `surface-volume.zarr` — the shipped
  configs always pass explicit `surface_volume_paths`, whose example values are
  `<segment>/surface-volume.zarr`).
- `volume_scale` (config) selects the pyramid level via `open_volume(...)`; the shipped
  configs use `volume_scale: 0`, which selects array `0` of the OME group — so pointing
  `surface_volume_paths["w035"]` directly at the S3 9.362 µm zarr group root works.
- Flat mode with explicit surface_volume_paths does not need tifxyz; `full_3d*` modes
  additionally require `x.tif` under the segment dir and a `volume_path` per dataset.
- Aligned corpus caveat: the 21-slice `surface-volume.zarr` for the aligned segments is
  **not published anywhere** — it must be generated per segment from the public 2.399 µm
  zarr with `vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input <raw.zarr>
  <out>/surface-volume.zarr --level 2` (writes one `(21,Y,X)` array tagged
  `level2-zmean4-21slice-v1`).

Recommended on-disk shape for a minimal native-9 run (mirrors the shipped config):

```
ink_9um/labels/native9-scrollprizeorg-21slices/
  w035/w035_inklabels.zarr, w035_supervision_mask.zarr   <- hf buckets sync (tiny)
  w039/..., w040/..., w041/..., w044/...
```

with config `surface_volume_paths` pointing at
`s3://vesuvius-challenge-open-data/PHerc0139/segments/<seg-dir>/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr`
(or the https:// equivalent, or a local `aws s3 cp` copy).

## Dead ends / negatives

- HF org `scrollprize` lists **no dataset repos** — `scrollprize/ink_9um` is checkpoints only
  (`hybrid_3d2d-seed42/`, `-seed43/`, 1.94 GB). The data lives in the Buckets product instead.
- `s3://vesuvius-challenge-open-data/PHerc0139/representations/` contains only `predictions/`
  — no ink labels or surface volumes there.
- `dl.ash2txt.org` is up and anonymous (200): `/datasets/` has fiber-skeletons,
  grand-prize-banner-region, seg-derived-recto-surfaces, spiral_datasets — no ink_9um.
  `/community-uploads/bruniss/ink_train_patches/` exists but is 2025-era `auto_grown_*`
  community patches, NOT this corpus.
- Bucket tree page reports 3.68 TB / 11.66 M files — that appears to be a whole-bucket
  (scrollprize `datasets` bucket) statistic, not the ink_9um labels subtree (which is tiny).
- API `tree/` on a nonexistent path returns `[]` (e.g. no `surface-volume.zarr` anywhere in
  the bucket — confirming labels-only).

## Best path (recommended)

1. `hf buckets sync hf://buckets/scrollprize/datasets/ink_9um/labels/native9-scrollprizeorg-21slices ./ink_9um/labels/native9-scrollprizeorg-21slices` (< 5 MB).
2. `aws s3 cp --recursive --no-sign-request` the w035 (+ w039/w040/w041) 9.362 µm surface
   volumes (~1 GB each), or skip the copy and put the `s3://`/`https://` URLs directly in the
   config's `surface_volume_paths` — the loader streams them anonymously.
3. Only if the aligned 24-segment corpus is needed: sync its labels the same way, then run
   `prepare_9um_isotropic_input` per segment against the mapped public 2.399 µm zarrs
   (mapping table in the bucket README / above).
