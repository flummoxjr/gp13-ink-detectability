# Data manifest — what is not in this repository

This repository ships all code, all documentation, all figures, and every small-to-medium
result file (the index, the separability axis, the corpus screen, control scores, mesh QC).

The items below are excluded by `.gitignore` for one of two reasons:

1. **Upstream data** — copies of public scroll data that are not ours to redistribute. This work
   redistributes no bulk scroll data; excluding these keeps that statement literally true.
2. **Large regenerable binaries** — derived products that would make the repository unclonable,
   each rebuildable from code in this repository plus public inputs.

| artifact | size | files | what it is | how to rebuild |
|---|---:|---:|---|---|
| `out/survey/maps_shard0` | 104 MB | 92 | 150 downsampled prediction maps (75 segments x 2 z-directions), shard 0 | `runpod/survey_segments.py  (pod, ~$5.5 for all 80 segments)` |
| `out/survey/maps_shard1` | 29 MB | 37 | prediction maps, shard 1 | `runpod/survey_segments.py` |
| `out/survey/maps_shard2` | 27 MB | 35 | prediction maps, shard 2 | `runpod/survey_segments.py` |
| `out/survey/maps_shard3` | 33 MB | 41 | prediction maps, shard 3 | `runpod/survey_segments.py` |
| `out/ink9um_w035` | 80 MB | 6 | w035 positive-control predictions, forward and reverse, 2 seeds | `runpod/render_tifxyz_sv.py then vesuvius.ink_detection.inference.infer` |
| `out/ink9um_1203` | 72 MB | 8 | PHerc1203 probe predictions | `same chain, PHerc1203 segment` |
| `salvage/tiles.parquet` | 13 MB | 1 | frozen ink_3d benchmark: per-tile scores any 1203-adapted model must beat | `salvage/ pipeline over the ink_3d run` |
| `salvage/proxies.parquet` | 2 MB | 1 | per-tile covariates for the ink_3d transfer analysis | `salvage/ pipeline` |
| `salvage/atlas_1203.npy` | 2 MB | 1 | ink_3d firing atlas for PHerc1203 | `salvage/ pipeline` |
| `out/smoke_1203_prob.npy` | 579 MB | 1 | early PHerc1203 smoke-test probability volume (superseded) | `not needed; kept locally only` |
| `hunt/corpuscache` | 32 MB | 308 | cached upstream scroll meshes (UPSTREAM DATA, not ours to redistribute) | `re-downloaded automatically by hunt/ scripts` |
| `hunt/meshcache` | 5 MB | 41 | cached upstream meshes (UPSTREAM DATA) | `re-downloaded automatically` |
| `out/survey/v2cache` | 22 MB | 3 | intermediate arrays for the v2 corpus screen | `analyze_survey_corpus_v2.py regenerates` |

**Total excluded: 1000 MB.**

## If you want the prediction maps

The 150 saved prediction maps are the most directly reusable product here — they are the raw
material for any re-analysis of the corpus screen without re-running inference. They are excluded
only for size. Open an issue and they can be attached to a GitHub Release, or regenerate them with
`runpod/survey_segments.py` (about $5.5 of GPU time for all 80 segments).

## Inputs (never vendored, always public)

- `s3://vesuvius-challenge-open-data` — scroll volumes, segments, surface volumes
- `https://dl.ash2txt.org` — legacy tree, detached fragments with infrared ground truth
- `https://huggingface.co/scrollprize` — released ink models

See `report/REPRODUCIBILITY.md` for exact paths, environment pins and per-figure commands.
