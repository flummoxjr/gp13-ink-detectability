# Measure Before You Hunt

**A scan-quality index, a validated ink instrument, and a corpus-wide screen of the
Vesuvius Challenge Grand-Prize scrolls.** — Ben Black, August 2026

Nobody had published a measurement of which Grand-Prize scrolls can actually be read,
and by what. This repository supplies one. No ink was found and none is claimed; what
is claimed is the measurement — and the machinery that kept it honest.

> **Mirror note:** this research also exists at [`measure-before-you-hunt`](https://github.com/flummoxjr/measure-before-you-hunt);
> the two repositories are kept content-identical. Whichever URL brought you here, you are reading the same work.

**Start here → [`report/REPORT.md`](report/REPORT.md)** (executive summary, five-minute
table, limitations). Then the sections: [index & separability](report/sections/01_index.md) ·
[instrument & corpus screen](report/sections/02_instrument.md) ·
[two screens, two nulls](report/sections/03_screens.md) ·
[adversarial QC](report/sections/04_methodology.md) ·
[reproducibility](report/REPRODUCIBILITY.md).

## Headline results

| Result | Number |
|---|---|
| Scan-quality index, all 13 GP scrolls + calibrator | two tiers, 3.0× gap, campaign correlation Fisher p = 0.007 |
| Sheet-separability axis (near-orthogonal to scan quality, ρ = +0.34) | the one scroll with proven letters ranks **1st of 14** |
| Published letters reproduced at native 9 µm | pixel AUC **0.9991** (independently re-verified) |
| Entire published GP segment corpus screened | 80/80 rows; **0 of 71** pass a five-gate protocol the control passes **5/5** |
| The index's own ROI rule audited | intensity-picked windows are incrustation: random sampling scores 2.95× higher, 14/14 scrolls |
| Corrections ledger | **16 published corrections, 5 against results in this report** |

## Reusable tools

- **Pre-fleet model gate** (~$1, ~30 tiles) — stopped a doomed fleet run at $4.46 of an $80 budget (`qc_live/`)
- **Text-signature battery + pre-registered tripwire** (~5 min/segment, CPU) (`salvage/verdict_*.py`)
- **Mesh-vs-lamella alignment gate** (seconds, on data you already have) (`hunt/mesh_lamella_alignment.py`)
- **tifxyz → surface-volume renderer** — makes flat-mode `ink_9um` checkpoints runnable on any 9 µm segment (`runpod/render_tifxyz_sv.py`)
- **Per-scroll index + per-ROI coordinates** for all 168 sampled cubes (`out/k2c_separability/`)

Every headline number is asserted against its primary artifact by
[`report/scripts/verify_report.py`](report/scripts/verify_report.py) — run it after any edit.

## Disclosure

AI agents were used in this work, under my direction. I set the questions, made the
judgment calls, and checked the numbers. `LOG.md` is the unedited running log — it shows
the order things were actually found in, including the wrong turns.

## License

Code: [MIT](LICENSE). Derived data: CC-BY-NC 4.0 (see [DATA_LICENSE.md](DATA_LICENSE.md)).
All inputs are public (`s3://vesuvius-challenge-open-data`, `huggingface.co/scrollprize`).
