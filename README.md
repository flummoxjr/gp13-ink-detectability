# Measure Before You Hunt

**A scan-quality index, a sheet-separability axis, a validated ink instrument, and a corpus-wide
screen of the published Grand-Prize scroll segments.**

Vesuvius Challenge submission, August 2026 · This repository and [gp13-ink-detectability](https://github.com/flummoxjr/gp13-ink-detectability) are **content-identical mirrors**; gp13 also hosts the prebuilt tracer binary as a release asset. Both stay public through judging. · [scrollprize.org](https://scrollprize.org)

I'm Ben Black, and this is my first month on the Vesuvius Challenge. This repository is one
question, measured: before you spend compute hunting ink in an unread scroll, can you tell whether
reading it is even possible? I did not find ink, and the report says so throughout. AI agents were
used in this work, under my direction; I set the questions, made the judgment calls, and checked
the numbers.

---

## The thesis

Before spending compute hunting ink in an unread scroll, measure whether its released scan and your
model are in the class where reading is even possible. Nobody had published that measurement. This
repository is that measurement, plus the acceptance gates that keep it honest.

**Total cloud compute: ≈ $20 all-in — the report itemises it, waste included.**

## What it found

- **Two scan-quality tiers across all 13 Grand-Prize scrolls**, separated by a 3.0× gap with no
  scroll inside it. Degradation correlates with acquisition campaign (4/4 vs 1/9, Fisher p = 0.007).
- **Scan quality is not readability.** A second, near-orthogonal axis — sheet separability, measured
  as local structure-tensor planarity — correlates only ρ = +0.34 (p = 0.24) with the quality index.
  It ranks PHerc0139, the one scroll with proven letters, **1st of 14 without knowing which scroll
  has letters**. The two axes disagree sharply: PHerc1447 scans worst of all 14 yet carries more
  published segments than any other scroll.
- **The index's own ROI rule was sampling the wrong material** — each scroll's *brightest* windows,
  which in a carbonised scroll is mineral incrustation rather than papyrus, by 2.95× in 14 of 14
  scrolls. The tier split survives; individual ROI values do not. We found this by building the
  second axis, not by auditing the first.
- **No text in the published corpus.** Every published segment of every Grand-Prize scroll that has
  one — 80 catalogue rows, 416 cm² of rendered papyrus — screened under a hardened protocol with
  200 permutations and five pre-registered gates. **0 of 71 scorable segments pass; the
  human-verified control passes 5 of 5** (z = +16.3, empirical p = 0.005).
- **What that negative does and does not cover** is stated as prominently as the result: the
  published corpus lives on index ranks 8, 13 and 14 of 14, many surfaces are debug dumps, and some
  lie partly outside the scanned volume. Only one scroll in the corpus was ever a fair test.

**No ink was found, anywhere, by anything, in this work.** Two model outputs were characterised in
detail; neither is a detection.

## The part that is easiest to check, and the point of the whole thing

Sixteen corrections are published as a ledger in [`report/sections/04_methodology.md`](report/sections/04_methodology.md),
five of them against results in this very report. Among them: our own newly grown PHerc0813
surfaces turned out to sit at a median **68.1° to the sheets they were meant to follow**, so the ink
test planned for them was **withdrawn rather than reported**.

## Layout

| path | what |
|---|---|
| `report/` | the submission: `REPORT.md` + `sections/01`–`04`, `REPRODUCIBILITY.md`, figures |
| `report/scripts/verify_report.py` | re-asserts every headline number against its primary artifact — **run this after any edit** |
| `k2b_detectability_index.py` | the scan-quality index (resumable, streaming) |
| `k2c_separability.py`, `k2c_analyze.py` | the sheet-separability axis, its isotropic floor and sensitivity sweep |
| `analyze_survey_corpus_v2.py` | the hardened corpus screen (200 permutations, empirical p, five gates) |
| `runpod/render_tifxyz_sv.py` | tifxyz → surface-volume renderer (validated r = 0.813 end-to-end) |
| `runpod/survey_segments.py`, `runpod/segment_catalog.json` | the corpus survey harness and the catalogue of every published GP segment |
| `hunt/mesh_lamella_alignment.py` | free QC gate: angle between a mesh normal and the local sheet normal |
| `hunt/check_air.py` | free pre-render QC gate: reject segments grown into empty volume |
| `out/` | results as JSON — index, separability, corpus screen, control scores |
| `qc/`, `qc_live/`, `verify_flag/`, `comb/`, `salvage/` | the audit trail, shipped as first-class results |

## Reproducing

Environment pins, public data URLs and per-figure regeneration commands are in
[`report/REPRODUCIBILITY.md`](report/REPRODUCIBILITY.md). Everything streams from public sources;
no bulk download is required.

```bash
python report/scripts/verify_report.py    # 18 checks against primary artifacts
```

## What is not in this repository

Large derived binaries — the 150 saved prediction maps, control prediction TIFFs, the frozen
`ink_3d` benchmark tables — are excluded to keep the repository clonable, and because this work
redistributes no bulk scroll data. Every one is regenerable; [`DATA.md`](DATA.md) lists each with
its size and the exact command that rebuilds it.

## Licence

Code MIT ([`LICENSE`](LICENSE)). Derived data CC BY-NC 4.0 ([`DATA_LICENSE.md`](DATA_LICENSE.md)).
All inputs are public and remain under the Vesuvius Challenge data agreement.
