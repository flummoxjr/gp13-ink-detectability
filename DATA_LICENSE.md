# Data license

**Code** in this repository is MIT (see `LICENSE`).

**Derived data products** produced by this work — index values, per-segment
statistics, screen scores, prediction maps, mesh QC, figures — are released
under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

## What this repository does *not* license

It redistributes no bulk scroll data. Every input is public and stays with its
original terms:

| Input | Source | Terms |
|---|---|---|
| Scroll CT volumes, segments, surface volumes | `s3://vesuvius-challenge-open-data`, `https://dl.ash2txt.org` | Vesuvius Challenge data agreement, accepted per-user |
| Released ink models (`ink_9um`, `ink_3d_dino_guided`) | `huggingface.co/scrollprize` | as published by the Vesuvius Challenge |
| Detached-fragment infrared photographs and ink labels | `dl.ash2txt.org/fragments/` | Vesuvius Challenge data agreement |

Anyone reproducing this work must accept the Vesuvius Challenge data agreement
themselves. Local caches of upstream data are excluded from version control by
`.gitignore` so that the statement above remains literally true.
