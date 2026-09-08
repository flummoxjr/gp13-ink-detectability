# PREREG — Bet B step C2a: the canonical 2 µm model on the in-band PHerc1203 surface (2026-09-08)

Committed before launch; the pod embeds `parts/prereg.json` and logs its sha before any data contact.
Plan of record: `SEPTEMBER_PLAN.md` §2.2 (C0 done 2026-09-03, r = 0.9999997; C1 done 2026-09-08,
`out/bet_b/c1_rank.md`). Cost cap **$3**, guard 5 h. Nothing here is a claim on PHerc1203; a positive
outcome is an *escalation* to human inspection, and the rules question on 2.4 µm-derived evidence stays
with Ben.

## 1. Question

Does `scrollprize/ink_canonical_2um`, run through villa `optimized_inference` on surfaces **we** render
from the PHerc1203 2.403 µm band (volume `20260319130212`), produce ink-like output on the in-band
surface where the 9.362 µm along-normal contrast is w035-like — and not on the merged-stack surface?

## 2. Inputs (all public)

- **ctl (positive control, runs first).** The published w035 9.362 µm mesh transformed into the 2.399 µm
  PHerc0139 volume `20260102150214` with the catalogue matrix (`hunt/pherc0139_2399um_to_9362um.json`,
  inverted; `hunt/tifxyz_transform.py` reproduces the team's published on-2.399 mesh to a median 1 px along
  z / 7 px in-plane), restricted to the sub-crop rows 1024:2432, cols 896:2560 of the 9.362 canvas (the
  middle 1408 × 1664 of the harness's control crop; human labels: `curvelib.load_ctl_labels()`), rendered
  by `runpod/render_tifxyz_sv.py` as **78 slices at 1 voxel step, mesh-centred**.
- **c2a.** The 12 PHerc1203 auto-grown segments that carry a selected tile, transformed 9.362 → 2.403 µm
  with our derived matrix (`hunt/pherc1203_2403um_to_9362um.json`, translation + nominal scale, residual
  22/16/10 µm), rendered the same way, valid only on the selected 9 × 9-cell tiles (+1 cell margin):
  - **set H** — all 48 tiles with C1 contrast ≥ 0.24 (w035's 10th percentile; 1.36 cm²; contrast 0.240–0.703);
  - **set L** — the 48 lowest-contrast tiles (0.023–0.056), the merged-stack control.
- Model, tiling, window: `MODEL_TYPE=resnet3d-152-3d-decoder`, tile 256 / stride 128, a **62-layer window**
  chosen on the ctl (start 8 = centred; then 0 and 16 if r < 0.90), then fixed for every 1203 run.
  Forward and `FORCE_REVERSE=true` passes on everything.

## 3. Gates, in order

- **Gate C (the 2.4 µm contract).** ctl forward map vs the published w035 map on the rendered box:
  Pearson r (ds4, joint support) **≥ 0.90**; forward pixel AUC on the sub-crop labels **≥ 0.95**; reverse
  AUC **≤ 0.80**. Any failure → `CONTRACT_UNVERIFIED`: the 1203 numbers are reported and **not read**.
- **Gate T (the tripwire is informative).** The tripwire — value > blank p99 (p99 of the ctl forward map on
  the sub-crop's labelled blank), component area ≥ 0.876 mm², bbox width ≥ 0.28 mm (the 0358-screen rule
  in physical units) — must fire on ≥ 1 component touching labelled ink on the ctl. Otherwise the
  tripwire readout is `INCONCLUSIVE` (reported; the fraction statistics still are).

## 4. Readout (frozen)

Per tile (covered > 50 %): p50/p99 forward and reverse, fraction of pixels above blank p99 (forward,
reverse), tripwire components forward/reverse, forward/reverse Pearson r (ds4, joint support).
An H tile is an **escalation candidate** iff `trip_fwd > 0 ∧ trip_rev = 0 ∧ frac_fwd > 3 × frac_rev`.

- **ESCALATE** ("region worth human inspection", never letter language) iff all of:
  (a) ≥ 1 escalation candidate in H; (b) the H tripwire tile-rate exceeds L's — L has 0 tripwire tiles, or
  a one-sided Fisher exact test (H vs L, tiles with ≥ 1 forward tripwire) gives p < 0.05; (c) the median
  per-tile forward/reverse r over H is < 0.20.
  Pre-stated action: Ben looks at the previews; a robustness re-render at window ±8 before any mention
  outside this repo; the rules question is asked before anything derived from it is used on 1203.
- **NULL** otherwise: no evidence, at the sensitivity the ctl bounds (ctl ink p50 vs blank p99 and the ctl
  tripwire count are reported as that bound), that the canonical 2 µm model reads ink on the in-band 1203
  surface. Bet B's 2.4 µm route is closed for September; C1 ranking, the renderer validated at 2.4 µm, and
  this null ship.
- Descriptives reported whatever the verdict: H vs L `frac_fwd` (Mann-Whitney), tripwire tiles H/L,
  fwd/rev r distributions, previews of the top three tiles per segment by `frac_fwd`.

## 5. What this cannot show

1.7 mm tiles cannot hold ruling cycles, so PROTOCOL_V2's periodicity battery does not run; the readout is
tripwire + depth asymmetry + a matched substrate control, weaker than the corpus battery. A NULL here does
not exclude ink on 1203; it bounds what this model sees on these surfaces. Prior nulls (Drobkov at 2.403 µm
with another model; TAUIL's `ink_3d_dino_guided` firing "within the Paris4 range") stay priors.

## 6. Cost

Renders: ctl ≈ 5.5 k × 6.5 k px × 78 slices; 1203 ≈ 96 tiles × 700² px × 78 slices; reads ≈ 30–60 GB from
S3. Inference: ≈ 0.04 Gpx (ctl) + 0.05 Gpx (1203), × 2 directions, plus up to two ctl window re-runs.
Estimate 1.5–2.5 h on an A6000/5090 → $1–2; cap $3. Balance before launch $23.
