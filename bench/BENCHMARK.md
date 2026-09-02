# The 9 µm transfer benchmark — v0.1 (2026-09-01)

_The yardstick for every ink-model claim in the 2026–27 plan (RESEARCH_PLAN_2026-27.md §2).
A fixed set of held-out native-scale surfaces with public human labels, a fixed scorer, fixed
nulls, and — the part nobody else ships — positive controls for the two known fake-null faults.
A model number that is not on this harness is not a benchmark number._

## 1. What it measures

Pixel-level ink detection on surfaces the model has never trained on, at the pitch the model
was trained for, with the input pipeline itself certified on an in-domain control before any
held-out number is read.

## 2. The evaluation set (`manifest.json`)

| tier | surface | why it is in | labels | status |
|---|---|---|---|---|
| **anchor** | PHerc0500P2 `500p2a` win1/win2/win3 (HF `ink/unused/500p2a`) | the only scroll-derived, human-labelled surface absent from ink_9um's 29-representation manifest; 65-layer surface volume at **2.215 µm** (see `P2A_PITCH_RESOLUTION.md`) | `500p2a_inklabels.tif`, `500p2a_supervision_mask.tif` (counts fingerprinted) | first corrected measurement: `p2a_v3` (in flight) |
| **native held-out** | PHerc0139 native 9.362 µm: w035, w039, w040, w041, w044 | the only native ~9 µm human labels in existence; held out via leave-one-scroll-out (khj1222's `make_holdout_config.py`, `bench/vendor/`) — the whole scroll's supervision mask becomes honest test data | HF `ink_9um/labels/native9-scrollprizeorg-21slices/<w>/` (`_inklabels.zarr`, `_supervision_mask.zarr`; plane z=14 carries the label) | label planes cached at `D:/vesuvius-data/trackD/ink9um_planes/`; LOSO retrain needed (Bet A arm 0) |
| **fragments** | Frag1–Frag4 (PHercParis2Fr47, Fr143, Paris1Fr34, Fr39; 3.24 µm, 54 keV), Frag5 (PHerc1667Cr1Fr3; 3.24 + 7.91 µm, 70 keV), Frag6 (PHerc0051Cr04Fr08; 3.24 µm at 53/70/88 keV, 7.91 µm at 53 keV) | optical-IR labelled, wholly outside every scroll model's training; resampled to 9.36 µm | `working/…/inklabels.png` + `mask.png` under each volpkg on dl.ash2txt.org | Frag1 fingerprinted (8181×6330×65, sha256s in `p2a_v3` parts); others on first use |
| **control** | PHerc0139 w035 native crop rows 512:2944, cols 384:3072 | in-domain; the on-record 0.9991 control | embedded in the harness (packbits/zlib/base64, sha256 `23ad57ae…`) | live |

Excluded on purpose: PHerc0009B / 0343P (a week of registration each; keep for a
contamination challenge), the 2.4 µm-pooled ink_9um training segments (in-domain), and any
segment counted twice through the PHerc0139 duplicate-wrap problem (w045/w046, w041/w042 are
one sheet each — Track F).

## 3. Input certification — the positive controls (run before any held-out number)

| control | input | pass rule | why |
|---|---|---|---|
| harness | w035 crop as released | forward AUC ≥ 0.95 (on record 0.9991) | a harness that cannot read the in-domain control certifies nothing |
| depth-order fault | w035 crop, `--direction reverse` | AUC ≤ 0.80 (on record 0.5123) | the harness must *reproduce* the known fault, or it cannot detect it in a held-out run |
| in-plane level fault | w035 crop upsampled ×1.9504 in-plane and depth | reported; expectation: collapse (< 0.75) | the fault that voided the Aug-25 anchor (nerln #1648 / PR #1580: level-0 vs level-2 costs 0.31 AUC); its measured effect size on w035 is the calibration every held-out null is read against |
| coarse level fault | w035 crop downsampled ×0.5 | reported | the opposite error |

A held-out null is reported only next to these four numbers from the same run.

## 4. Scorer (fixed; `p2a_v3` curvelib is the reference implementation)

- **AUC**: exact tie-corrected pixel rank-AUC on the *native label grid*; predictions
  bilinearly upsampled (anti-aliased when downsampled); maps quantized to 16-bit bins (×257).
  Positives = label ∧ mask; negatives = mask ∧ ¬label; label pixels outside the mask are in
  neither class. Both z-directions always run; the reported cell is the better direction, the
  other is reported beside it (the fwd/rev ratio is itself a signal — Track D gate 5).
- **Resampling to model pitch**: Gaussian anti-alias σ = (f−1)/2 source px, linear, area-aligned
  `grid_mode`; depth resampled with the same rule; the pitch is taken from the volume's geometry,
  never from a name string (lesson of `P2A_PITCH_RESOLUTION.md`).
- **Hallucination null** (per surface): 40 rigid translations of the label shapes inside the
  annotated blank, |shift| in [1.5 × median letter height, 0.6 × window]; SHAPE_CONFOUNDED if
  the null median ≥ 0.60 or ≥ 20% of draws ≥ 0.60; GENUINE if real > max(null) and gap ≥ 0.15.
- **Block-mosaic null** for any *positive* claim on an unlabelled surface: block size from the
  measured correlation length (`out/null_scaling/`: n_eff 566 of 2560; never a permutation null).

## 5. Reporting format

One JSON per model per run: `{model, checkpoint_sha, harness_version, controls:{harness,
depth_order, level_x1.95, level_x0.5}, surfaces:{id:{pitch_um, depth_mode, auc_fwd, auc_rev,
best, null_median, null_max, gap, verdict}}, anchor_500p2a: best-of-win1-iso}`. The
`p2a_v3` run's `results.json` is the first instance and the schema reference.

## 6. Gates the plan reads off this harness

| bet | gate (SEPTEMBER_PLAN / RESEARCH_PLAN) | harness number |
|---|---|---|
| A (noise-matched LOSO) | ≥ +0.05 over the LOSO baseline on held-out native 0139 **and** 500p2a ≥ max(0.65, corrected anchor + 0.05) | native tier + anchor |
| C (max-corpus generalist) | ≥ 0.75 on a held-out native scroll | native tier (two scrolls in rotation once PHerc0172 / Scroll 1 labels are in) |
| D (autoresearch on transfer) | leaderboard on held-out AUC | all tiers |
| E (legal bootstrapping) | runs only once some model reads a held-out native scroll ≥ 0.75 | native tier |

## 7. Open items

1. Resolve the five native-0139 S3 segment ids (`<timestamp>-w0xx_<date>`) from the anonymous
   bucket listing and fingerprint their surface volumes (sizes, chunk layout).
2. Fingerprint Frag2–Frag6 labels and layers (three TIFF layouts exist — `STRIPOFFSETS_REPORT_DRAFT.md`).
3. Add the 2023 Scroll 1 legacy 7.91 µm labels and PHerc0172 as *training* sources (Bet C); they are
   not evaluation surfaces because the corpus models will train on them.
4. Release: once `p2a_v3` reports, publish `bench/` with the manifest, the scorer, and the first
   results JSON (organizers' wishlist: "stronger diagnostics… evaluation suites").
