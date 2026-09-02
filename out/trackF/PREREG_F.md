# PREREG_F — Track F: battery screen of the unread high-agreement PHerc0139 segments

Written 2026-08-25, BEFORE any battery computation on any candidate map, control jpg,
or fwd/rev pair. Premise checks (Step 0) were run first and are recorded below with
their evidence; no battery statistic has been computed on anything yet.

## Candidates (survived Step 0)

| candidate | public segment | xacq r | L99 |
|---|---|---|---|
| w059 | PHerc0139 20250223000000-w059_2025022312 | 0.6735 | 20.848 |
| w046 | PHerc0139 20260325000000-w046_20260325 | 0.6940 | 37.684 |
| w042 | PHerc0139 20260206000000-w042_2026020613 | 0.6449 | 36.096 |

(xacq numbers quoted from `trackD\out\xacq\corpus_summary.json`, banked earlier work.)

## Step 0 — premise verification (run TODAY, before this prereg was finished)

Claim to verify: each candidate has NO ink_9um label and NO published reading.

(a) **Label corpus** — fresh fetch this session of the ink_9um bucket README
(`.../resolve/ink_9um/README.md`, 10,739 B) and a fresh `api/buckets/.../tree/ink/0139`
listing. PHerc0139 labeled public segments (aligned + native9, after applying the
README's corpus-name → public-segment mapping): 20250108000004-w029,
20250108000005-w030, 20260115000000-w044, 20260126000000-w045, 20260317000000-w035,
20260302000000-w039, 20250831000000-w040, 20260108000000-w041, 20260112000000-w043.
The human ink-annotation corpus `ink/0139` holds exactly 11 dirs (w016, w017, w028,
w029, w030, w035, w039, w040, w041, w043, w044 — annotation-dataset names). **No
candidate appears in either.** All three candidates sit in `ink/unused/0139/`
(predictions only, no labels). → PASS for all three.

(b) **Preprint readings** — `trackD\GROUND_TRUTH_AUDIT.json` (re-read this session)
records published readings for w025 (20250108000000), w034 (20260303000000), w047
(20260206000001), w049 (20260318000000) and the title strip (20260422000000) only,
and states for the 29 held-out segments (which include w042): "NO published
per-segment transcription". Note 20260206000000-w042 ≠ 20260206000001-w047 — same
scan date, different segments. → PASS for all three.

(c) **Anything published since** — web search this session (2026-08-25) surfaces only
the known June-2026 preprint readings and press coverage; nothing naming w042, w046
or w059. Limitation: a web search is not exhaustive; a very recent reading could be
missed. → PASS (with that caveat).

**All three candidates survive Step 0.**

## Inputs and their limitation (stated up front)

- Arm A = published canonical ink prediction on the **2.399 µm / 78 keV** acquisition
  (volume 20260102150214), served as a **ds8 jpg** → 19.192 µm/px, 8-bit.
- Arm B = published canonical ink prediction on the **1.129 µm / 59 keV** acquisition
  (volume 20260413113053-L1), ds8 jpg → ≈18.04 µm/px (derived per segment as
  19.192 × mean(scale_h, scale_w) from the banked xacq registration).
- Fused-z = 0.5·(zA + zB) recomputed in float on A's raster via the banked
  `xacq_score.register` (resize + integer phase shift; all three candidates
  registered cleanly: pc_snr 21.9–43.3, zero flags).
- These are **8-bit, 8× downsampled jpgs of model predictions**, and the models were
  trained on sibling segments of this same scroll. Nothing here is a letter claim;
  the ceiling is "region worth human inspection".

## The battery

The operative validated screen is PROTOCOL_V2 (`trackD\analyze_survey_corpus_v2.py`),
the hardened 5-gate version of the salvage verdict battery; its control record is
w035 5/5 gates at z = +16.26, p = 0.00498, period 4.678 mm. I do NOT rewrite it: a
driver imports the module and calls its `prep / score / joint_permute / _perm_stats /
_detail` functions and its exact seeding (`20260818 + 1000·draw + crc32(key) % 997`).
The only new code is input preparation:

1. jpg → float map (0–255), validity = value > 2.0/255·255 (xacq VALID_T convention,
   needed because jpg background is compression-noisy rather than exactly 0);
2. resample map and validity to the protocol's analysis scale **37.448 µm/px**
   (cv2 INTER_AREA; validity thresholded at 0.5; map zeroed outside validity so the
   module's `valid_mask(a>0)` chain — closing 5×5, fill, erode 10 px — applies
   unchanged);
3. fused map stored affine-shifted to positive (fused − min + 1, 0 outside the joint
   mask) — the prominence statistic is invariant to affine rescaling of the map, so
   this is exact, not approximate.

Protocol constants, unchanged from v2: band 1.7–8.4 mm; detrend σ = 90 px; θ grid 3°
over [0,180); erode 10 px; MIN_ROWFRAC 0.25; perm tile 64 px (fallback 32);
N_PERM = 200; statistic = band peak power / band median power; null = joint
(map, mask) tile permutation; band-constrained secondary search reported.

### The five gates (thresholds identical to v2)

1. `gate_significance` — empirical p ≤ 0.05, p = (1+#{null ≥ obs})/201.
2. `gate_cycles` — ≥ 6.0 cycles of the claimed period in the profile.
3. `gate_autocorr` — ρ(2P) > 0 and ρ(3P) > 0 on the detrended profile.
4. `gate_band_bin` — peak not in the band's 2 lowest Fourier bins (index ≥ 2).
5. `gate_fwd_rev` — |fwd/rev r| < 0.20, computed from the published same-checkpoint
   forward+reverse pair `ps512_ema_ds_scse_resdecoder_wider_zskip2_*_ckpt_030000_*`
   in `ink/unused/0139/<seg>/preds/` (full-res Pearson over common nonzero support,
   the survey convention). **Computable for w046 and w042 only.** w059 publishes no
   reverse render (verified by tree listing this session) → gate 5 is n/a for w059.
   Caveat stated now: this pair is a different checkpoint from the canonical maps
   that form arms A/B; it tests one-sidedness of the segment's model response, and
   is per-segment (the same value enters both arms' scorecards, as v2 took fwd/rev
   from the survey record).

## Decision rule (fixed now, numbers first)

- A candidate is **"flagged for human inspection"** iff **arm A passes ≥ 4/5 gates
  AND arm B passes ≥ 4/5 gates**. A non-computable gate counts as NOT passed, so
  w059 must pass 4/4 of its computable gates on each arm.
- The fused-z map is scored and reported for all three candidates but is
  **descriptive only** — it does not enter the flag decision (fusion was KILLED for
  AUC gain in the xacq study; fused numbers are context, not evidence).
- **Multiplicity** (3 candidates × 3 maps = 9 significance tests): the gate uses raw
  p ≤ 0.05 per map, exactly as v2 §2.1, because 200 permutations bottom out at
  p = 1/201 = 0.00498 while Holm across 9 tests demands p < 0.05/9 = 0.00556 for the
  smallest — attainable only at the floor. Holm-9 p is reported beside every raw p.
  The flag conjunction is the real control: under the global null with arms
  independent, P(both arms pass gate 1 alone) ≤ 0.05² = 0.0025 per candidate,
  ≤ 0.0075 familywise across 3 candidates — before gates 2–5 bite.
- **Escalation, not announcement**: any flagged candidate gets (i) a dedicated
  ≥ 1000-permutation rerun of gate 1 before any public language, and (ii) at most
  the sentence "cross-scanner agreement region worth human inspection" — never
  letter-language. No flag → the result is a matched null for these three segments
  and the gallery still ships.

## Control gates (must pass BEFORE candidate numbers are interpreted)

- **C1 — protocol equivalence.** The driver run on the v2 control cache
  (`v2cache\w035_fwd_strided_ds4.npy`, px 37.448, key `w035_CONTROL_strided`, same
  seeding) must reproduce the v2 record: 5/5 gates, prominence 123.55 ± 1 %, period
  4.678 mm (± one Fourier bin), 11.0 cycles, peak bin 4, ρ(1P/2P/3P) =
  +0.646/+0.544/+0.458 each ± 0.02, p = 0.00498, z within [+13, +20] (identical
  seeding should land ≈ +16.26 exactly). C1 failure = the port is wrong → fix or
  KILL the battery step; candidates are not run until C1 passes.
- **C2 — modality control.** The full Track F path (ds8 jpg → resample → battery) on
  w035's own A and B jpgs (w035 carries human-verified Greek letters) must pass all
  4 map-internal gates (1–4) on arm A AND on arm B. If C2 fails on an arm, the
  battery is declared NON-INFORMATIVE for that arm's modality: candidate results on
  that arm are reported as descriptive, no null claim and no flag is issued from it,
  and the track outcome falls back to gallery + "battery not informative at ds8-jpg
  fidelity". w035's fused map is also run, descriptive, to calibrate the fused
  numbers. (w035's gate-5 value is quoted from earlier work: r = 0.0943.)
- Order of execution: C1 → C2 → candidates. Downloads for gate 5 happen after the
  map-internal battery, and gate-5 values are computed blind to nothing — they are a
  fixed threshold on a single correlation.

## Step 3 — gallery (runs regardless of battery outcome)

For each candidate: top-3 agreement crops, window 512×512 A-px (≈ 9.83 × 9.83 mm),
selected by box-filtered mean of min(zA, zB) over the joint mask (windows ≥ 50 %
joint-mask coverage, centers ≥ 256 px apart), rendered A | registered-B | fused
side by side, contrast-stretched p2–p98 per panel, 5 mm scale bar, every PNG
labeled **"cross-scanner agreement region — not verified text"**, plus one
full-segment overview per candidate. Output: `scratchpad\close\gallery_F\`.

## Numbers promised in the report

Per candidate × per map (A, B, fused): prominence, null mean ± sd, z with 95 % CI,
empirical p, Holm-9 p, period mm, θ, cycles, peak bin/nbins, ρ(1P/2P/3P), eroded
mask fraction, profile length mm; per candidate: fwd/rev r (or n/a), gates passed
per arm, flag decision. Controls C1/C2 with the same fields. All numbers measured
this session unless explicitly marked quoted.
