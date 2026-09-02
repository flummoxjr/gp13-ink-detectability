# Cross-acquisition ink confirmation — corpus run (research D)

Run 2026-08-24, harness `trackD/xacq_score.py` (xacq-1.1). Pre-registration:
`trackD/PREREG_XACQ.md`, git fb8b55b (2026-08-24T01:46:06-05:00), committed before any
corpus-wide number existed. Protocol followed exactly; every deviation is listed in
"Limits & deviations" below.

## Verdict: KILLED (by the pre-registered decision rule, clause 1)

The cross-acquisition **agreement signal is real and corpus-wide** (median L(0.99) = 25.1
across the 112 pairs vs the pre-registered kill floor of 5; nulls at 0.9–1.8), but the
**fusion layer fails its pre-registered calibration bar**: median AUC gain of fused over
best-single across the 18 usable human-labelled segments is **+0.0080 < 0.01**. Per the
committed rule the confirmation layer does not ship. A negative was pre-registered as
publishable: fusion gain does not replicate beyond PHerc0139 as a corpus-wide property —
it is a property of samples where both arms are comparably good.

## 1. Anchor reproduction (w035, gate passed before any corpus number)

The pilot harness was not on disk; the harness was reconstructed and validated against the
committed anchors before scoring the corpus.

| quantity | pilot anchor | this harness | tolerance | pass |
|---|---|---|---|---|
| registration scale | 0.9402 | 0.940199 (shape ratio) | — | exact |
| residual shift | ~2 ds8 px | 2.0 px ([-2,0]) | — | yes |
| peak SNR | ~25 | 25.03 | — | yes |
| Pearson r | 0.6608 | 0.6635 | ±0.03 | yes (+0.0027) |
| null r (rot180 / roll(700,700)) | 0.0767 / 0.0515 | 0.0780 / 0.0512 | — | yes |
| L(0.99) | 34.0 | 34.07 | ±20% | yes (+0.2%) |
| AUC A / B / fused | 0.8768 / 0.8771 / 0.9015 | 0.8763 / 0.8773 / 0.9014 | ±0.01 | yes (≤0.0005) |
| FPR@50% recall A/B/F | 0.0363 / 0.0321 / 0.0263 | 0.0342 / 0.0309 / 0.0237 | none stated | deltas 0.002–0.003* |

*FPR deltas trace to a label-resampling interpolation convention the pilot did not record
(INTER_LINEAR reproduces B exactly at 0.0321; the frozen harness uses INTER_AREA, which
reproduces all three AUCs to ≤0.0005). No gated quantity is affected.

Conventions recovered and frozen: joint-valid mask = (A > 2/255) & (registered B > 2/255);
L(q) is quantile-based (per-map q-quantile over joint-valid pixels); B→A resample by the
image shape ratio; unwindowed phase-correlation refine, peak restricted to ±60 px (v1.1 —
see deviations); fused = mean of per-map z-scores (moments from the joint-valid mask).
The extracted native w035 label plane matches the pilot's on-disk assets to 11 of 396k ink
pixels (Jaccard 1.0000) — the upstream zarr was trivially revised on 2026-08-18.

## 2. Corpus results (112 pairs; per-sample strata; Paris 4 separate — shared-energy caveat)

Pearson r on the joint-valid mask:

| sample | n | r median [q25, q75] | min / max | L(0.99) median | null r medians rot/roll/perm | null L(0.99) medians |
|---|---|---|---|---|---|---|
| PHerc0139 | 37 | 0.581 [0.516, 0.657] | 0.268 / 0.748 | 35.6 | -0.017 / 0.003 / 0.041 | 0.95 / 0.82 / 2.00 |
| PHerc1667 | 19 | 0.343 [0.266, 0.370] | 0.148 / 0.528 | 22.3 | 0.003 / 0.005 / 0.039 | 0.36 / 1.03 / 1.78 |
| PHerc0814 | 19 | 0.461 [0.215, 0.492] | -0.098 / 0.593 | 21.3 | -0.024 / -0.012 / 0.086 | 0.24 / 0.76 / 2.10 |
| **main stratum** | 75 | — | — | **25.6** | — | — |
| PHercParis4 (separate) | 37 | 0.625 [0.575, 0.735] | 0.073 / 0.838 | 23.6 | -0.000 / 0.029 / 0.069 | 1.04 / 1.38 / 1.64 |

All-112 medians: r = 0.559; L(0.90/0.95/0.98/0.99) = 4.91 / 9.40 / 17.08 / **25.09**.
Overall null-r medians: rot180 -0.008, roll 0.013, blockperm 0.054. 64/111 measurable
pairs have r > 0.5; 91/111 have r > 0.3; 9 have r < 0.1 (one pair is unmeasurable, see §5).

## 3. Calibration on human-labelled segments (ink_9um label set)

22 of the 112 public segments carry human ink labels (aligned corpus; 5 of them also have
native-9.362 µm labels, used as sensitivity rows). Full table:
`calibration_table.md/.csv`; provenance: `label_provenance.json`.

**Label semantics (stated per the project's reference-class rule):** positives = supervision-mask
pixels with ink > 0; negatives = supervision-mask pixels with ink == 0, i.e. **annotator-declared
blank**; pixels outside the supervision mask (including 62k ink px on w035 outside it) are
never used. Labels annotate one plane only (aligned: z=10 of 21; native: z=14 of 28) —
verified against the per-z nonzero profile for all 27 label sets.

**Usable: 18 of 22.** Four labelled PHerc0139 segments (pub w029, w043, w044, w045 — corpus
names pherc0139-w016/-w043/-w028/-w029, plus native w044) are **unusable: the annotated patch
has zero joint-valid pixels** because the 1.129 µm B volume covers only 30–40% of those
segments and renders the annotated region as nodata (B arm is exactly 0.0 there). This is a
B-coverage fact, not an annotation gap.

Per-stratum medians over usable labelled rows (primary = aligned):

| stratum | n | AUC A | AUC B | AUC fused | FPR@50 A / B / fused | fused 1%-FPR thr (z) |
|---|---|---|---|---|---|---|
| main (0139+1667+0814) | 12 | 0.901 | 0.832 | 0.911 | 0.0238 / 0.0646 / 0.0287 | 2.44 |
| PHercParis4 | 6 | 0.943 | 0.947 | 0.962 | 0.0088 / 0.0069 / 0.0057 | 2.42 |

Per-segment AUC gains (fused − best single), all 18 usable rows, sorted:
-0.0173 (1667-w031), -0.0161 (1667-w018), -0.0119 (0814-46527), -0.0111 (1667-w013),
-0.0097 (1667-w028), -0.0072 (1667-w029), +0.0032 (1667-w023), +0.0037 (0139-w041),
+0.0077 (P4-w01), +0.0084 (P4-w02), +0.0094 (P4-w06), +0.0097 (P4-w07), +0.0116 (P4-w03),
+0.0133 (P4-w05), +0.0162 (0139-w040), +0.0164 (0139-w017), +0.0239 (0139-w035),
+0.0279 (0139-w039).

Native-label sensitivity rows agree in sign and size (w035 +0.0241, w039 +0.0318,
w040 +0.0159, w041 −0.0029); aligned vs native w035 give nearly identical gains
(+0.0239 vs +0.0241) despite different label transfers.

**Why fusion fails corpus-wide:** on PHerc1667 the B arm is much weaker than A
(AUC_B 0.68–0.78 vs AUC_A 0.88–0.96) and equal-weight fusion drags the better map down on
5 of 6 segments; same on PHerc0814 (A 0.981 vs B 0.903, gain −0.0119). Fusion helps on all
5 usable PHerc0139 rows and all 6 Paris4 rows. Fusion gain is a property of
matched-quality arms, not of cross-acquisition fusion per se.

## 4. Decision rule, applied verbatim

- **Clause 1** — "across the labelled segments, AUC(fused) does not exceed AUC(best single)
  with DeLong p < 0.05 AND median gain ≥ 0.01":
  - Combined DeLong (Stouffer over 18 one-sided per-segment z's): z = 81.3, p ≈ 0 → the
    p < 0.05 leg passes, though heterogeneously: 12/18 segments significantly better,
    6/18 significantly *worse* (p > 0.95).
  - **Median gain = +0.0080 < 0.01 → this leg FAILS.** (Main stratum −0.0020; Paris4-only
    +0.0095; native-sensitivity swap leaves 0.0080 — below the bar under every split.)
  - Clause 1 therefore kills.
- **Clause 2** — "median L(0.99) across the 112 below 5": median = **25.09 ≥ 5** — does not kill.
- **Verdict: KILLED.** The confirmation-layer products (calibrated confirmed-ink maps at a
  1%-FPR fused threshold, disagreement maps as hallucination estimates, ranked
  confirmed-only list) are **not shipped as calibrated claims**.

## 5. Diagnostic agreement ranking (not a shipped confirmation layer)

Computed because the agreement signal itself passes its nulls broadly; framed strictly as
regions of cross-acquisition agreement meriting human inspection. Survival criterion
(operationalized before corpus numbers): survive(null) := r − r_null ≥ 0.15 AND L99 ≥ 5
AND L99 ≥ 3×max(L99_null, 0.5), applied to the rot180 and block-permutation nulls.

- Survive both nulls: **main stratum 65/75, Paris4 32/37** (97/112).
- Top main-stratum segments by r-margin over the worst null (all survive both):
  PHerc0139 w047 (r=0.748, L99=41), w059 (0.673, 21), w046 (0.694, 38), w041 (0.651, 36),
  w042 (0.645, 36), w039 (0.664, 40), w034 (0.707, 40), w049 (0.733, 54).
- Top Paris4 (separate stratum, shared-energy caveat): 20231012184424 (r=0.838, L99=21),
  20260623142658-w028-037 (0.800, 24), 20231031143852 (0.777, 33),
  20260623143441-w038-045 (0.758, 28), 20260604223808-20231210121321_v8 (0.735, 34).
- Notable non-survivors: the five PHercParis4 "5753_-N" segments (r = 0.07–0.10,
  L99 = 3.1–5.1) — the two acquisitions substantially disagree there; and five tiny
  PHerc0814 auto_grown segments (joint masks 42–54k px, r ≈ 0), plus one PHerc0814 pair
  with an empty joint mask (B coverage does not overlap A validity):
  20260226123353-auto_grown_20260226123353106.

Full ranked lists with per-segment nulls: `corpus_summary.json` (ranked_main /
ranked_paris4). Per-pair diagnostics PNGs (fused z, |zA−zB| disagreement, joint mask):
`out/xacq/maps/` (336 files). No claim about recoverable text is made anywhere in these
outputs, and nothing here has been looked at by human eyes.

## 6. Limits & deviations

1. **Pilot-harness reconstruction.** The pilot code was not preserved; conventions were
   recovered by matching the committed anchors (all gated anchors hit; FPR@50 differs by
   0.002–0.003 from an unrecorded interpolation convention).
2. **Refine-window deviation (v1.1).** The prereg fixes "refined by phase correlation" but
   not a search radius. For 5/112 pairs the unrestricted argmax landed on a distant
   self-similarity alias (|shift| 738–2352 px) while a clean local peak sat at ≤7 px; the
   frozen harness restricts the refine to ±60 px. The other 107 pairs are bit-identical
   under both definitions (max observed |shift| = 43 px). Decision numbers were also
   computed before this fix: the verdict is KILLED under both.
3. **B-arm coverage gaps.** The 1.129 µm B volumes cover segments only partially
   (median joint-valid fraction 0.33 across the 112; frac_valid_B ranges 0.00-0.68). All r/L
   are conditional on the covered region. Four labelled segments lost to this (§3).
4. **PHerc1667 registration anisotropy.** All 19 PHerc1667 pairs have ~1.3% h/w scale
   anisotropy (flagged aspect_mismatch); a per-axis global resample was used, but residual
   local warp may depress 1667's r and B-arm AUC — 1667's negative fusion gains partly
   reflect B-arm quality/registration, not only model error. The prereg protocol has no
   local-warp step, so none was added.
5. **Null degeneracy on small segments.** 7 pairs are too small for a fully valid ≥700 px
   roll (roll_null_weak); 6 lack enough fully-valid 64-px blocks for the permutation null
   (perm_null_insufficient); 1 pair has an empty joint mask entirely. All flagged per pair.
6. **Block-permutation parameters** (64 px blocks, 5×5-block envelope, 10 strata,
   per-segment CRC32 seed) were not pinned by the prereg (the pilot had only two nulls);
   they are frozen in the harness and recorded in every pair JSON.
7. **Maps are 8-bit JPEG ds8** (the published downsampled products); quantization and
   compression noise are shared by both arms' products equally but bound precision.
8. **PHercParis4 shares 78 keV across arms** — energy-correlated physics could correlate
   there; reported as a separate stratum throughout (prereg caveat).
9. **DeLong "across the labelled segments"** was operationalized as Stouffer-combined
   per-segment one-sided z (n=18) plus the median-gain clause; per-segment p's and z's are
   in `calibration_rows.json` so any other combination rule can be recomputed.

## Files

- `corpus_summary.json` — per-sample distributions, decision walk, ranked lists
- `pairs/*.json` (112) — per-pair registration, r, L(q), 3 nulls, flags, moments
- `maps/*.png` (336) — fused-z / disagreement / joint-mask diagnostics per pair
- `calibration_rows.json`, `calibration_table.md`, `calibration_table.csv`
- `label_provenance.json`, `label_sets.json`, `label_map.json`, `anchors_check.json`
- Harness: `trackD/xacq_score.py` (anchor / score / calib / ship subcommands)
- Label planes: `D:\vesuvius-data\trackD\ink9um_planes\*.npz` (+ extraction_report.json)
