# Round 1 QC verdict — PHerc1203 ink screen (ink_3d_dino_guided, Paris4-trained)

Date: 2026-08-16 (QC ran live at fleet ~45% done; fleet stopped on this verdict)

## VERDICT (read this first): GARBAGE (domain shift) — fleet stopped; salvage is model-side only

**Not a pipeline bug.** The screen pipeline is faithful to the villa CLI to within
blending noise, and the normalization difference ([1,99] of nonzero vs the CLI's
[1,99] of all voxels) is numerically irrelevant on interior tiles. The blanket
firing is the model's genuine behavior on PHerc1203: the villa CLI's own
reference run on this volume fires just as densely (f05 = 0.049 over the whole
smoke region; up to 0.086 per tile). The healthy in-domain reference — the
*released* Paris 4 ink3d prediction from the SAME checkpoint lineage
(v3-78k-fullsup) — is silent over most random blocks (5/8 blocks f05 = 0.000,
p90 of values 0.012) and sparse elsewhere. On 1203, **zero of 3,346 screened
tiles are silent**; the median tile fires on 5-7% of voxels. The per-tile stats
carry no usable ranking signal: firing scales with mask fill, the top-1% f08
tiles are spatially ANTI-clustered (noise), pmax==1.0 marks mask-edge tiles, and
the voxel maps trace cracks/void-boundaries/mask edges rather than letter-shaped
marks. Continuing the fleet to 100% buys more of the same distribution.
Finally, the histogram-matching probe (test 5B, run on the w0 pod) REFUTED the
cheap salvage path: matching 1203's intensity distribution to Paris 4 made
firing *worse* (f05 0.039 → 0.081, above the in-domain ink-active level), so
the domain shift is texture/morphology-level, not an intensity covariate fixable
at the input. **The fleet was stopped (pods terminated); the collected
stats+probL2, CLI reference, and QC scripts are secured locally.**

### Test scorecard

| # | Test | Result | Meaning |
|---|---|---|---|
| 1 | Normalization equivalence (local GPU, same ckpt, tiles inside CLI region) | screen-norm ≡ villa-norm to 3rd decimal; vs CLI blended ref r=0.64-0.75 @L2, 94.5-98.8% agreement @0.5 | Screen pipeline faithful; firing is the model's behavior, not a bug |
| 2 | Firing morphology (27 pulled tiles vs CT L2 + surface pred) | background f05 0.0007; edge band 0.139; bulk on/off-sheet ratio 0.64-1.62; ribbons, no glyphs | Crack/edge/texture follower inside material; not letter-shaped |
| 3 | Discrimination (3,346 tiles) | unimodal f05/f08; drivers = fill (ρ 0.43); top-1% anti-clustered (0.0 vs 0.11 rand); top-5% regionally clustered (~3σ) | No rankable tail; residual regional structure likely texture, unvalidated |
| 4 | In-domain reference (released Paris4 ink3d, same ckpt lineage) | 5/8 random blocks f05=0.000, p90 value 0.012; 1203: 0 silent tiles of 3,346 | 1203 output is out-of-family — domain shift, not calibration |
| 5A | In-domain control (fleet pipeline on Paris4 CT) | reproduces released pred: r=0.873 (active), 0.741 (quiet) | Checkpoint+pipeline healthy end-to-end |
| 5B | Histogram-match probe (w0 pod, single 256³ tile) | f05 0.0392 → **0.0810** after matching (matching verified: p1/p50/p99 = 25/96/154 vs target 26/97/155) | Input-side intensity adaptation REFUTED; shift is texture-level |

Fleet cost note: at ~$4.19/hr, finishing the remaining ~55% would have bought
data that this QC shows cannot rank tiles or reveal text. Stop executed.

---

## 1. Normalization equivalence (test 1) — PASS, pipeline faithful

No screened tile overlapped the CLI region yet (w5 is still on its first z-slab;
its z=7424 rows are mask-skip bookkeeping only), so the equivalence was run
locally: same checkpoint (`D:\vesuvius-data\trackD\models\ink3d\ckpt_78k_fullsup.pth`,
EMA weights), single 256³ tiles fully inside the CLI reference region, both
normalizations, compared to `out\smoke_1203_prob.npy` avg-pooled 4x
(`qc_norm_equiv.py` → `qc_norm_equiv_result.json`).

| tile (z,y,x) | norm | f05 | f08 | corr vs CLI (L2, core) | MAD (core) | agree@0.5 |
|---|---|---|---|---|---|---|
| 7424,7168,11776 | screen [1,99] nonzero | 0.0392 | 0.0309 | 0.637 | 0.018 | 98.8% |
| 7424,7168,11776 | villa [1,99] all      | 0.0394 | 0.0310 | 0.639 | 0.018 | 98.8% |
| 7680,7168,11776 | screen                | 0.0907 | 0.0706 | 0.753 | 0.054 | 94.5% |
| 7680,7168,11776 | villa                 | 0.0909 | 0.0707 | 0.753 | 0.054 | 94.5% |
| CLI ref, same subvolumes | (its own) | 0.022 / 0.086 | 0.013 / 0.058 | — | — | — |

- **Screen norm vs villa norm: identical to the 3rd decimal.** On interior tiles
  (>99.7% nonzero) the percentile cuts differ by ~2 gray levels (16-196 vs
  14-196). The checkpoint config (`image_normalization: "percentile_minmax"`)
  matches the CLI implementation in `villa/vesuvius/src/vesuvius/data/volume.py`
  (`np.percentile(patch, (1.0, 99.0))` over ALL voxels, clip, scale).
- **Mismatch that does exist:** on low-fill edge tiles (>1% zeros) the CLI's p1
  is 0 while the screen's p1 cuts into the material histogram, stretching
  contrast harder. This inflates edge-tile responses and explains why every
  pmax=1.0 "top" tile is a low-fill (0.03-0.17) mask-edge tile. It amplifies an
  artifact but does not cause the blanket firing (full tiles fire 5-9% under
  both norms; the CLI overlay shows halos at mask edges too).
- Screen single-tile vs CLI (50% overlap, blended): r = 0.64-0.75 at the L2
  grid, threshold agreement 94.5-98.8% — the expected signature of
  blending-vs-single-pass, not of a bug. Same firing regime, same blob
  locations (verified visually on `out\smoke_1203_overlay.png`).

**Conclusion: the fleet's numbers are the model's real output on 1203.**

## 2. Firing morphology (test 2) — crack/edge-follower, not letter-shaped

All 27 pulled probL2 tiles vs CT L2 + released surface prediction
(`qc_morph.py` → `qc_morph_result.json`, gallery `qc_morph_gallery.png`;
interior = CT>5, sheet = surf>127):

| support | f05 median | mean-p median |
|---|---|---|
| on-sheet | 0.129 | 0.143 |
| off-sheet interior | 0.077 | 0.099 |
| mask-edge band (≤8 µm-vox of background) | 0.139 | 0.145 |
| background (air) | **0.0007** | 0.013 |

- Background is clean — the model is not hallucinating in air; it fires *inside
  material*.
- **In the scroll bulk (full-interior tiles) on/off-sheet ratio is 0.64-1.62 —
  effectively no discrimination.** The apparent "sheet-following" tiles
  (ratios 3-15x) are all low-fill fragments where the sheet support is tiny
  (down to 85 voxels) and coincides with the mask edge.
- Tile verdicts: 8 sheet-following / 8 diffuse-noise / 8 cavity-blobbing /
  3 edge-halo. The gallery and the CLI overlay agree: firing forms elongated
  ribbons along cracks, sheet boundaries and damage, with strong halos at the
  scroll surface — no letter-like clusters anywhere in the sample.

## 3. Discrimination check (test 3) — no usable relative signal

All 3,346 completed tiles from w0/w4/w5 (z slabs 0, 1024, 1280;
`qc_stats.py` → `qc_stats_result.json`):

- Distribution is tight and unimodal: f05 p5-p95 = 0.018-0.096; full-fill tiles
  f08 p5-p95 = 0.028-0.080. **There is no threshold that separates a sparse
  firing population**: <5% of tiles fire only past f05 > ~0.096 / f08 > ~0.079,
  i.e. cutting the smooth tail of one noise mode, not isolating a second mode.
- What f05/f08 correlate with: **fill fraction** (Spearman 0.43) — more material,
  more firing — a mild z trend (0.20, confounded with fill), and winding radius
  (-0.17/-0.20). Nothing text-plausible.
- pmax anti-correlates with fill (-0.23): the pmax ranking is an edge-tile
  artifact (edge tiles median pmax 0.999 with LOW f08 ~0.007).
- **Top-1% f08 tiles are spatially anti-clustered**: fraction with an adjacent
  top tile = 0.0 in both testable slabs vs 0.10-0.12 for random subsets of the
  same size. The extreme tail — the tiles a screener would flag — is noise.
- Nuance: the broader upper tail HAS regional structure. Top-5% f08 tiles
  cluster above chance (z=1280: 0.571 with-neighbor vs 0.294±0.094 random,
  ~3σ; z=1024: 0.619 vs 0.482±0.062; top-10% similar). So the f08 field is a
  smooth regional intensity map, not pure white noise — but given the
  morphology (section 2: fat crack/sheet ribbons, on/off-sheet ratio ≤1.6x, no
  glyph shapes), the plausible driver is regional CT texture
  (compression/damage/density), not ink. Unvalidated as a text signal; worth
  keeping the data to cross-reference against any future working detector, not
  worth fleet-hours to complete.

## 4. In-domain reference (test 4) — released Paris 4 predictions exist and are sparse

`s3://vesuvius-challenge-open-data/PHercParis4/representations/predictions/ink-3d/
20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr` (75784³ grid, uint8) —
same checkpoint lineage as the fleet's `ckpt_78k_fullsup.pth`
(`qc_paris4.py` → `qc_paris4_result.json`):

- 8 random interior 64³ blocks: f05 = {0, 0, 0.082, 0, 0.004, 0.023, 0, 0};
  pmax = {0.016, 0.008, 0.976, 0.012, 0.953, 0.969, 0.0, 0.337}.
- Healthy profile: **most blocks silent (pmax ~0.01), sparse confident firing
  where ink is.** Value percentiles p50/p90/p99 = 0.008/0.012/0.55.
- 1203 profile: pmax > 0.96 in 100% of 3,346 tiles, median f05 6-7%, no silent
  tiles. The output is out-of-family, not merely mis-calibrated: in-domain
  positives reach f05 ~0.08 — the same magnitude as 1203's *median* tile — so no
  recalibrated threshold can recover Paris4-like sparsity.
- Granularity caveat: at 256³-tile granularity the released Paris 4 pred also
  fires 1.4-5% in the two regions probed in test 5 (it contains its own FPs).
  The decisive contrast is the shape of the distribution: in-domain firing is
  regionally concentrated with many near-zero blocks (5/8 random 64³ blocks
  exactly 0.000); 1203 is a uniform 5-7% blanket with a hard floor
  (p1 of f05 = 0.0075) and no quiet tiles anywhere.

## 5. In-domain control + histogram-matching salvage probe (test 5)

`qc_indomain_histmatch.py` — (A) the screen pipeline run on Paris 4 CT at a
released-pred-active and a released-pred-silent location (end-to-end health
check of ckpt+normalize+inference); (B) a 1203 tile histogram-matched to the
Paris 4 nonzero-CT distribution, re-inferred — the cheapest input-side
domain-adaptation lever.

**A) In-domain control — PASS (pipeline + checkpoint healthy on Paris 4):**

| Paris4 256³ tile | released pred f05 | our screen f05 | corr(ours, released) |
|---|---|---|---|
| active region (21700,8347,10941) | 0.0512 | 0.0695 | **0.873** |
| quiet region (51081,18488,16432) | 0.0137 | 0.0282 | 0.741 |

The exact fleet pipeline (checkpoint, screen normalization, single-pass 256³)
reproduces the released Paris 4 prediction where it fires and stays sparse
where it is quiet. Everything upstream of the volume is healthy — the 1203
behavior is the volume's doing (domain shift), closing the case for (a) bug.

**Intensity covariate shift, nonzero-CT percentiles [p1,p25,p50,p75,p99]:**
Paris 4 = [26, 49, 91, 116, 158] (mean 86.4, sd 36.4) vs 1203 tile =
[16, 39, 47, 103, 196] (mean 72.3, sd 44.9). 1203's material histogram has a
much darker median (47 vs 91) and a longer bright tail — a distribution-shape
shift that percentile min-max normalization cannot correct (it only pins the
1st/99th percentiles). This is a concrete, fixable input-side cause for
out-of-family activations.

**B) Histogram-matching probe — input-side adaptation REFUTED.**
(Local GPU run stalled behind Track A training; final numbers from the minimal
pod-side rerun on w0, `qc_histmatch_pod.py` → `qc_histmatch_result.json`,
`qc_hm.log`, probL2 maps `qc_hm_base.npy` / `qc_hm_matched.npy`.)

| condition | pmax | pmean | f05 | f08 |
|---|---|---|---|---|
| Paris4 control tile (ink-active region) | 0.989 | 0.081 | 0.0695 | 0.0575 |
| 1203 tile (7424,7168,11776), baseline | 0.993 | 0.049 | 0.0392 | 0.0309 |
| same tile, histogram-matched to Paris4 | 0.980 | **0.096** | **0.0810** | **0.0590** |

The pod run also reproduced the laptop's Paris4 control (f05 0.06952 on both
machines) and the laptop's baseline for this exact tile (f05 0.03924, test 1) —
cross-machine consistency of the whole harness. The matching itself is verified
correct (matched nonzero percentiles p1/p50/p99 = 25/96/154 vs target
26/97/155), yet firing **doubled**, exceeding the in-domain ink-active level. Interpretation: the model's false positives on
1203 are driven by texture/morphology (crack patterns, sheet compression,
fiber lattice at 2.4µm) — giving 1203's texture Paris 4's intensity profile
makes it look *more* ink-like to the model, not less. Intensity covariate
adaptation (histmatch, and by extension mean/std alignment or percentile
tweaks) cannot rescue this checkpoint on 1203. The remaining salvage paths are
model-side, not input-side.

## Salvage-path verdict (post-probe)

**Executed: fleet stopped** (pods terminated). Forgone: remaining ~55%
coverage; saved ~$4.19/hr. The collected artifacts are kept — they are a
complete, quantified negative control plus a working QC methodology.

**Refuted paths — do not spend more on these:**
- *Threshold recalibration / per-tile z-scoring of this output.* The tail is
  anti-clustered noise; no recalibration creates a ranking (test 3).
- *On-sheet gating in post.* Median bulk contrast ≤1.6x over a 5-7%
  false-positive floor (test 2) — cannot surface text.
- *Input-side intensity adaptation (histmatch / mean-std alignment /
  percentile tweaks).* Directly refuted by test 5B — firing doubled. The shift
  is texture/morphology-level; no input intensity transform fixes it.

**Remaining viable paths, in order of preference:**
1. **Scroll-specific fine-tuning per the 1667 pseudo-label playbook**: generate
   conservative pseudo-labels on 1203 (or annotate a few high-confidence
   patches), fine-tune the v3-78k checkpoint with 1203 patches in the loader
   (the training config already supports self-distill dynamic labels), then
   re-screen. Highest expected value; cost = one fine-tune run + one screen
   relaunch. The saved band stats/probL2 become the before/after benchmark.
2. **Cross-scroll-trained models**: run ink_9um on the 9µm 1203 volume (a model
   trained across scrolls should not carry Paris4-specific texture priors).
   Cheap to probe on a few tiles before committing a fleet.
3. **Bank the negative**: no further compute on 1203 2.4µm ink until a model
   demonstrably transfers; the QC harness here (norm-equivalence, morphology
   vs surface pred, in-domain distribution comparison, single-tile adaptation
   probe) is the acceptance test any candidate must pass on ~30 tiles BEFORE
   any fleet launch.

If any relaunch happens, also fix two screen-side artifacts regardless:
(a) use the CLI's percentile-over-all-voxels normalization to stop edge-tile
contrast stretching; (b) gate stats to on-sheet voxels (surf>127 at the probL2
grid) so pmax/f05 stop rewarding mask edges and cracks.

## What this means for the Aug 31 report

This round is submission content, not just an operational loss:

- **A quantified cross-scroll transfer failure.** Same checkpoint, same
  pipeline, verified end-to-end healthy on Paris 4 (r=0.87 against the released
  prediction), produces a 100%-tiles-firing, unimodal, unrankable output on
  PHerc1203 2.4µm — with the failure localized to texture/morphology rather
  than intensity by a controlled histogram-matching experiment. That is a
  clean, reproducible negative result about ink_3d_dino_guided's domain
  robustness that (to our knowledge) nobody has published numbers on.
- **A working live-QC methodology for fleet screens**: reduced per-tile stats +
  downsampled prob pulls + a 5-test battery (pipeline equivalence, morphology
  vs surface pred, tail clustering, in-domain distribution reference,
  single-tile adaptation probe) that reached a defensible stop decision at
  ~45% spend. The battery is checkpoint-agnostic and is the acceptance gate
  for the next candidate model.
- **Concrete numbers to cite**: in-domain sparsity profile (5/8 blocks silent,
  p90=0.012) vs out-of-domain blanket (median f05 6-7%, 0 silent tiles);
  on/off-sheet contrast 1.4-1.6x; histmatch f05 0.039→0.081; edge-tile pmax
  artifact and its normalization-support cause.
- Framing suggestion: "negative screen, positive methodology" — the prize
  narrative is that Track D now has a validated, cheap gate that prevents
  burning fleet budget on non-transferring models, plus the first quantified
  characterization of that non-transfer.

## Files

All in `C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\`:

- `qc_norm_equiv.py` / `qc_norm_equiv_result.json` — test 1 (local GPU)
- `qc_morph.py` / `qc_morph_result.json` / `qc_morph_gallery.png` — test 2;
  `qc_zoom_tile.py` / `qc_zoom_w5_1280_9984_12032.png` — zoomed look at the
  highest-f05 full tile
- `qc_stats.py` / `qc_stats_result.json` — test 3
- `qc_paris4.py` / `qc_paris4_result.json` — test 4
- `qc_indomain_histmatch.py` — test 5A + intensity stats (local run; part B
  stalled behind Track A GPU training and was killed)
- `qc_histmatch_pod.py` / `qc_histmatch_result.json` / `qc_hm.log` /
  `qc_hm_base.npy` / `qc_hm_matched.npy` — test 5B (w0 pod rerun; pod since
  terminated)
- `round_1\` — pulled fleet samples (stats jsonl + 27 probL2 tiles + manifest)
