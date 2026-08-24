# Track D — Ink-First Forensics: running log

Proposal: claude.ai/code/artifact/cd138b85-1f8c-45b8-8e29-09c8fce7f7f1 · Plan: WORKPLAN.md §Track D

## 2026-08-16 (day 1)

### Data groundwork
- Catalog subtrees dumped to `trackD/meta/*.json` (from open-data metadata.json local copy).
- All 12 GP 9µm volumes share uint8 window f32 [-0.03, 0.145] → cross-scroll DN comparability.
- GP 9µm processing recorded per scan: Paganin delta_beta=1000, unsharp coeff 4.0 sigma 1.2.
- Paris 4 45.5µm 74/110keV pair: delta_beta=10, **unsharp 0.0** (near-attenuation, unsharpened);
  windows 74keV [-0.058,0.27], 110keV [-0.04,0.2]; shapes (4071|4066, 2264, 2264).
- Scout: ink_9um corpus mapped (`trackD/meta/ink9um_corpus.md`) — w035 + 4 native labeled
  segments verified downloadable (S3 surface volumes + HF-bucket labels).
- Scout: 1667 ground truth mapped (`trackD/meta/pherc1667_groundtruth.md`) — per-segment ink TIFs
  (the June-2026 read) + on-7.91µm tifxyz grids = per-vertex letter coords in the low-res frame,
  zero matrix math. Full transform chain also published and verified self-consistent.
- PHerc1203 2.4µm band HAS a released m7-L2 surface prediction + normal grids
  (`representations/predictions/surfaces/20260319130212-...`), plus ~22 auto-grown raw segments (9µm).

### K1 — intensity-AUC kill test → **CHANNEL DEAD (clean kill)**
- w035 (strongest known 9µm ink): AUC 0.4956–0.5059 across detrending scales (raw, σ=25/50/100).
- Pre-registered rule |AUC−0.5|<0.02 met → absolute-intensity channel carries no linearly-recoverable
  ink signal at 9.362µm even in the best case. Candidate D's retraining branch cancelled.
- Files: `k1_intensity_auc.py`, `out/k1_w035_auc.json`, `out/k1_w035_maps.png`. QC audit in flight.

### K2 — spectral information ceiling → quantization NOT binding; real cliff at ~15–19µm
- PHerc0813 / PHerc1203 / PHerc0139 (256³ papyrus ROIs, level 0): PSD stays 1–6 decades above the
  uint8 quantization floor through the whole per-axis band → the release format is not the ceiling.
- Real ceiling: PSD cliff at ~0.25–0.35 cyc/px (half-period ~15–19µm) — the recon's own noise/MTF.
  This cliff (per scroll) is the detectability-index core. Crossover-at-floor stats (0.67–0.71) are
  3D-corner artifacts; do not quote them per-axis.
- Files: `k2_spectral_ceiling.py`, `out/k2_spectra.{png,json}`. QC audit in flight (incl. air-ROI
  reference discriminator + multi-ROI variance).

### K3 — dual-energy screen: stage 1 done; QC caught a sign error → rebuild stage 2
- Volumes are **natively voxel-aligned** (mask IoU 0.9816, phase-corr shift (0,0,0) at L3).
- Papyrus ratio median 1.248 (Compton expectation ~1.13 — offset problem, see below).
- QC audit (`qc/k3_stage1_review.md`) — 4 blockers, key findings:
  1. **Pb K-edge (88keV) sits BETWEEN 74 and 110keV → Pb ink = LOW ratio (µ74/µ110≈0.67), co-bright.**
     The high-ratio screen was structurally blind to lead. High tail = mid-Z (Ca/Fe) channel.
  2. Dense-cluster ratio 1.705 inflated by rim/selection bias → ~1.5–1.56 after erosion; surviving
     dense phase = spatially coherent Ca/Fe-like incrustation 2.5–3mm under the surface, one flank,
     clumped (not writing). 74keV volume ~22% blurrier (Paganin width ∝ √λ) → rim artifacts.
  3. Air offset: f74≈+0.0025, f110≈+0.0080 where air should be ~0 → ratios biased ~10%;
     calibrate zero on interior air before any chemistry claims.
  4. Alignment constrained only to ±4 L0 voxels; sub-voxel refine before scaling.
- GOOD NEWS: σ_ratio≈0.12/voxel (quantization <1% of variance) → realistic Pb loadings = >5σ signal.
- Stage 2 rebuild: low-ratio+co-bright channel, air-offset zero, 3-voxel rim erosion, blur
  equalization (match 74/110 PSFs), sub-voxel alignment, then full-scroll recto screen incl. title region.
- Files: `k3_dualenergy_stage1.py`, `k3_dense_clusters.py`, `out/k3_*.{json,png}`, caches on D:.

### Stage 1a — 1667 letter-contrast (model-free): first run had a selection bug, rerunning
- w032 data cached: 2.4µm-model ink TIF + 1.129µm-model TIF + on-7.91µm grids (`D:\vesuvius-data\trackD\w032`).
- Bug found via renders: densest-"ink" tiles were saturated model garbage over the segment's EMPTY
  margin (maps off-papyrus, black renders, degenerate AUC 0.5). Mapping chain itself verified —
  tile (15,4) rendered real papyrus at the letter coordinates.
- Fix: select letter-like tiles (ink frac 0.02–0.30, bg present, grid ≥98% valid), on-papyrus check
  (≥70% nonzero at k=0), sample mask restricted to on-papyrus. Rerun in flight.

### Crown jewel — PHerc1203 2.4µm band smoke test: in flight
- Papyrus-dense ROI at band center: bbox 7392:7776, 7168:7552, 11776:12160 (probe fill 1.00).
- `vesuvius.predict --model_path hf://scrollprize/ink_3d_dino_guided` streaming volume from S3,
  .venv314, cuda. First-ever (public) ink inference on GP-eligible 2.4µm data.

### QC round 2 (evening) — both Stage-0 conclusions corrected by audit
- **K1 kill OVERTURNED** (`qc/k1_k2_review.md`): the 17-slice z-mean structurally cancels an
  ANTISYMMETRIC ink depth profile (+8.6 DN at z=12, −6.3 DN at z=5-6). Depth-resolved:
  per-slice AUC 0.553 @ z=12; frozen depth-contrast statistic mean(z10-14)−mean(z4-8) →
  **AUC 0.5715** (3.6× kill threshold), replicated across supervision patches. Texture channels:
  z-std AUC 0.564, in-plane texture 0.423 (inverted, |dev| 0.077). Labels are 2D (z=14 only) so
  max-projection was lossless. → Intensity is ALIVE at 9µm when depth-resolved. K1b validation
  on held-out w039/w040/w041/w044 launched (`k1b_depth_validation.py`).
- **K2 corrected, optimistically**: vs streamed air-ROI noise references, papyrus structure stays
  above 2× air power until q=0.43-0.50 (half-period ~9.4-10.8µm) → releases are SAMPLING-limited,
  not information-limited. My "cliff=ceiling at 15-19µm" claim withdrawn; the shipped
  `smallest_recoverable_halfperiod_um` numbers are 3D-corner artifacts — do not quote. One ROI per
  scroll insufficient (25× site spread at q=0.5); final index = ≥5 ROIs/scroll, median±IQR of
  (a) air-referenced structural bandwidth, (b) structural SNR at q=0.25, (c) DN headroom.
- **Stage 1a first positive**: pooled AUC 0.651 at k=0 (peak k=-1..0), best tile (7,5) 0.917;
  one tile inverted (~0.44). Renders show background sets may be contaminated by off-segment
  no-data zones mapping to bright off-surface material — would UNDERSTATE letters-brighter signal.
  Adversarial verification agent in flight (placebo rotation, independent 1.129µm mask,
  no-data exclusion, stroke-core erosion, overlay money-figure).
- Ops note: ~4:31 PM system event killed all python processes (Dataset059 download restarted,
  K3 stage2 + 1203 smoke died mid-run; both relaunched — K3 resumes from L1 cache).

### Round 3 (late evening) — S1a refuted then rebuilt; the heterogeneity story emerges
- **S1a-v1 REFUTED by verifier** (`qc/s1a_verification.md`): background sets were 100% no-data
  pixels (ink TIF nonzero floor ~28; my <20 threshold selected only 0s), and tile selection
  REQUIRED >=30% no-data → all tiles on the segment's central tear. The 0.651/0.917 measured
  "papyrus vs segmentation-failure zone". Placebo nulls bracketed the corrected AUC (0.427).
  Verdict: v1 evidence dead; thesis untested. DO NOT cite v1 anywhere.
- **S1a-v2 (corrected design) — controlled positive, pending re-verification**:
  dual-model letters (ink24>=200 AND ink11>=150), on-prediction background [28,60] >=8px from
  no-data, dual-coverage text-region tiles (overview render shows unmistakable text lines in both
  models' maps), rot180+shift placebo nulls through the identical pipeline.
  Pooled AUC: 0.513(k=-4) → **0.614(k=-1), 0.609(k=0)** → 0.512(k=+4) — surface-localized profile;
  both nulls flat/driftig without a peak. Tile (9,9): **0.902 at k=-1** (nulls 0.433/0.380),
  peaked profile. Tile (11,12): coherently INVERTED (0.366, peaked) — letters darker there.
  → mixed-polarity, spatially heterogeneous GP-resolution signal. Verifier re-audit in flight
  (deep null distribution, profile-shape specificity, autocorrelation-aware CIs).
- **K1b held-out validation**: w035 reproduces 0.5717, but w039=0.513, w040=0.501 (w041 HF 429,
  w044 label-path issue — retry pending). Depth-contrast intensity channel = REAL BUT
  SEGMENT-SPECIFIC. Consistent with S1a-v2 tile heterogeneity.
- **Emerging unified finding**: 9µm-class signal exists but is spatially heterogeneous in
  strength AND polarity — per-region detectability measurement (our index) is the right product;
  naive uniform models will average it away.
- Ops: repeated background-task stops (~5:20-5:30 PM) hit only my session's tasks (Ben's
  downloader survived) → switched to foreground-serial execution + agent-based QC.
- 1203 smoke: vesuvius.predict hf:// loader expects nnUNet layout; ink_3d_dino_guided is a
  vesuvius.train ckpt (use ema_model) → route = local .pth download + train_py loader path.
  Checkpoint download pending (was killed twice).

### Round 4 (night) — S1a-v2 also refuted; K3 null lands; 1203 pipeline proven; 2 Track B bugs
- **S1a-v2 REFUTED** (`qc/s1a_v2_verification.md`): (9,9)'s 0.902 = off-sheet air in background
  again (depth-flat samples, raw0 40-50, fooled the raw0>5 gate); on-sheet corrected pooled
  excl (9,9) = 0.5035 (z=0.34, 18/40 nulls above). Profile shape fakeable (12-29% of nulls).
  ink11 resample misregistered ~(-36,+39) ds4 px → dual-confirmation premise broken. ESS ×146
  deflation. Inverted tile = surface-offset geometry. Verifier's v3 requirements documented.
  **DECISION: park w032 letter-contrast (v3 only if time allows). The defensible letter-level
  result is w035 depth-contrast (0.5717, human labels, QC-validated) + K1b heterogeneity.
  The two verification reports become methodological content for the submission
  (how detectability claims fail + a verified protocol = the anti-hallucination story).**
- **K3 stage 2 COMPLETE** (slab-streaming rewrite after 2 OOMs): full-scroll Paris 4 screen at
  L1 (91µm). **LOW/heavy-metal channel: 2 singleton voxels in ~268M tested = clean calibrated
  NULL — no Pb-bearing ink above sensitivity floor anywhere incl. title region** (left tail far
  sub-Gaussian → genuine absence, not noise). HIGH/mid-Z channel: 51,343 vox / 30,258 clusters,
  top = the known flank incrustation (z_L0~1240, Ca/Fe-like) + z~3900-4020 group. σ_det=0.126.
  TODO for writeup: convert σ → Pb-fill-fraction sensitivity bound (QC framework in
  qc/k3_stage1_review.md); L0 verification ROIs on top HIGH clusters.
- **PHerc1203 2.4µm band ink inference WORKS end-to-end on the laptop**: vesuvius.predict
  (local ckpt, train_py loader, --input_anon s3://, TORCH_COMPILE_DISABLE=1, 192³ patches)
  → blend_logits → sigmoid → overlay. 216 patches on ROI (7281,7007,11615)+672³.
  Result imagery: predictions concentrate in damage cavities/sheet-void edges (domain-shift
  FP morphology), ROI was a damaged zone. Next: cleaner parallel-sheet ROIs, higher threshold,
  on-sheet vs in-void prediction-density artifact metric.
- **Track B: two genuinely-encountered defects captured tonight**:
  1. torch.compile lazy TritonMissing crash on Windows+CUDA in INFERENCE path too
     (inference.py:712-719 guard wraps only the compile() call; no CLI opt-out;
     workaround TORCH_COMPILE_DISABLE=1). Same root cause as held train.py finding →
     one issue covering both sites, motivated by the 1203 First Letters run.
  2. NEW: vesuvius.finalize_outputs crashes on EVERY invocation — build_parser() refactor
     left `parser` unbound in main() (finalize_outputs.py:588/590/598 reference it;
     NameError at 590). 2-line fix. Hit while finalizing the 1203 smoke logits.
  → Draft both issues tomorrow; Ben writes/edits commentary per CONTRIBUTING.md.

## 2026-08-16 session 2 (evening)

- Dataset059 COMPLETE (3,509 files; the ~92GB estimate was wrong — 17.4GB total). A parallel
  session owns Track A: smoke runs done, ablation cycle 1 (v0-v3) authored, v1_sheetcomp trained
  to epoch 40+ while observed; ALSO filed PR #1480 (train-side triton fallback) at 00:25Z.
  Coordination rule: I stay off the GPU + experiments/trackA; Track D = CPU/streaming lanes.
- K1b completed (minus w044, HF 429): held-out mean 0.4934 — w039 0.513, w040 0.501,
  w041 **0.4657 INVERTED**. Depth-contrast does not transfer; polarity flips by segment.
  Report line: 9µm intensity statistics are region-sign-unstable → per-region index + learned
  models are the right tools. (w044 retry pending.)
- K2b detectability index running (detached): first results differentiate sharply —
  0125/0191/0211 at Nyquist bandwidth, SNR@0.25 ≈ 100-128; **PHerc0257 degraded (SNR ~23,
  bw 0.445, DN headroom 143)** → ranking is meaningful.
- 1203 clean-ROI probe v2 (interior-masked, surface-pred-scored): 6 verified ROIs; sweep script
  + on-sheet/off-sheet artifact-metric report ready (sweep_1203_clean.ps1) — queued for GPU.
- K3 mid-Z verification imagery: nodular incrustation confirmed (#1/#2/#4); streaky fiber-ratio
  patterns flagged as PSF-mismatch suspects (#3/#5).
- **K3 sensitivity bound (workflow + hostile referee, 6 errors fixed)**: realistic Herculaneum Pb
  loadings (Tack 84, Brun 16 µg/cm²) = ~1.0σ/0.3σ at stroke geometry → the LOW-channel null
  validates FPR calibration but does NOT exclude Tack/Brun-level metallic ink. Supersedes the
  stage-1 ">5σ" claim. Upgrade path: surface-conformal matched filter (~5× σ reduction) would
  make Tack-level ink ~5σ-detectable. Files: out/k3_sensitivity_bound.{md,json} + referee notes.
- **Track B status after adversarial review (trackD/issue_drafts/REVIEW.md)**:
  - torch-compile → NEEDS-BEN, rewritten predict-focused (inference.py:709-719 + no opt-out
    flag; cites #1480 for the train half). Blocking gap: paste the predict-side traceback
    (re-capture queued behind Track A GPU).
  - finalize_outputs → BLOCKED-duplicate (axiosdevs PR #1430, Aug 13, strict superset).
    Support-comment draft for Ben included in the draft file instead.
  - All file:line claims re-verified at origin/main 465cab06; re-run duplicate searches at filing time.
- RunPod costing done for Ben: 1203 full band screen ≈ $45-60 (community 4090 fleet, --num_parts
  sharding), month of everything ≈ $150-300. Awaiting his decision + API key if he wants it.

## 2026-08-16 session 2, cloud phase (~10 PM)

- Ben provided RunPod API key, cap $80. Account balance $170.36 at start.
- Smoke pod (RTX 5090 32GB, $0.69/hr): env recipe hardened — uv sync installs almost nothing
  (torch left to user by design); working recipe = uv sync + torch==2.11.0+cu128 &
  torchvision==0.26.0+cu128 (host driver caps at CUDA 12.8; PyPI torch 2.13 incompatible)
  + explicit models-extra dep list (minus volume-cartographer/cucim). Linux env: 5 min total.
- Cloud throughput: CLI 192³ = 0.86 s/patch (~6× laptop); native 256³ FITS (32GB) at
  ~2.8× voxel throughput.
- **Custom streaming screener built+validated** (runpod/screen_band.py): loads ema via villa's
  train_py path (NetworkFromConfig + _select_train_py_state_dict), non-overlap 256³ tiles,
  L4-mask gating (~73% skipped), bounded 6-thread S3 prefetch → **1.49 tiles/s** sustained.
  Reduced artifacts only: per-tile stats jsonl + 4×-downsampled prob (uint8).
- **FLEET LIVE ~9:55 PM: 6× RTX 5090 screening the ENTIRE PHerc1203 2.4µm band**, interleaved
  z-tile sharding (worker i takes iz % 6 == i). Projected ~169k tiles, ~5.5h wall, ~$19-25.
  Burn $3.49/hr; spend at launch $0.62. fleet.json has pod ids/endpoints; fleet_status.py
  = one-shot status + auto-terminate-all past $70 session spend.
- Next: heartbeat checks; on completion — pull stats + probL2, assemble band-level ink map,
  rank hotspots, then dense/overlap re-run on top candidates + cross-model confirmation pass.

## 2026-08-17 early AM — cloud screen outcome: pipeline proven, model refuted on 1203

- Fleet ran 6× RTX 5090; one bad host cost 3 pods (caught by heartbeat, ~$0.40); replacements
  CUDA-verified pre-provision. ~30k tiles screened with full per-tile stats (all six shards'
  jsonl secured to qc_live/final_stats/ before termination).
- **Live QC verdict (agent, 5 controls)**: screen pipeline FAITHFUL (reproduces released Paris 4
  predictions r=0.873 active / r=0.74 quiet); 1203 output = scroll-wide blanket firing
  (median tile fires at Paris4 ink-active rate, 5× quiet rate); NO rankable tail; morphology
  crack/edge-driven. Root cause quantified: covariate shift (1203 material p50 DN 47 vs 91,
  longer bright tail) — **but histogram matching REFUTED as fix: matched tile f05 0.039→0.081
  (worse than in-domain active 0.070) with mapping verified correct → shift is
  texture/morphology-level, not first-order intensity.**
- Fleet stopped on QC verdict (before completion — raw output family cannot rank or reveal text);
  all pods terminated. **Total cloud spend: $4.46 of $80.**
- Remaining paths for 1203 ink: (a) scroll-specific fine-tune (1667 pseudo-label playbook,
  A100 80GB ≈ $8-17/round, ~6 rounds ≈ $50-100, days), (b) cross-scroll-trained ink_9um on the
  9µm volume via released recto surfaces (cheap, designed for transfer), (c) bank the quantified
  negative for the Aug 31 report. RECOMMENDATION: (b) first, then decide on (a).
- Reusable assets: runpod/rp.py (API + budget guard), provision.sh (validated Linux env recipe:
  uv sync + torch 2.11.0+cu128/torchvision 0.26.0 + explicit dep list), screen_band.py
  (streaming screener, 1.5-2.7 tiles/s/pod), fleet_{launch,add,stop,status}.py, qc_pull_samples.py.
  Any relaunch ≈ 10 min + $0.
- QC agent finalizing round1_verdict.md (full test table + salvage verdict + report framing).

## 2026-08-17 — salvage analysis of the failed screen (3-agent workflow, all perm-null disciplined)

- **Failure mechanism REVISED (morphology, 63 prob maps)**: not crack-following — the model
  PAINTS INTACT SHEET SURFACES: 79% of fired voxels = thin (25-40µm) sheet-conformal percolating
  membranes; through-sheet crack ribbons ≈ 0.2%. Coplanarity real (17.5° vs 50-60° nulls) but
  evidentially neutral. No ink-plausible subpopulation (patch class = round flecks, unclustered).
  → On-sheet gating can NEVER rescue this output; any fine-tune needs 1203 sheet-surface
  HARD NEGATIVES. (Concrete training recipe for the fine-tune option.)
- **QC round-1 correction**: raw top-1% "anti-clustering" was a coverage/small-n artifact —
  full 29,748-tile data shows strong clustering (z=-12.6). Operational stop-verdict unchanged.
- **Scalar field (H1/H5 SUPPORTED)**: firing driver = mean CT density (Spearman 0.73, R²=0.62
  nuisance model); rim wraps fire 2.5× less; field traces the wrap spiral. Residual ~36% variance
  carries real spatial structure (Moran's I 0.289 vs 0.213 block null, z=+31) explained by NO
  tested covariate — parked as cross-reference layer (salvage/tiles.parquet).
- **Damage-atlas reframe (H6) REFUTED by sign inversion**: f05 ANTI-correlates with damage
  (interior ρ=-0.54 with crack texture; LOW where m7 has holes) — it's a preservation/density
  readout, strictly dominated by free CT stats (meanCT hole-AUC 0.892 vs 0.855; incremental
  value exactly zero: partial ρ=-0.001, LOSO ΔR²=0.0000). No community atlas; honest
  failure-mode map built anyway (salvage/atlas_1203.{npy,png}) as the fine-tune before/after baseline.
- Also learned: released m7 surface predictions are BINARY at every level — no confidence field
  exists anywhere (relevant to future difficulty-proxy work).
- Net: last night's data = fully understood failure + hard-negative recipe + corrected QC
  methodology + one unexplained-residual thread. Nothing ink-like anywhere in it.

## 2026-08-17 midday — ink_9um probe executed end-to-end (pod, total cloud spend now $5.61)

- **w035 positive control: AUC 0.9991/0.9982** (seed42/43 forward; reverses ~0.5 → z-orientation
  matters and ours is right). The published "letters found at native 9µm on w035" is independently
  reproduced by our stack — figure out/ink9um_w035_control.png (letters plainly legible).
- **Custom tifxyz renderer validated**: re-rendered w035 prediction correlates r=0.813 with the
  prebuilt-SV prediction. (Runbook: trackD/runpod/ink9um_runbook.md; native-3D path impossible —
  all 14 ckpts are flat-mode.)
- Ops lessons: /workspace is 20GB not 40 (disk-fill killed round 1 of renders); villa volume_io
  disk-cache path triggers zarr-3 "ContextVar created in a different Context" on the 1203 renders
  (no-cache works) — possible Track B item, uninvestigated; segment auto_grown_20251005221856743
  fails render even without cache (unknown cause, skipped).
- **PHerc1203 result (2 segments × 2 seeds × 2 dirs)**: NOT ink_3d-style blanket (quiet regions
  exist; patchy), firing fraction 0.10-0.16 ≈ control's 0.12, seeds agree — but NO visible text-row
  organization; dense worm-like stroke-scale filaments. Gallery: out/ink9um_1203_gallery.png.
  Verdict agent running: line-periodicity (ruling wavelength) + stroke morphology + quiet-band
  analysis vs the w035 template → TEXT-LIKE / TEXTURE-ONLY / INCONCLUSIVE.

## 2026-08-17 — ink_9um 1203 VERDICT: stroke-scale texture only; instrument proven; coverage now the limit

- Full verdict: salvage/ink9um_1203_verdict.md + verdict_gallery.png. All 8 maps fail every text
  signature simultaneously; control passes all decisively:
  - Ruling periodicity: control 4.68-4.73mm @ stable θ, z=+25.6/+26.5/+34.1 (matches label-derived
    row spacing 4.59mm within 3%); 1203 z=-2.7..+1.0, orientations wander, band-edge pileup = null.
  - Morphology: letters vs 1203 KS D=0.92-0.97 (complete separation); control-BLANK vs 1203
    D=0.06-0.23 (same family). 1203 filaments = blank-papyrus response (5-9px wide, wormy).
  - **z-symmetry diagnostic (keep this)**: control fwd-vs-reverse r=0.076 (ink on ONE face);
    1203 r=0.51-0.60 (symmetric through sheet = texture). Powerful, cheap discriminator.
  - Intensity: 0.06-0.11% of 1203 px above control-blank p99 vs 69.4% of true letter px.
  - ink_9um ≠ ink_3d failure: quiet regions kept, seed-consistent, letters proven on w035
    (pixel AUC 0.964 recomputed) → a WELL-BEHAVED "no ink found" reading.
  - Calibration note: w035 letters are 242px/2.26mm tall, rows 4.59mm — my 35-55px prior was 6× off.
- **Coverage caveat**: only ~28.5cm² of arbitrary auto-grown patches — could be uninscribed papyrus.
  Battery is now turnkey (~5 min/segment). Next options: (1) 2-3 segments from distinct winding
  radii/faces, (2) z-offset sweep ±1-2 layers on segment A (fwd/rev symmetry hints mid-sheet
  sampling), (3) standing tripwire: component value>195 & area>10⁴px & width>30px → human look.
- Cloud spend total: $5.61. Aug 31 report now has: proven instrument + calibrated positive control +
  two characterized model behaviors on 1203 + turnkey text-detection battery + per-scroll index.

## 2026-08-17 — full-data comb (3 hunters + skeptic): nothing ink-like survives; 2 keepers

- Verdict doc: comb/COMB_VERDICT.md. All 8 hunter flags null-tested; scorecard:
  - w035 beyond-labels catalog (18 letter-class / 14 cross-seed components): real catalog,
    ALL THREE statistical supports died under nulls (cross-seed reproduction explained by
    size+strength alone, P=0.61; ruling alignment fails texture-bootstrap p=0.088-0.43;
    z-reversal patterning = denominator artifact). One 4-component row keeps p=0.013-0.017
    but post-hoc. → honest gallery figure only (comb/w035_beyond_labels.png).
  - 1203 local-symmetry hunt: EMPTY + the premise corrected — tile-scale z-symmetry is NOT
    ink-specific (letters trend HIGHER tile-r, p=0.97; only the MAP-scale contrast is real).
    Earlier LOG note "keep z-symmetry diagnostic" now qualified: map-scale only.
  - Tripwires at 4 thresholds × 8 maps: zero (control trips at every threshold — no blind spot).
  - Dense-ink scan (57 cubes, 14 scrolls): no PHerc172-class signature anywhere. CONFIRMED
    REPORTING CAVEAT instead: uint8 window clipping saturates 5-33% of in-mask voxels in 8/14
    scrolls (0191 33%!) → bright-ink screens + the k2b DN-headroom metric are censored there;
    PHerc0139's low saturation anti-tracks with the density ranking. Goes in the report.
  - K3 z~3900-4020 regional ratio elevation: REFUTED by 28-window matched null (ordinary
    patchy variation, Ca/Fe family). Streak clusters #3/#5 CONFIRMED PSF artifacts (sigma_eq
    0.56-0.62 px; collapse 3.6-5.5× at match).
  - 36%-residual thread: half-decomposed (7/20 top tiles = off-sheet FP floor at rim,
    13/20 = damage-morphology response); remains parked, mystery reduced.
- NET ANSWER to "is there anything there": no recoverable ink signal exists in any data collected
  so far; every apparent signal died under a null with numbers attached. Keepers: the clipping
  caveat, the symmetry-scale correction, the beyond-labels gallery, the two-family residual story.

## 2026-08-17 — REPORT BUILT (trackD/report/)

- Structure: REPORT.md (exec summary + 5-min reader table + section index + limitations +
  released artifacts) → sections/01-04 → REPRODUCIBILITY.md → BEN_TODO.md.
  ~14,000 words + 16 figures. Sections drafted by 4 agents; integration/harmonization done
  in-session after the workflow's integrate+review agents died on the monthly spend limit.
- **CORRECTION to my own 2026-08-17 LOG claim** ("all four 8.64µm degraded vs all nine 9.362µm
  good"): PHerc0257 is 9.362µm/113keV AND degraded (SNR 22.7). True split = 4/4 vs 1/9,
  Fisher p=0.007. §1.4 states it correctly; the degraded tier is FIVE scrolls
  (1218, 0268, 0257, 0800, 1447), not four.
- Integration fixes applied: duplicated index table in §4.5 removed (it carried the superseded
  "three degraded scrolls" count + a wrong residual-reference bias direction — residual is
  conservative for BANDWIDTH only, anti-conservative for mid-band SNR); duplicated fleet-battery
  table in §4.2 → pointer to §3.7; dangling "§[X]" → §3.6-3.9; **all figure paths in all four
  sections were broken** (`figures/` from `sections/`) → `../figures/`.
- Self-run verification (report/scripts/verify_report.py, replaces the killed fact-check agent):
  26 headline claims checked against primary JSONs — 14 index SNRs, 4 control AUCs, 1203 firing
  range, K3 sigma + both channels, K1b 4 segments + held-out mean, tier gap, plus every figure
  and internal link. **0 problems.**
- NOT done (spend limit): independent line-by-line fact-check + external prize-fit review.
  Both are one workflow re-run away; flagged in BEN_TODO.md §D.
- BEN_TODO.md: 7 must-do items (voice/claims/license/title), 4 judgement calls with
  recommendations, 3 Track B items, suggested work order (~2h of Ben's time).

## 2026-08-17 evening — CORPUS SURVEY COMPLETE (80/80) + cache-fix PR verified on real data

- **Corpus screen: every published segment of the GP scrolls with meshes — 80/80, 0 errors,
  0 tripwire hits.** PHerc1203 22/22, PHerc1447 52/52, PHerc0800 6/6. 4-pod 5090 fleet,
  render→infer(seed42 fwd+rev)→stats per segment; 150 prediction maps (134MB) pulled home.
  Answers the "we only tested 2 segments" gap: coverage is now the ENTIRE public segment corpus.
- **Corpus periodicity analysis** (analyze_survey_corpus.py; per-map block-permutation nulls):
  69 scorable segments, ruling z from **−1.30 to +5.94, median −0.27**, vs control **+25.6..+34.1**.
  One flag: PHerc1447/z_dbg_gen_00166_inp_hr z=5.94 @ 7.26mm — verification agent running
  (200+ perms, corpus max-z null over other segments, mesh/tile-artifact check, full battery).
  Suspect: small-null instability (N_PERM=16) + multiple comparisons across 80 segments.
- **Cache-budget fix VERIFIED ON PRODUCTION DATA** (fills the PR's last empty cell):
  same pod, same PHerc1203 render, same 4GB budget, pointed at the already-bloated 5.98GB dir →
  **5,979,366,890 B / 2,854 files → 3,999,538,943 B / 1,907 files in a single open** (disk 79%→62%),
  render exit 0, output intact (mid-slice mean 77.04 DN, 65.6% nonzero). The fix RECOVERS an
  overshot directory, not just caps growth. Evidence now spans 3 orders of magnitude:
  pod 4GB (1.49×→1.00×), workstation 30MB (1.96×→0.98×), synthetic (2.00×→1.00×).
  NOTE: `git apply` of the patch FAILED on the pod (clone at a different commit) — file copy works.
- Ops: shared villa checkout was rebased by an agent onto origin/main; **restored to `cycle2`**
  so the parallel Track A session isn't disrupted (fix branch preserved at ee14289f).
- **Budget: account spend $33.38 of $80** — roughly $11 mine (survey ~$5.5), ~$22 the parallel
  session's. Two sessions share one cap and can't see each other; needs an explicit split.
- Report reframed: title now "Measure Before You Hunt", exec summary leads with what was built,
  not with negatives. COMMUNITY_POST.md drafted (3 staged Discord posts; adoption is prize-weighted).

## 2026-08-18 — CORPUS SCREEN v2: all four FLAG_VERDICT fixes shipped; corpus re-scored; 0/71 survive

`analyze_survey_corpus_v2.py` + `corpus_v2_figure.py` → `out/survey/corpus_analysis_v2.json`,
`corpus_screen_v2.png`, **`out/survey/PROTOCOL_V2.md`** (full before/after, machine-emitted
tables via `v2_report_numbers.py`). 71.1 min on 30 workers, 15,477 scored maps.

- **All four prescriptions implemented**: (1) empirical p at **N_PERM=200** + bootstrap CI on z
  + Holm; (2) validated preprocessing restored — 40-px rim erosion, σ=90 detrending, 3° grid,
  joint (map,mask) 64-ds4-px permutation, ds4 (no extra decimation); (3) periodicity gates
  ≥6 cycles / ρ(2P),ρ(3P)>0 / peak not in the band's 2 lowest bins; (4) |fwd/rev r| < 0.20.
  Two additions: a **band-constrained second search** sharing the same spectrum and null (so the
  gates can't cause a false negative), and a memory fix (`rot_cache_light`; the first run hit
  41 GB across 30 workers and swapped — now 3 GB, identical arithmetic).
- **CONTROL PASSES 5/5**: prom 123.5 vs null 20.0±6.4, **z=+16.3 (CI +13.5..+20.1),
  empirical p=0.00498 — 0 of 200 shuffles beat it**, 4.678 mm, 11.0 cycles, ρ=+0.65/+0.54/+0.46,
  bin 4 of 24, r=0.094. Block-mean variant z=+18.8, same period/cycles/bin — the survey's
  `pred[::4,::4]` decimation costs nothing. Note the correction is symmetric: the control's own
  z falls +25.6 → +16.3 because the 5-draw null sd (4.91) was an underestimate too.
- **CORPUS: 0 of 71 pass all five gates**; 0 pass even the four map-internal gates.
  Per gate: significance 4/71 (**3.55 expected under the null**), cycles 19/71, autocorr 23/71,
  band-bin 23/71, **fwd/rev 0/71**. Corrected z −1.20..+4.64, median −0.09. Min Holm p **0.354**.
  Band-constrained search: **0 survivors, corpus-best p=0.0597**.
- **THE FLAG IS DEAD THROUGH THE SCREEN ITSELF**: `z_dbg_gen_00166_inp_hr` +5.94 → **+0.97,
  p=0.159 (31/200), 0 gates of 5** (4.0 cycles, band bin 0/12, ρ(2P)=−0.004, r=0.754);
  band-constrained z=**−0.55**, p=0.692. Reproduces the verification's independent 400-perm
  result (+0.54, p=0.180).
- **How miscalibrated v1 was**: run v1's own scorer on the control → **z=+13.74**, i.e. v1
  separated *known Greek letters* from its worst false positive by only **2.3×** (v2: 3.5×).
  The `+25.6..+34.1` control line on `corpus_ranking.png` was never computed by v1 — it was
  quoted from the validated battery, so that figure's reference was apples-to-oranges.
  Spearman ρ(v1 z, v2 z) = **+0.370**: four of v1's top five collapse, its #2 rises to the new
  top, and three segments v1 put at/below zero rise into v2's top eight. v1's ordering was
  substantially permutation noise.
- **Two corpus facts that need no periodicity machinery**: the lowest fwd/rev r in the entire
  80-segment survey is **0.222** (control 0.055–0.094), and **48 of 71** segments put their
  "best period" in the band's lowest 1–2 Fourier bins — band-edge leakage, not a period.
- **Coverage**: 80 surveyed → 75 have saved maps → **71 scored**, 4 skipped and each documented
  (1 degenerate: 0.22% nonzero and fwd/rev r = **1.000**, i.e. reverse ≡ forward; 3 too small —
  post-erosion extent 5.4–6.4 mm < 4× the band's shortest period). The 5 segments with no saved
  map have r = 0.33–0.64, so gate 4 is decided for all 80. Tripwire hits: 0.
- **Honest blind spot, documented**: run on w035's *human ink labels* the screen returns
  z=+0.26, p=0.328 (validated battery: +1.31 on the same input). This is a page-scale ruling
  test; a handful of letters on an otherwise blank sheet would not trip it. The tripwire and
  morphology battery cover that regime — also clean across all 80.

### Open threads
- [ ] K1/K2 QC audit (agent in flight)
- [ ] Stage 1a rerun (in flight) → then per-depth AUC + texture stats + uv renders
- [ ] 1203 smoke (in flight) → then band-scale plan + blend/finalize + overlay
- [ ] K3 stage 2 rebuild per QC blockers
- [ ] Gate test: scouting stack (ink-coverage-32um, dinovol cosine) vs w035 pos/neg controls
- [ ] Track A: Dataset059 still downloading (labels not started); smoke run when labels land,
      compile_policy: off, 128³, fine-tune from bruniss weights
- [ ] Track B: file cucim fix once a genuine models-extra install block is hit in Track D work
