# Reproducibility appendix

Everything in this report derives from public data and publicly released model checkpoints; no credentials are required beyond accepting the Vesuvius Challenge data terms (`uv run vesuvius.accept_terms` or the PyPI equivalent). Every headline number in the report exists in a JSON/MD artifact produced by a named script in this repository; where a number appears in prose, the artifact is the authority. Bulk scroll data is **not** redistributed here — scripts stream or cache it from the URLs below.

Path convention: all paths are relative to the repository root. Historical audit documents written during the project reference the same files with a `trackD/` prefix (the working-directory name at the time); the mapping is 1:1 (`trackD/qc/...` → `qc/...`).

---

## A. Data sources (all public)

### A.1 Endpoints

| Endpoint | Access | Notes |
|---|---|---|
| `s3://vesuvius-challenge-open-data` | anonymous S3; HTTPS mirror `https://vesuvius-challenge-open-data.s3.amazonaws.com` | primary source for all volumes, segments, released predictions. `vesuvius`-lib code paths open it with `anon=True` automatically |
| `https://data.aws.ash2txt.org` | anonymous HTTPS | legacy data server (used for the PHerc1667 7.91 µm volume). Rate-limits aggressively — use retry/backoff |
| `https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/...` | anonymous HTTPS | ink_9um label zarrs. Rate-limits (HTTP 429) — this blocked the w044 K1b run (§4.4) |
| `https://huggingface.co/scrollprize/...` | anonymous (public repos) | model checkpoints. `scrollprize/dinovol_v2` is gated and was **not** used |

### A.2 Datasets used, by object

All S3 keys below are under `s3://vesuvius-challenge-open-data/` unless a full URL is given.

| Object | Key / URL | Used by |
|---|---|---|
| PHerc0139 9.362 µm masked volume | `PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr` | K2, K2b anchor, ink_9um render control |
| w035 segment (surface volume, 28×5820×5240 u8) | `PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr` | K1, K1b, ink_9um positive control, comb |
| w035 mesh (tifxyz quadmesh) | `PHerc0139/segments/20260317000000-w035_2026031718/mesh/20260317000000-on-20250728140407-9.362um.tifxyz/` | renderer validation |
| ink_9um human ink labels + supervision masks (w035/w039/w040/w041/w044) | `https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/native9-scrollprizeorg-21slices/<seg>/<seg>_inklabels.zarr` and `<seg>_supervision_mask.zarr` | K1, K1b, verdict battery calibration |
| Sibling segments' surface volumes (w039/w040/w041/w044) | same layout as w035 under `PHerc0139/segments/` | K1b |
| GP 9 µm-class masked volumes, 13 scrolls | `PHerc<id>/volumes/<scanid>-*.zarr`; scan IDs: 0125=20250821151825, 0191=20250821151635, 0211=20250821151803, 0257=20250821151750, 0268=20251110183117, 0358=20250821151737, 0800=20250521135224, 0813=20250821151723, 0826=20250821151701, 1218=20250521120456, 1447=20250521151220, 1545=20250821151648, 1203=20250820131727. Exact volume names resolved at runtime from `meta/<scroll>.json` | K2 (0813/1203/0139), K2b (all), comb dense scan |
| PHerc1203 2.403 µm 77 keV band | `PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr` | fleet screen, live QC, salvage |
| PHerc1203 released m7 surface predictions (+ normal grids) | `PHerc1203/representations/predictions/surfaces/20260319130212-...` | ROI probes, live-QC morphology, salvage atlas |
| PHerc1203 9.362 µm masked volume | `PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr` | ink_9um probe renders |
| PHerc1203 auto-grown segment meshes (22 dirs, tifxyz only) | `PHerc1203/segments/raw/auto_grown_*` | ink_9um probe (segments A/B; a third failed to render — §I) |
| PHerc.Paris.4 dual-energy pair, 45.532 µm | `PHercParis4/volumes/20260310170716-45.532um-11.0m-74keV-masked.zarr` and `.../20260310173927-45.532um-11.0m-110keV-masked.zarr` | K3 stages 1–2, sensitivity bound, comb skeptic |
| PHerc.Paris.4 2.400 µm 78 keV CT | `PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr` | live-QC in-domain control (tests 2, 5) |
| PHerc.Paris.4 released ink-3d prediction | `PHercParis4/representations/predictions/ink-3d/20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr` | live-QC in-domain distribution reference (test 3) |
| PHerc1667 w032 segment (2.4 µm-model + 1.129 µm-model ink TIFs, on-7.91 µm tifxyz grids) | `PHerc1667/segments/20260105050000-w032_2026010505_flatboi/` | S1a (refuted; retained as methodology) |
| PHerc1667 7.910 µm 53 keV masked volume | `https://data.aws.ash2txt.org/samples/PHerc1667/volumes/20231117161658-7.910um-53keV-masked.zarr` | S1a |
| Open-data catalog `metadata.json` | bucket root (snapshot dumped locally) | `dump_gp_meta.py` → `meta/*.json` (scan windows, processing parameters, per-scroll subtrees) |

### A.3 Model checkpoints

| Model | Source | Files used | Notes |
|---|---|---|---|
| ink_9um | `https://huggingface.co/scrollprize/ink_9um` | `hybrid_3d2d-seed42/step-075000.pth`, `hybrid_3d2d-seed43/step-075000.pth` (138 MB each) | all 14 released checkpoints embed `mode: "flat"`; config embedded in the .pth (`patch_size [17,128,128]`, `robust_mad` 1/99 norm). Native-3D tifxyz path refuses flat checkpoints — hence the custom renderer (§2.3) |
| ink_3d_dino_guided | `https://huggingface.co/scrollprize/ink_3d_dino_guided` | `ckpt_78k_fullsup.pth` (v3-78k-fullsup lineage; EMA weights) | loaded via villa's `train_py` path (`NetworkFromConfig` + EMA state-dict selection); `hf://` CLI loading fails on this repo (checkpoint is a vesuvius.train ckpt, not nnUNet layout) — download locally |

---

## B. Script inventory

Grouped by experiment. Columns: scripts (repo-relative) → inputs (from §A) → primary outputs. Verdict/status annotations mark refuted lines whose scripts are retained for the record but whose results must not be cited.

### B.1 K1 / K1b — intensity statistics at native 9 µm (report §4.4)

| Scripts | Inputs | Outputs |
|---|---|---|
| `k1_intensity_auc.py` | w035 surface volume + ink labels + supervision mask | `out/k1_w035_auc.json`, `out/k1_w035_maps.png` |
| `qc/qc_k1.py`, `qc/qc_k1b_depth.py` (audit: reproduction, per-slice sweep, depth-contrast, texture, within-patch controls) | cached w035 arrays | `qc/qc_k1_results.json`, `qc/qc_k1b_results.json`; findings in `qc/k1_k2_review.md` |
| `k1b_depth_validation.py` (frozen statistic, held-out segments) | w035/w039/w040/w041(/w044) surface volumes + labels | `out/k1b_depth_validation.json` |

### B.2 K2 / K2b — spectral analysis and per-scroll detectability index (report §4.5)

| Scripts | Inputs | Outputs |
|---|---|---|
| `k2_spectral_ceiling.py` | 256³ level-0 papyrus ROIs, PHerc0813/1203/0139 | `out/k2_spectra.{json,png}` (note: two JSON keys deprecated by audit — see `qc/k1_k2_review.md` B2) |
| `qc/qc_k2_local.py`, `qc/qc_k2_stream.py`, `qc/qc_k2_air0139.py` (audit: floor validation, air references, multi-ROI variance) | same volumes + streamed air ROIs | `qc/qc_k2_local.json`, `qc/qc_k2_stream.json`, `qc/qc_k2_air0139.json`, `qc/qc_k2_air.png` |
| `k2b_detectability_index.py` (production index; resumable) | 14 scrolls' 9 µm-class volumes via `meta/*.json` | `out/k2b_index/<scroll>.json`, `out/k2b_index/k2b_index_summary.{json,png}` |
| `k2c_separability.py` (sheet-separability axis; resumable, 12 uniform-random ROIs/scroll, seed 20260818) | same 14 volumes, same central-z band and `fill > 0.98` frame as K2b | `out/k2c_separability/<scroll>.json`, cubes cached to `D:/vesuvius-data/trackD/k2c` |
| `k2c_analyze.py` (per-scroll medians + bootstrap CIs, SNR rank-correlation, paired intensity-max-vs-random picker test) | `out/k2c_separability/*.json`, cached cubes under `D:/vesuvius-data/trackD/k2b` and `.../k2c` | `out/k2c_separability/k2c_analysis.json` |
| `report/scripts/make_separability_figure.py` | `k2c_analysis.json` | `report/figures/separability_axis.png` |
| `k2c_isotropy_floor.py` (the measured null floor: 28 in-scan air windows + synthetic isotropic noise) | K2b air cubes | `out/k2c_separability/isotropy_floor.json` |
| `k2c_sensitivity.py` (block 16/32/64 x sigma 0.5/1/2 sweep; rank-correlates each setting against the shipped one) | K2c random-frame cubes | `out/k2c_separability/sensitivity.json` |
| `hunt/seed_separability.py` (separability at the PHerc0813 growth seeds) | `hunt/seeds_0813.json`, PHerc0813 level 0 | `out/k2c_separability/pherc0813_seed_separability.json` |
| `hunt/mesh_lamella_alignment.py` (mesh normal vs local structure-tensor normal) | `hunt/pherc0813_meshes/`, cached seed cubes | `out/k2c_separability/pherc0813_mesh_alignment.json` |
| `hunt/alignment_control.py` (**positive control**: identical code on published GP meshes + the w035/w032 controls) | `hunt/meshcache/`, level-0 cubes at each mesh centre | `out/k2c_separability/published_mesh_alignment.json` |

### B.3 K3 — Paris 4 dual-energy screen and sensitivity bound

| Scripts | Inputs | Outputs |
|---|---|---|
| `k3_dualenergy_stage1.py` (L3 alignment + ratio prototype) | Paris 4 74/110 keV volumes, level 3 | `out/k3_stage1_stats.json`, `out/k3_stage1_overview.png` |
| `k3_dense_clusters.py` (rim/selection-bias controls) | cached L3 arrays | `out/k3_dense_clusters.json`, `out/k3_dense_gallery.png` |
| `k3_stage2_screen.py` (full-scroll L1 slab-streaming screen, two channels, air-offset zero, detrending) | Paris 4 pair, level 1 | `out/k3_s2_stats.json`, `out/k3_s2_{low,high}_clusters.json` |
| `k3_verify_clusters.py` (L0 imagery of top HIGH clusters) | Paris 4 pair, level 0 windows | `out/k3_s2_high_verify.png` |
| `k3_sensitivity_calc.py` (closed-form Pb bound; hostile-refereed) | measured σ, thresholds, windows | `out/k3_sensitivity_bound.{md,json}` |
| `comb/comb_k3_psfmatch{,2,3}.py`, `comb/comb_skeptic_k3.py` (PSF-mismatch tests; 28-window matched-radius null) | Paris 4 pair windows | `comb/k3_psfmatch*.{json,png}`, `comb/comb_skeptic_k3.json` |

Audit: `qc/k3_stage1_review.md` (4 blockers incl. the Pb-sign correction, §4.3 row 4).

### B.4 S1a — PHerc1667 w032 letter contrast — **REFUTED ×2; methodology content only**

| Scripts | Inputs | Outputs / status |
|---|---|---|
| `s1a_prep_w032.py` | w032 segment TIFs + grids | cache manifest |
| `s1a_letter_contrast.py` (v1) | w032 + 7.91 µm volume | `out/s1a_w032_stats.json` — **refuted** (`qc/s1a_verification.md`); do not cite |
| `s1a_v2_contrast.py`, `s1a_v2_overview.py`, `s1a_v2_gallery.py` (v2) | same | `out/s1a_v2_*` — **refuted** (`qc/s1a_v2_verification.md`); do not cite |
| `qc/qc_s1a_extract.py`, `qc_s1a_analyze.py`, `qc_s1a_overlay.py`; `qc/qc_s1a_v2_extract.py`, `qc_s1a_v2_register.py`, `qc_s1a_v2_nulls.py`, `qc_s1a_v2_final.py`, `qc_s1a_v2_overlay.py` | independent re-extraction | `qc/qc_s1a*_results/*.json`, overlay PNGs, and the two verification reports |

### B.5 PHerc1203 2.4 µm fleet screen, live QC, and salvage (report §4.2; §[X] main text)

| Scripts | Inputs | Outputs |
|---|---|---|
| `probe_1203_band.py`, `probe_1203_clean_rois.py`, `probe_s3_paris4.py` (ROI discovery/verification) | 1203 band + surface predictions | `out/probe_1203_*.{json,png}` |
| `smoke_1203_overlay.py` (+ `vesuvius.predict` laptop smoke), `sweep_1203_clean.ps1`, `sweep_1203_report.py` | 1203 band, ink_3d ckpt | `out/smoke_1203_*`, sweep reports |
| `runpod/rp.py` (RunPod API + budget guard), `runpod/provision.sh` / `provision_idle.sh` (env recipe, §D.3), `runpod/fleet_{launch,add,stop,status,recover}.py`, `runpod/qc_pull_samples.py` | — | `runpod/fleet.json`, heartbeat/status |
| `runpod/screen_band.py` (streaming screener: 256³ non-overlap tiles, L4 mask gating, S3 prefetch; 1.5–2.7 tiles/s per pod sustained) | 1203 band (S3), local ckpt | per-tile stats jsonl + 4×-downsampled uint8 prob maps (`qc_live/final_stats/`, `qc_live/round_1/`) |
| `qc_live/qc_norm_equiv.py`, `qc_morph.py`, `qc_zoom_tile.py`, `qc_stats.py`, `qc_paris4.py`, `qc_indomain_histmatch.py`, `qc_histmatch_pod.py` (the 5-test acceptance battery) | fleet output + Paris 4 references | `qc_live/qc_*_result.json`, `qc_live/qc_morph_gallery.png`, verdict `qc_live/round1_verdict.md` |
| `salvage/m0_inventory.py`–`m3_gallery.py` (morphology census), `salvage/r1_proxies.py`–`r3_atlas.py`, `salvage/validate_ct.py` (scalar-field + atlas), `salvage/analysis.py`, `salvage/assemble.py` | secured fleet stats + prob maps + CT/surface refs | `salvage/{morph_report.md, scalar_report.md, reframe_report.md}`, `salvage/tiles.parquet`, `salvage/atlas_1203.{npy,png}` |

### B.6 ink_9um probe and text-signature battery (report §2)

| Scripts | Inputs | Outputs |
|---|---|---|
| `runpod/pod_ink9um.sh`, `pod_ink9um_part2.sh`, `pod_ink9um_part3.sh` (pod orchestration), `runpod/render_tifxyz_sv.py` (custom tifxyz→surface-volume renderer), runbook `runpod/ink9um_runbook.md` | ink_9um ckpts, w035 segment, PHerc0139 + PHerc1203 9 µm volumes, auto_grown meshes | `out/ink9um_w035/`, `out/ink9um_1203/`, galleries |
| `score_w035_control.py` (positive-control AUC) | w035 predictions + labels | `out/ink9um_w035_scores.json`, `out/ink9um_w035_control.png` |
| `analyze_1203_ink9um.py` | 1203 prediction maps | `out/ink9um_1203_stats.json`, `out/ink9um_1203_gallery.png` |
| `salvage/verdict_prep.py`, `verdict_labelgeom.py`, `verdict_periodicity.py`, `verdict_morph.py`, `verdict_consistency.py`, `verdict_intensity.py`, `verdict_figure.py`, `verdict_common.py` (the four-test battery + tripwire) | the 8 maps + control + labels | `salvage/verdict_*.json`, `salvage/ink9um_1203_verdict.md`, `salvage/verdict_gallery.png` |

### B.7 Full-data comb (hunters + skeptic; report §2.7, §4.6)

| Scripts | Inputs | Outputs |
|---|---|---|
| `comb/comb_00_inspect.py`–`comb_03_gallery.py`, `comb_flags.py` (w035 beyond-labels hunter) | w035 predictions + labels | `comb/comb_catalog.json`, `comb/w035_beyond_labels.{json,png}` |
| `comb/comb_symmetry.py`, `comb_tripwire.py` (1203 symmetry hunter + tripwire rescans) | 8 maps + control | `comb/sym_*.{json,png,npz}`, `comb/comb_tripwire.json` |
| `comb/comb_dense_scan.py`, `comb_residual_eyeball.py` (cross-anomaly hunter) | 57 cubes / 14 scrolls; residual top tiles | `comb/dense_scan.json`, `comb/residual_*.{json,png}` |
| `comb/comb_skeptic_w035.py`, `comb_skeptic_w035b.py`, `comb_skeptic_sym.py`, `comb_skeptic_k3.py`, `comb_skeptic_misc.py` (matched-null skeptic pass) | all of the above | `comb/comb_skeptic_*.json`, verdict `comb/COMB_VERDICT.md` |

### B.8 Metadata

`dump_gp_meta.py` → `meta/*.json` (per-scroll catalog subtrees, uint8 windows, Paganin/unsharp processing parameters); `meta/ink9um_corpus.md`, `meta/pherc1667_groundtruth.md` (scouting notes with verified layouts).

Local caches: scripts cache fetched arrays under `D:\vesuvius-data\trackD\` (a machine-local path); all caches are regenerable from the URLs above and are **not** part of the repository.

---

## C. Determinism and pre-registration notes

- All AUC computations use seeded subsampling (`np.random.default_rng(0)`); QC re-runs use average-rank (tie-correct) AUC per `qc/k1_k2_review.md` W1.
- Pre-registered items: K1 kill rule (|AUC−0.5| < 0.02, in the script docstring before data fetch); K1b frozen statistic (frozen in `qc/k1_k2_review.md` B1 before held-out segments were touched); ink_9um tripwire thresholds (set from the positive control before scoring PHerc1203).
- GPU inference is run at fp16/TF32 defaults; cross-machine agreement was measured rather than assumed (laptop vs pod f05 identical to 5 decimals on the shared test tile, `qc_live/qc_histmatch_result.json`).
- Fleet sharding is deterministic (worker i takes z-tiles with iz % N == i).

---

## D. Environments

### D.1 Local analysis environment (`.venv`, Windows 11, Python 3.12.10)

Used for all CPU/streaming analysis (K1/K1b/K2/K2b/K3, S1a, comb, salvage, verdict battery) and laptop GPU smoke runs (RTX 4090 Laptop 16 GB). Key pins (from `pip freeze`, 2026-08-17):

```
numpy==2.5.2            scipy==1.18.0           scikit-image==0.26.0
zarr==2.18.7            numcodecs==0.15.1       fsspec==2026.7.0
s3fs==2026.7.0          aiohttp==3.14.3         requests==2.34.2
tifffile==2026.8.16     imageio==2.37.4         pillow==12.3.0
matplotlib==3.11.1      pandas==3.0.5           jupyter==1.1.1
torch==2.11.0+cu128     torchvision==0.26.0+cu128
vesuvius==0.2.4
```

### D.2 Villa environment (`.venv314`, Python 3.14.7)

Used for `vesuvius.predict` / villa inference paths on Windows. Key pins: numpy 2.5.2, scipy 1.18.0, **zarr 3.3.0**, fsspec/s3fs 2026.7.0, torch 2.11.0+cu128, torchvision 0.26.0, huggingface-hub 1.27.0, tensorstore 0.1.85, tifffile 2026.8.16, plus `villa/vesuvius` as an editable checkout. Villa code pins for this report: every file:line claim was verified at `origin/main 465cab06` (2026-08-16) and re-verified at `97621bcf` (2026-08-20); pod inference runs cloned main `--depth 1` on their run dates (2026-08-16 → 08-21) and are not commit-pinned in the scripts, so those two hashes are the reference commits for any line-number claim. Windows caveat: set `TORCH_COMPILE_DISABLE=1` for inference (no triton on Windows wheels; see villa PRs #1480/#1492).

### D.3 Cloud pod recipe (validated; `runpod/provision.sh`, idle variant `runpod/provision_idle.sh`)

Target: RunPod RTX 5090 32 GB (community, ~$0.69/hr) or similar; **host driver caps CUDA at 12.8** — PyPI torch 2.13 wheels are incompatible, pin cu128. Provision time ≈ 5 min:

```bash
apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 https://github.com/ScrollPrize/villa.git && cd villa/vesuvius
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra models          # installs almost nothing heavy by design (torch left to user)
uv pip install "torch==2.11.0" torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install tqdm scipy scikit-image pandas einops opencv-python-headless tifffile aiohttp \
  numba monai timm accelerate pytorch-lightning pytorch-optimizer huggingface-hub \
  dynamic-network-architectures nnunetv2 batchgenerators fft-conv-pytorch fvcore \
  connected-components-3d tensorstore typed-argument-parser psutil nest-asyncio \
  blosc2 lxml imagecodecs pynrrd cachetools edt wandb
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run vesuvius.accept_terms --yes
```

Excluded on purpose: `volume-cartographer`, `cucim` (manylinux-only; see villa PR #1479). Operational caveats learned the expensive way: `/workspace` is 20 GB, not 40 (disk-fill killed one render round); villa's `volume_io` disk-cache path can raise a zarr-3 `ContextVar` error on some workloads (run with cache disabled); verify CUDA before accepting a community host (one bad host cost 3 pods / ~$0.40 — `runpod/fleet_recover.py` now checks first).

---

## E. Cost and time to reproduce

| Item | Data moved | Compute | Cost |
|---|---|---|---|
| K1 (w035) | ~1.1 GB | CPU, minutes | $0 |
| K1b (3 held-out segments) | ~1 GB/segment | CPU, minutes each | $0 |
| K2 + audits | ~10 256³/128³ ROIs streamed | CPU, ~1 h | $0 |
| K2b full index (14 scrolls) | ~60 ROIs streamed | CPU, hours (resumable) | $0 |
| K3 stage 1 (L3) | ~80 MB | CPU, minutes | $0 |
| K3 stage 2 (full-scroll L1) | streamed slabs | CPU, hours | $0 |
| Acceptance-gate battery on a new model | ≤30 tiles | single GPU, minutes | ≈$0–1 |
| ink_9um probe (control + 2 segments × 2 seeds × 2 dirs + battery) | ~3 GB | 1 pod, ~2 h | ≈$1.15 |
| Full 2.4 µm band screen (if a model ever passes the gate) | streamed | 6-pod fleet, ~5.5 h | ≈$19–25 projected |
| Corpus survey, 80 published GP segments (render + infer, both z-directions) | ~80 meshes + volume slabs streamed | 4-pod 5090 fleet, 6.34 GPU-segment-hours | ≈$5.5 |
| **Total Track D cloud spend, entire project** | | | **≈$11.1** ($5.61 core + ≈$5.5 corpus survey) of an $80 cap |

---

## F. Suggested repository layout

Repository root = the current `trackD/` working tree, pruned of large binaries. Suggested name: `gp13-ink-detectability`. Repository: `https://github.com/flummoxjr/gp13-ink-detectability` (created 2026-08-24; private until submission, public at submission time).

```
gp13-ink-detectability/
├── README.md                  # landing page: abstract + links into report/
├── LICENSE                    # MIT (code) — see §G
├── DATA_LICENSE.md            # data terms note + attribution — see §G
├── report/
│   ├── sections/01_*.md … 04_methodology.md
│   ├── figures/               # all PNGs referenced by sections
│   └── REPRODUCIBILITY.md     # this file
├── k1_intensity_auc.py …      # top-level experiment scripts, as-run (B.1–B.6)
├── qc/                        # audit scripts + the 4 verification/review reports
├── qc_live/                   # fleet acceptance battery + round1_verdict.md
├── salvage/                   # morphology/scalar/atlas + verdict battery + reports
├── comb/                      # hunters + skeptic pass + COMB_VERDICT.md
├── runpod/                    # pod recipes, fleet tooling, renderer, runbook
├── meta/                      # per-scroll catalog snapshots (JSON) + scouting notes
├── out/                       # result JSONs + report-grade PNGs (keep); exclude *.npy
└── LOG.md                     # the unedited running log (recommended: ship it — it is the provenance trail)
```

Packaging rules: keep every `.json`, `.md`, `.png` under `out/`, `qc*/`, `salvage/`, `comb/`; exclude `*.npy`/`*.npz`/`*.parquet` caches above ~5 MB (all regenerable; `salvage/tiles.parquet` at full resolution is the one debatable keeper — decision: ship it — it is the frozen benchmark any 1203-adapted model must beat, and the cross-reference layer future detectors will want). Exclude `runpod/fleet.json` and anything containing pod IDs; the RunPod API key lives outside the tree (`.runpod_key`, gitignored) and must never ship.

---

## G. Licensing and data-use notes

- **Code: MIT.** The prize rules require submissions to be open-source; MIT is the project's choice for all scripts and report text in this repository. Named on the submission form as: MIT.
- **Scroll data: Vesuvius Challenge terms, non-commercial.** All CT volumes, segments, labels, and released predictions are distributed by the Vesuvius Challenge under its data agreement (CC BY-NC 4.0 non-commercial terms at last check, accepted programmatically via `vesuvius.accept_terms`). This repository redistributes **no** bulk scroll data — only URLs, small derived JSON statistics, and report figures. Derived rasters and figures that visually reproduce scan content (overlay PNGs, galleries) inherit the non-commercial restriction and are marked as data-derived in `DATA_LICENSE.md`. The CC BY-NC characterization comes from the data-acceptance flow, not a legal review; the current text at scrollprize.org/data is the authority at submission time.
- **Model checkpoints** (`scrollprize/ink_9um`, `scrollprize/ink_3d_dino_guided`) are used unmodified under their Hugging Face repo terms; no weights are redistributed here.
- **Attribution.** Data: Vesuvius Challenge / EduceLab and the ESRF (BM18 beamline scans). Literature values used in the K3 bound: Tack et al. 2016 (Sci. Rep. 6:20763) and Brun et al. 2016 (PNAS 113:3751); attenuation coefficients from NIST/XCOM. Villa/vesuvius tooling: ScrollPrize/villa (our upstream fixes are listed in the main report).

---

## H. Submission form fields

The Google Form at scrollprize.org/prizes asks for the following; the values are filled in below.

- Submitter: Ben Black — benblack211@gmail.com — GitHub `flummoxjr`. Discord handle to be entered directly on the form.
- Category: Monthly Progress Prize. Deadline 31 August 2026, 11:59 pm Pacific.
- Submission description: "A per-scroll detectability index for the thirteen Grand-Prize scrolls on two near-orthogonal axes (scan quality and sheet separability), an ink instrument validated against human-verified letters at AUC 0.999, a screen of every published Grand-Prize segment with that instrument (80/80 rows, 0 of 71 passing a five-gate protocol the control passes 5/5), two whole-volume screens, and three reusable QC gates. No ink was found and none is claimed. Sixteen corrections are published as a ledger, five of them against results inside the report." A longer variant, if the form allows: append "Also included: a quantified transfer-failure characterization of ink_3d on PHerc1203 with a fine-tune recipe, and upstream tool fixes (merged PR #1480; open PRs #1479/#1487/#1492; issues #1488–#1490, #1520, #1524–#1525)."
- Repository URL: `https://github.com/flummoxjr/gp13-ink-detectability` (public at submission). License: MIT.
- Link `LOG.md` as the provenance trail. It is unflattering in places and that is the point: it shows the order things were actually found in, including the wrong turns.
- Claim wording on the form follows the house rule carried through this report: model detections are never called "letters" or "ink" unless human-verified; the only human-verified letters anywhere in this work are the w035 labels.

---

## I. Known reproducibility gaps (honest list)

1. **K1b is 3 of 4 planned held-out segments**: w044 never ran (HuggingFace bucket HTTP 429, retry not completed). The held-out mean (0.4934) and the non-transfer conclusion are stated over w039/w040/w041 only.
2. **The fleet screen covers ~30k of a projected ~169k tiles** (stopped at ~45% by the QC verdict, deliberately). The banked negative and all per-tile statistics apply to the covered z-slabs, not the full band.
3. **PHerc0139's K2b noise reference is a papyrus-residual floor**, not air (no candidate air window passed validation); its bandwidth/SNR are conservative. PHerc0800's air came from the fallback full-z search. Both are flagged in the per-scroll JSONs (`noise_ref` field).
4. **One PHerc1203 segment fails to render** (`auto_grown_20251005221856743`), cause undiagnosed; excluded from the ink_9um probe. Villa's `volume_io` zarr-3 disk-cache `ContextVar` error (worked around with no-cache) is reported but uninvestigated.
5. **Windows-specific paths**: scripts cache to `D:\vesuvius-data\trackD\` and reference `C:\...\trackD\out`; a re-runner on another machine must adjust two path constants per script (they are top-of-file). Accepted as-is for a report repository — both constants are top-of-file and documented here.
6. **The exact villa commit for the pod runs is not pinned in the scripts** (clone `--depth 1` of main at run time); the reference hashes in §D.2 (`465cab06`, `97621bcf`) are the pins for all line-number claims.
7. **S1a v1/v2 outputs are retained but refuted** — any reproduction will faithfully reproduce numbers that the verification reports show measure artifacts. The scripts stay for the methodological record; the reports (`qc/s1a_verification.md`, `qc/s1a_v2_verification.md`) are the citable objects.
8. **uint8 saturation censoring** (8/14 scrolls, 5–34% of in-mask voxels) right-censors the K2b DN-headroom metric and any bright-ink screen on those scrolls; documented in §4.5 and `comb/COMB_VERDICT.md` flag 7.
9. **K3's beam spectra are nominal**: the 74/110 keV predictions carry ±10% systematics (pink beam, inconsistent filtration metadata), and a ~8–10% papyrus-ratio discrepancy remains unexplained after offset correction. All K3 sensitivity numbers are quoted per calibration frame for this reason.
10. **One-time environment events** (a system update killed all Python processes mid-run on 2026-08-16; repeated background-task kills later that evening) forced re-runs; all affected results were regenerated from scratch or resumed from verified caches — no partial-state results are reported.
