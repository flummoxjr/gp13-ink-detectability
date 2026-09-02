# Bet A — arm 0: LOSO (no-PHerc0139) retrain of the ink_9um recipe — launch-ready plan

_Written 2026-09-02 from a read-only survey (villa `merge-ink-pipelines`, villa-pin, HF `ink_9um`,
S3 open-data, khj1222's write-ups). Companion files: `data_manifest.json` (every store the pod
fetches, with shapes, chunk plans and sha256 of the small metadata files). Prereg:
`trackD/PREREG_BET_A_DRAFT.md` — §4 needs the addendum in §9 below **before** it is committed and
before any training starts._

## 0. Bottom line

| question | answer |
|---|---|
| Is the training code runnable on our pod pattern? | **Yes, in the proven env.** villa-pin (`a3f2c29` = vesuvius @ `37e300d3`) already carries the whole merged pipeline as `vesuvius.ink_detection.{training,data,models,inference,preprocessing}`: trainer, `FixedScrollPriorStratifiedBatchSampler`, the `surface_volume_paths` / `sampling_*` / `fixed_scroll_prior` schema, the 2.4 µm→9.6 µm pooling script, and the inference CLI p2a_v3 already ran. `pyrun -m vesuvius.ink_detection.training.train <cfg>` is the entry point. No koine_machines install needed. Verified offline: the generated LOSO config parses through villa-pin's `TrainingConfig.from_mapping` and `stage_training_request()`; the model builds (34.55 M params). |
| Data the pod must pull | labels ≈ 1 GB (HF, ~32k non-empty chunk files); training source volumes **29.3 GB** as a sparse level-2 fetch (16,482 chunks; the full level-2 stores would be ≈ 246 GB — chunks are full-z 109×128×128 and uncompressed); native eval crops 1.95 GB; released checkpoint 138 MB. |
| Cost / wall time | 8–11.5 h on a 5090 community pod ($0.69/h) ≈ **$5.5–8** for both seeds; ~3–4.5 h per seed of training (21–30 min per 10k steps). Guard deadline 14 h, 120 GB disk. |
| khj1222's anchor (native 0139, LOSO no-0139, best-of-grid over 7 steps × 2 seeds, best-F1 threshold sweep on the supervision mask) | native-5 mean **0.653** (seed42 0.627, seed43 0.653), floor mean 0.541 → margin **+0.112**; all-14 representations 0.678 / +0.169 (the "+0.17" in the prereg is this all-14 number, not the native one). |

## 1. What was established

**Code.** `ink-detection/` on `merge-ink-pipelines` (tip `3ea17f5`, 2026-08-14) is the package
`koine_machines` (pyproject: `torch==2.10.0`, `torchvision==0.25.0`, `zarr==2.18.7`,
`numcodecs==0.15.1`, `numpy<=2.2`, `python>=3.11`, uv.lock, `vesuvius = {path="../vesuvius", editable}`).
It is *not* independent of `vesuvius`: it imports `vesuvius.models.augmentation.pipelines.training_transforms`,
`vesuvius.image_proc.intensity.normalization`, `vesuvius.tifxyz`, `vesuvius.models.training.{optimizers,lr_schedulers}`,
`vesuvius.models.build.build_network_from_config`. villa main later merged the same pipeline into
`vesuvius/src/vesuvius/ink_detection/` — that merged copy is what villa-pin (`37e300d3`) ships, with
identical `aligned21_fixed_scroll_prior.json` and an `aligned21_hybrid_3d2d.json` whose only difference
from the vendored one is the description string and a pre-expanded 29-entry `datasets` block. Because our
generator takes `--recipe/--contract`, we train from the vendored (khj1222-verified) files. Caveat: the
villa-pin trainer and khj1222's koine_machines are the same lineage but not byte-identical code paths
(sampler RNG, augmentation); the anchor tolerance is what absorbs that.

**Data layout the trainer expects** (from `vesuvius.ink_detection.data` and the HF README):
`<segments_path>/<segment>/<segment>_inklabels.zarr`, `<segment>_supervision_mask.zarr`, optional
`<segment>_validation_mask.zarr` (only pherc1667-w029 and pherc0814-46527 among the kept 15), labels
annotated at plane z=10 of 21 (aligned) / z=14 of 28 (native) = `shape[0]//2`; the surface volume for a
segment is whatever `surface_volume_paths[<segment>]` points at, opened at `volume_scale: 0`, and must
be the 21-slice ~9.6 µm pooled group with array `"0"` (native volumes are used as-is). Patch finding
reads **only** the supervision/validation mask plane: patch corners `(y//32*32, x//32*32)` for every
supervised pixel, 128×128 patches, `patch_min_labeled_coverage 0.0`. The image is never consulted to
choose patches, which is what makes a sparse source fetch legitimate.

**Where the volumes live.** No pooled 9.6 µm renders are published for 1667/Paris4/0814 (their S3
segment dirs hold only 1.129 µm, 2.4 µm, and for Paris4 45 µm renders). The recipe's inputs are
level 2 (9.596 µm XY, still 2.399 µm in z, 109 planes) of the 2.4 µm surface-volume OME-Zarr, planes
13:97 mean-pooled ×4 → 21 slices, exactly what `prepare_9um_isotropic_input.py` does and what the label
`.zattrs` record (`source_z_slice [13, 97]`). S3 level-2 chunks are `(109,128,128)` uint8, blosc
clevel 3 but effectively incompressible (≈ 1.79 MB each), so a z-window cannot be sliced — the only
way to avoid 190–250 GB is to fetch only the chunk columns training patches touch (§3).

**Training cost.** khj1222 (RTX 5090, Windows, 12 workers): 78,125 steps at ~7–8 it/s = 2 h 54 m /
3 h 14 m per seed (leave-Paris4-out, 21 reps); leave-1667-out 2 h 46 m / 3 h 20 m; the two-segment
holdout 3 h 55 m. Our micro-benchmark of the recipe model (batch 64, fp16, (17,128,128)): 360 ms/step
compute-only on the laptop 4090 (2.8 it/s), **5.43 GiB peak allocated / 7.23 GiB reserved** — a 5090 has
~2.3× the throughput, i.e. a 6–7 it/s compute ceiling, so the dataloader (robust-MAD + augmentation on
CPU) is the likely limiter on a pod with few vCPU. Checkpoints: `ckpt_{step+1:06}.pth` every 5,000
steps + `best_val_balanced_accuracy.pth`, ≈ 138 MB each (the released files are 138,360,039 B).

## 2. Environment recipe (pod; matches `pod_p2a_v3.sh` provision verbatim)

```bash
# image runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04, ports 8000/http 22/tcp, disk 120 GB
cd /workspace && apt-get update -qq && apt-get install -y -qq git
git clone --depth 1 https://github.com/flummoxjr/villa-pin-37e300d3.git villa   # expect a3f2c29
cd /workspace/villa/vesuvius
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
uv sync --extra models
uv pip install "torch==2.11.0" torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install tqdm scipy scikit-image pandas einops opencv-python-headless tifffile aiohttp numba monai timm \
  accelerate pytorch-lightning pytorch-optimizer huggingface-hub dynamic-network-architectures nnunetv2 \
  batchgenerators fft-conv-pytorch fvcore connected-components-3d tensorstore typed-argument-parser psutil \
  nest-asyncio blosc2 lxml imagecodecs pynrrd cachetools edt wandb s3fs pillow
pyrun() { (cd /workspace/villa/vesuvius && uv run --no-sync --extra models python "$@"); }
export TORCH_COMPILE_DISABLE=1 WANDB_MODE=disabled
# provision gate (new for this run): the trainer must import and the model must build on the GPU
pyrun -c "import torch, accelerate; import vesuvius.ink_detection.training.train as T; \
from vesuvius.ink_detection.models.model import make_model; print('TRAIN_IMPORT_OK', torch.__version__, torch.cuda.get_device_name(0))"
```

The trainer needs `accelerate` (villa-pin's train.py imports it lazily and raises "ink training
requires the models extra with accelerate installed" otherwise) — it is in the pip line above, as in
p2a_v3. wandb is only initialised if `wandb_project` is in the config; the recipe has none.

## 3. Data manifest (details and sha256s in `data_manifest.json`)

| block | source | items | size | fetch rule |
|---|---|---|---|---|
| labels, 15 kept aligned reps | HF bucket `ink_9um/labels/aligned-scrollprizeorg-21slices/<seg>/` | inklabels + supervision_mask (+ validation_mask for w029, 0814); `(21,H,W)` u8, chunks `(21,128,128)`, 78-byte all-zero chunks everywhere else | ≈ 30k non-empty chunk files, < 1 GB | list with the tree API (`?limit=1000`, cursor pagination), skip files whose `xetHash` is the all-zero chunk (`50cce7bb…` aligned, `4f3a3229…` native) — do **not** filter by size (134-byte native chunks with real content exist); 16–32 threads, retry/backoff, resumable |
| labels, 5 native eval segs | `…/native9-scrollprizeorg-21slices/<w>/` | `(28,H,W)`, plane z=14 | ≈ 1.2k files | same |
| training source volumes | S3 `<segment>/surface-volumes/2.399um-…zarr/2/` (Paris4: `2.4um-…zarr/2/`) | 15 stores, level 2 `(109,H,W)` u8, chunks `(109,128,128)`, fill 0; **`pherc1667-w013` uses `.` separators, all others `/`** | **16,482 chunks ≈ 29.3 GB** (full: 139,133 ≈ 246 GB) | plan from the downloaded supervision mask: chunk columns covering every patch `[y//32*32, +128)`, +1 chunk margin; verify `.zarray`/`.zattrs` sha256 against the manifest first; 404 = zero chunk (zero-length marker); 48 threads |
| native eval volumes | S3 `PHerc0139/segments/<id>/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/0/` | 5 stores `(28,H,W)` u8, chunks `(28,128,128)`, **compressor None**, fill 0 | supervision-bbox crops padded to 128: 4,248 chunks ≈ 1.95 GB (full 7.0 GB) | p2a_v3 `ctl_build.py` fetcher verbatim; w035 crop (399 chunks) = the certified control crop |
| released checkpoint | HF `scrollprize/ink_9um` `hybrid_3d2d-seed42/step-075000.pth` | 138,360,039 B | — | `hf_hub_download`, size-exact gate (p2a_v3 stage `ckpt`) |
| optional anchor | HF `ink/unused/500p2a` | 3,675 chunks + rasters | ≈ 2 GB | p2a_v3 `fetch_windows`/`build_windows` verbatim (523 s measured) |

Per-representation sparse plan (chunks / GB): w013 565/0.9, w018 2233/4.0, w023 935/1.7, w028 182/0.3,
w029 299/0.5, w031 447/0.8, P4-w00 583/1.0, w01 1407/2.5, w02 811/1.5, w03 1033/1.8, w05 3044/5.4,
w06 1748/3.1, w07 2475/4.4, w09 645/1.2, 0814 75/0.1. (w00/w09 planned from the HF chunk inventory,
the rest from the locally cached label planes; the pod recomputes all 15 and requires agreement within
±25 % or it stops.) Measured pod-side throughput to plan against: p2a_v3 pulled S3 chunks at 8 threads
≈ 9 chunks/s (latency-bound) and HF chunks ≈ 7/s; hence the higher thread counts above.

Pooling (per representation, CPU): `pyrun -m vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input
data/level2/<seg>.zarr data/volumes/aligned9/<seg>.zarr --level 2` (local group = copied `.zgroup`,
`.zattrs`, `2/.zarray` + fetched chunks; missing chunks read as 0). Output `(21,H,W)` u8 blosc-zstd —
its shape must equal the label shape (gate). Delete `data/level2/` afterwards (disk).

## 4. Pod script stages (`pod_betA_arm0.sh`, gist-launched; same skeleton as `pod_p2a_v3.sh`)

Status server on :8000 first, `prereg.json` locked before provisioning, `stage_open/close` markers,
`retry`, `die`→linger, heartbeat, `FORCE`/`DRY`, **`ALL DONE` only from `finalize`**. Env knobs:
`SEEDS="42 43"`, `RUN_P2A=0|1`, `WORKERS_FETCH=48`, `TRAIN_WORKERS=auto`.

| # | stage | does | gate (FATAL unless noted) |
|---|---|---|---|
| 1 | `provision` | §2 | `TRAIN_IMPORT_OK`, CUDA device present |
| 2 | `ckpt_ref` | released seed42 step-075000 | size 138,360,039; embedded config mode=flat crop=(17,128,128) norm=robust_mad |
| 3 | `labels_fetch` | HF tree listing → non-empty chunks → `data/labels/{aligned-…,native9-…}/<seg>/…` (+ verbatim `.zattrs`, `.zgroup`, `0/.zarray`) | sha256 of the 3 metadata files per store = manifest; supervision plane non-zero count within 0.5 % of the manifest `sup_px` (where recorded) |
| 4 | `sv_plan` | from each supervision mask (z=10): patch corners → chunk columns → +1 chunk; write `out/sv_plan.json` | count within ±25 % of manifest `chunks_planned_sparse`; total ≤ 45 GB |
| 5 | `sv_fetch` | 48-thread resumable fetch of level-2 chunks into `data/level2/<seg>.zarr/2/…` honouring each store's separator | S3 `2/.zarray` sha256 = manifest; absent fraction per store < 60 % |
| 6 | `pool` | 15× `prepare_9um_isotropic_input`; up to 3 in parallel if `nproc ≥ 8` | pooled shape == label shape; for 20 random supervised patches per rep, the pooled 17-slice patch is not all-zero; then `rm -rf data/level2` |
| 7 | `config_gen` | `make_holdout_config.py --exclude-scroll 0139 --seed S --recipe … --contract …` per seed; post-edit `dataloader_workers = min(12, nproc-2)` (recorded in results, khj1222 did the same); `TrainingConfig` parse; **smoke run** with `--iterations 30 --save-every 30 --val-every 30` into a scratch dir | quotas `{1667:40, Paris4:20, 0814:4}`, 15 reps, 3 entries; `sampling_observed.json` exists after the smoke; every patch bbox in the smoke's patch index lies inside `sv_plan` (the silent-zeros guard) |
| 8 | `native_fetch` + `ctl` | 5 native crops (§3) built into 28-layer zarrs; the four p2a_v3 controls with the released checkpoint on the w035 crop | ctl_native fwd ≥ 0.95 and rev ≤ 0.80 (fatal); scale-fault ×1.9504 and ×0.5 reported |
| 9 | `train_s42` | `pyrun -m vesuvius.ink_detection.training.train out/cfg/loso_no0139_s42.json`, `out_dir=out/runs/s42` (served); trainer stdout → `out/logs/train_s42.log`; heartbeat parses `it/s`, writes ETA to status; re-entry resumes by setting `checkpoint` to the newest `ckpt_*.pth` | reaches `ckpt_075000.pth`; loss finite; `sampling_observed.json` matches quotas |
| 10 | `eval_s42` | for steps 10k,20k,30k,40k,50k,60k,75k: forward inference on the 5 crops (`--direction forward --batch-size 16 --num-workers 8 --no-compile`), uint8 TIFFs; per crop: khj1222 F1 sweep (§6) **and** the benchmark AUC (`p2a_v3` curvelib, fwd); reverse direction for the best-of-grid step; `results.json` rewritten after every checkpoint | all 35 cells present |
| 11 | `train_s43`, `eval_s43` | as 9–10 | |
| 12 | `ref` | released checkpoint forward on the 5 crops (khj1222's "ref" row, in-scroll) | reported |
| 13 | `anchor_500p2a` (only if `RUN_P2A=1`) | p2a_v3 fetch/build; win1 iso fwd for the 14 checkpoints | reported (prereg secondary) |
| 14 | `finalize` | aggregate, inventory, `bundle.tgz` (results, preds, previews, prereg, configs, `sampling_observed.json`, `validation_metrics.jsonl`, logs), `ckpts_keep.tgz` (7 canonical steps × 2 seeds ≈ 1.9 GB, served separately) | then `ALL DONE` |

Launch: `python trackD/bench/tools/launch_pod.py --name betA0 --gist <pinned raw url> --script
pod_betA_arm0.sh --out experiments/betA_arm0 --deadline-hours 14 --disk 120`; guard with
`--fetch-files ckpts_keep.tgz,results.json --no-status-min 20` (checkpoints and `results.json` are
served incrementally, so a deadline kill still yields a scored partial trajectory).

## 5. Config generation (verified offline 2026-09-02)

```
python make_holdout_config.py --labels-root data/labels --volumes-root data/volumes \
  --exclude-scroll 0139 --seed 42 --recipe configs/aligned21_hybrid_3d2d.json \
  --contract configs/aligned21_fixed_scroll_prior.json --out out/cfg/loso_no0139_s42.json \
  --run-dir /workspace/betA/out/runs/s42
→ 15 kept / 14 held out; quotas {'1667': 40, 'Paris4': 20, '0814': 4} (batch 64); 3 dataset entries
```
Kept volumes must be discoverable as `data/volumes/aligned9/<segment>.zarr` (the generator's
`<segment>.zarr` branch; khj1222's own layout). The emitted config parsed through villa-pin's
`TrainingConfig` with `num_iterations 78125, batch 64, save_every 5000, val_every 5000, fp16,
best_checkpoint_metric val_balanced_accuracy`, mode flat, crop (17,128,128), robust_mad, model
`vesuvius_unet_3d_stem_2d` (autoconfigured 2D U-Net, 6 stages, features 32→320, stem 16 → 32 in-ch).
Seed 43 = same command with `--seed 43`. The two extra keys the generator writes
(`held_out_representations`, `description`) are tolerated.

## 6. Evaluation (what the anchor gate compares, and what the benchmark records)

*khj1222 replica (the reproduction check).* `tools/eval_validation.py --region-kind supervision_mask`:
prediction = infer's uint8 TIFF (sigmoid × 255, truncated), pixel positive iff `score ≥ t`, sweep
t = 0..255 from 256-bin histograms over the supervision-mask region, report `max_t F1`. Floor per
segment = `2p/(1+p)` with p the ink fraction of the region. Their region is found from the mask
pyramid at level 3, so their `scored_px` differ from our full-plane counts by ≤ 0.15 % (w035:
1,069,617 vs 1,071,121; ink 333,586 vs 334,035) — negligible against the tolerance. Default
forward direction, default centred 17-of-28 window.

*Benchmark (BENCHMARK.md §4).* Exact tie-corrected pixel AUC on the native grid, pos = ink ∧ mask,
neg = mask ∧ ¬ink, both directions on the selected checkpoint, controls from the same run; results
JSON per the §5 schema plus a `khj1222_f1` block: per seed × step × segment `{best_f1, threshold,
floor, margin}` and the derived `native5_mean_best_of_grid`, `native5_mean_step20k`, seed |Δ|.

## 7. Cost and wall time (5090 community, $0.69/h; measured numbers marked *m*)

| stage | estimate |
|---|---|
| provision | 1–2 min (*m* 50 s in p2a_v3) |
| labels (≈ 32k small HF files) | 10–30 min |
| level-2 sparse fetch, 29 GB | 15–45 min (48 threads; *m* 8 threads ≈ 9 chunks/s) |
| pooling ×15 | 15–30 min |
| config + smoke, native crops + controls | 8–12 min (*m* controls 130 s + fetch 44 s) |
| training, per seed | **3–4.5 h** (*m* khj1222 2 h 54 m – 3 h 14 m; 21–30 min per 10k steps) |
| eval, per seed | ~12 min (7 ckpts × 5 crops fwd ≈ 77 s each from *m* 0.9 Mpx/s; + reverse on the best) |
| ref + optional 500p2a | 2 min / +15 min |
| **total, two seeds** | **8–11.5 h → $5.5–8**; balance floor for `launch_pod.py` is $40 (have ≈ $80) |

GPU memory: ≈ 7.5 GB reserved for training; the 5090's 32 GB leaves room but running both seeds
concurrently would halve the dataloader per run — keep sequential. Disk peak ≈ 75 GB (§3) → 120 GB.

## 8. khj1222's LOSO no-0139 numbers (the anchor; `runs/ink9um_scorecard/no0139_matrix*.{csv,json}`)

Per-segment best-of-grid (7 steps × 2 seeds), native representations:

| seg | floor | seed42 best (step) | seed43 best (step) | best | margin |
|---|---|---|---|---|---|
| w035 | 0.4755 | 0.6527 (20k) | 0.6789 (20k) | 0.6789 | +0.203 |
| w039 | 0.4817 | 0.5451 (10k) | 0.5848 (75k) | 0.5848 | +0.103 |
| w040 | 0.6115 | 0.6502 (20k) | 0.6519 (20k) | 0.6519 | +0.040 |
| w041 | 0.5859 | 0.6751 (10k) | 0.7025 (20k) | 0.7025 | +0.117 |
| w044 | 0.5501 | 0.6139 (10k) | 0.6458 (20k) | 0.6458 | +0.096 |
| **native-5 mean** | 0.5409 | 0.6274 | 0.6528 | **0.6528** | **+0.112** |

Native-5 mean best-F1 by step: seed42 10k 0.623 · 20k 0.615 · 30k 0.606 · 40k 0.606 · 50k 0.600 ·
60k 0.592 · 75k 0.602; seed43 0.592 · **0.651** · 0.628 · 0.602 · 0.622 · 0.616 · 0.617; mean-of-seeds
at 20k 0.633 (+0.092). All-14 (9 aligned + 5 native): mean 0.678, margin +0.169, peak 20k (0.654),
seed |Δ| mean 0.032; aligned-9 0.691 / +0.201. Reference (released, in-scroll) on the native five:
0.966–0.986. Ensembling the two seeds adds only +0.007. Their arms: 78,125 steps, seeds 42/43,
`dataloader_workers 12`, trained on Windows at `merge-ink-pipelines` tip.

## 9. Proposed anchor gate (addendum for PREREG §4 — commit before launch)

> Arm 0 is anchored on the **native-5 best-of-grid mean best-F1** with khj1222's scorer (§6):
> published 0.653 (seed42 0.627, seed43 0.653; floor 0.541; margin +0.112). PASS the reproduction iff
> our best-of-both ≥ **0.603** (within 0.05) **and** our mean-of-seeds margin over the same floors is
> ≥ **+0.06** **and** the trajectory peaks at 10–30k with 75k below the peak on both seeds. The
> "+0.17" in the draft refers to all 14 representations; the native tier alone is +0.112. Pixel AUC
> (both directions, best-of-grid checkpoint per seed) is reported alongside and becomes the arm-0
> baseline for §5; the arm-0 seed spread in AUC is the noise floor.

## 10. Top 5 things most likely to break (and the mitigation built into §4)

1. **Dataloader starvation / worker crashes on the pod.** khj1222 saw a `MemoryError` in a DataLoader
   worker on Windows and lowered workers to 6; community pods may expose only 6–8 vCPU, so 12 workers
   can thrash and the 7–8 it/s becomes 4 → 5.5 h per seed. *Mitigation:* `dataloader_workers =
   min(12, nproc-2)`, `pin_memory` kept, it/s parsed into the heartbeat with an ETA, deadline 14 h,
   checkpoints served as written, resume-by-`checkpoint` on stage re-entry.
2. **A sparse-fetch miss feeding silent zeros into training.** *Mitigation:* the chunk plan is derived
   from the trainer's own patch rule (+1 chunk), the smoke run's patch index is checked against the
   fetched set (FATAL on any patch outside it), and 20 pooled patches per representation are asserted
   non-zero.
3. **Per-store zarr quirks.** `pherc1667-w013` level 2 uses `.` chunk separators (others `/`); native
   stores have `compressor: None`; absent chunks are 404s meaning zero. *Mitigation:* copy `.zarray`
   verbatim and derive the key format from it; sha256-check `.zarray`/`.zattrs` against the manifest
   before any chunk fetch; cap the absent fraction; zero-length markers for resumability.
4. **Environment drift vs khj1222's run** (villa-pin's merged trainer vs koine_machines at
   `3ea17f5`; torch 2.11 cu128 vs 2.10). Sampler RNG and augmentation streams will differ, so
   bit-identical numbers are impossible. *Mitigation:* the 0.05 tolerance and the seed-spread rule in
   §9; the provision gate imports the trainer and builds the model; the 30-iteration smoke run catches
   API breakage before the data fetch.
5. **HF small-file rate limits / slow chunk GETs** (p2a_v3 measured ≈ 7 files/s at 8 threads; ≈ 32k
   label files). *Mitigation:* xetHash-filtered listing (skip all-zero chunks), 16–32 threads with
   backoff, resumable markers, and `hf buckets sync hf://buckets/scrollprize/datasets/ink_9um/labels/…
   ./data/labels/…` (with `--include`) as the fallback path; the labels stage has its own 60-min budget
   before the guard's no-status abort matters.

Also watched: the online validation uses the leaky `_validation_mask` splits of w029/0814 (villa
#1638) — fine for `best_val_balanced_accuracy.pth` bookkeeping, never for selection; and the guard
must not be asked to `--fetch-dirs runs` (30 checkpoints ≈ 4 GB) — use `ckpts_keep.tgz`.

## 11. Not determined offline

- Pod vCPU count and real end-to-end it/s on Linux (only khj1222's Windows 7–8 it/s and our
  compute-only ceiling are known); real HF/S3 throughput at 32–48 threads.
- Whether villa-pin's trainer reproduces koine_machines numerically at the same seed (expected: no,
  by RNG; the gate is designed for that).
- khj1222 published F1 only; no AUC exists for their arms, so the AUC baseline is ours to create.
- The exact all-zero-chunk filtering saves fetching ~230k 78-byte files; it was verified on four stores
  (0814 both masks, Paris4 w00/w09 supervision, native w035), not all 40.

## 12. Launch checklist

1. Commit `PREREG_BET_A_DRAFT.md` with the §9 addendum (A₃ already filled) — before anything else.
2. Write `pod_betA_arm0.sh` from `pod_p2a_v3.sh` (§4), embed `make_holdout_config.py`, the two
   vendored configs, `data_manifest.json`, the p2a_v3 `ctl_build/ctl_score/curvelib` parts and the F1
   sweep; `bash -n`, py_compile, `DRY=1` walk, synthetic end-to-end for the plan/verify logic.
3. Gist it (pinned raw URL), `launch_pod.py --deadline-hours 14 --disk 120`, laptop awake, guard on.
4. On `ALL DONE`: `results.json` → `bench/betA_arm0/RESULTS.md`, gate verdict, sync mirrors.
