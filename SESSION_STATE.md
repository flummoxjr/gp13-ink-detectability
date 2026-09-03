# SESSION STATE — written 2026-09-02 ~03:00 UTC (supersedes the 2026-08-25 version; read this before HANDOFF.md / NEXT_SESSION.md, which are historical)

Plan of record: `SEPTEMBER_PLAN.md` (month 1) inside `RESEARCH_PLAN_2026-27.md` (to June 2027, First Letters).
Artifact: "Road to First Letters" (claude.ai/code/artifact/4cae25e3-…). Balance $80.95 before the run below.

## The one thing that changed the picture (2026-09-01)

**The "ink_9um reads at chance (0.54) on a clean foreign scroll" anchor is VOID.** The 500p2a surface
volume is **2.215 µm/px, not 4.32** (three independent proofs in `bench/P2A_PITCH_RESOLUTION.md`: the
mesh bbox fits only the 2.215 µm public volume; the canvas is 1:1 with those voxels; letters are 2.5 mm
tall / 0.6 mm strokes at 2.215 vs 4.9 / 1.2 mm at 4.32). The Aug-25 run fed the model 4.80 µm data
presented as 9.36 µm — the "wrong in-plane level" fake-null fault (nerln #1648). Every "0.54 on a clean
foreign scroll" sentence in SEPTEMBER_PLAN §0/§1.2, RESEARCH_PLAN §0, the artifact's requirement-1 box,
and the 0358 v1 prereg is unverified until the corrected number below lands. Nothing was posted publicly.

## In flight / just finished

- **`bench/p2a_v3`** (DONE) — pod `hq7soy0hkbl6xi` (5090 community $0.69/h, launched 02:35 UTC, guard deadline
  05:06 UTC, `experiments/p2a_v3/`): the corrected anchor (win1/2/3 at 2.215→9.36, iso + fit17 depth
  modes, fwd+rev) **with positive controls first** (w035 crop native ≥ 0.95 fwd; reverse ≤ 0.80;
  the crop upsampled by exactly v2's factor 1.9504 = the fault reproduced; ×0.5). Pre-registered in the
  script header + `out/prereg.json` (sha 49097685225a). Results → `experiments/p2a_v3/p2a_v3_results.json`
  + `bundle.tgz`. Pre-stated readings: ≥ 0.85 → Aug-25 was an input fault and ink_9um transfers
  in-modality; 0.65–0.85 partial; < 0.65 transfer failure at the correct pitch (read with the CTL verdict).
  **LANDED 02:54 UTC (19 min, ≈ $0.22; `bench/p2a_v3/RESULTS.md`):** ctl native fwd/rev = **0.9991 / 0.5118**
  (on record 0.9991 / 0.5123); scale-fault ×1.95 = 0.7489 → FAULT_REPRODUCED; ×0.5 = 0.5227; **corrected
  anchor A₃ = 0.5211** (win1 iso forward; fit17 0.5301); win2 0.434/0.497, win3 0.491/0.496; expB all three
  SHAPE_CONFOUNDED (real at chance, translated-label nulls up to 0.92 → texture-driven output, not ink).
  **Verdict: transfer failure confirmed at the correct pitch; the Aug-25 number was void, its conclusion holds.**
  Consequence: Bet A's premise weakened (a pooled-class foreign input reads at chance) → run arm 0 only, hold
  arms 1/2 for Bet C's first number (`PREREG_BET_A_DRAFT.md` §1b). Plan docs + artifact corrected.
- Local validation before launch: `bash -n`, py_compile ×9, DRY walk, synthetic end-to-end (every verdict
  path, both fatal exits, translation annulus non-empty at [3.75, 5.90] mm, depth-upsampling path).

## Done this session (all $0 except the pod)

| item | where | state |
|---|---|---|
| Mirrors reconciled, verifier `__file__`-relative | trackD `3c80122`, gp13 `b26fede`; trees identical (`113b1d7…`); fresh-clone verifier 23 checks / 0 problems | pushed both |
| 500p2a pitch resolution | `bench/P2A_PITCH_RESOLUTION.md`, `bench/p2a/` (labels, x/y/z, meta) | done |
| Transfer benchmark v0.1 | `bench/BENCHMARK.md`, `bench/manifest.json`, scorer = p2a_v3 curvelib, `bench/vendor/make_holdout_config.py` (khj1222 PR #1608 @ dc9edb6) + recipe configs; offline LOSO test reproduces quotas {1667: 40, Paris4: 20, 0814: 4} | done; release after p2a_v3 |
| Bet A prereg | `PREREG_BET_A_DRAFT.md` — gate needs A₃ filled, then commit before any training | draft |
| Canvas off-by-one root cause | villa `tifxyz/types.py` `int(h/scale)` with float32-rounded scale 0.05000000074505806 → 3039 (and w035's 5819×5239); fixed in `runpod/render_tifxyz_sv.py` (canonical `round(h/scale)` canvas); Track B draft `issue_drafts/filing/tifxyz_fullres_shape_truncation.md` (Ben-gated) | done |
| 0358 screen v2 | `bench/screen0358_v2/` — v1 gist patched with the canvas fix + the corrected honesty frame (A₃ = 0.5211 filled); `PREREG_0358_v2.md` (sha dcfa3de67cf0); gist `gist_raw_url.txt`. **DONE 03:27 UTC** (pod `pou3y6h4s7gp7i`, 32 min, ≈ $0.37): 8/8 healthy on the canonical 3040 canvas, 0 tripwires, fwd/rev r 0.45–0.59 on every patch (control 0.094; corpus min 0.22 → gate 5 fails on all 8 before any periodicity test), hot pixels 0.2–1 % speckle. `bench/screen0358_v2/RESULTS.md`; records `out/screen0358_v2/`, `experiments/screen0358c/` (guard + mirror). The pod script does NOT bundle — `bench/tools/mirror_pod_out.sh` + `scripts/pod_guard.py --fetch-dirs maps,previews,results` saved the maps. Local PROTOCOL_V2 battery done 04:26 UTC: **control 5/5 (z +16.26, exact record); 0 of 8 patches flagged** (best 2/5; 0 at p ≤ 0.05 vs 0.4 expected; corrected z −0.75..+1.09; fwd/rev 0/8) → `out/screen0358_v2/battery_0358.json`, `bench/screen0358_v2/RESULTS.md` | DONE — null as pre-registered |
| pod_guard "ALL DONE" substring bug | already fixed 2026-08-25 (line-anchored regex) | verified |

## Parallel tracks started 2026-09-02 03:45 UTC (Ben asked "what other tracks should we start")

| track | what | state |
|---|---|---|
| Bet C corpus reconnaissance ($0) | agent wrote `bench/betC_corpus_manifest.json` (140 KB, all URLs live-checked) then was cut off by the Claude usage cap before the memo; totals: new native-7.91 µm ink labels **181 cm²** (Scroll 1 GP legacy 44 segments 153.7; Scroll 5 13 segments 23.1; Frag5/6 0.65), **92.4 cm² clean vs ink_9um**, 2,362 cm² labelled surface; raw 606 GiB, **139 GiB with bbox + layers 20–44 cropping**. Memo `bench/BETC_CORPUS.md` written from the manifest 2026-09-02 | manifest done; memo from manifest |
| Bet A arm 0 = LOSO baseline plan ($0 now, ~$5.5–8 to run) | **PLAN DONE** `bench/betA_arm0/PLAN.md` + `data_manifest.json` (119 KB): trainer already in villa-pin as `vesuvius.ink_detection.training.train`; data = labels ~1 GB + sparse level-2 chunk fetch 29.3 GB (16,482 chunks) + native eval crops 2 GB; khj1222: 21–25 min per 10k steps on a 5090; 2 seeds ≈ 8–11.5 h ≈ $5.5–8; anchor = native-5 F1 0.653 (floor 0.541) — prereg §4 corrected. Next: write `pod_betA_arm0.sh` per PLAN §4/§12 (30-iteration smoke before the data fetch), gist, launch with deadline 14 h, disk 120 GB | **SCRIPT BUILT + OFFLINE-TESTED 2026-09-03** (`bench/betA_arm0/pod_betA_arm0.sh`, 5,965 lines from `parts/` via `build.py`; tests: F1 sweep, sparse-plan geometry, config generator, gate paths, real 0814 label sync). **SMOKE launched 01:20 UTC 2026-09-03**: pod `2io34vg4fuuhdf` (5090 community, 27 vCPU/53 GB, $0.69/h), SMOKE_ONLY=1 (seed 42, 2,000 steps), guard 3 h, fetch-files results.json,bundle.tgz,ckpts_keep.tgz → `experiments/betA0smoke/`. `LAUNCH.md` has the full-run command (SMOKE_ONLY=0, deadline 14 h). **Smoke #1 (pod 2io34vg4fuuhdf) died at my own sparse-plan gate** after 62 min: trainer contract OK (30 synthetic its on real 0814 labels), all 15+5 label stores synced with exact checksums, then w029's plan 377 vs manifest 299 = 1.26× > the 1.25× band (our rule is systematically 1.15–1.26× the manifest's estimate); band widened to 1.6×, hfsync hardened (logs failures, low-concurrency retry), ≈ $0.75 spent. **Smoke #2 = pod `9wj9g2rrwkkjcr`** (5090, 12 vCPU/94 GB), launched 02:34 UTC 2026-09-03, guard 4 h → `experiments/betA0smoke2/` |
| Geometry substrate v2 (~$0.5–1) | pod `yd7zzui374fg4i` (builder image, gist `bench/grow_v2/gist_raw_url.txt`): 24 support-gated seeds each on PHerc0813 (`hunt/seeds_0813.json`, 8 kept + 16 new) and PHerc0826 (`hunt/seeds_0826.json`), tracer vc-tracer-de3c2494 (build fallback), waves of nproc, min_area 0.3; guard → `experiments/grow_0813_0826/` (bundle.tgz = paths_0813 + paths_0826 + logs). Then `hunt/gate_patches.py --scroll … --paths … --seeds …` locally; release v1 = gate-PASS tifxyz + QC JSON | pod `dbxycnf0fmllyk` (A6000, 32 vCPU) grew **PHerc0813 24/24 in 30 min** then DIED in the 0826 wave: one seed hit the 1800 s timeout, the inherited ERR trap parked that subshell, `wait` never returned, guard deadline-terminated at 06:54 — the 24 finished 0813 meshes were LOST (no per-scroll bundle, launch had no `--fetch-files`; ≈ $1). Fixed in `pod_grow_template.sh` (trap off in the seed subshell; per-scroll `out/paths_<id>.tgz` written immediately) + launcher forwards `--fetch-files`. **Relaunched 13:49 UTC as pod `bhlq4yipw6b8kb`** (5090 community, 37 vCPU, $0.69/h; guard 2.5 h, fetch-files paths_0813.tgz,paths_0826.tgz) → `experiments/grow_0813_0826e/`. **PHerc0813 DONE 14:08 UTC: 24/24 grown in 16 min; alignment gate (exact log pairing + LOCAL normal, `hunt/gate_patches.py`) 18/24 PASS = 154.6 cm²** → `hunt/pherc0813_grow_v2/` (meshes + gate) and **`hunt/pherc0813_release_v1/`** (catalogue.json + README + the 18 PASS tifxyz). Gate lesson re-confirmed: whole-patch (global) angles read 7–88° on these curved sheets while local angles are 0.4–25° for the passes; the first gate pass used bbox seed matching and was wrong (13 ambiguous) — use `--logs`. **PHerc0826 DONE 14:24 UTC: 24/24 grown; gate 21/24 PASS = 173.0 cm²** → `hunt/pherc0826_grow_v2/`, **`hunt/pherc0826_release_v1/`**. Pod ALL DONE 14:25, harvested + terminated (36 min, ≈ $0.41). **Substrate now: 0358 8/8 (69 cm²) + 0813 18/24 (155) + 0826 21/24 (173) = 47 gate-PASS patches ≈ 397 cm² on the three Tier-1 scrolls** — the Sept-14 release-v1 milestone is met on 2026-09-02; next is publishing it (Ben: Discord post + the team's catalogue format) and the model to run on it |
| w059 escalation (~$3–5) | Track F arm B full-coverage render from the 1.129 µm volume + same-model reverse + identical battery (`out/trackF/RESULTS_F.md`); queued after the growth pod (one pod at a time) | next |
| xacq null re-audit ($0 local) | **DONE 2026-09-03** `xacq_reaudit.py` → `out/xacq/reaudit_masks.{json,md}`: all 112 pairs re-scored under the frozen joint-valid mask (reproduces r 0.559 / L99 25.09 / nulls roll 0.013, rot −0.008 exactly) and under a sheet-footprint mask with B's footprint rolled WITH B's calls: L99 25.12, nulls 1.11 / 0.86, 103/112 ≥ 5, 101/112 survive both nulls — identical, because the model leaves essentially no exact-zero pixels inside the sheet (zero-frac 0.000/0.002). The 14.2× vs 25× gap is NOT a sheet-mask convention; next candidate is resolution/compression (our maps are downsampled JPEGs; block-averaging raises top-1% co-occurrence) — **scale addendum**: L99 24.7/25.0/26.6 at ds16/32/64 (flat); **registration addendum**: L99 25.2/25.2/24.8/24.0 at 8/16/24/32 native px mis-shift (flat) → the lift is a stroke-scale agreement invariant to mask, resolution and registration; 14.2× is not reproducible under any convention (L98 = 17.1, L95 = 9.4 → a q≈0.975 quantile would give ~14). Write-up `out/xacq/REAUDIT.md` | **DONE** |

## Next actions, in order

1. (done 2026-09-02 03:00) results read, plan docs + artifact corrected, A₃ in the Bet A draft, manifest updated,
   committed; mirrors synced with `bench/tools/sync_gp13.sh`.
2. (done 04:30) 0358 screen v2: measured, battery run, 0/8 flagged, recorded, mirrors synced.
3. Lane A Week-2 items: Bet A arm 0 (LOSO anchor reproduction) on a pod — the training data is HF
   `ink_9um/labels/{aligned-scrollprizeorg-21slices,native9-scrollprizeorg-21slices}` + S3 surface volumes
   (24 pooled + 5 native); recipe `bench/vendor/configs/`. zroll ladder (`gist_raw.txt`, `experiments/zroll`
   guard dir) — the Aug-19 pod ran blind; v2 script LAUNCH_READY per Aug 25 review.
4. Ben-gated: Discord rules questions (a)–(e) (RESEARCH_PLAN appendix), the tifxyz filing, Track F w059
   result post, the STRIPOFFSETS report.

## Infrastructure facts (do not relearn)

- Launch pattern that works (used tonight): gist raw URL (pinned sha) → `dockerStartCmd` = curl-with-retries
  + `exec bash script` → GraphQL `uptimeInSeconds` poll (container up in 47 s) → `scripts/pod_guard.py
  <id>:<name> --deadline-hours H --out experiments/<name> --no-status-min 15` (harvests status.txt,
  results.json, bundle.tgz from the pod's :8000, then terminates). Image
  `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`, ports `8000/http` + `22/tcp`,
  80 GB container disk. Provision (villa-pin + uv sync + torch 2.11 cu128) takes 50–100 s.
- Pods provision from `github.com/flummoxjr/villa-pin-37e300d3`; never `pkill -f` a pattern in your own
  command line; `set +e` does not suppress `trap ERR` — use `cmd || RC=$?`.
- Both repos: `measure-before-you-hunt` (private, canonical working truth = local trackD) and
  `gp13-ink-detectability` (public, what the Aug form cited); keep trees identical.
- `.venv` (3.12) has numpy/scipy/tifffile/PIL; anything importing `vesuvius` needs `.venv314`.
- **Pod sizing (2026-09-02):** for CPU-bound jobs (the tracer) use `launch_pod.py --min-vcpu N --min-ram G` — it goes through GraphQL
  `podFindAndDeployOnDemand(minVcpuCount, minMemoryInGb)`, which filters hosts; REST `minVCPUPerGPU` did not. Judge a pod by
  `pod.vcpuCount`/`memoryInGb`, NOT `machine.cpuCount` (host-level, misleading; two pods were killed on it needlessly ~$0.05).
  RunPod CPU-only pods (cpu3c/cpu5c…) returned 'no instances' for the builder image; untested further. An A6000 community
  host gave 32 vCPU / 156 GB at $0.33/h.
- `scripts/pod_guard.py` lives OUTSIDE the trackD git repo (project root is not a repo); its 2026-09-02 `--fetch-dirs/--fetch-files`
  change is on disk only. Pod scripts that write results under served subdirs need those flags or a bundle.tgz.
- GPU laptop was busy (87%) all evening — no local runs were attempted.
- **Claude usage cap** hit ~04:05 UTC 2026-09-02 (monthly spend limit; resets 2 am America/Chicago): the two research agents were
  killed mid-task (Bet A pod-script agent wrote nothing; corpus agent wrote the manifest only). Until it resets, no subagents.

### 2026-09-03 02:28 UTC — w059/C0 pod launched (ef8tzdyssyk0b8)
- `bench/w059_c0/pod_w059_c0.sh` (sha 418747f13a99597d) committed as 8c856b2 BEFORE launch (prereg locked in-script; C0 gate r>=0.90 vs the published w035 A map; then w035 B-arm L1 control; then w059 B-arm L1 fwd+rev at full coverage; ds4/ds16 maps for the local PROTOCOL_V2 battery). Gist raw URL in `bench/w059_c0/gist_raw_url.txt`.
- Pod ef8tzdyssyk0b8 (5090 community, 12 vCPU, $0.69/h), guard 5 h / no-status 20 min, `--fetch-files results.json,bundle.tgz,status.txt`, out dir `experiments/w059_c0/`. Runs CONCURRENTLY with smoke #2 (9wj9g2rrwkkjcr) via `--allow-concurrent`; both landed on host suffix 644111c4 and both were still runtime=null at 02:31 (10 min for the smoke pod) — if a no-status abort fires, relaunch on a different GPU/cloud attempt order.
- gp13 resynced at 3212cad (tree 364d10b3…) after 8c856b2.
- **Smoke #2 (9wj9g2rrwkkjcr) aborted by the guard 02:43 UTC**: the community 5090 host (…-644111c4) never started the container in 21 min (runtime=null throughout; nothing of ours ran; ≈$0.25). **Smoke #3 = pod `33xqiggp0xvfhi`** (5090 SECURE, $0.99/h, 12+ vCPU), launched 02:44 with the same gist; BOOT 02:43:57 UTC within a minute of creation, PREREG locked sha df6492905823, guard 4 h, out `experiments/betA0smoke3/`. The w059/C0 pod ef8tzdyssyk0b8 is on the SAME host; its guard's no-status abort is due ~02:56.
- **Smoke #3 (33xqiggp0xvfhi, 5090 SECURE) FAILED in provision 02:51 UTC**: `uv sync` could not fetch the CUDA-13 wheels from pypi.nvidia.com from that host (nvidia-curand: "Request failed after 3 retries in 126.6s ... operation timed out"); the pod sat in FAILED billing until terminated by hand at 02:54 (≈$0.17). Fixes: (1) `scripts/pod_guard.py` now treats a line-anchored `FAILED -- run is dead` as terminal (harvest + terminate); (2) `bench/betA_arm0/build.py` injects a PREFLIGHT before uv sync (8 MB range download from pypi.nvidia.com, die if < 1 MB/s; reachability of pypi.org / HF / S3 logged) and exports `UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=6` with `timeout 2400` per try. Rebuilt script sha d1abe3ec3bbdd7a2, gist re-pinned. **Smoke #4 launched ~02:58** with attempt order 5090 COMMUNITY → 5090 SECURE → 4090 SECURE, out `experiments/betA0smoke4/`.
- **w059/C0 pod ef8tzdyssyk0b8 aborted by its guard 02:57 UTC** (never booted on the same community host as smoke #2; ≈$0.35). **Relaunched as `wluhqd2196l2o6`** (5090 SECURE, $0.99/h) at 02:58, same gist/prereg, guard 5 h, out `experiments/w059_c0_b/`. Smoke #4 `x7ijyqlk5dbx1c` (5090 COMMUNITY) booted 02:57:22, PREFLIGHT pypi.nvidia.com 4.97 MB/s, uv sync running.
- **w059/C0 pod wluhqd2196l2o6 (5090 SECURE, US host …644120fb) terminated by hand 03:23 UTC**: pip read-timed-out on files.pythonhosted.org, then the second attempt made no visible progress for 20 min; a host with a slow external route cannot stream the ~140 GB of S3 zarr this job reads (≈$0.42). `parts/stages.sh` now preflights files.pythonhosted.org throughput (die < 1 MB/s) and runs pip with `--timeout 180 --retries 8` (script sha aa1db27ce65d8f72, gist re-pinned). **Third launch = pod `0q61zt4fkn0nki`** (5090 COMMUNITY, $0.69/h) 03:24, out `experiments/w059_c0_c/`. Lesson: for I/O-bound pods prefer COMMUNITY hosts that pass the preflight; both SECURE 5090 hosts tonight had crippled PyPI/NVIDIA-index routes.
- **w059/C0 pod 0q61zt4fkn0nki FAILED 03:36 UTC at the import check** (guard terminated it on its own within 20 s — the new dead-run rule works; ≈$0.15). Cause in `experiments/w059_c0_c/provision.log`: `torch>=2.0.0` in optimized_inference's requirements.txt let pip install torch 2.14.0 into the venv over the image's 2.8 build (torchvision/torchaudio mismatched). Fix in `parts/stages.sh`: pip runs with `-c constraints.txt` pinning torch/torchvision/torchaudio to the image's versions, and the import check now writes its traceback to the log + status. Script sha 5ad12a6342eb7c99, gist re-pinned; **fourth launch = pod `ckqw3dbr2jgyvv`** (5090 COMMUNITY, $0.69/h) 03:39, out `experiments/w059_c0_d/`.
- **w059/C0 pod ckqw3dbr2jgyvv terminated 03:48 UTC**: it landed on the same FR community host `…-644111c4` that never booted smoke #2 and the first w059 pod (runtime=null 8.5 min; ≈$0.10). `bench/tools/launch_pod.py` gained `--avoid-host <suffix,...>` (queries `machine.podHostId` after create; terminates and re-runs the attempt list, up to 4 times). **Fifth launch ~03:50** with `--avoid-host 644111c4`, order 5090 COMMUNITY → 4090 COMMUNITY → 5090 SECURE, out `experiments/w059_c0_e/`.
- **Fifth launch (03:50) burned three creates**: RunPod placed every `5090/COMMUNITY minVcpu 8` request on `…-644111c4` (vzpd9w32za8wk7, a95m5hi39zp0pg, blhtjdq5hfx598 — each terminated within ~10 s by `--avoid-host`, ≈$0). Launcher now drops the offending GPU/cloud pool from the attempt list on retry. **Sixth launch ~03:51**: 4090 COMMUNITY → 5090 SECURE → 5090 COMMUNITY, out `experiments/w059_c0_f/` (BATCH_OI=8 fits a 24 GB card).

