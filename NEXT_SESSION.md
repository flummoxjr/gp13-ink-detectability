# COORDINATION NOTE 2026-08-25 (Track A/B/close session)
NAMESPACE PROTOCOL (agreed after a shared-filename race cost a run + ~$1.15): NO session writes
un-prefixed filenames in runpod/ or scratchpad shared dirs. Track D prefixes trackD_*, this
session prefixes tAB_*. Same rule class as never-switch-branches-in-shared-villa.
0358 SPLIT AGREED via cross-session message: Track D session owns GROWTH + GEOMETRY QC
(pod 1ixkk647u1ylf7, results -> hunt/pherc0358_first_surfaces/); this session owns
RENDER + ink_9um + BATTERY on gate-passing patches (<30 deg). Do not double-launch.
My redundant 0358 pod mtj9zto9lpqq5a was terminated by Track D (correctly, per cost rules).

# Track D — guide for the next session

_Written 2026-08-18. Read this first, then `LOG.md` for the chronological detail._
_Canonical plan: `../WORKPLAN.md`. Ben's action list: `report/BEN_TODO.md`._

---

## 0. Sixty-second orientation

Ben is competing for Vesuvius Challenge prizes. Deadline for the monthly progress prize:
**Aug 31 2026, 11:59pm Pacific**, Google Form at scrollprize.org/prizes.

**SHARED-TREE RULE (2026-08-25, learned at ~$1.15):** two sessions raced on the same filename in
`trackD/runpod/` — one overwrote the other's pod script minutes before it shipped, and the mismatched
watchdog deadline-terminated a healthy run with an empty fetch. Extension of the villa-checkout rule:
**namespace every pod script and watcher you write (`trackD_*` for this session; pick your own prefix),
never write to un-prefixed shared filenames in runpod/, and make watchdogs tar `paths_*` + all logs on
every exit path** so a label mismatch degrades to mislabeled data instead of data loss.

Track D is the contrarian track: **measure whether ink *could* be found before hunting it.**
Tracks A (nnUNet surface models) and B (villa bug fixes) are owned partly by a **parallel
session working in the same repo and on the same RunPod account** — coordinate, don't collide.

The submission is written and verified: **`report/REPORT.md`** + `report/sections/01-04` +
`report/REPRODUCIBILITY.md`. `report/scripts/verify_report.py` re-asserts every headline number
against primary artifacts — **run it after any edit to the report** (currently 11 checks, 0 problems — now including the v2 corpus screen and a sweep that fails if a superseded v1 number reappears unmarked).

---

## 1. What is PROVEN (do not re-litigate)

| Result | Number | Primary artifact |
|---|---|---|
| Our stack reproduces published letters at 9 µm | pixel AUC **0.9991 / 0.9982** (2 seeds) on PHerc0139 w035 | `out/ink9um_w035_scores.json` |
| Our tifxyz→surface-volume renderer is faithful | r = **0.813** vs the published surface volume, passes the ruling test | `runpod/render_tifxyz_sv.py`, LOG 08-17 |
| Entire published GP segment corpus screened | **80/80 rows, 66 unique surfaces, 416 cm², 0 tripwire hits** | `out/survey/survey_all.json` |
| Corpus shows no text under a hardened protocol | **0 of 71 pass all 5 gates**; control passes 5/5 at z=+16.3, p=0.005, 4.678 mm, 11 cycles | `out/survey/corpus_analysis_v2.json`, `PROTOCOL_V2.md` |
| Scan-quality index for all 13 GP scrolls | two tiers, **3.0× gap**, campaign correlation Fisher p=0.007 | `out/k2b_index/*.json` |
| villa cache bug + fix | 4 GB budget → **5.98 GB**; fixed → **4.00 GB**, real PHerc1203 data | `issue_drafts/PR_cache_budget.md` |
| Paris 4 dual-energy Pb screen | 2 hits in 2.7×10⁸ (clean null) — but only excludes ~10× beyond published Pb loadings | `out/k3_sensitivity_bound.md` |
| First surfaces ever grown on PHerc0813 | 8 patches, **99.9 cm²**, 5 on material (68.5 cm²) — but all 8 sit at a median **68.1° to the lamellae** and cannot be used (§3) | `hunt/pherc0813_meshes/`, `hunt/pherc0813_mesh_qc.json`, `out/k2c_separability/pherc0813_mesh_alignment.json` |
| Sheet-separability axis, all 14 scrolls | control ranks **#1** (0.744 at n=24 vs a 0.105 isotropic floor); ρ=+0.27 with SNR, i.e. a genuinely different axis | `out/k2c_separability/k2c_analysis.json` |
| K2b's ROI picker samples incrustation, not papyrus | random vs intensity-picked **3.00×**, higher in **14/14** scrolls, p=3.6e-28 | `out/k2c_separability/k2c_analysis.json` |

## 2. What is REFUTED (killed with numbers — don't resurrect without new data)

- **S1a v1 and v2** (letter-contrast on PHerc1667 w032): backgrounds were no-data / off-sheet air. `qc/s1a_verification.md`, `qc/s1a_v2_verification.md`
- **The corpus flag** (`z_dbg_gen_00166_inp_hr`, z=+5.94): → +0.97, p=0.16, 0/5 gates. E[#z≥5 among 80] = **1.07 vs 1 observed**. `verify_flag/FLAG_VERDICT.md`
- **Mesh-misplacement hypothesis for the PUBLISHED corpus**: GP meshes are as well-placed as the control (worst 2.6 vox; ±3 vox costs only 2.4% of signal), and independently confirmed by orientation — published meshes sit 13.1° from their local sheet normal, 7/9 within 30°. `hunt/geometry_compare.md`, `out/k2c_separability/published_mesh_alignment.json`. **This does NOT extend to our own PHerc0813 patches**, which are misaligned by 68.1° (§3) — do not conflate the two.
- **Input-side domain adaptation** for ink_3d: verified histogram matching **doubled** false firing (f05 0.039→0.081).
- **"Damage atlas" reframe**: sign-inverted (fires on *intact* material, ρ=−0.54 with crack texture) and dominated by free CT statistics.
- **K1 depth-contrast as a portable detector**: 0.5717 on discovery segment, 0.4934 mean on held-out (one *inverted*). Real but region-unstable.
- **ContextVar/zarr cache crash**: could not be reproduced on a live pod → not filable.

## 3. The separability axis — BUILT, and it overturned this section's original claim (2026-08-18, late)

This section previously said: *"PHerc0813 ranks #1 of 14 on structural SNR… High SNR, unresolvable
sheets."* **That was wrong, and the correction is the finding.** Scripts: `k2c_separability.py`,
`k2c_analyze.py`, `hunt/mesh_lamella_alignment.py`, `hunt/alignment_control.py`.
Artifacts: `out/k2c_separability/`.

**The axis.** Sheet separability = median over 32³ blocks of structure-tensor planarity
(λ1−λ2)/(λ1+λ2). Measured on 12 uniformly-random ROIs per scroll, all 14 scrolls, 168 cubes.

| finding | number |
|---|---|
| Control PHerc0139 (letters proven at 9 µm) ranks **#1 of 14** | **0.748** [0.738, 0.775], tightest CI of any scroll |
| Isotropic floor (28 real in-scan air windows) | **0.105** |
| Separability vs K2b structural SNR | Spearman ρ = **+0.336, p = 0.24** — genuinely different axes |
| Ordering is parameter-invariant (block 16/32/64, σ 0.5/1/2) | Spearman ρ = **+0.978 … +0.996** (n=14) |
| Induced clipping does **not** lower it (confound refuted, conservative direction) | sat 0.001→0.37 moves 0.456→0.525 |

**PHerc0813 scores 0.665 — rank 3 of 14, six times the floor.** Its sheets are *not* unresolvable;
the images show clean laminae at every depth from z=4,736 to z=13,360.

**Why the grown patches looked flat — MECHANISM FOUND.** In absolute coordinates our 8 meshes have
**|n_z| = 0.876–1.000 (median 0.974, 8/8 above 0.7)**; the 9 published/control meshes have **0.004–0.307
(median 0.223, 0/9)**. Complete separation, no overlap. **Our surfaces lie flat in the xy slice plane, cutting
across the scroll's wraps; published ones run vertically through slices, following them.** Visible directly:
ours put 24–34 vertices in one slice, published 0–4 (a sheet-following surface is seen edge-on).
Everything else matches — vertex spacing 187.9 µm vs their 172.8–194.6, mesh flatness 0.045–0.056 vs 0.048–0.075.
Probable cause: our v3 run dropped `normal_grid_path` after the streaming grid errored, so the tracer defaulted to
a z-axis orientation. NOT established — only 5 of 10 comparison meshes record `vc_gsfs_params`, and among them it
is 2 with a grid (13.1°, 15.2°) vs 2 without (11.3°, 47.5°). Do **not** confuse `normal_grid_path` (released for
PHerc0813, tutorial says REQUIRED) with `direction_fields` (structure tensors, not on the bucket, truly optional).
**NORMAL-GRID HYPOTHESIS: TESTED TWICE, REFUTED.** Attempt 1 (08-21) was invalid — the script also
switched `--volume` from the m7 surface-prediction zarr to the raw masked CT. Attempt 2 changed ONLY
`normal_grid_path`: **|n_z| 0.974 (n=8) → 0.986 (n=3)**, still all >0.7 vs published 0.223 / 0-of-9. The grid
was in play (recorded in the new meshes' meta). **The missing direction field is NOT the cause.**
Meshes: `hunt/pherc0813_meshes_ngrid2/`. 3 of 8 seeds finished; 5 hit the 1500 s cap.
**TWO OPERATIONAL TRAPS, both cost hours:**
  1. The tracer runs on the **surface-prediction zarr**, never the raw CT. Check `--volume` first.
  2. `nohup … & disown` inside a one-shot ssh exec **dies when the channel closes** on this image — the job
     never even creates `/work`. Hold the SSH channel open (`runpod/run_grow_held.py`).
**UPSTREAM BUG worth filing:** `vc_grow_seg_from_seed` opens `<normal_grid_path>/metadata.json` LOCALLY even
for a remote store, so the documented `normal-grids-remote.json` recipe core-dumps. Fetch that one file.
**LEAD CLOSED:** `area_cm2: 0.0` is a bare-prediction-zarr artefact (our own v3 header says so). Real area
from vertices is 13.70 cm².
**WHAT SURVIVES — SHARPENED 2026-08-25 by the mode A/B forensics:** growth on a bare zarr URL is
**data-blind**. In a 16-run A/B (8× `mode: random_seed` at genuinely scroll-wide random locations, 8× our
explicit seeds), every worker produced ~8.70M vx² surfaces on an IDENTICAL expansion schedule — fringe +8
per generation, done=21,900 at gen 74, everywhere, both modes. Candidate rejection never fires. The
published runs that stop at sheet boundaries (extents 850–2,530, real cm² areas) all ran inside a
**volpkg** (`/volpkgs/PHerc1203.volpkg/...`); ours pass a bare `https://...zarr`. One root plausibly
explains the whole family: no volpkg context → no voxelsize (area 0.000000, re-confirmed: `min_area_cm
0.3` silently discarded 16/16 grown surfaces), and possibly no data term in candidate acceptance →
schedule-driven flat growth. Mode/seed choice is NOT the cause (both arms identical). Also confirmed:
`random_seed` mode genuinely randomizes (8 distinct locations). **MODE A/B COMPLETED 2026-08-25 (min_area 0.0): both arms flat — random_seed |n_z| median 0.979, explicit
0.989, 16/16 above 0.7 (`out/k2c_separability/modetest_nz.json`). Mode and seeds exonerated by experiment.**

**GROWTH MYSTERY SOLVED (2026-08-25).** Cause: the `:edge`/`:main` container images are a **2026-05-13
build** whose tracer misreads the volume (reads 103 at a genuinely-255 voxel — clue from @Bullo27 on #1588).
Rebuilt `vc_grow_seg_from_seed` from main (`de3c2494`, ~8 min compile on a builder image) and re-grew the
identical 8 seeds: **|n_z| 0.974 → 0.236** (published 0.223), **angle 68.1° → 20.5°** (published 13.1°),
**areas 0.000000 → 8.08–9.73 cm²** (which also retires the volpkg theory — a stale-build artifact too).
One binary explained all five symptoms. Artifacts: `out/k2c_separability/mainbuild_{nz,angles}.json`,
meshes `hunt/pherc0813_mainbuild/`. Posted to #1588 + #191.
**OPERATIONAL:** the built binary died with the pod — next growth run rebuilds in ~8 min on a builder image (my earlier 5 h figure was a local-vs-UTC
timestamp misread) — still tar the binary home for reuse. **NEVER use `:edge`/`:main` images until republished.**
**DONE 2026-08-25: PHerc0358 FIRST SURFACES — 8 patches, ≈69 cm², 8/8 PASS the alignment gate** (local angles 3.6–18.6°, |n_z| 0.005–0.351, areas 8.41–9.12 cm², all seeds read 255). Grown with the current-main tracer from this axis's own top ROIs; gate artifact `hunt/pherc0358_first_surfaces/alignment_gate.json`. **Render + ink_9um + battery is the OTHER SESSION'S lane (agreed split)** — results land in out/. Reusable tracer published: github.com/flummoxjr/gp13-ink-detectability/releases/tag/vc-tracer-de3c2494 (159 MB, skips all future builds). Gate lesson: use the LOCALLY-RESTRICTED angle (corpus_alignment_local.local_normal) — the global measure read 33–75° on these healthy curved sheets.
Track B same day: #1479 MERGED 08-22 (second merge); Bullo27 byte-identically verified #1586/#1587 on
zarr 3.2.1; spread recorded on zarr#4282.**

**The bigger finding: K2b's ROI picker samples the wrong material.** `pick_rois` scores candidates
`np.where(fill > 0.98, inten, 0)` and takes the **highest** — the brightest dense material in each
scroll, which is mineral incrustation, not papyrus. Sampling the *identical* frame uniformly at
random instead: **random 0.564 vs intensity-picked 0.168, median ratio 2.95×, random higher in
14 of 14 scrolls, Mann-Whitney p = 5.4e-25.** Picked ROIs run 24–68 DN brighter in every scroll, at a median saturation of
0.039 against 0.0004 for random. This is the project's own recurring trap — *a reference
class that silently contains something other than what it names* — sitting in the flagship §1 index.

**External check.** The three scrolls the community actually segmented (PHerc1447 52 segments,
PHerc1203 22, PHerc0800 6) sit at separability ranks **5, 6, 7 of 14** but SNR ranks **8, 13, 14**.
PHerc1447 has the **worst scan quality of all 14** (SNR 8.4) and **the most published segments**.
SNR says don't bother; separability says rank 5. (n = 3 vs 11, so this is a qualitative inversion,
not a significant test — MW p = 0.277.)

**Method caveat, stated because it bit me:** a phase-randomisation null was tried first and is
**invalid** for this statistic — `J_ij = Σ_q q_i q_j |F(q)|²`, so the structure tensor is phase-blind
by construction and that null could never have failed. Removed. The statistic is the *angular*
anisotropy of gradient power; K2b's SNR is a *radial* property of the same spectrum.

### Ranked separability, all 14

| rank | scroll | separability | 95% CI | SNR rank |
|---|---|---|---|---|
| 1 | **PHerc0139** (control) | 0.748 | [0.738, 0.775] | 2 |
| 2 | **PHerc0358** | 0.713 | [0.689, 0.749] | 7 |
| 3 | **PHerc0813** | 0.665 | [0.600, 0.735] | 1 |
| 4 | PHerc0826 | 0.634 | [0.505, 0.704] | 9 |
| 5 | PHerc1447 | 0.605 | [0.566, 0.681] | 14 |
| 6 | PHerc1203 | 0.570 | [0.532, 0.603] | 8 |
| 7 | PHerc0800 | 0.563 | [0.511, 0.674] | 13 |
| 8 | PHerc1545 | 0.541 | [0.455, 0.608] | 4 |
| 9 | PHerc0211 | 0.530 | [0.468, 0.599] | 5 |
| 10 | PHerc0191 | 0.506 | [0.418, 0.643] | 6 |
| 11 | PHerc0125 | 0.415 | [0.360, 0.516] | 3 |
| 12 | PHerc1218 | 0.389 | [0.334, 0.611] | 10 |
| 13 | PHerc0257 | 0.374 | [0.295, 0.483] | 12 |
| 14 | PHerc0268 | 0.337 | [0.291, 0.397] | 11 |

Many CIs overlap, so treat this as **tiers, not a strict ranking**. PHerc0358 is the standout
non-control target and is *not* near the top on SNR.

---

## 3b. RECALIBRATION (2026-08-23) — read before doing anything else

**Eight days to the deadline. The critical path is shipping, not research.** Everything below §4's rows 1–2
is Ben-gated and none of it needs a GPU: make the repo public (A6 — eligibility), Ben's 30-minute review of
the voice pass (A1 figure check above all), the Google Form, the Track B filings (verified ready 08-20), and
the Discord posts. The growth mystery is interesting and is a trap: three hypotheses died, ≈$8 went to pods
idling, and the report is already honest without the answer. **Parked** — two candidates remain
(`mode: random_seed`, `min_area_cm: 0.3`), test them once, attended, after submission, cap $1.

**Cloud-ops rule, learned four times:** never rely on a local watcher to stop a pod — this machine's DNS
flakes and killed one mid-watch. Bake self-termination into the pod's own start command (hard `sleep`+kill
via the RunPod API from inside, or a dockerStartCmd timeout), and only launch what will be attended.
`nohup … & disown` in a one-shot ssh exec dies with the channel on the VC image — hold the channel open
(`runpod/run_grow_held.py`).

**n=24 EXPANSION: PARKED IN A CLEAN STATE (2026-08-23).** For release correctness the tree was rolled back to the shipped n=12 everywhere: `k2c_separability.py` is back at `N_RANDOM = 12`, the 14 n=12 per-scroll JSONs are restored in `out/k2c_separability/`, and the completed n=24 partials (13 of 14 scrolls; 0139 died on a DNS flake) are parked in `out/k2c_separability/_n24_partial/` with all cubes still cached on D:. Shipped code now reproduces shipped numbers exactly. To finish the expansion later: set N_RANDOM=24, run the sweep (only 0139 fetches), re-run `k2c_analyze.py`, and update §1.8 + `verify_report.py` in ONE pass. Early n=24 medians track n=12 within ±0.02 (0813: 0.665→0.662), so expect tightened CIs, not a re-ranking.

**Measurement debt:** the corpus alignment audit's angle has a curvature confound for large meshes (w032
control reads 59° under the same method — now caveated in §2.7). The fix is a locally-restricted recompute
over cubes already cached in `D:/vesuvius-data/trackD/corpus_cubes` — pure local compute, no downloads,
~10 min. Do it before anyone quotes "19 of 56" as a census.

## 4. Next actions, ranked

| # | Action | Cost | Why now |
|---|---|---|---|
| ~~1~~ | ~~Add the separability axis to the index~~ — **DONE**, see §3. All 14 scrolls; control ranks #1; ROI-picker bias found | $0 | Superseded by the two rows below |
| 1 | **Write the separability axis + the ROI-picker correction into report §1**, and add k2c assertions to `report/scripts/verify_report.py` | $0, local | The index is the flagship deliverable and its ROI values are drawn from the wrong material (§3). This is now the highest-value edit in the report. |
| 2 | **Re-grow PHerc0813 with `normal_grid_path` and read |n_z|** | **≈$0.10** | Decisive test of the mechanism in §3. Growth only — no render, no inference. If |n_z| drops 0.97 → ~0.2 the cause is settled and surface growth is unblocked for every unsegmented scroll, including PHerc0358. If it does not, the seed rule is the next suspect. This gates all further growth spend. |
| ~~2b~~ | ~~Render + infer + battery the 5 on-material patches as grown~~ | — | **Do not.** They are oblique to the lamellae: a null would be uninformative and a positive would be luck. §2.9.1's closing status line records that they were withdrawn and why. |
| ~~3~~ | ~~Track B filing~~ — **DONE 2026-08-24**: villa #1586 (issue) + **#1587 (PR, mergeable)** + #1588 (new normal-grid bug) + zarr #4282, cross-linked; context comment on #1492. The PR-creation 404 is gone. | $0 | Optional leftover: cucim comment on #1479 (verify #1268 first). |
| 4 | **Ben's report voice pass** (`report/BEN_TODO.md` items A1–A9) | ~2 h | The report cannot ship without it. |
| 5 | Grow surfaces on the next-best scrolls — **PHerc0358 (sep 0.713, rank 2) first**, then PHerc0826 (0.634). NOT 0125/1545/0211: they rank 8, 9, 11 on separability despite good SNR | ~$0.35 each | §3 now says where to aim. Always pass a normal direction field. |
| 6 | The 2.4 µm path on PHerc1203 (`hunt/two_micron_path.md`) — transform already derived | ~$0.50 control first | Partly independent physics; laminae measured as merged, so odds moved down. |

---

## 5. Operational knowledge (hard-won — re-reading this saves hours)

**Budget.** `$45.54 of the $80 cap spent` as of writing; **the cap is shared with the parallel
session** (its pods are named `c2-*`/`c3-*` — do NOT terminate them). Burn was $2.25/hr with three
of their pods up. `runpod/rp.py status` shows spend/burn; `rp.py` refuses new pods past $70.
**Kill your own pods the moment a result is read** — an idle pod cost us ~$1 before it was noticed.

**The villa checkout is shared.** It moves between branches (`cycle2` → `cycle3` today) because the
parallel session works there. **Never switch branches in it.** Use `git worktree add` to a temp dir,
and note that `.venv314` has vesuvius installed *editable* pointing at the shared tree, so pytest in
a worktree still imports the shared code unless you set `PYTHONPATH=<worktree>/vesuvius/src`.

**Pod recipe that works** (`runpod/provision_survey.sh`): RunPod pytorch image → `uv sync --extra
models` → **pin `torch==2.11.0` + `torchvision==0.26.0` cu128** (host driver caps at CUDA 12.8;
PyPI torch 2.13 fails) → explicit dep list → `TORCH_COMPILE_DISABLE=1` (no triton on some images).
For **custom images** (e.g. `ghcr.io/scrollprize/villa/volume-cartographer:edge`) there is **no sshd** —
inject one via `dockerStartCmd` (see `runpod/launch_grow_0813.py`).
Throughput: render+infer ≈ 2.5–5 min/segment on a 5090; 80 segments ≈ $5.5.

**Traps that have already bitten us:**
- **Zero-fill ambiguity**: a failed chunk fetch and an empty volume both read as zeros. Count
  server 404s separately (`hunt/qc_new_meshes.py`). This is the #1 recurring trap in this project.
- **Small-null z inflation**: 16 permutations underestimated the null sd by 2.2×. Use ≥200 and
  report **empirical p**, not z.
- **`gh search issues` silently omits PRs** — duplicates here arrive as PRs. Use
  `gh api -X GET search/issues`. (gh IS authenticated as `flummoxjr`.)
- **`git apply` fails across differing checkouts** — copy the whole file instead.
- **PowerShell mangles `$(...)`, `<`, quotes** in remote commands — write a `.sh`, `scp` it, run it.
- **Tiled inference creates patch-boundary halos**: 90.8% of one segment's hot pixels lived in the
  stride lattice's halo ring. Erode rims before scoring prediction maps.
- **villa's cache overshoots its budget ~2×** until our fix lands — budget disk accordingly.

**QC discipline that has repeatedly saved us** (11-row ledger in `report/sections/04_methodology.md`):
reproduce the numbers exactly → attack the meaning with a *new* experiment → every positive must
beat a **matched** null → name the mundane explanation → account for autocorrelation. Five of the
eleven corrections were the same trap: *a reference class that silently contained something other
than what it named*.

---

## 6. Where things live

```
trackD/
  LOG.md                     chronological record (read after this file)
  NEXT_SESSION.md            this file
  report/                    THE SUBMISSION — REPORT.md, sections/, figures/, BEN_TODO.md,
                             COMMUNITY_POST.md, REPRODUCIBILITY.md, scripts/verify_report.py
  out/k2b_index/             the 14-scroll index
  out/survey/                corpus screen: survey_all.json, corpus_analysis_v2.json,
                             PROTOCOL_V2.md, maps_shard*/ (150 prediction maps)
  out/ink9um_w035*           the positive control
  out/k3_*                   Paris 4 dual-energy screen + sensitivity bound
  hunt/                      the five letter-hunt investigations + HUNT_PLAN.md +
                             pherc0813_meshes/ + geometry_compare.md
  qc/, qc_live/, salvage/,
  comb/, verify_flag/        the audit trail (shipped as first-class artifacts)
  issue_drafts/              Track B: FILING_CHECKLIST.md + filing/ (paste-ready bodies)
  runpod/                    all cloud tooling (rp.py, fleets, screener, renderer)
```

---

## 7. Honest odds (from `hunt/HUNT_PLAN.md`, unchanged by today's results)

P(defensible First Letters claim by Jun 2027): **12–18%**. By Aug 31 2026: **4–7%**.
Failure across the six good scrolls is **correlated**, not six independent draws — our index sees
scan quality, and (as of today) not sheet separability, so it tells us where we can best *look*,
not what is there.

**Plan the monthly submission around the deliverables, not around letters.** The deliverables are
real, verified, and unclaimed by anyone else: the index, the validated instrument, the corpus
screen, the two reusable gates, the transfer-failure characterization with its fine-tune recipe,
the dual-energy null, and the corrections ledger.
