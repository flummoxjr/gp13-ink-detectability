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
| 0358 screen v2 | `bench/screen0358_v2/` — v1 gist patched with the canvas fix + the corrected honesty frame (A₃ = 0.5211 filled); `PREREG_0358_v2.md` (sha dcfa3de67cf0); gist `gist_raw_url.txt`. **LAUNCHED 02:58 UTC** via `bench/tools/launch_pod.py` → `experiments/screen0358c/` (guard deadline 3 h). Then: local battery on the ds4 npys | in flight |
| pod_guard "ALL DONE" substring bug | already fixed 2026-08-25 (line-anchored regex) | verified |

## Next actions, in order

1. (done 2026-09-02 03:00) results read, plan docs + artifact corrected, A₃ in the Bet A draft, manifest updated,
   committed; mirrors synced with `bench/tools/sync_gp13.sh`.
2. 0358 screen v2 in flight (`experiments/screen0358c/`): when the guard harvests `bundle.tgz`, run the local battery
   (`analyze_survey_corpus_v2.py` on the ds4 npys; w035_CONTROL_strided must reproduce 5/5 first); any flag →
   escalation only, never announcement.
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
- GPU laptop was busy (87%) all evening — no local runs were attempted.
