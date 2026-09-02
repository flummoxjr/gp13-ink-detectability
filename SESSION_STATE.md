# SESSION STATE — written 2026-08-25 00:15 UTC (supersedes NEXT_SESSION.md and HANDOFF.md where they conflict)

**Deadline: Progress Prize form closes 2026-08-31 23:59 Pacific.** Balance ~$84.9, zero pods billing.

## The four pre-registered verdicts now banked (all committed, all pushed to backup)

| study | prereg | verdict | the number |
|---|---|---|---|
| Ink Mass Budget v2 | `PREREG_G1_V2.md` @ a2e0001 | **CEILING** | pooled 2σ ≤ **0.578 vox-equiv** (block) / 0.691 (rigid) — first bounded upper limit on carbon-ink areal contrast; `out/g1v2/` |
| Cross-acquisition fusion | `PREREG_XACQ.md` @ fb8b55b | **KILLED** (gain +0.0080 < 0.01 bar) | but agreement itself is real: L(0.99) median **25×** vs nulls ~1; strongest agreement exactly on the human-read segments; `out/xacq/` |
| Fragment transfer (curve baseline) | pod prereg sha ab3e8d9d4c6d | **KILLED_BASELINE** | first-ever ink_9um score on a fragment: **fwd 0.6925 / rev 0.4477** vs 0.9991 in-domain; fwd/rev asymmetry persists out-of-domain; `out/curve_audit/` |
| Null methods | `out/null_scaling/` | SHIP_READY | n_eff 566/2560; sd·√n non-transferable 2.4×; **0-of-71 corpus verdict survives corrected nulls** (passers 4→2 vs 3.55 expected by chance) |

## Live threads

1. **Curve v2**: re-anchor the detectability curve on **500p2a** (scroll-derived, clean for ink_9um, P1-proven pipeline). Needs a fresh prereg BEFORE launch. The pod script `runpod/pod_curve_audit.sh` works end-to-end now — the whole infra war is over (see below).
2. **Hallucination audit**: also blocked by the Frag1 baseline kill; re-scope onto 500p2a windows (already on disk in the metrology scratchpad + re-fetchable) under the same v2 prereg.
3. **Depth-encoding ladder** (`scratchpad/zroll/pod_zroll.sh`, reviewed LAUNCH_READY): the 0.69/0.45 fragment asymmetry strengthens its motivation.
4. **Jinhojeong swap**: harness delivered (github.com/flummoxjr/facing-pairs-harness, #191 comment 2026-08-24). Await their twin scoring; their v5 tight-gap set is the cycle-4 val target.
5. **BEN-GATED, unchanged**: report voice pass (36 [BEN:] blocks), A1 figure look, PUBLIC release = flip `measure-before-you-hunt` visibility, A/B/C filings (corrected bodies in `issue_drafts/filing/`), community posts, the submission form.

## Infrastructure facts that cost money to learn (do not relearn)

- **Pods provision from `github.com/flummoxjr/villa-pin-37e300d3`** (27 MB snapshot; vesuvius + 2 path-deps, volume-cartographer excised — `vesuvius[models]` needs Ceres/CMake and cannot build on stock pods; candidate Track B bug report, Ben-gated).
- **Never `pkill -f` a pattern that appears in your own command line** — it killed 5 pods ($~1 total) before a 3-probe bisect found it. Boot wrapper now uses a PID file.
- Launch pattern that works: gist raw URL → dockerStartCmd wrapper (bootstat server first, PID-file kill) → foreground GraphQL `uptimeInSeconds` poll (~2 min tells you if the container started; negative/oscillating = crash-loop or dead host, kill for $0.05) → `pod_guard.py` (deadline + `--no-status-min` + auto fetch-terminate) → filtered Monitor.
- Guard gap known: a live heartbeat with a never-advancing stage isn't caught (v5 burned $1.55 in a 2 h git fetch). Add stage-stall abort if fleets grow.
- Fragment TIFFs: **three layouts** (Frag1 MM front-IFD, per-layer SO 260/262; Frag3/Frag6 II end-IFD, data@8). Parse per file. Draft report: `STRIPOFFSETS_REPORT_DRAFT.md` (Ben-gated).

## Repos

- `measure-before-you-hunt` (PRIVATE) — full research repo backup, push after every commit batch.
- `facing-pairs-harness` (public) — delivered instrument.
- `villa-pin-37e300d3` (public) — provisioning snapshot.
- Local `trackD/` is the working truth; latest commit at time of writing: a0f52d9.

## Standing discipline (the reason this month worked)

Prereg before data, committed to git; kill conditions honored (five held so far, two by margins under 0.002); anchors reproduced before any new number; every positive beats a matched null; blank = mask∧¬label; report both framings; the kill carries its measurement out with it.
