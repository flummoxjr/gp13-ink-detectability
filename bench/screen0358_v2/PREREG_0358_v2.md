# PREREG — PHerc0358 first-surfaces ink screen, v2 (2026-09-01)

Supersedes PREREG_0358.md (2026-08-25, sha256 1a870223…; the v1 pod run died at patch 1 and the
document itself was lost with that session's scratchpad — its decision rules survive verbatim in
the v1 script's embedded `prereg.json`, reproduced below unchanged except where marked **v2**).

## Experiment (unchanged)

First-ever ink screen of PHerc0358's first-ever surfaces: the 8 gate-PASS patches grown
2026-08-25 (`hunt/pherc0358_first_surfaces/paths_0358/`, commit 0d902ef; alignment gate
`alignment_gate.json`, local angles 3.6–18.6°). Per patch: render a 21-slice surface volume from the
masked 9.362 µm CT (`20250821151737-9.362um-1.2m-113keV-masked.zarr`), infer ink_9um seed42
step-075000 forward and reverse, save full-res maps, strided ds4 npys (survey-verbatim battery
inputs), ds8 previews, survey stats, full-res fwd/rev Pearson r. **Measurement only** — no battery
statistic, gate, or verdict is computed on the pod.

## Honesty frame (**v2**)

v1 cited "ink_9um reads at chance (0.5382 fwd / 0.5055 rev) on the clean foreign-scroll anchor
500p2a win1". That number is **void**: the 500p2a surface volume is 2.215 µm/px, not 4.32, so the
model was fed 4.80 µm data presented as 9.36 µm (`bench/P2A_PITCH_RESOLUTION.md`). The corrected
anchor is measured by `bench/p2a_v3` with in-domain positive controls; its value and pre-stated
reading are copied into the v2 script's `prereg.json` **before launch** and logged as status line 3.

Expected outcome: NULL, with sensitivity bounded by that corrected anchor. A blank screen is not
evidence these patches carry no text. Value if null: first-ever look at any recoverable surface of this
scroll; maps + correctly-oriented meshes as community artifacts; a matched null for the local battery.

## Local battery and flag rule (unchanged)

PROTOCOL_V2 (`analyze_survey_corpus_v2.py`), 5 gates, run locally after fetch on the pod's strided
ds4 npys (`map[::4, ::4]` uint8, 37.448 µm/px — the exact `survey_segments.py` decimation the
71-segment corpus and the `w035_CONTROL_strided` control used) of the FORWARD maps; reverse maps
descriptive only; gate 5 from the pod's full-res fwd/rev Pearson r. `w035_CONTROL_strided` must
reproduce 5/5 before any patch is scored.

A patch is "flagged: region worth human inspection" iff its forward map passes ≥ 4 of 5 computable
gates (non-computable = not passed); identical to Track F. Never letter-language.

Multiplicity: 8 flag-bearing tests; gate-significance raw p ≤ 0.05 per v2 with Holm-8 beside it. Base
rates (corpus_analysis_v2.json + null_scaling/ns4): the ≥ 4/5 rule fired 0/71 on the GP corpus
(gate_fwd_rev failed all 71, min |r| = 0.2215); gate_significance 4/71 raw (3.55 expected by chance),
2/71 under corrected nulls. Foreign-scroll fwd/rev bleed is unmeasured: if gate_fwd_rev passes here a
flag needs ≥ 3 of the other 4 gates, an 11/71 (15.5%) base rate, so expect 0 to ~1.2 chance flags
among 8; up to 2 flags are fully consistent with chance.

Escalation: any flag → ≥ 1000-permutation gate-1 rerun + human eyes on map and CT before any public
language; at most "region worth human inspection"; report which gates carried the flag.

## Infra vs science (**v2** canvas rule)

Data-gate mismatch or render/infer hard failure kills the run (infra). Canvas ≠ 3040×3040 or forward
nonzero_frac < 0.05 = INFRA_INCOMPLETE for that patch; the run continues; ALL DONE requires 8/8
healthy; a null verdict may only come from a healthy map. **v2:** the renderer now allocates the
canonical canvas round(h/scale) = 3040 (villa's `Tifxyz.shape` truncates 152/0.05000000074505806 to
3039 — `issue_drafts/filing/tifxyz_fullres_shape_truncation.md`); the 3040 gate is unchanged.

## Frozen inputs (unchanged)

Volume level 0 [14744, 7783, 7783] uint8, chunks 128³, raw, separator "/". Model
scrollprize/ink_9um hybrid_3d2d-seed42/step-075000.pth, 138,360,039 bytes, mode flat, crop
(17,128,128), robust_mad. Meshes: 8 × 152×152 tifxyz, scale 0.05, per-file sha256 manifest embedded in
the script; tarball sha256 a18408e8…; raw-URL fallback from `measure-before-you-hunt` master. Render:
21 slices, step 1.0, `render_tifxyz_sv.py` (r = 0.813 validated).

Patches: auto_grown_20260825155611879, …155613379, …155615680, …155619178, …155619482,
…155621418, …155624780, …155625980.
