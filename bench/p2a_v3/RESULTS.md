# p2a_v3 — the corrected 500p2a anchor, with positive controls (2026-09-02)

Pod `hq7soy0hkbl6xi` (RTX 5090, community, $0.69/h), 02:35 → 02:54 UTC, **≈ $0.22**. Pre-registration:
script header + `out/prereg.json` (sha256 49097685225a), locked before provisioning. Artifacts:
`experiments/p2a_v3/` (guard log, results.json, bundle), unpacked to `trackD/out/p2a_v3/` (results,
previews, all prediction maps). Every data gate passed (raster and window label counts exact; window
volume stats within ±0.5 DN of the Aug-25 record; embedded w035 labels sha256-exact; checkpoint exact).

## The harness certified itself first (CTL, PHerc0139 w035 crop, in-domain)

| arm | forward | reverse | pre-registered rule | outcome |
|---|---|---|---|---|
| native (as released) | **0.9991** | 0.5118 | fwd ≥ 0.95; rev ≤ 0.80 | both gates passed; on record 0.9991 / 0.5123 — reproduced to 4 decimals |
| scale fault ×1.9504 (4.80 µm presented as 9.36; exactly the Aug-25 error) | **0.7489** | 0.5381 | < 0.75 → FAULT_REPRODUCED | **FAULT_REPRODUCED** (by 0.001 — read it as "the fault costs ~0.25 AUC on a 0.999 read", not as a cliff) |
| coarse fault ×0.5 (18.7 µm presented as 9.36) | 0.5227 | 0.4844 | reported | a 2× too-coarse input takes the model to chance |

The level-fault effect sizes are now measured on the control the community calibrates against:
**0.999 → 0.749 (×1.95 too fine), 0.999 → 0.523 (×2 too coarse).**

## The corrected anchor (500p2a, 2.215 µm → 9.36 µm)

| window | iso fwd | iso rev | fit17 fwd | fit17 rev |
|---|---|---|---|---|
| **win1** (the anchor; 4.39 M ink px / 8.58 M blank) | **0.5211** | 0.5106 | 0.5301 | 0.5099 |
| win2 | 0.4339 | 0.4971 | 0.4481 | 0.4897 |
| win3 | 0.4914 | 0.4956 | 0.4761 | 0.4756 |

**Corrected anchor A₃ = 0.5211** (win1 iso forward). Pre-stated reading (< 0.65): *transfer failure
confirmed at the correct pitch.* The curve gate (0.85) was not met, so the detectability rungs did not
run. The depth mode does not matter (iso vs fit17 within 0.01). No fwd/rev asymmetry (0.01, vs 0.49
in-domain) — Track D gate 5 corroborated again.

**How this relates to the Aug-25 number.** The Aug-25 0.5382 / 0.5055 was measured through the ×1.95
scale fault and is void as a measurement. Its *conclusion* survives: with the fault removed the model
still reads at chance, and the CTL arm shows the fault alone would have taken a genuinely readable
surface only to ~0.75, not to ~0.52. Both facts together say the 500p2a result is a real transfer
failure, not an artefact.

**What it says about the transfer problem.** 500p2a is a 2.215 µm scan pooled to 9.36 µm — the same
representation family as 24 of ink_9um's 29 training inputs (2.399 µm pooled to 9.6 µm). A
pooled-class foreign surface reads at chance while the in-domain control reads at 0.999 through the
identical harness. The gap is scroll-specific (texture / ink prior), not acquisition domain — which
weakens Bet A's premise before it runs (`PREREG_BET_A_DRAFT.md` §1b) and is the premise of Bet C.

## Hallucination audit (EXP B, 40 rigid translations per window, annulus [3.75 mm, 0.60·size])

All three windows: **SHAPE_CONFOUNDED** — the real AUC sits at chance while translated label shapes
reach null AUCs up to 0.75 (win1), 0.83 (win2), 0.92 (win3), with ≥ 20% of draws above 0.60. Read
together with the chance-level real AUC this means the model's output on 500p2a has *spatial structure
at letter-row scale that is not ink* (win3 forward: null median 0.38, max 0.92 — strongly periodic).
The model responds to the fragment's texture, not its text. This is the failure mode the audit was
built to detect; it is a property of the model on this scroll, not of the labels.

## Corrections applied

SEPTEMBER_PLAN §0 / §1.2 / §5.3, RESEARCH_PLAN §0, the Road to First Letters artifact (requirement 1
+ a dated callout), `PREREG_BET_A_DRAFT.md` (A₃ and §1b), the 0358 screen v2 prereg text,
`bench/manifest.json`. `out/transfer_ladder/` stays as the record of the void measurement.

## Not yet done

The five native-0139 held-out surfaces and Frag2–6 are in the manifest but unfingerprinted; the
benchmark release (`bench/`) should ship with this run's `results.json` as the schema reference.
