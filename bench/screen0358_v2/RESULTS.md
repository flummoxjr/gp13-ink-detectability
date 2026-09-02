# PHerc0358 first-surfaces ink screen v2 — results (2026-09-02)

Pod `pou3y6h4s7gp7i` (RTX 5090 community, $0.69/h), 02:55 → 03:27 UTC (≈ 32 min, ≈ $0.37).
Pre-registration: `PREREG_0358_v2.md` (sha256 dcfa3de67cf0) + the script's `prereg.json`, locked as
status line 3 before provisioning; honesty frame re-anchored on the corrected 500p2a number
(0.5211, `bench/p2a_v3/RESULTS.md`). Records: `experiments/screen0358c/` (guard + mirror),
`trackD/out/screen0358_v2/` (pod stats, previews, battery inputs and output).

## What ran

The 8 gate-PASS patches grown 2026-08-25 (`hunt/pherc0358_first_surfaces/paths_0358/`, 32 files
sha256-verified on the pod), rendered as 21-slice surface volumes from the masked 9.362 µm CT
(`20250821151737`), inferred with ink_9um seed42 step-075000 forward and reverse, survey-verbatim
stats, strided ds4 maps for the local battery. Every data gate passed (volume identity, mesh bbox,
checkpoint). **The canvas fix worked: every map is 3040×3040** (v1 died at 3039 on patch 1).
Renders took 160–240 s per patch on this host (v1's host: 1,619 s).

## Pod-side measurement (no verdicts are computed on the pod)

| patch (…suffix) | fwd p50 | fwd p99 | frac > blank p99 (195) | hot comps | tripwires | fwd/rev r |
|---|---|---|---|---|---|---|
| 611879 | 95 | 193 | 0.0057 | 1896 | 0 | 0.482 |
| 613379 | 88 | 186 | 0.0021 | 661 | 0 | 0.552 |
| 615680 | 95 | 191 | 0.0031 | 1506 | 0 | 0.536 |
| 619178 | 106 | 195 | 0.0100 | 3680 | 0 | 0.586 |
| 619482 | 98 | 189 | 0.0027 | 1114 | 0 | 0.474 |
| 621418 | 99 | 193 | 0.0062 | 1755 | 0 | 0.447 |
| 624780 | 95 | 189 | 0.0028 | 1206 | 0 | 0.468 |
| 625980 | 96 | 191 | 0.0030 | 1615 | 0 | 0.585 |

- **Tripwire: 0 of 8** (value > blank p99 ∧ area ≥ 10⁴ px ∧ width ≥ 30 px). The hot pixels above
  the blank threshold are 0.2–1.0 % of each map and form only small components (hundreds to a few
  thousand, none reaching the area/width floor) — speckle, not strokes.
- **fwd/rev r = 0.45–0.59 on every patch.** The in-domain control reads 0.094 and the 71-segment GP
  corpus minimum was 0.22; gate 5 of the battery (|r| < 0.20) therefore fails on all 8 before any
  periodicity statistic is computed. This is the foreign-scroll signature: the model's output does
  not depend on depth order here, which is exactly what it does on 500p2a (0.01 asymmetry) and not
  what it does on a scroll it can read (0.49).
- Reverse-direction p99 186–193: the reverse maps are statistically indistinguishable from forward.

## Local five-gate battery (PROTOCOL_V2, 200 permutations, control first)

`analyze_survey_corpus_v2.py` unchanged, driven by `run_battery_0358.py` (10 units × 200 joint 64-tile permutations,
28 min on 8 cores); output `trackD/out/screen0358_v2/battery_0358.json`.

| unit | gates passed | empirical p | period (mm) | fwd/rev r | note |
|---|---|---|---|---|---|
| w035_CONTROL_strided | **5/5** | 0.00498 (0/200) | 4.678 | 0.094 | z +16.26 — reproduces the protocol record exactly (prom 123.5, 11 cycles, bin 4/24) |
| …619482 | 2/5 | 0.134 | 4.40 | 0.474 | cycles+band only |
| …611879 | 0/5 | 0.174 | 7.41 | 0.482 | none |
| …613379 | 0/5 | 0.507 | 7.20 | 0.552 | none |
| …625980 | 2/5 | 0.567 | 1.92 | 0.585 | cycles+band only |
| …621418 | 1/5 | 0.572 | 6.16 | 0.447 | autocorr only |
| …615680 | 2/5 | 0.706 | 4.37 | 0.536 | cycles+band only |
| …619178 | 2/5 | 0.751 | 5.13 | 0.586 | cycles+band only |
| …624780 | 0/5 | 0.771 | 7.64 | 0.468 | none |

**Verdict under PREREG_0358_v2: 0 of 8 flagged** (flag rule ≥ 4 of 5 computable gates; best patch 2/5). 0 of 8 at
p ≤ 0.05 against 0.4 expected under the null; corrected z from −0.75 to +1.09; constrained (band-restricted) search: 0
survivors; gate_fwd_rev 0/8 (r 0.45–0.59). The control passing 5/5 through the identical code path certifies the run.
Per gate: significance 0/8, cycles 4/8, autocorr 1/8, band-bin 4/8, fwd/rev 0/8 — the same profile the 71-segment GP
corpus showed (0/71).

## Reading (pre-registered)

Expected outcome was NULL with sensitivity bounded by the corrected foreign-scroll anchor. The
measurement and the battery are consistent with that: no tripwire, no depth asymmetry, hot pixels at speckle scale.
**A blank screen here is not evidence that these patches carry no text** — ink_9um reads 500p2a at
0.52 with the pitch fault removed, so its sensitivity on a foreign scroll is near zero. The value of the
run is what the prereg said it would be: the first-ever look at any recoverable surface of PHerc0358,
eight correctly-oriented meshes and their maps as community artifacts, and a matched null for the
battery. A model that transfers (Bet C) reruns on these same meshes for ≈ $0.35.
