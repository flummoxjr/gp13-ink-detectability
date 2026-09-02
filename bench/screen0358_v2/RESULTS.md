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

_Running (`experiments/screen0358c/battery.log`); results are written to
`trackD/out/screen0358_v2/battery_0358.json` and summarised here when done._

| unit | gates passed | empirical p | note |
|---|---|---|---|
| w035_CONTROL_strided | pending | | must reproduce 5/5 or the battery is invalid |
| 8 patches | pending | | flag rule ≥ 4 of 5 computable gates → "region worth human inspection", never letter language |

## Reading (pre-registered)

Expected outcome was NULL with sensitivity bounded by the corrected foreign-scroll anchor. The
measurement is consistent with that: no tripwire, no depth asymmetry, hot pixels at speckle scale.
**A blank screen here is not evidence that these patches carry no text** — ink_9um reads 500p2a at
0.52 with the pitch fault removed, so its sensitivity on a foreign scroll is near zero. The value of the
run is what the prereg said it would be: the first-ever look at any recoverable surface of PHerc0358,
eight correctly-oriented meshes and their maps as community artifacts, and a matched null for the
battery. A model that transfers (Bet C) reruns on these same meshes for ≈ $0.35.
