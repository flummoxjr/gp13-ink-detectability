# Addendum — four pre-registered verdicts after the report closed (2026-08-24/25)

Each study below had its decision rule committed to git **before its data existed**; each verdict is
the rule applied mechanically, and two of the kills landed within 0.002 of their bars without the
bars moving. Artifacts and pre-registrations ship in this repository.

## A1. An upper limit on carbon-ink areal contrast (CEILING)

Depth-integrated column excess at infrared-verified ink on the detached fragments, under a
converged block-bootstrap null (`PREREG_G1_V2.md`; v1 at `PREREG_G1.md` returned NO_BOUND because
its own 40-draw null estimator was under-powered — measured, not assumed: the sd carried a 19.4%
standard error). Three fragments admitted (G0 IR AUC 0.9433/0.9461/0.9616), no detection
(max z = +1.81 vs a fixed threshold of 4.0), so the pre-registered outcome is the bound:

> **Pooled 2σ ceiling: 0.578 papyrus-voxel-equivalents** (block null) / 0.691 (converged rigid) —
> an ink stroke adds less depth-integrated opacity than ~1.9–2.2 µm of bulk papyrus. Secondary
> mass figure 0.112–0.134 mg/cm², valid only under a stated DN-linearity assumption.

To our knowledge the first bounded upper limit of its kind in these scans. `out/g1v2/`.

## A2. Translation nulls do not average down (methods)

The 40-translation rigid-shift null used across this field is a single global shift of an
autocorrelated field: effective sample size **566 of a nominal 2,560 tiles** (a 4.5× overcount),
heavy-tailed (one draw carried 55% of the null variance), and its noise constant sd·√n is **not
transferable** — 2.4× between plates, 2.5× across regions of one plate. A block bootstrap at the
measured correlation length is stable where the rigid null is not (seed-cv 0.4–0.7% vs 6.5–23.6%).
Applied back to this report's own flagship negative: **0 of 71 survives** — the four null-free gates
alone exclude every segment, and under corrected nulls the raw-significance passers fall from 4 to
2 against 3.55 expected by chance. The negative is robust to its own null's pathology.
`out/null_scaling/`.

## A3. Cross-acquisition confirmation (KILLED, and the kill is informative)

112 segments carry ink maps from two physically independent acquisitions (verified from volume
physics; the known same-volume-two-recipes case excluded). Agreement is real and large —
confirmation lift **L(0.99) median 25×** against matched nulls of ~1 — and is strongest exactly on
the segments where humans have read text. But the pre-registered fusion claim dies:
median labelled-segment AUC gain **+0.0080 < 0.01** (`PREREG_XACQ.md`); fusion helps only when both
acquisitions are of matched quality. The confirmation layer does not ship; the negative and the
per-segment agreement/disagreement diagnostics do. `out/xacq/`.

## A4. The flat ink model does not transfer to fragments (KILLED_BASELINE)

First score of `ink_9um` on a detached fragment with photographic ground truth, outside its
training manifest: **forward AUC 0.6925, reverse 0.4477** against 0.9991 in-domain, failing a
pre-registered 0.85 baseline gate that aborted the planned degradation-curve experiment before it
could measure rungs from an uninterpretable anchor. The forward/reverse asymmetry persists even
where the model barely works. The curve re-anchors on a scroll-derived clean surface in future
work. `out/curve_audit/` (prereg locked in the run itself, sha `ab3e8d9d`).

---

# Addendum II — verdicts pre-registered and decided 2026-09-01 → 09-04

Same discipline as above: rule in git before data, verdict mechanical, artifacts shipped either way.

## A5. The foreign-scroll anchor, re-measured with positive controls (KILLED_BASELINE, corrected)

The Aug-25 "0.538 on a clean foreign scroll" was measured through an input fault: the 500p2a surface
volume is **2.215 µm**, not the 4.32 µm its metadata field names (`bench/P2A_PITCH_RESOLUTION.md`; the
mesh bounding box fits only the 2.215 µm volume), so the model was fed data ×1.95 too fine. Re-run at the
correct pitch with the harness certified in the same pod (`bench/p2a_v3/`, prereg sha 49097685225a): the
in-domain control reads **0.9991 / 0.5118** (forward / reverse; the on-record numbers to four decimals),
the same fault deliberately re-applied to the control costs **0.999 → 0.749**, and a ×2 too-coarse input
gives 0.523. The corrected anchor is **A₃ = 0.5211** forward / 0.5106 reverse (iso; fit17 0.5301 / 0.5099;
win2 0.434, win3 0.491), inside its own hallucination null (median 0.504, max 0.750). Conclusion unchanged,
now fault-free: the released model reads nothing on a scroll it was not trained on. The level-fault effect
sizes (−0.25 / −0.48 AUC) are the first measured on the community's calibration control. `out/p2a_v3/`.

## A6. Input-noise-matched training does not close the 9 µm transfer gap (KILLED — premise inverted)

Bet A (`PREREG_BET_A.md`, final v1 committed before launch): leave-PHerc0139-out on the published
`ink_9um` recipe, two seeds per arm, 78,125 steps, native-5 held-out evaluation with the A5 controls in
every pod. Arm 0 (recipe unchanged) reproduced khj1222's LOSO baseline within seed spread — best native-5
F1 0.627 / 0.631 (floor 0.540, anchor 0.653), forward AUC **0.746 / 0.757** — and passed its anchor gate.
Arm 1 (every training crop degraded to a k2b-index target, calibrated to the estimator) ended at a two-seed
mean AUC of **0.759 (+0.008)**; arm 2 (per-volume in-plane PSD whitening at train and test) at
**0.705 (−0.047)**; the frozen gate was +0.05. The decisive number came before training: measured with the
very estimator arm 1 degrades with, the pooled 2.4 → 9.6 µm training volumes have structural SNR ≈ 6 where
the native 9.36 µm PHerc0139 crops have ≈ 30 — the pooled inputs are *noisier* and narrower-band than the
native target, the opposite of the premise. Input noise, as defined and measured here, is not the gap.
Reverse-order AUC stayed at chance (0.50–0.54) on every checkpoint of every arm. Ships: the degradation /
whitening code (fork branch `betA-arms` @ 45d5e03, absent keys byte-identical), the LOSO pod harness with
the measure stage, four results files, `bench/betA_arm0/RESULTS.md`, `bench/RESULTS_LADDER.md`. ≈ $40.

## A7. Track F: the w059 lead does not clear the second scanner (NOT CONFIRMED)

The unread, unlabelled PHerc0139 w059 region flagged on the 2.4 µm arm (4/4 gates, z +5.66) was re-run
on the second acquisition at full coverage (1.129 µm volume, both depth orders, the canonical 2 µm model
whose published w035 map we first reproduced to r = 0.9999997 — the C0 contract): fwd/rev r 0.0039, **no
ruling periodicity** (z −1.10, p 0.94), while the modality control on w035 arm B detects the signal only
marginally (z +2.07; the volume covers 39 % of the sheet). Under the pre-registered protocol the lead is
closed for escalation; no human look, no letter language. `bench/w059_c0/RESULTS.md`.

## A3, re-audited (2026-09-03)

williamshermer-pixel's 14.2× (density-preserving null) against our 25×: the 25× lift is invariant to the
mask convention (sheet footprint rolled with the calls: 25.1×), to resolution (ds8–ds64) and to registration
residual (0–32 native px); 14.2× is not reproducible under any convention we can construct. A3's numbers
stand as written. `out/xacq/REAUDIT.md`.
