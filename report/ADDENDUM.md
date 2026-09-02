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
