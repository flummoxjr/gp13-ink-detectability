# Pre-registration — Cross-acquisition ink confirmation (research D)

**Written 2026-08-24, before any corpus-wide number exists.** Committed to git before the scoring
run. Only the w035 pilot numbers below were measured earlier; every corpus number is future.

## Premise, verified this session

112 published segments across four samples carry TWO ink-detection maps computed from two
**physically independent acquisitions** of the same material — verified from volume physics, not
assumed:

| sample | volume A | volume B | independent? |
|---|---|---|---|
| PHerc0139 (37 segs) | 20260102150214 · 2.399 µm · 78 keV | 20260413113053 · 1.129 µm · 59 keV | yes — date, pitch, energy all differ |
| PHerc1667 (19) | 20251217075048 · 2.399 µm · 78 keV | 20260323082859 · 1.129 µm · 59 keV | yes |
| PHerc0814 (19) | 20260309142202 · 2.399 µm · 78 keV | 20260521123630 · 1.129 µm · 59 keV | yes |
| PHercParis4 (37) | 20260411134726 · 2.400 µm · 78 keV | 20260608103018 · 1.129 µm · 78 keV | yes, with caveat: same energy — energy-dependent effects could correlate; reconstruction noise, rings, and model error still cannot |

The known-contaminated case (PHerc0500P2: one volume, two recipes) is **not** in this set.

## The physics claim

Real ink is a physical deposit and must appear in every acquisition of the same material.
Ring/streak artifacts, Paganin haze, detector fixed-pattern, reconstruction noise, and model
hallucination are acquisition-specific and are re-drawn independently by each scan. Agreement
across acquisitions is therefore evidence of a deposit; disagreement maps the error.

## Protocol (fixed)

Per segment: similarity resample of map B onto map A's raster, refined by phase correlation
(w035 pilot: scale 0.9402, residual 2 ds8 px, peak SNR 25.4). On the joint-valid mask compute
Pearson r, and confirmation lift `L(q) = P(both > q) / [P(A > q) · P(B > q)]` at q ∈ {0.90, 0.95,
0.98, 0.99}. Fusion score = mean of the per-map z-scores.

**Three matched nulls per segment**: 180° rotation of B; a ≥700 px roll of B; within-segment block
permutation preserving each map's marginal and low-frequency envelope.

**Calibration** on the segments carrying human ink labels (the ink_9um label set; w035 among them —
on-training-data for the *model* but the labels are human and the cross-acquisition comparison does
not involve the model's training): DeLong test of AUC(fused) vs AUC(best single), and FPR at
matched 50% recall.

## Pilot anchors (measured earlier, to reproduce)

w035: r = 0.6608 vs nulls 0.0767 / 0.0515; L(0.99) = 34.0; AUC 0.8768 / 0.8771 single,
0.9015 fused; FPR at 50% recall 0.0363 / 0.0321 → 0.0263.

## Decision rule

- **KILLED** if, across the labelled segments, AUC(fused) does not exceed AUC(best single) with
  DeLong p < 0.05 AND median gain ≥ 0.01 — or if median L(0.99) across the 112 is below 5.
- **CONFIRMATION LAYER SHIPS** otherwise: per-segment confirmed-ink maps (fused score over the
  1%-FPR threshold calibrated on labelled segments), the per-segment disagreement map as the first
  direct estimate of published-map hallucination, and the ranked list of segments whose
  confirmed-only regions merit human inspection.
- Any candidate unread text claim requires the confirmed-only region to survive the rotation null
  AND the block-permutation null, and goes to human eyes before any words like "letters" are used.
- PHercParis4 results are reported in a separate stratum because of the shared-energy caveat.

## What this is not

It is downstream of the community's models (it consumes their published maps), so it is not an
independent detector. It is a **validity and confirmation layer** the field does not have. A
negative (fusion gain fails to replicate beyond w035) is publishable as-is.
