# Pre-registration — Ink Mass Budget v2 (converged null)

**Written 2026-08-24, before any v2 number exists.** Supersedes nothing: the v1 verdict
(`PREREG_G1.md`, NO_BOUND_QUOTABLE) stands as committed and is reported alongside v2. This file
exists because v1's own null estimator was measured to be under-powered — a 40-draw sd carries a
19.4% standard error, and Frag1's gate-failing 2755.3 was a +1.6 SE fluctuation on a converged
value of 2091.9 — so the honest continuation is a re-run under a converged, pre-registered null,
not a reinterpretation of v1.

## What changes from v1 (and only this)

| parameter | v1 | v2 |
|---|---|---|
| null | 40 rigid translations | **2×8-tile block bootstrap, 4000 replicates, matched-n** (measured seed-cv 0.43–0.69%; block-size sensitivity flat 1134–1256 across 1×4..4×12) |
| everything else | — | **identical**: T=32 non-overlapping, origin (0,0), min 64 px/class, positive = label∧mask, negative = mask∧¬label, weight min(n_ink, n_blank), S = Σ_z (I(z) − I_air), per-layer-parsed StripOffsets |

The rigid-translation null is still computed and reported as a diagnostic, at 400 draws.

## Gates

- **G0** unchanged: IR paired AUC ≥ 0.90 on identical tiles; failing fragments dropped, not
  explained. (v1 admitted Frag1, Frag2, Frag6; Frag3/Frag4 failed G0; Frag5 structurally too small.)
- **G1-v2**: at ≥ 1,700 admitted T=32 tiles, the **block-bootstrap** null sd ≤ 2,300 DN·vox.
  Measured-before-this-prereg values: Frag1 1227.6 (whole plate). Frag2 and Frag6 to be measured;
  their v1 rigid sds (934.0, 1839.1) already sat under the gate.
- **Pooling requires ≥ 3 admitted fragments**, unchanged.

## Decision rule (numbers fixed now)

- **DETECTION** iff z > 4.0 on an admitted fragment under the block-bootstrap null, Bonferroni
  across the admitted set. Known-before-this-prereg: Frag1 whole-plate z = +1.56, so detection is
  **not** the expected outcome and the threshold does not move.
- **CEILING** otherwise, and this is the expected outcome: pooled 2-sigma ceiling
  = |obs − null_mean| + 2·null_sd per fragment, pooled inverse-variance across admitted fragments,
  expressed **primarily in papyrus-voxel-equivalents** (BULK = 6,434 DN per 3.24 µm voxel on Frag1;
  per-plate BULK measured per fragment), with mg/cm² secondary and the Paganin linearity systematic
  stated. This would be the first bounded upper limit on carbon-ink areal contrast in these scans.
- The v1 NO_BOUND verdict is reported next to the v2 ceiling with the estimator-artifact
  explanation. v2 does not erase v1; it explains it.

## Honest framing constraints

1. The framing choice measured in the null-scaling work — block bootstrap answers a
   matched-configuration question and sits at ~0.65× the converged rigid ensemble because the rigid
   null also carries between-region nonstationarity — is stated in the writeup, with the ceiling
   quoted under BOTH framings. Under either, Frag1's sd < 2,300 and z ≪ 4.
2. No number from this file's "known-before" rows is re-derived post hoc; they are quoted as the
   motivation and the v2 run must reproduce them within stated tolerances (sd within 5%, z within
   0.2) or the run is VOID.
3. Compute: CPU-only, local, on data already on disk. No pod.
