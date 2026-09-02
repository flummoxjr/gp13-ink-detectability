# G1 v2 — Ink Mass Budget under the converged null (PREREG_G1_V2.md, git a2e0001)

**Outcome: CEILING.** All three admitted fragments pass G1-v2; pooling is licensed (3 >= 3);
no fragment approaches detection. This is the first bounded upper limit on carbon-ink areal
contrast in these scans.

## Headline

Pooled 2-sigma ceiling on the depth-integrated column excess an ink stroke adds, quoted under
BOTH framings per the prereg's honest-framing constraint:

| framing | pooled ceiling | in papyrus thickness | mg/cm2 (secondary) |
|---|---|---|---|
| **block bootstrap** (matched configuration, the prereg null) | **0.578 papyrus-voxel-equivalents** | 1.87 um of bulk papyrus | 0.112 |
| **converged rigid ensemble** (400 draws, diagnostic) | **0.691 papyrus-voxel-equivalents** | 2.24 um | 0.134 |

The mg/cm2 figures assume DN linear in mu at papyrus density 0.6 g/cm3; these are Paganin
phase-contrast reconstructions rescaled by an unpublished window, so the dimensionless
voxel-equivalent number is the claim and the mass is the interpretation (v1 units discipline,
unchanged).

## VOID gate — anchors reproduced

| anchor | committed | measured | tolerance | dev |
|---|---|---|---|---|
| Frag1 block-bootstrap sd (2x8, 4000 reps, matched-n) | 1227.6 | 1227.6 | 5% | 0.0% |
| Frag1 z under block null | +1.56 | +1.559 | 0.2 | 0.000 |
| Frag1 converged rigid sd (400 draws) | 2091.9 | 2091.9 | 5% | 0.001% |

The Frag1 observed tile field recomputed from the raw volume is bit-identical to the
null-scaling work's `ns2_obsfield.npz`; `block_boot_sd` is a mechanically verified verbatim
copy of `ns3_corrected.py`; rigid draws 0..39 equal v1's committed 40 null values on all
three fragments (same generator, seed 20260820).

## Measured, per fragment (identical tiles, code, and I_air as v1)

| | n tiles | obs (DN vox) | G0 IR AUC | block sd (seed-cv) | z_block | rigid-400 sd / mean | BULK (DN/vox) |
|---|---|---|---|---|---|---|---|
| Frag1 (whole plate) | 2,560 | +1,914.07 | 0.9433 | 1,227.6 (0.64%) | +1.559 | 2,091.9 / -246.0 | 6,434 (committed) |
| Frag2 (band 7392..12415) | 4,014 | +1,678.14 | 0.9461 | 929.2 (0.58%) | +1.806 | 892.6 / +71.4 | 3,311.3 |
| Frag6 (band 640..8351) | 1,857 | +1,113.53 | 0.9616 | 1,229.2 (0.53%) | +0.906 | 1,596.6 / +139.6 | 5,262.1 |

Every v1 reproduction check passes exactly: obs, n_tiles, G0 AUC, I_air (23,263 / 24,674 /
27,115) all match the committed v1 values bit-for-bit on identical tiles.

**G1-v2 gate** (n >= 1,700 AND block sd <= 2,300): PASS / PASS / PASS -> admitted = {Frag1,
Frag2, Frag6}. The v1 gate-failing Frag1 sd 2,755.3 is explained, not erased: a 40-draw
estimate with 19.4% self-noise of a converged 2,091.9 (rigid) / 1,227.6 (block); both
converged values sit under 2,300.

**Detection** (z > 4.0 under the block null, Bonferroni across 3 admitted, family alpha
9.5e-5): max z = +1.806 (Frag2). Not close. CEILING branch taken, as the prereg expected.

**Block-size sensitivity** (1x4 .. 4x12 tiles): Frag1 1,134..1,256; Frag2 865..973; Frag6
1,148..1,293 — flat on all three, as pre-measured on Frag1.

## Per-fragment 2-sigma ceilings

|  | block: DN vox / vox-equiv | rigid: DN vox / vox-equiv | block/rigid sd ratio |
|---|---|---|---|
| Frag1 | 4,369.3 / 0.679 | 6,343.9 / 0.986 | 0.587 |
| Frag2 | 3,536.5 / 1.068 | 3,391.9 / 1.024 | 1.041 |
| Frag6 | 3,571.9 / 0.679 | 4,167.1 / 0.792 | 0.770 |

Pooling: fixed-effect inverse variance on delta_i = (obs_i - null_mean_i)/BULK_i with
sigma_i = null_sd_i/BULK_i (block framing zero-centered by construction — the anchor's own
z convention; rigid framing uses its measured ensemble mean). Pooled delta +0.316 (block) /
+0.348 (rigid) vox-equiv, pooled sd 0.131 / 0.171 -> ceiling = |delta| + 2 sd = **0.578 /
0.691**. Alternative reading (inverse-variance-weighted mean of the per-fragment ceilings):
0.763 / 0.940 — looser; reported for robustness.

## Honest limits

1. **Framing.** The block bootstrap answers a matched-configuration question; the rigid
   ensemble also carries between-region nonstationarity. The measured block/rigid sd ratio is
   0.587 on Frag1 but 1.041 on Frag2 and 0.770 on Frag6 — the "~0.65x" from the null-scaling
   work is Frag1-specific, and on Frag2 the two nulls agree. Both framings are quoted
   everywhere; under either, every sd < 2,300 and every z << 4.
2. **Pooled excess is positive.** All three fragments sit high (+1.6, +1.8, +0.9 sigma);
   the descriptive pooled z is +2.42 (block) / +2.03 (rigid). Under the committed rule this
   is not a detection (per-fragment z > 4.0 Bonferroni; nothing approaches it) and no
   post-hoc pooled test is licensed. It is reported so nobody has to discover it.
3. **BULK calibration systematic.** Per-plate BULK uses the rule that replicates the pilot
   anchor exactly (in-mask mean at the mean-profile peak layer minus I_air: 6,434.3 vs
   committed 6,434). Frag1 whole-plate under the same rule gives 5,947 (-7.6% vs the pilot
   band); Frag2's value ranges 2,785..3,311 across rule variants. The voxel-equivalent
   conversion therefore carries a ~10-20% per-plate calibration systematic on top of the
   statistical bound; the DN-vox ceilings are free of it.
4. **mg/cm2 linearity systematic.** DN-linear-in-mu is assumed, unverified (Paganin filter +
   unpublished uint8 window). Never quote the mass figure without this sentence.
5. **Rigid-400 provenance.** Frag2/Frag6 ensembles computed fresh this run (first 40 verified
   equal to v1's committed null values); Frag1's from the null-scaling cache, itself
   regenerated from scratch and verified bit-identical in that ship item.
6. **Scope.** The ceiling bounds the weighted-mean paired column excess at IR-verified ink
   pixels under the committed estimator (T=32, min 64 px/class, weight min(n_ink, n_blank))
   on three fragments from three plates; Frag3/Frag4 remain dropped by G0, Frag5 by
   structural capacity, as committed in v1.

## v1 verdict, reported alongside (prereg requirement)

PREREG_G1 v1 (git 502ae65) returned **NO_BOUND_QUOTABLE**: Frag1's 40-draw rigid sd 2,755.3
failed the 2,300 gate, leaving 2 admitted fragments against a pooling floor of 3. That verdict
stands as committed. v2 changed exactly one thing — the null estimator — because the 40-draw
sd carries a 19.4% standard error and the converged value (2,091.9 rigid; 1,227.6 block) sits
under the gate: the v1 FAIL was a +1.6 SE estimator fluctuation, not a property of the plate.
v2 does not erase v1; it explains it, and both ship together.

## Files

- `G1V2_RESULTS.json` — the rule walk (VOID gate, reproduction, gates, detection, ceilings, pooling)
- `g1v2_f1.json`, `g1v2_f2.json`, `g1v2_f6.json` — per-fragment measurements
- `g1v2_bulk.json`, `g1v2_bulk_step1.json` — per-plate BULK and the pilot-rule replication
- `f1/f2/f6_obsfield_v2.npz`, `f2/f6_rigid400.npz`, `f2/f6_P.npy`, `f2/f6_layers.json` — fields, ensembles, caches
- `g1v2_lib.py` (verbatim-verified bootstrap + committed v1 estimator import), `g1v2_build.py`, `g1v2_meas.py`, `g1v2_f1.py`, `g1v2_bulk.py`, `g1v2_verdict.py`
