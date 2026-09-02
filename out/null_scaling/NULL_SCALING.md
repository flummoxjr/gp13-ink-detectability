# The null-scaling methods result
CPU-only, data already local. All numbers measured 2026-08-23/24 on this machine
(python = Vsuvious\.venv). Every stage re-run from scratch this session and
verified bit-identical to its first run (ns1, ns2 including regeneration of the
400-draw ensemble, ns3).

## 1. Premise — reproduced (ns1_premise.py -> ns1_premise.json)

From the on-disk Frag1 arrays and the committed estimator (m1_lib, T=32,
seed 20260820, 40 rigid rolls, |dx|>=128):

* obs +1914.07 DN*vox on 2560 tiles; null sd 2755.28, mean -39.08 — matches
  G1_RESULTS.json to the printed digits; all 40 draws reproduce to <0.02.
* Heavy tail: worst draw -12,833 against IQR [-992, +1334]; excess kurtosis
  +9.84; that single draw carries 55.3% of the null variance (sd without it
  1836.6, which would PASS the 2300 gate Frag1 failed at 2755.3).
* sd*sqrt(n) is not transferable: 1024-row slabs of Frag1 span 57,650..146,740
  (2.55x within one plate); plates span Frag2 59,173 / Frag6 79,252 /
  Frag1 139,407 (2.36x).

Premise REPRODUCED; item not killed.

## 2. Effective sample size of the single-global-shift null (ns2_ess.py)

400 draws of the identical generator (first 40 ARE the committed 40):

* Converged whole-plate null sd = 2091.9. The committed 40-draw estimate
  (2755.3) sits +1.6 SE high: with kurtosis 4.03 the SE of a 40-draw sd
  estimate is 406, i.e. 19.4% relative error. G1's sd<=2300 gate thresholds
  a number whose own 1-sigma band is +/-406.
* One global shift moves every tile coherently: per-tile null variance says
  independent tiles would give sd 1480.5; actual 2091.9 -> design effect
  DEFF = 2.0. Kish n (weights) median 1130.6 -> n_eff = 566 vs the nominal
  2560 obs-configuration tiles: the null behaves as if it had 22% of the
  nominal sample, a 4.5x overcount.
* Mismatch in n: null draws admit only 401..2533 tiles (median 1400) vs 2560
  observed, because rolling ink off the mask deletes tiles.
* Measured null-field correlation (tile units): x-lags 1..5 = .69 .40 .20 .12
  .03 (1/e at lag 3, <0.1 at lag 5); y is white beyond lag 1. Obs field
  similar (x: .45 .31 .19 .19 .18).

## 3. One corrected null: block bootstrap at the measured correlation length (ns3_corrected.py)

Cluster bootstrap of T = sum(w d)/sum(w) over 2x8-tile blocks (y-corr dead at
lag 1, x-corr <0.1 by lag 5-8), 4000 replicates, 4 anchor offsets, evaluated
on the OBSERVED tile configuration (matched n — which the rigid null never was).

* Whole plate: boot sd 1227.6 (10 seeds: cv 0.64%); obs +1914.1 -> z +1.56.
  Block-size sensitivity 1x4..4x12: 1134..1256 (flat above the corr length;
  1x1 gives 929, the independence floor).
* Stability across the same 7 slabs where the rigid null is unstable:
  - estimator repeatability: boot seed-cv 0.43..0.69% vs ten disjoint 40-draw
    rigid estimates cv 6.5..23.6% per slab (e.g. slab 0: rigid-40 ranges
    5763..13,680 across ten draws of the SAME generator).
  - transferability: boot sd*sqrt(n) spans 40,202..67,321 across slabs (1.67x)
    and the whole-plate value 62,112 lies inside that span; the rigid-shift
    constant spans 60,361..169,265 (2.80x at 400 draws — more draws do not fix
    it, it is real nonstationarity).
* Honest caveat: boot/converged-rigid-ref median 0.65 per slab. The two nulls
  answer different questions. The rigid ensemble adds between-region variance
  (each draw parks the label on a different piece of a nonstationary plate);
  the bootstrap is the sampling variability of the statistic at the observed
  configuration. V1 check: bootstrap applied to single null fields gives
  median 1452 vs ensemble 2092 — the shortfall IS the nonstationarity term.
* Consequences, both directions: (a) Frag1's G1 sd-gate FAIL (2755.3 > 2300)
  is an artifact of thresholding a 40-draw sd of a heavy-tailed null — the
  converged rigid value is 2091.9 and the matched-configuration corrected null
  is 1227.6, both under 2300; (b) detection is NOT rescued: corrected z +1.56,
  still nowhere near the committed z>4.

## 4. The payoff — the corpus screen's "0 of 71 pass" SURVIVES (ns4a/b/c/d)

The v2 screen (corpus_analysis_v2.json, 71 scored maps) gates on 5 conditions;
only gate_significance touches the null.

Structural: 0 of 71 pass the four NULL-FREE gates (recomputed independently
from the raw JSON, and again via ns4a using the screen's own module). The
fwd/rev-consistency gate alone passes 0 of 71 (min r 0.2215 vs <0.2 required);
each of the 4 significance passers also fails 1-3 of the periodicity gates.
Since a pass requires all gates, NO correction to the null — up to and
including deleting the significance gate — can raise the pass count above 0.

Measured anyway (the screen's own prep/score/joint_permute, obs asserted equal
to the stored value before any draw; fresh seeds; 50 perms; block 128 >=
the maps' measured L(0.1), median 26, max 200; 19/74 maps exceed the v2
tile of 64):

  map                    p @64,200perm  ->  p @128,50perm   z 64->128
  z_dbg_gen_00260            .00498          .0588 (2/50)   4.64 -> 2.24
  ..._00320 (20251105...)    .0199           .0392 (1/50)   3.96 -> 2.70
  z_dbg_gen_00357            .0348           .1961 (9/50)   2.56 -> 1.22
  z_dbg_gen_00215            .0398           .0196 (0/50)   2.07 -> 2.63
  w035_CONTROL (real ruling) .00498          .0196 (0/50)  16.26 -> 5.14

2 of 4 passers keep raw p<=0.05 at the corrected block size — against the
screen's own null expectation of 3.55 passers among 71. The strongest passer
(00260, whose x correlation length L(0.1)=93 exceeds the 64-px tile) loses
significance once blocks exceed the correlation length: the tile-64 null was
mildly ANTI-conservative there, i.e. the screen was if anything too generous
to candidates. The genuine-ruling control still clears the corrected null with
0/50 nulls at or above obs (its null mean doubles, 20.0 -> 41.7, because
128-px blocks approach its 125-px period — power shrinks but survives).

VERDICT: the flagship negative is NOT overconfident with respect to the null.
"0 of 71 pass" stands under the committed null, under a converged version of
it, under a correlation-length-matched null, and under no significance gate
at all. The only overconfidence found anywhere in this item runs the OTHER
way (a too-small permutation tile flattering candidate maps), and correcting
it strengthens the negative.

Caveats: re-nulls use 50 perms (floor p = .0196) vs the screen's 200; block
128 remains below the passers' claimed 199-230-px periods, so a real ruling
at those periods would survive the permutation (the control demonstrates
retained power at 0.98x its period, a harder case).

## Files
ns1_premise.py/.json            premise reproduction (KILL check) — PASS
ns2_ess.py/.json, ns2_draws.npz, ns2_slab*.npz, ns2_obsfield.npz
                                400-draw ensemble, DEFF/n_eff, correlation lengths
ns3_corrected.py/.json          block bootstrap + stability validation
ns4a_corpus_structure.py/.json  gate decomposition + map correlation lengths
ns4b_renull.py, ns4b_z_dbg_gen_00215.json   tile-64+128 re-null (validated path)
ns4c_renull_resume.py, ns4c_*_t128.json     resumable tile-128 re-nulls + control
ns4d_summary.py, ns4_corpus_verdict.json    consolidated corpus verdict
