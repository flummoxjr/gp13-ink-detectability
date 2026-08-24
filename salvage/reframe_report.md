# Reframe analysis (H6) — is the failed ink screen a crackle/damage-density atlas?

Date: 2026-08-17. Analyst: reframe subagent. Inputs: `proxies.parquet` (this
analysis; tiles.parquet + streamed TRUE CT-L4 and surf-m7-L2 blocks for all
29,748 scored tiles), building on `scalar_report.md` and `morph_report.md`.
Discipline: every correlation below carries a 4³-tile block-permutation null
(200 perms) alongside a naive null; all quoted p-values are the block ones.

## VERDICT: H6 REFUTED — the sign is backwards, and the information is free elsewhere

The field is a genuine, strongly structured **material-condition read-out — but it
is a PRESERVATION/DENSITY map, not a damage map, and everything it measures is
available at ~zero cost from the CT pyramid itself.** f05 *anti*-correlates with
every crack/damage proxy tested and adds exactly zero incremental information
about the one independently measurable condition target (m7 segmentation holes)
once cheap CT covariates are controlled. The atlas deliverables were built anyway
(`atlas_1203.png`, `atlas_1203.npy`, `atlas_1203_README.md`) with honest framing:
they document the failure mode and provide the before/after baseline for any
future fine-tune — they are not a community condition tool.

## 1. Condition proxies vs f05 (task 1)

New per-tile proxies for **all 29,748 scored tiles** (not sampled):
`ct_std_all`/`ct_std_mat` = std of the TRUE streamed CT L4 16³ block (all voxels /
material-only, crack-texture proxies); `surf_frac` = density of the released m7
surface mask (SURF L2, same 16³ grid); `surf_recovery` = surf_frac / material
fraction (sheet yield per material voxel — the segmentation-difficulty proxy);
plus fill and meanct_mat from tiles.parquet. Frame checks: streamed-L4 mean vs
cached-L5 meanct corr 0.99999987; mat_frac vs fill corr 0.9998.

**Caveat found on the way:** the released m7 surface prediction ("th0.2") is
**binary at every pyramid level** (values {0,255} at L2 and L4) — no graded
confidence exists in the release. Coverage/recovery is therefore the only
available difficulty proxy; "confidence" per se cannot be tested.

Spearman with f05, block-permutation null (200 perms; naive nulls all ±≤0.006):

| proxy | ρ (all tiles) | block null | ρ (interior, fill ≥ 0.9) | p_block |
|---|---|---|---|---|
| meanct_mat (density) | **+0.731** | +0.143 ± 0.014 | +0.676 | < 0.005 |
| surf_frac (sheet density) | +0.516 | +0.120 ± 0.011 | +0.417 | < 0.005 |
| surf_recovery (seg. yield) | +0.487 | +0.099 ± 0.011 | +0.412 | < 0.005 |
| fill | +0.380 | +0.198 ± 0.011 | — | < 0.005 |
| ct_std_all (crack texture) | **−0.379** | −0.032 ± 0.010 | **−0.507** | < 0.005 |
| ct_std_mat (material texture) | −0.401 | −0.029 ± 0.011 | −0.541 | < 0.005 |

(Block nulls are nonzero because the proxies are themselves spatially organized;
every observed ρ sits far outside its null — 0/200 exceedances, ~40σ.)

**The sign pattern is the story.** A crackle/damage atlas requires f05 HIGH where
texture is rough, sheets are broken, and segmentation struggles. The opposite
holds on every axis: f05 is high where material is **dense** (+0.73), texture is
**smooth** (−0.54 interior), the m7 mask is **dense** (+0.52), and sheet recovery
is **good** (+0.49). Consistent with morph_report's mechanism (the model paints
intact sheet surfaces as membranes): more intact, tightly-wound, well-preserved
wraps → more sheet surface → more firing. Damage *silences* the model.

## 2. Tautology decomposition (tasks 1+3)

Incremental R² for f05 (n = 29,721 complete rows):

| order | sequence |
|---|---|
| density first | meanct 0.615 → +fill 0.616 → +ct_std 0.634 → +surf_frac 0.634 → +recovery **0.635** |
| texture first | ct_std 0.070 → +surf 0.346 → +recovery 0.359 → +meanct 0.634 → +fill 0.635 |

- Not the trivial tautology of the task prompt (f05 ≈ fill gives only R² 0.24),
  but the next-cheapest one: **f05 ≈ monotone(mean CT) — one free covariate
  captures 61.5% of variance; texture and the entire m7 surface field together
  add 2.0 points.** All condition proxies combined: R² 0.635, matching the
  scalar analyst's nuisance ceiling (0.62–0.63).
- The residual ~36% retains spatial structure (scalar H1: Moran I 0.29 over
  block null 0.213) but correlates with **no independent condition proxy tested
  here** — it remains unexplained, not demonstrably meaningful.

## 3. Community value: does f05 predict m7 segmentation holes? (task 3)

Interior tiles (fill ≥ 0.9, n = 27,728); "hole" = bottom decile of surf_recovery.

| test | result |
|---|---|
| raw ρ(f05, recovery) | **+0.41** — f05 is LOW in holes (inverted flag at best) |
| partial ρ(f05, recovery \| meanct, ct_std, fill, rnorm) | **−0.001** (block null −0.012 ± 0.010, p ≈ 1) |
| leave-slab-out CV R² for recovery: cheap covariates | 0.3555 |
| same + f05 | 0.3555 (**Δ = −0.0000**) |
| AUC flagging hole tiles: low f05 | 0.855 |
| AUC flagging hole tiles: low meanCT (free) | **0.892** |

Low f05 does flag segmentation holes (AUC 0.855) — but **mean CT, computable
from the public zarr pyramid in seconds without any GPU, does it strictly
better**, and f05 carries zero information about holes beyond it. As a
segmentation-difficulty predictor the screen output is **dominated**: the fleet
compute bought a noisy copy of a free covariate.

## 4. Deliverables built (task 2, honest framing)

- `atlas_1203.npy` — (60,104,104) float32 tile grid of f05, NaN = unscored;
  1 cell = 0.615 mm; frame documented in the README.
- `atlas_1203.png` — 14-slab montage, mm axes, material-mask outlines, sequential
  colorbar, embedded reading guide stating the correlations and the NOT-ink /
  NOT-damage framing. The in-plane field visibly traces the wrap spiral — i.e.
  the density field, as expected.
- `atlas_1203_README.md` — what it does and does not measure.
- Analysis artifacts: `r1_proxies.py`, `r2_correlate.py`, `r3_atlas.py`,
  `proxies.parquet`, `reframe_correlations.json`, `reframe_interior_check.json`.

## 5. Verdict on H6 (task 4)

**H6 REFUTED.** The screen output is a legitimate *condition-correlated* field —
its structure is real and survives block-permutation nulls everywhere — but it is
(a) a **preservation/density** map with the **opposite sign** to crackle/damage
(interior ρ with crack-texture std = −0.54; damage and holes silence the model),
and (b) **strictly dominated** as a condition tool by free CT statistics
(meanCT: R² 0.615 of f05 itself; hole-AUC 0.892 vs 0.855; incremental value of
f05 ≈ exactly 0 by partial-ρ and blocked CV). "Crackle/damage-density atlas of
the band" is not a defensible community framing; "documented failure-mode map of
ink3d under cross-scroll domain shift" is, and that is how the atlas files are
labeled.

### What last night's data is actually good for (project-log summary)

Last night's 30k-tile screen is a clean, quantified negative result, not an
atlas: a Paris4-trained ink3d checkpoint, verified healthy in-domain, responds
on PHerc1203 with a field that is 63%-explained by local CT density plus mild
texture/surface terms and carries zero incremental information about anything
independently measurable. The response is a preservation map, not a damage map —
the model fires on dense, smooth, well-segmentable wraps and goes quiet on
cracked or damaged ones — so neither ink reading nor condition-atlas reframing
survives. Every condition quantity it tracks is available for free from the
public CT pyramid (meanCT predicts m7 segmentation holes better than f05 does),
so the screen's outputs have documentation value only: the failure-mode
characterization for the Aug 31 report, hard-negative guidance for any 1203
fine-tune (mine dense intact sheet stacks, not cracks), and a fixed
before/after benchmark (tiles.parquet + proxies.parquet + atlas_1203) that any
future 1203-adapted model must beat. The reusable positives are the QC battery
as a pre-fleet acceptance gate and the nuisance-covariate stack assembled here.
The residual field (Moran I 0.29 after nuisance removal) is the one unexplained
remnant — keep it purely as a cross-reference layer for a future working
detector; it predicts nothing measurable today.
