# Scalar-field salvage — H1 residual structure + H5 geography

Date: 2026-08-17. Input: full six-worker per-tile stats from the stopped PHerc1203
2.4 µm ink screen (ink_3d_dino_guided, 256³ tiles, stride 256, L0 frame).
Analyst discipline: every structure/clustering claim below carries a ≥200-permutation
null; where spatial autocorrelation would inflate naive p-values, a **block-permutation
null** (4³-tile blocks ≈ 2.5 mm) is reported alongside.

## Headline

1. **H1 SUPPORTED — the residual field has real spatial structure.** After
   regressing out fill, mean CT, z and winding radius (R² = 0.62), the residual
   f05 field still has Moran's I = 0.289 (naive null 0.000 ± 0.004; block null
   0.213 ± 0.002, z = +31, p < 0.005), and the top-1% residual tiles are
   spatially **clustered**: mean NN distance 2.10 tile units vs 3.14 ± 0.08
   random-subset null (z = −12.6, p < 0.005), still below the block-permutation
   null 2.39 ± 0.06 (p < 0.005).
2. **The QC round's raw-field "anti-clustering" does not replicate on the full
   data — it was a coverage/small-n artifact.** On all 29,748 scored tiles the
   raw top-1% f05/f08 tiles are strongly clustered (adjacency fraction 0.39 vs
   0.05 random; NN z = −12.1). Restricted to QC's three slabs at full coverage
   the top-1% adjacency is 0.475; thinning those slabs to QC's n = 3,346 drops
   it to a mean of 0.24 with a min–max range 0.00–0.45 across 50 thinnings —
   QC's observed 0.0 sits inside that range. The "no rankable tail" conclusion
   survives for other reasons (blanket firing level), but the anti-clustering
   evidence line should be retired from the report.
3. **Mean CT density is the dominant nuisance driver, not fill.** Spearman
   (meanCT_material, f05) = **0.73** — far above fill (0.38). Fill alone gives
   R² = 0.24; adding mean CT brings R² to 0.617; z and radius together add
   < 0.004. The QC round never had the CT covariate; this is the single most
   useful nuisance regressor for any future screen.
4. **H5 SUPPORTED — geography is structured, but the effect sizes are small.**
   Radial structure is the strongest (η² = 0.067 vs block null 0.021 ± 0.004,
   p < 0.005): firing is flat through the scroll body and falls ~2.5× in the
   outermost wraps (r/R95 > 1.0). Angular structure is real but mild
   (η² = 0.023 vs 0.005 ± 0.002, p < 0.005; ±10% swings). The z-trend is
   marginal once block-scale autocorrelation is accounted for (η² = 0.016 vs
   0.012 ± 0.002, **p = 0.010**) and is confounded with partial coverage of the
   two sparse slabs.

Interpretation caveat, up front: **structure ≠ ink.** These tests establish that
the residual field is not exchangeable noise; they say nothing about the cause.
Given the QC morphology findings (crack/edge-ribbons, on/off-sheet contrast
≤ 1.6×) and the shapes of the residual hotspots (below), regional CT
texture/damage remains the most plausible driver. The residual field is the
right thing to cross-reference against any future working detector — nothing
more is claimed.

## 1. Assembly (task 1)

`tiles.parquet` — one row per unique tile: worker, z/y/x (L0 origin), fill,
skipped, pmax, f05, f08, meanct, meanct_mat, radius, rnorm, theta.

| count | value |
|---|---|
| raw rows in six jsonl files | 502,186 (35,148 scored + 467,038 skip) |
| rows sharing a tile coordinate | 10,800 (5,400 tiles double-scored by two workers) |
| unique tiles after dedupe (scored kept over skipped) | 496,786 |
| **unique scored tiles** | **29,748** |
| scored z-slabs (L0 origins) | 0, 256, 512, 768, 1024, 1280, 1536, 1792, 2560, 2816, 3072, 4096, 4352, 5888 |
| coverage | 14 slabs, all in the lower 40% of the band (z ≤ 5888 of 14,848); slabs 4096 and 5888 sparse (321 / 118 tiles), all others ≥ 1,700 |

Grid: stride 256, tile indices (ti,tj,tk) = (z,y,x)/256; volume L0 ≈
(15.2k, 26.5k, 26.5k); 1 tile unit = 0.615 mm.

**Mean CT covariate.** Computed as 8³-block means of the cached L5 array
(`ct1203_L5.npy`, 474×828×828), divided by fill for a material-only mean
(`meanct_mat`, background is 0 in the masked volume). Validated against the
task-preferred source by streaming six random 128³ chunks of the S3 L4 array
and computing 16³ block means for the 990 scored tiles they cover:
**corr = 1.000000, max |Δ| = 0.04 DN** (`ct_proxy_validation.json`) — L5 is
mean-pooled L4, so the covariate is exactly the L4 16³ block mean at zero
streaming cost.

## 2. Geography (task 2) — `geography.png`

Profiles of f05 (median + IQR) with η² (share of variance explained by binning)
against two nulls, 200 permutations each: naive value shuffle, and block
permutation (value-vectors swapped between size-matched 4³-tile blocks,
within-block arrangement preserved; 99.3% of tiles sit in swappable blocks).

| profile | η² obs | naive null (mean ± sd) | p | block null (mean ± sd) | p |
|---|---|---|---|---|---|
| z (14 slabs) | 0.016 | 0.0004 ± 0.0002 | <0.005 | 0.0116 ± 0.0019 | **0.010** |
| radial (12 bins of r/R95) | 0.067 | 0.0004 ± 0.0002 | <0.005 | 0.0209 ± 0.0041 | <0.005 |
| angular (16 bins) | 0.023 | 0.0005 ± 0.0002 | <0.005 | 0.0047 ± 0.0016 | <0.005 |

- **Radial** (strongest): flat ~0.06–0.07 from centre to r/R95 ≈ 0.85, then a
  monotone fall to ~0.025 at the rim. The outer wraps fire markedly less. Sign
  matches QC's winding-radius ρ = −0.17; the profile shows it is a rim effect,
  not a linear gradient (linear rnorm term adds <0.3% R² in the regression).
- **Angular**: real but small — median f05 0.055 (θ ≈ +10°) to 0.071 (θ ≈ −60°).
- **z**: flat ~0.065 through the well-covered slabs; the apparent drop past
  z = 3072 rides on the two sparse, partially-screened slabs and survives the
  block null only at p = 0.010. Treat as weak/confounded.
- The slab maps (fig. panels D–F) show the in-plane f05 field visibly tracing
  the spiral wrap structure — and the mean-CT map of the same slab shows the
  same spiral. That is the geography's proximate cause: **firing tracks local
  CT density** (Spearman 0.73), and density itself is organized by the wraps.

So: firing is not flat noise — it tracks scroll structure — but the structure
it tracks is the material density field, not anything letter-scale.

## 3. Nuisance regression (task 3)

OLS, standardized covariates, n = 29,748 (zero rows dropped).

**M1 (spec): f05 ~ fill + meanCT_mat + z + rnorm → R² = 0.621** (f08: 0.597)

| term | incremental R² (cumulative) | std beta (f05 units) |
|---|---|---|
| fill | 0.241 | −0.0002 |
| + meanCT_mat | 0.617 | +0.0185 |
| + z | 0.618 | +0.0007 |
| + rnorm | 0.621 | −0.0013 |

Mean CT eats fill's entire contribution once included (fill's marginal beta ≈ 0):
"more material → more firing" was really "denser material → more firing".
M2 (flexible: slab dummies + quadratics in fill/meanCT/rnorm) reaches only
R² = 0.627 — the linear spec model is essentially complete; residuals from M1
and M2 give the same downstream answers everywhere below.

Residual sd = 0.0145 (vs raw f05 sd 0.0236).

## 4. Residual clustering (task 4) — `residuals.png`

Top-1% = 297 tiles. Distances in tile units (1 = 0.615 mm). 200 permutations
per null. "adj" = fraction of top tiles with ≥1 6-neighbor also in the top set.

| field | top-1% mean NN obs | random-subset null | z | block-perm null | adj obs / null |
|---|---|---|---|---|---|
| raw f05 | 2.132 | 3.139 ± 0.083 | −12.1 | 2.261 ± 0.049 (p<0.005) | 0.391 / 0.052 |
| raw f08 | 2.128 | 3.137 ± 0.084 | −12.1 | — | 0.384 / 0.052 |
| **resid f05 (M1)** | **2.098** | **3.144 ± 0.083** | **−12.6** | **2.390 ± 0.057 (p<0.005)** | **0.401 / 0.053** |
| resid f05 (M2) | 2.205 | 3.150 ± 0.084 | −11.2 | — | 0.370 / 0.051 |
| resid f08 (M1) | 2.142 | 3.133 ± 0.088 | −11.3 | — | 0.357 / 0.050 |

Top-5% (n = 1,487): same picture, z ≈ −18 to −23 for all fields.

The block-permutation null is the honest bar for the NN statistic (a smoothly
autocorrelated field clusters its top tail "for free"): block-permuted fields
give mean NN 2.39 ± 0.06, and the observed 2.10 is still 5σ below it — the
clustering exceeds what within-2.5mm-block autocorrelation alone produces.

Moran's I (6-connectivity, 78,911 neighbor pairs):

| field | I obs | naive null | block null | z vs block |
|---|---|---|---|---|
| raw f05 | 0.529 | 0.0002 ± 0.0034 (p<0.005) | 0.404 ± 0.0035, max 0.415 (p<0.005) | +35 |
| resid f05 M1 | 0.289 | −0.0005 ± 0.0035 (p<0.005) | 0.213 ± 0.0025, max 0.220 (p<0.005) | +31 |
| resid f05 M2 | 0.290 | −0.0002 ± 0.0031 | 0.214 ± 0.0024 (p<0.005) | +31 |
| resid f08 M1 | 0.304 | −0.0003 ± 0.0037 | 0.225 ± 0.0024 (p<0.005) | +32 |

The nuisance model removes ~45% of the raw autocorrelation (0.53 → 0.29); what
remains is overwhelmingly non-random at both short range (naive null) and
beyond block scale (block null).

**Where the residual hotspots are.** Top-1% residual tiles appear in all 13
covered slabs (5–33 per slab) — not one privileged region. 213 connected
components (6-conn): mostly singletons/pairs, 14 components of size ≥3, largest
14 tiles. The large components are **elongated, axis-aligned runs**, e.g. 14
tiles spanning slabs 3–7 at constant ty=60, tx 45–51 (a wall in x–z); 11 tiles
at constant ty=35, tx 59–65 (slabs 10–11); 8 tiles at ty 61–63, tx 19–21
crossing slabs 10–12. Sheet/crack-like geometry, not compact letter-suggestive
patches — consistent with the QC morphology verdict (texture/damage follower).

**Comparison with QC's raw-field result (task 4 requirement).** QC test 3
(3,346 tiles, 3 slabs, top-1% n≈33) reported adjacency 0.0 vs 0.10–0.12 random
→ "anti-clustered". Replication here: same three slabs at full coverage give
adjacency 0.475 vs 0.042 ± 0.029 null (clustered, >10σ); randomly thinning
those slabs to n = 3,346 gives observed adjacency 0.00–0.45 (mean 0.24) across
50 thinnings. The QC value 0.0 is inside the thinned range: the anti-clustering
was an artifact of partial coverage plus a 33-tile top set, not a property of
the field. (QC's own top-5% caveat — "regional structure ~3σ" — was pointing
at the truth.)

## 5. Verdicts (task 5)

**H1 (residual signal exists): SUPPORTED — with a texture-not-ink caveat.**
After removing fill + meanCT + z + radius (R² = 0.62), the residual f05 field
retains Moran's I = 0.289 vs block-permutation null 0.213 ± 0.0025 (z = +31,
p < 0.005, 200 perms), and top-1% residual tiles cluster at mean NN 2.10 vs
3.14 ± 0.08 random-subset null (z = −12.6) and vs 2.39 ± 0.06 block null
(p < 0.005). The structure is robust to a flexible nuisance model (M2) and to
f05↔f08. Nothing here validates it as ink: hotspot geometry is elongated and
axis-aligned, and the QC morphology evidence points at regional texture/damage.
Value: a ranked residual map to cross-reference against any future detector.

**H5 (geography is structured): SUPPORTED.** Firing tracks scroll structure,
not flat noise: radial η² = 0.067 (block null 0.021 ± 0.004, p < 0.005; rim
wraps fire ~2.5× less), angular η² = 0.023 (block null 0.005 ± 0.002,
p < 0.005), z marginal (η² = 0.016, block null 0.012 ± 0.002, p = 0.010,
confounded with sparse-slab coverage). The proximate driver is the CT density
field (Spearman meanCT↔f05 = 0.73), which itself follows the wrap geometry.

## Corrections to the round-1 verdict

- Retire "top-1% tiles are spatially ANTI-clustered (noise)" (§3 of the
  verdict): full-data raw top-1% is clustered at z = −12; the original result
  is reproduced only as a coverage artifact.
- Amend "drivers = fill (ρ 0.43)": the driver is mean CT density (ρ 0.73);
  fill is a proxy that mean CT subsumes.
- The stop decision itself is unaffected — blanket firing level, no silent
  tiles, and the morphology evidence stand.

## Method caveats

- Block permutation swaps value-vectors between equal-size 4³-tile blocks
  (within-block arrangement preserved, local-offset order matching); 0.7% of
  tiles sit in blocks with no size partner and keep their values — marginally
  conservative. Block scale 4 tiles ≈ 2.5 mm; structure at or below that scale
  is inside the null, so all "beyond block scale" claims are conservative.
- Scored coverage is 45% of the band, biased to low z; H5's z-verdict applies
  to the covered range only.
- 5,400 double-scored tiles deduped by keep-first; worker pairs agreed at the
  reported precision on spot checks.
- meanct_mat = meanct/fill is undefined below fill 0.02; zero scored tiles hit
  that (min scored fill > 0.02).

## Files

All under `C:\Users\benbl\Desktop\Vsuvious\trackD\salvage\`:

- `tiles.parquet` — 496,786 unique tiles, full covariate set; `zslab_geometry.parquet` — per-slab centroid/R50/R95
- `assemble.py`, `validate_ct.py`, `analysis.py`, `figures.py` — reproduce everything
- `ct_proxy_validation.json` — L5-vs-L4 covariate check (corr 1.000000, n=990)
- `analysis_results.json` — every statistic + null quoted above
- `fig_data.npz` — arrays behind the figures (incl. residual fields)
- `geography.png` — profiles + slab maps (task 2 deliverable)
- `residuals.png` — residual maps with top-1% overlay + null-vs-observed panels
