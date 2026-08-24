# Track D data-comb — SKEPTIC VERDICT

Date: 2026-08-17. Three hunters combed the Track D data (w035-beyond,
1203-symmetry, cross-anomalies) and raised 8 flags. This pass tried to kill
each one with nulls the hunters did not run. Skeptic scripts/outputs:
`comb_skeptic_w035.py/.json`, `comb_skeptic_w035b.py/.json`,
`comb_skeptic_sym.py/.json`, `comb_skeptic_k3.py/.json`,
`comb_skeptic_misc.py/.json` (all in `trackD\comb\`).

Project discipline applied: two prior refuted false positives; every verdict
below names the surviving mundane explanation. Bottom line first:

> **Nothing in this comb is ink. One flag (a methodological catch) is
> CONFIRMED; two are PLAUSIBLE with heavy downgrades; five are REFUTED or
> collapse into other flags.** The comb's real product is a set of matched
> nulls that future "beyond-the-labels" claims must pass.

---

## Verdict table

| # | Flag (hunter) | Hunter interest | Skeptic verdict | One-line evidence |
|---|---|---|---|---|
| 1 | w035: 18 letter-class comps beyond labels (w035-beyond) | MEDIUM | **PLAUSIBLE-NEEDS-DATA (downgraded)** | All 3 statistical supports die under matched nulls: cross-seed P(18/18)=0.61 vs size/strength-matched texture (97.3% baseline); alignment bootstrap p=0.088/0.26 vs texture subsets; rev/fwd "patterning" is a denominator artifact (reverse response is FLAT at 75–77 DN for labels, candidates, and texture alike; matched rev-mean MW p=0.38). |
| 2 | 4-component coherent row 2.5 periods below labels (w035-beyond) | LOW | **PLAUSIBLE-NEEDS-DATA (weak)** | Sole surviving structural stat: P(≥4 of 18 in one 60-px row band, cx-span ≥2500) = 0.013–0.017 vs both uniform and texture-resampled nulls — nominally significant but post-hoc (one of many patterns that could have been flagged) and the null does not preserve local field clustering. |
| 3 | Tile z-symmetry NOT ink-specific (1203-symmetry) | MEDIUM | **CONFIRMED-INTERESTING** | Survived two quantitative tests I ran: (a) letter tiles do not have lower r than blank tiles — they trend HIGHER (pooled MW one-sided letter<blank p=0.97; KS p=0.018 in the wrong direction); (b) at component level the reverse-map response is flat (~75–77 DN) under labeled letters, candidates, and texture alike. Per-tile/per-component z-symmetry is dead as an ink localizer; LOG's "keep this" note must say map-scale only. |
| 4 | Tripwire near-miss, 1203A_s43r p80 (1203-symmetry) | LOW | **REFUTED** | Chance graze, quantified: 356 hot(≥196) clusters map-wide → expected 1.50 inside a 14,931-px blob, Poisson P(≥1)=0.78; the comp's value distribution is blank-papyrus (p50 142, p90 178). |
| 5 | w035 patch 8 on unlabeled tiles (1203-symmetry) | LOW | **REFUTED as independent evidence** | It IS flag-1 candidate id4535: 100% of the candidate's footprint lies inside patch 8, area matches exactly (17,549 px). One detection counted twice by two hunters; it fails the value template (LV=false) and inherits flag 1's verdict. |
| 6 | K3 z~3900–4020 regional median-ratio elevation (cross-anomalies) | MEDIUM | **REFUTED** | 28-window matched-radius/z null (same pipeline, rings at the clusters' r and z plus ±z planes): whole-window median ratio spans 1.17–1.52, IQR 1.26–1.38, **5/28 ≥ 1.43, max 1.52 > the flagged 1.46/1.48**. The "elevation" is ordinary patchy regional variation of the lower scroll (angular variation at fixed radius dominates) — same family as the confirmed Ca/Fe incrustation; and the flagged windows were pre-selected as top stage-2 clusters, so a ~p80 placement is nothing. |
| 7 | uint8 window censors bright-ink screens in 8/14 scrolls (cross-anomalies) | LOW | **CONFIRMED (reporting caveat, not an anomaly)** | sat_frac recomputed independently from the raw cubes, exact match (0191_pap0 0.34045, 1203_pap0 0.26724, 0139_pap0 0.0019); the flag itself is the mundane explanation — carry it as a K2b-index caveat line. |
| 8 | Residual top-20 = two mundane families (cross-anomalies) | LOW | **PLAUSIBLE-NEEDS-DATA (mechanism corrected)** | Family split is real (7 tiles meanct 38–40 vs 13 at 95–113; gallery shows empty/off-sheet wisps vs frothy damage, zero letterforms) BUT the proposed family-A mechanism fails quantitatively: meanct²+meanct³ terms absorb only 15% of the rim-tile residual (M1 pred 0.011 → 0.020 vs observed 0.073). Family A is a genuine uncaptured off-sheet FP floor — the 36%-residual thread stays parked, not closed. |

---

## Flag 1 in detail (the headline flag) — why it was downgraded

The 18-component catalog is real, reproducible, and worth its report figure.
But every quantitative argument that these are *text* rather than the model's
generic strong-response texture failed a proper null:

1. **Cross-seed confirmation (14/18) carries zero information.**
   Texture components reproduce across seeds at 60% baseline (IoU>0.3,
   same matching method); size-matched (area≥10k) at 81%; size+strength-matched
   (area≥10k, v_p90≥190, n=37) at **97.3%**. Under the same IoU method the
   candidates are 18/18 — P = 0.61 against the matched baseline. Cross-seed
   IoU is just a size/strength readout (Spearman IoU~v_p90 = 0.43, IoU~area
   = 0.32); both seeds share training data anyway.
2. **Ruling alignment is a property of the whole response field, not of the
   candidates.** Bootstrap of size-18 texture subsets: candidates' mean_cos
   0.383 sits at p=0.088; the 14 cross-seed survivors (0.290) at p=0.26; the
   "6 within ±0.15 vs 4.2 expected" claim used a uniform expectation — the
   correct texture-based expectation is 5.2, observed 6, p=0.43.
3. **The z-reversal ratio (0.49 "patterns with letters") is a mechanical
   artifact of the gate.** Decomposition: reverse-map mean response is flat —
   labeled letters 74.7, candidates 77.5, texture 77.3 DN (strength-matched
   MW p=0.38). The entire ratio gradient (0.41/0.49/0.60) is the *forward*
   strength in the denominator (labels 187, cands 155, texture 127), and the
   gate selects on forward strength (v_p90≥195). Spearman(rev/fwd, fwd_mean)
   = −0.59 (p=2e-30) among texture. Even true letters show no suppressed
   absolute reverse response — the map-level r=0.076 signature does not
   localize to components. (This independently confirms flag 3.)
4. **Reverse-render gate null is ambiguous, not supporting.** The identical
   gate on the z-reversed renders yields 2 (s42) / 1 (s43) letter-class
   components vs 18 forward — but the reverse maps' bright tail is globally
   10× weaker (frac≥195: 0.0021 vs 0.0212), so this measures the map-wide
   forward/reverse asymmetry, not candidate-specific one-sidedness.
5. **Visual check:** in the gallery the 12 top candidates are amorphous
   blobs; the labeled letters in the same overview are glyph-shaped. The
   candidates' weakness/blobbiness (frac195 0.19 vs 0.69) remains equally
   consistent with training-adjacency memorization of the labels.

What remains: PHerc0139 was actually read, so unlabeled text under this
segment is plausible *a priori*; the map-wide ruling periodicity (z=+25.6)
does show row-organized response beyond the 12 labels — but texture aligns to
the same grid, so it cannot arbitrate. **The honest report line: "the model
produces strong seed-stable responses beyond the labels on a scroll known to
bear text; no measured statistic distinguishes them from its generic texture
response."** Settling it needs new data, not new statistics on this map:
higher-quality render / z-offset sweep at the candidate rows, or a
human-labeling pass on the dense bottom cluster.

## Flag 6 in detail — the K3 kill

Null design: mask-centroid axis per z-plane; rings at the flagged clusters'
exact radius and z (8 angles each, ±30° exclusion around the cluster), plus
3 angles each at two other z planes per cluster; identical pipeline (window
offsets, papyrus mask, 2-vox erosion, masked 110 keV blur σ=0.62,
whole-window median 74/110 ratio). Result (n=28 accepted windows):

```
min 1.17 · p25 1.26 · p50 1.32 · p75 1.38 · max 1.52
flagged clusters: #3 = 1.48, #5 = 1.46  →  5/28 null windows ≥ 1.43
```

Two null windows on the *same ring, same z* as #3 hit 1.52. The dual-energy
ratio field in the lower scroll is patchy at the ±0.15 level everywhere at
these radii; a pre-selected window at 1.46–1.48 is unremarkable. Mundane
family: the already-verified clumped Ca/Fe incrustation + regional recon
systematics. No follow-up warranted.

---

## What the comb definitively did NOT find

- **No text organization on PHerc1203 at 9 µm**, at any threshold: tripwire
  0/32 rescans across p60–p90 on all 8 maps, positive control trips at every
  threshold, and the single near-miss is quantified chance (P=0.78).
- **No per-tile or per-component z-symmetry ink localizer exists.** Letters
  are not locally z-asymmetric in tile r or in absolute reverse response;
  the fwd/rev discriminator is valid at map scale only.
- **No statistical signature separating the w035 beyond-label candidates
  from the model's generic strong texture response** (cross-seed, ruling
  alignment, z-reversal all fail matched nulls).
- **No localized dual-energy anomaly at K3 #3/#5** beyond the scroll's
  normal regional ratio variation (and no stroke geometry, per the hunter's
  own PSF matching).
- **No PHerc172-class dense-ink signature in the 57 K2b cubes** — with the
  now-documented caveat that 8/14 scrolls are 5–34% window-clipped and a
  bright-anomaly screen is censored there (readability anti-tracks
  saturation: PHerc0139 is among the least clipped).
- **No ink-like or letterform content in the top-20 unexplained-residual
  tiles** of the 1203 ink_3d screen (empty off-sheet wisps + frothy damage).

## What survives the comb

1. **Flag 3 (CONFIRMED):** tile-scale z-symmetry is not ink-specific —
   qualify the LOG's "z-symmetry diagnostic (keep this)" to map-scale use.
   New corollary from this pass: even at component scale, reverse-map
   response is flat; all per-region rev/fwd ratios are forward-strength
   readouts. Any future use of the diagnostic must be a whole-map statistic.
2. **Flag 1/2 (PLAUSIBLE, downgraded):** the beyond-labels catalog is
   report-figure material with the honest framing above; the 4-component
   row (p≈0.015, post-hoc) is the only structural stat that survived at all.
3. **Flag 7 (CONFIRMED caveat):** add the sat_frac censoring line to the
   K2b index writeup.
4. **Flag 8 (corrected):** the 36%-residual thread remains genuinely
   unexplained by covariates (the low-density nonlinearity absorbs only 15%
   of the rim-tile residual) — keep parked as before, now with the top-20
   visually cleared of anything ink-like.

## Three most valuable artifacts produced by the whole comb

1. **`trackD\comb\w035_beyond_labels.png` + `.json`** (hunter w035-beyond) —
   the ranked beyond-labels catalog and overview/crop gallery: the report
   figure for the Aug 31 writeup, now to be captioned with the downgraded
   framing from this verdict.
2. **`trackD\comb\comb_skeptic_w035.json` + `comb_skeptic_w035b.json`
   (+ scripts)** — the matched-null battery (size/strength-matched cross-seed
   baseline, texture-bootstrap alignment null, strength-matched rev/fwd
   decomposition, reverse-render gate null). This is the reusable template
   any future "model found something beyond the labels" claim must pass, and
   the anti-hallucination-story content for the submission.
3. **`trackD\comb\comb_skeptic_k3.json`** — the 28-window matched-radius
   base-rate map for the Paris 4 dual-energy ratio (IQR 1.26–1.38, tail to
   1.52): closes K3 #3/#5 and provides the reference distribution for any
   future ratio-anomaly claim on this scan pair.

(Honorable mention: `sym_calibration.png` + `sym_summary.json` — the
calibration that killed the per-tile symmetry premise before it could
generate false positives.)
