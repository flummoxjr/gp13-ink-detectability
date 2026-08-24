# Corpus screen protocol v2 — what changed, and what it does to the control and to the flag

Date: 2026-08-18. Scripts: `trackD\analyze_survey_corpus_v2.py` (screen),
`trackD\corpus_v2_figure.py` (figure), `trackD\out\survey\v2_report_numbers.py`
(every v2 number below is machine-emitted from the JSON by that script; the v1
and validated-battery comparison figures are quoted from the sources named
beside them, and the v1-on-control row was produced by re-running v1's own
`analyze()` on the control map). Outputs: `trackD\out\survey\corpus_analysis_v2.json`,
`trackD\out\survey\corpus_screen_v2.png`, log `corpus_v2_run.log`
(71.1 min, 30 workers, 15,477 scored maps).

## Verdict

> **Zero of 71 scored segments pass the hardened screen, and the human-verified
> control passes it 5 gates out of 5** — prominence 123.5 against a null of
> 20.0 ± 6.4, **z = +16.3 (95 % CI +13.5…+20.1), empirical p = 0.00498 (0 of 200
> permutations beat it)**, at 4.678 mm, 11.0 cycles, ρ(1P/2P/3P) =
> +0.65/+0.54/+0.46, fwd/rev r = 0.094. The v1 flag
> `PHerc1447/z_dbg_gen_00166_inp_hr` goes from **z = +5.94** to **z = +0.97,
> p = 0.159 (31 of 200 shuffles of its own map beat it), 0 gates of 5.**
>
> The permutation test alone no longer separates the two populations — both the
> control and one corpus segment (`z_dbg_gen_00260`) bottom out at the
> attainable p floor of 1/201. **What separates them is everything else.** The
> control's peak repeats 11 times, autocorrelates positively at 1P, 2P and 3P,
> and sits 4 bins inside the search band. `z_dbg_gen_00260`'s "period" occupies
> the band's **lowest** bin, repeats 5 times, and its map survives z-reversal at
> r = 0.64. Under the band-constrained search (§2.5) the control holds
> p = 0.00498 while the **entire corpus bottoms out at p = 0.060**.

Two corpus-level facts do most of the work, and neither depends on the
periodicity machinery:

- **`gate_fwd_rev`: 0 of 71.** The lowest forward/reverse map correlation in the
  whole 80-segment survey is **0.222**; the control is 0.055–0.094. Ink lives on
  one face of the sheet. Nothing in this corpus does.
- **48 of 71 segments (68 %) put their "best period" in the band's lowest one or
  two Fourier bins** — the band-edge leakage mode, not a period.

---

## 1. Why v2 exists

v1 (`analyze_survey_corpus.py`) flagged exactly one segment out of 80,
`PHerc1447/z_dbg_gen_00166_inp_hr`, at `ruling_z = +5.94`. The verification
(`trackD\verify_flag\FLAG_VERDICT.md`) refuted the flag and, in doing so,
refuted the screen: its z-scores did not mean what they appeared to mean. Its
four prescriptions are implemented here, plus two additions of our own (§2.5,
§2.6). The statistic, the band and the null family are unchanged, so v2 stays
directly comparable to the validated battery that scored the control.

**How badly was v1 calibrated?** Run v1's own `analyze()` on the control map:
**z = +13.74** (prominence 69.33, null 12.73 ± 4.12, period 4.39 mm). So under
v1, the separation between *known Greek letters* and its worst false positive
was a factor of **2.3**. (The `+25.6…+34.1` printed on v1's `corpus_ranking.png`
as the control line was never computed by v1 — it was quoted from the validated
battery, a different protocol, making that figure's reference line
apples-to-oranges.)

**And how much of v1's ranking was signal?** Spearman ρ between v1's z and v2's
corrected z across the 69 segments both scored is **+0.370**. Four of v1's top
five collapse (#1 +5.94 → +0.97, #3 +3.36 → +0.18, #4 +3.03 → +0.59,
#5 +3.02 → +0.69) while its #2 rises (+3.52 → +4.64), and three segments v1
ranked at or below zero rise into v2's top eight (−0.66 → +3.96, −0.31 → +2.07,
−1.30 → +1.08). 33 of 69 fall, 36 rise: v1's ordering was substantially
permutation noise, not a weak signal.

---

## 2. The changes

### 2.1 Empirical p, and ≥ 200 permutations (prescription 1)

v1 drew **16** block permutations per map and reported only
`z = (obs − mean)/sd`. The prominence statistic (band peak power ÷ band median
power) is heavy-tailed, so 16 draws underestimated its own sd by 2.2× — the
flag's true null sd was 8.43, not 3.92. On v1's own numbers `z ≥ 5` meant
`p ≈ 0.01`, and `E[# segments with z ≥ 5 among 80] = 1.07` — exactly the one
observed.

v2 draws **200** permutations per map and reports
`empirical_p = (1 + #{null ≥ obs})/(n_null + 1)`, `z_corrected` with a
**bootstrap 95 % CI** (2,000 resamples), the full null summary
(`null_mean`, `null_sd`, `null_max`, `null_p95`) so the tail is visible, and
`holm_p` across the scored corpus.

The correction is symmetric, not a thumb on the scale: **the control's own z
falls from +25.6 (5 nulls, validated battery) to +16.3 (200 nulls)**, because
the 5-draw null sd of 4.91 was also an underestimate of the true 6.37. The
control still clears by 16 standard deviations with p at the floor.

**Multiplicity, stated honestly.** At 200 permutations the smallest attainable p
is `1/201 = 0.00498`. Holm across 71 tests requires `p < 7.04e-4`, which that
permutation count **cannot produce** — gating on Holm would make the gate
vacuously unpassable and "zero survivors" meaningless. So `gate_significance`
uses the **raw** p ≤ 0.05; `holm_p` is reported beside it (corpus minimum:
**0.354**); and the design is explicitly two-stage — **any segment clearing all
five gates gets escalated to a dedicated high-permutation run** (the
400-permutation machinery in `verify_flag\vf_perm.py` already exists for this).
Observed at p ≤ 0.05: **4 of 71, against 3.55 expected under the null.**

### 2.2 The validated preprocessing, restored (prescription 2)

v1 dropped four steps of `salvage\verdict_periodicity.py` — the protocol that
scored the human-verified control — for speed. All four are back:

| step | v1 | v2 | why it matters |
|---|---|---|---|
| rim erosion | none | **40 full-res px** (10 ds4 px), after closing + hole-fill | 90.8 % of the flag's above-blank-p99 pixels lived in the `ink_9um` inference patch-boundary halo, on an exact 64-px lattice; eroding it dropped the flag's prominence 47.3 → 13.3 |
| profile detrending | none | **σ = 90 ds4 px** gaussian high-pass | without it, 1/f drift leaks into the band and a step function scores as a ruling |
| orientation grid | 15° | **3°** | the control's ruling sits at θ = 177° |
| null | map only, 32 ds8-px blocks | **joint (map, mask), 64 ds4-px tiles** | permuting the map but not the mask leaves the mask's own long-range structure intact, so the null is too easy to beat |
| analysis scale | extra ds2 (ds8) | **ds4**, no extra decimation | σ = 90 and the 64-px tile are both defined at ds4 |

### 2.3 Periodicity sanity gates (prescription 3)

A peak in a power spectrum is not a period. At the winning orientation v2
requires all three:

- **≥ 6 cycles** of the claimed period inside the profile. Flag: **4.00**
  (838 ds4 px = 29.0 mm; 29.0 / 7.24 = 4.00). Control: **11.0**.
  Corpus: **52 of 71 below the gate**, median 4.0.
- **positive autocorrelation at 2P and 3P** of the detrended profile. Flag, as
  the verification measured it on the undetrended profile: ρ(1P) = −0.269,
  ρ(2P) = +0.013, ρ(3P) = −0.025; through v2's detrended pipeline
  +0.065 / −0.004 / +0.013. Control: **+0.646 / +0.544 / +0.458**. Corpus ρ(2P)
  median +0.094, max +0.302 — no segment reaches even half the control's.
- **peak not in the band's two lowest Fourier bins.** Flag: bin **0** of 12.
  Control: bin **4** of 24. Corpus: **48 of 71 in bin 0 or 1**.

### 2.4 Forward/reverse symmetry gate (prescription 4)

Requires **|fwd/rev r| < 0.20** at map scale, using the correlation the survey
already computed at full resolution over the common non-zero support.

Calibration: control 0.055 (eroded mask) / 0.076 (verdict) / **0.0943**
(recomputed here, common non-zero support). Corpus **minimum over all 80
surveyed segments: 0.222** (`auto_grown_20250702235910292` — the one segment
that matches the control on every mesh-geometry statistic, per
`hunt\geometry_compare.md`). The threshold sits in the gap; it is not tuned to
any segment. **0 of 71 pass.** The five segments with survey stats but no saved
map (§4) score 0.33–0.64, so this gate is decided for all **80** surveyed
segments, not just the 71 scored.

### 2.5 Addition — a band-constrained second search (guards against false negatives)

Applying §2.3 only to the argmax has its own failure mode: a genuine 4.7 mm
ruling could lose the argmax to a band-edge artifact and then be discarded by
the very gate meant to catch the artifact. So v2 computes, **from the same
spectrum and with the same band-median normaliser** (no extra rotation cost), a
second statistic whose search is restricted to bins that could physically be a
ruling — not in the band's two lowest bins, repeating ≥ 6 times — and runs it
through the **same 200-permutation null**. Reported per segment under
`constrained_search`.

It costs nothing on the control: the constrained search selects the identical
peak (4.678 mm, θ = 177°), **z = +18.9, p = 0.00498**. And it independently
confirms the corpus null: **0 survivors, corpus-best p = 0.0597**; the flag
scores prominence 10.2 against a null of 12.8 ± 4.6, **z = −0.55, p = 0.692
(138 of 200 shuffles beat it)**.

### 2.6 Addition — two implementation fixes

- **Memory.** `vf_common.rot_cache` keeps all 60 rotated mask *images* — ~1 GB
  per worker on the larger maps (the first full run reached 41 GB across 30
  workers and began swapping). `rot_cache_light` keeps only what
  `vf_common.profile` reads (column sums, coverage mask, shape): identical
  arithmetic, ~1/60 the memory. Verified — the flag's observed prominence is
  24.771 through both paths.
- **Crop to the eroded mask's bounding box** before the orientation search. Rows
  outside carry mask weight 0 and are already dropped by the coverage test, so
  the retained profile is the same up to interpolation edge effects — measured
  at 0.4 % on the flag's prominence (24.68 uncropped, 24.77 cropped) — for ~20 %
  less work.

### 2.7 Cross-check that v2 really is the validated protocol

Run on the human ink labels of w035, v2 returns prominence **39.09** at period
**6.033 mm**, θ = 18°. `salvage\verdict_periodicity.json` records **39.231** at
**6.033 mm**, θ = 18.0° for the same input. Three significant figures on an
independent implementation path. (What that number *means* is §6.)

---

## 3. Exact specification

```
n_perm 200 | erode 40 full-res px (10 ds4) | detrend sigma 90 ds4 px
theta grid 3 deg over [0,180) | perm tile 64 ds4 px (fallback 32) | band 1.7-8.4 mm
statistic: band peak power / band median power   null: joint (map, mask) block permutation
gates: raw empirical p <= 0.05 | >= 6.0 cycles | rho(2P) > 0 and rho(3P) > 0
       peak bin index >= 2 | |fwd/rev r| < 0.20
```

Per segment, on the saved forward `*_forward_ds4.npy` prediction map:

1. valid region = `binary_closing(map > 0, 5×5)` → `binary_fill_holes` →
   `binary_erosion(3×3, 10 iterations)`; crop to its bbox; zero the map outside.
2. for θ = 0°…177° step 3°: rotate map and mask (bilinear, `reshape=True`), take
   the mask-weighted row-mean profile over rows with coverage > 0.25 × max.
3. detrend (subtract the σ = 90 gaussian), Hann-window, `rfft`, power.
4. prominence = band peak ÷ band median; keep the best θ. In the same pass keep
   the best θ for the band-constrained variant (§2.5).
5. null: 200 joint (map, mask) 64-ds4-px tile permutations, each through the
   identical search. 70 of 71 segments used the 64-px tile; one small map fell
   back to 32.
6. gates as above.

Scale: `ink_9um` predictions are full-resolution; the survey saved
`pred[::4, ::4]` as uint8. 9.362 µm/px scrolls (PHerc1203; PHerc0139 for the
control) → 37.448 µm/px at ds4. 8.64 µm/px scrolls (PHerc1447, PHerc0800) →
34.56 µm/px.

---

## 4. Coverage

| | n |
|---|---|
| segments in the survey | 80 |
| with a saved forward ds4 map | 75 |
| **scored by v2** | **71** (PHerc1447 48, PHerc1203 20, PHerc0800 3) |
| skipped | 4 |

Skips, all documented and none silent:

- `z_dbg_gen_00325_inp_hr` — eroded mask 0.001 of canvas (0.22 % non-zero, and
  fwd/rev r = **1.000**: its reverse render is bit-identical to its forward, a
  degenerate map).
- `auto_grown_20250929222256117`, `auto_grown_20250930000321811`,
  `z_dbg_gen_00070_inp_hr` — after 40-px rim erosion the surviving surface spans
  only **5.4–6.4 mm**, less than the 6.8 mm (4 × the band's shortest period) the
  band test requires. Too small to support a ruling claim.

The 5 surveyed segments with no saved map (`z_dbg_gen_00329`,
`z_dbg_gen_00701`, and three PHerc0800 `auto_grown_2025102822…`) were screened
on the pod (0 tripwire hits each); their maps were not among the 150 pulled
home. Their fwd/rev r values are 0.33–0.64, i.e. they fail `gate_fwd_rev` on the
survey record alone.

Tripwire hits across all scored segments: **0**.

---

## 5. RESULTS

### 5.1 Gate cascade

| gate | passing | of 71 |
|---|---|---|
| `gate_significance` (raw p ≤ 0.05) | 4 | 71 |
| `gate_cycles` (≥ 6 cycles) | 19 | 71 |
| `gate_autocorr` (ρ(2P) > 0 and ρ(3P) > 0) | 23 | 71 |
| `gate_band_bin` (not the band's 2 lowest bins) | 23 | 71 |
| `gate_fwd_rev` (\|r\| < 0.20) | **0** | 71 |
| the three periodicity gates together | 10 | 71 |
| the four map-internal gates (drop fwd/rev) | **0** | 71 |
| **ALL FIVE** | **0** | 71 |
| band-constrained search (§2.5) | **0** | 71 |

The result does not rest on the symmetry gate alone: **dropping `gate_fwd_rev`
entirely still leaves zero survivors.**

### 5.2 Corpus distributions

| quantity | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| corrected z | −1.20 | −0.51 | **−0.09** | +0.35 | **+4.64** |
| empirical p | 0.0050 | 0.231 | 0.408 | 0.761 | 1.000 |
| period at winning θ (mm) | 1.75 | 4.46 | 5.50 | 7.51 | 8.27 |
| cycles in profile | 2.0 | 3.0 | **4.0** | 6.0 | 21.0 |
| fwd/rev r | **0.222** | 0.463 | 0.637 | 0.758 | 0.914 |
| ρ(2P) | −0.094 | +0.019 | +0.094 | +0.158 | +0.302 |

Minimum Holm-adjusted p across the corpus: **0.354**. Profile lengths after
erosion: 7.0–54.1 mm (median 22.5). Eroded mask fraction: 0.037–0.790
(median 0.517).

### 5.3 Top 12 by empirical p

| # | scroll | segment | v1 z | v2 z | p | Holm p | period | cycles | bin | ρ(2P) | fwd/rev r | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1447 | `z_dbg_gen_00260` | +3.52 | **+4.64** | **0.0050** | 0.354 | 7.95 mm | 5.0 | 0/19 | +0.302 | 0.639 | 2/5 |
| 2 | 1447 | `20251105093211-z_dbg_gen_00320` | −0.66 | +3.96 | 0.0199 | 1.000 | 7.60 mm | 4.0 | 0/14 | −0.029 | 0.538 | 1/5 |
| 3 | 1447 | `z_dbg_gen_00357` | +0.89 | +2.56 | 0.0348 | 1.000 | 6.90 mm | 7.0 | 1/23 | +0.204 | 0.625 | 3/5 |
| 4 | 1447 | `z_dbg_gen_00215` | −0.31 | +2.07 | 0.0398 | 1.000 | 7.11 mm | 4.0 | 0/13 | +0.121 | 0.765 | 2/5 |
| 5 | 1203 | `auto_grown_20251005230830031` | +1.11 | +1.49 | 0.0796 | 1.000 | 4.83 mm | 2.0 | 0/4 | — | 0.747 | 0/5 |
| 6 | 1203 | `auto_grown_20250923164713356` | −0.13 | +1.12 | 0.0796 | 1.000 | 7.61 mm | 3.0 | 0/11 | −0.044 | 0.399 | 0/5 |
| 7 | 1447 | `auto_grown_20250703025628283` | −0.49 | +0.84 | 0.1144 | 1.000 | 3.13 mm | 8.0 | 5/12 | +0.265 | 0.314 | 3/5 |
| 8 | 1447 | `20250502182456-auto_grown_20250502161202782` | −1.30 | +1.08 | 0.1194 | 1.000 | 7.76 mm | 3.0 | 0/11 | +0.110 | 0.790 | 0/5 |
| 9 | 1447 | `z_dbg_gen_00283_inp_hr` | — | +1.15 | 0.1293 | 1.000 | 8.10 mm | 3.0 | 0/12 | +0.044 | 0.892 | 0/5 |
| 10 | 1447 | `auto_grown_20250502161744358` | +3.02 | +0.69 | 0.1293 | 1.000 | 7.84 mm | 2.0 | 0/8 | — | 0.743 | 0/5 |
| 11 | 0800 | `20251028220955-auto_grown_20251028220955262` | −0.57 | +0.69 | 0.1443 | 1.000 | 5.15 mm | 3.0 | 1/8 | +0.218 | 0.522 | 0/5 |
| 12 | 1447 | `20250703025628-auto_grown_20250703025628283` | −0.54 | +0.73 | 0.1492 | 1.000 | 3.63 mm | 8.0 | 4/14 | −0.036 | 0.329 | 2/5 |

The new leader, `z_dbg_gen_00260`, is the same failure family the flag
verification already identified ("period 7.88 mm, i.e. the band edge again"):
its peak is the **lowest bin of a 19-bin band**, it repeats 5 times, its map
survives z-reversal at r = 0.64, and its Holm p is 0.354. Its band-constrained
score is p = 0.0597 — the corpus maximum, and still not significant.

### 5.4 Figure

`trackD\out\survey\corpus_screen_v2.png` — A the gate cascade; B the empirical-p
ECDF against the uniform null with the control at the floor; C the two decisive
axes (fwd/rev r vs corrected z), control vs corpus; D v1 z against v2 z with the
flag's collapse marked; E cycles-in-profile against claimed period with the
6-cycle gate; F the control and the flag side by side through the identical code
path.

---

## 6. Before / after

### 6.1 The control — `out\ink9um_w035\w035_seed42-075000.tif` (PHerc0139, human-verified Greek letters)

| protocol | prominence | null mean ± sd | z | empirical p | period | verdict |
|---|---|---|---|---|---|---|
| **v1 screen** (16 perms, ds8, 15°, no erosion, no detrend) | 69.33 | 12.73 ± 4.12 | **+13.74** | not reported | 4.39 mm | — |
| validated battery (5 perms, block-mean ds4) | 143.6 | 17.67 ± 4.91 | +25.63 | not reported | 4.678 mm | — |
| **v2, strided ds4** (like-for-like with the survey maps) | 123.55 | 20.02 ± 6.37 | **+16.26** (CI +13.47…+20.10) | **0.00498** (0/200) | 4.678 mm | **5/5 PASS** |
| **v2, block-mean ds4** (cross-check) | 123.80 | 19.41 ± 5.56 | **+18.78** (CI +15.97…+22.23) | **0.00498** (0/200) | 4.678 mm | **5/5 PASS** |

Full v2 control record: θ = 177°, profile 51.5 mm, **11.0 cycles**, peak at band
bin **4 of 24**, ρ(1P/2P/3P) = **+0.646 / +0.544 / +0.458**, fwd/rev r = 0.0943,
eroded mask 0.859 of canvas, null max 55.04. The strided-vs-block-mean rows
differ by 2.5 z and agree exactly on period, cycles, bin and autocorrelation:
the survey's choice of `pred[::4,::4]` over block-mean costs nothing that
matters.

**The protocol detects the known letters decisively. That is the licence to
report the corpus null.**

### 6.2 The previously-flagged segment — `PHerc1447/z_dbg_gen_00166_inp_hr`

| protocol | prominence | null mean ± sd | z | empirical p | period | cycles | bin | ρ(2P) | gates |
|---|---|---|---|---|---|---|---|---|---|
| **v1 screen** (16 perms) | 35.42 | 12.12 ± 3.92 | **+5.94** | not reported | 7.26 mm | 4.0 | 0/14 | +0.013 | — |
| verification, 400 perms, same geometry | 35.42 | 14.88 ± 8.43 | +2.43 | 0.0374 | 7.26 mm | 4.0 | 0/14 | +0.013 | — |
| verification, validated protocol, 400 perms | 24.68 | 19.46 ± 9.69 | +0.54 | 0.180 | — | — | — | — | — |
| **v2** (200 perms) | 24.77 | 18.13 ± 6.87 | **+0.97** (CI +0.75…+1.24) | **0.159** (31/200) | 6.748 mm | **4.0** | **0/12** | −0.004 | **0/5** |
| **v2, band-constrained search** | 10.24 | 12.78 ± 4.60 | **−0.55** | **0.692** (138/200) | 3.219 mm | 7.0 | 4/12 | +0.111 | fail |

v2 reproduces the verification's independent 400-permutation result (+0.54,
p = 0.180) to within the difference the bbox crop makes. The segment fails
**all five** gates: p = 0.159; 4.0 cycles; ρ(2P) = −0.004; band bin 0 of 12;
fwd/rev r = 0.754.

### 6.3 Net effect of the four fixes

| | v1 | v2 |
|---|---|---|
| control z | +13.74 | **+16.26** |
| worst corpus false positive (z) | +5.94 | +4.64 |
| control ÷ worst false positive | 2.3× | 3.5× |
| corpus median z | −0.27 | −0.09 |
| segments flagged | 1 (`z ≥ 5`) | **0** (all five gates) |
| corpus best p, band-constrained | not computed | **0.0597** |
| Spearman ρ(v1 z, v2 z) | — | **+0.370** |

---

## 7. What v2 still cannot do

Stated plainly, because a screen's blind spots are part of its result.

1. **It cannot resolve p below 0.005.** 200 permutations bottom out at 1/201, and
   Holm across 71 tests needs 7.0e-4. The gate is therefore raw-p, with
   escalation for survivors (§2.1). There were no survivors to escalate.
2. **It is a page-scale test, not a letter-scale one.** Run on w035's *human ink
   labels* — the ground truth itself — v2 returns z = +0.26, p = 0.328, 3 gates
   of 5 (the validated battery gets z = +1.31 on the same input). The labels are
   a sparse subset (13 letter components) on a 54 mm profile; the prediction map
   carries the model's dense response over the whole sheet, and that is what the
   ruling test measures. **A handful of letters on an otherwise blank sheet
   would not trip this screen** — the survey's tripwire and the morphology
   battery cover that regime, and both are also clean (0 hits across 80).
3. **The gates are calibrated on one control.** A hand whose ruling falls outside
   1.7–8.4 mm, or a fragment showing fewer than 6 ruled lines, is missed by
   construction. `MIN_CYCLES = 6` costs sensitivity on small segments: 52 of 71
   segments have profiles too short (or periods too long) to clear it.
4. **The 71 tests are not 71 independent surfaces.** `hunt\geometry_compare.md`
   found the 80 published segments are only **66 unique surfaces** (23 are
   `z_dbg` debug dumps, and several appear twice under timestamped and bare
   names — e.g. `auto_grown_20250703025628283` and
   `20250703025628-auto_grown_20250703025628283` both appear in §5.3). The
   multiplicity arithmetic in §2.1 is therefore mildly conservative.
5. **The corpus null is a statement about GP index ranks 8, 13 and 14.** 18 of
   the 66 unique meshes — all PHerc1447 — put part of their surface outside the
   reconstructed volume; PHerc1447 publishes segments grown into empty volume
   (one with 91 % zero-valued vertices). Only PHerc1203 was a fair test. v2 does
   not change that, and no re-screening of these maps can.
6. **Only forward maps are screened for periodicity.** The reverse renders enter
   solely through `fwd_rev_r`.
7. **The saved maps are 8-bit and 4× strided.** §6.1 shows this costs nothing
   measurable on the control, but the screen never sees full-resolution
   predictions; anything finer than ~35 µm/px is invisible to it by construction.

---

## 8. Files

- `trackD\analyze_survey_corpus_v2.py` — the screen (four prescriptions + §2.5, §2.6)
- `trackD\corpus_v2_figure.py` — the figure
- `trackD\out\survey\corpus_analysis_v2.json` — per-segment records, control records, skip list
- `trackD\out\survey\corpus_screen_v2.png` — publication figure
- `trackD\out\survey\v2_report_numbers.py` — emits every table above
- `trackD\out\survey\corpus_v2_run.log` — run log
- `trackD\out\survey\v2cache\` — the control's ds4 arrays (strided, block-mean, labels)

Superseded, retained for the before/after: `trackD\analyze_survey_corpus.py`,
`trackD\out\survey\corpus_analysis.json`, `corpus_ranking.png`.

AI assistance: the v2 screen, the figure and this document were produced by
Claude (Opus 5) under the project author's direction, who specified the four
fixes from the flag verification. All v2 numbers are machine-emitted from
`corpus_analysis_v2.json` via `v2_report_numbers.py`; v1 and validated-battery
figures are quoted from `corpus_analysis.json`, `verify_flag\FLAG_VERDICT.md`
and `salvage\verdict_periodicity.json`.

[BEN: one line on whether the v2 screen (or just its verdict) belongs in the
public report / community post, and whether the "0 of 71, and 0 even without the
symmetry gate" framing is the one you want to lead with.]
