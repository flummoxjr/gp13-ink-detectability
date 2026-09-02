# Discord / community release drafts

Community adoption is explicitly weighted in the monthly progress prizes, and the prize history
shows tools that people actually pick up outscore tools that merely exist. These are drafts for
Ben to post in his own voice — short, useful, no marketing.

---

## Post 1 — the index (post first; it is the most immediately useful thing we have)

> **Scan-quality index for the 13 GP scrolls (+ PHerc0139 as a read anchor)**
>
> I wanted to know which scrolls were worth GPU time before spending any, and I could not find that
> published anywhere. So I measured it.
>
> I measured every released 9 µm-class GP volume against **its own in-scan noise** — five papyrus
> ROIs per scroll, air-window noise references validated by a flatness gate, three metrics
> (air-referenced structural bandwidth, structural SNR at q = 0.25 cyc/px, DN headroom), reported
> as median + IQR.
>
> Two things came out that might be useful to others:
>
> 1. **The cohort splits into two tiers with a 3× gap and nothing in between.** Nine volumes at
>    mid-band SNR 72–160; five at 8.5–24.4 (PHerc1447 is the outlier at 8.5, an order of magnitude
>    below the cohort). If you are choosing where to spend compute, the bottom five are where a
>    model failure cannot be blamed on the model.
> 2. **It correlates with acquisition campaign**: all four 8.64 µm / 116 keV scrolls are in the
>    degraded tier vs one of nine at 9.362 µm / 113 keV (Fisher p = 0.007). Stated as correlation —
>    campaign, pitch, energy and batch are confounded, and PHerc0257 is degraded *inside* the good
>    campaign, so it is not the whole story. Still, if a rescan list is being drawn up, this is data.
>
> Caveat worth knowing regardless of my index: **the shared uint8 render window clips 5–34 % of
> in-mask voxels in nine of fourteen volumes** (PHerc0191 worst at 34 %). Any bright-ink screen or
> dynamic-range metric on those volumes is right-censored — PHerc0139, the one that has been read,
> is among the least clipped.
>
> **Two corrections to the above, from building a second axis afterwards — worth reading before you
> use the numbers.**
>
> **(a) Scan quality is not the axis that predicts segmentability.** I added a sheet-separability
> measure (local structure-tensor planarity: high where lamellae are distinct and parallel, low in
> granular incrustation, and invariant to contrast so it can't just restate SNR). It correlates only
> ρ = +0.27 (p = 0.36) with the SNR index. The validation I did not tune for: **PHerc0139, the one
> scroll with letters proven at 9 µm, ranks first of 14** on it, against a measured isotropic floor
> of 0.105 from 28 in-scan air windows. And the inversion that matters to this community —
> **PHerc1447 is the worst-scanning volume in the whole index and has more published segments than
> any other scroll**; separability ranks it 5th where SNR ranks it 14th. If you are picking a target,
> the two axes disagree, and the separability one is the one about whether a surface can be traced.
>
> **(b) My own ROI rule was sampling the wrong material.** The index picks each scroll's *highest
> mean intensity* windows subject to a fill gate. In a carbonised scroll that upper tail is mineral
> incrustation, not papyrus. Re-sampling the identical frame uniformly at random instead scores
> **3.0× higher on separability, in 14 of 14 scrolls** — the intensity-picked windows sit barely
> above the isotropic floor. The tier split and the campaign correlation survive (every scroll is
> biased the same way), but **no individual ROI number in my index describes that scroll's typical
> material**, and I did not say so originally. If you run anything that scores "the densest window"
> in these volumes, this will bite you too.
>
> Code + per-scroll JSON + figure: https://github.com/flummoxjr/measure-before-you-hunt. The separability output includes **per-ROI
> coordinates and scores for all 328 sampled cubes**, so you can seed growth at material measured
> to be laminated rather than wherever a picker lands. Happy to run it on any volume you care about.

---

## Post 2 — the tools (post after the index lands, so it reads as follow-through)

> **Three reusable gates from a month of failing to find ink**
>
> I burned a fleet learning these, so you do not have to.
>
> 1. **Pre-fleet model acceptance gate** (~30 tiles, minutes, ≈$1): pipeline-equivalence check,
>    in-domain control, in-domain distribution reference, firing morphology vs the surface
>    prediction, and a single-tile input-adaptation probe. It caught a released model blanket-firing
>    on an out-of-domain scroll at 45 % fleet completion and stopped the run at $4.46. Run it before
>    you rent anything.
> 2. **Text-signature battery** (~5 min/segment, CPU): four tests against texture-preserving nulls —
>    line-ruling periodicity, stroke morphology vs a blank-papyrus reference, map-scale z-orientation
>    asymmetry, and intensity calibration — plus a pre-registered tripwire (value > blank p99 ∧ area
>    > 10⁴ px ∧ width > 30 px → mandatory human look) so nothing gets eyeballed into a claim.
>    Calibrated on w035, where it locks onto the 4.68 mm ruling at z = +26 while the null sits at 0.
>
> 3. **Mesh-vs-lamella alignment check** (seconds, on data you already have). Take the angle between a
>    grown surface's own normal and the local structure-tensor normal of the volume underneath it. It has a
>    calibrated reference: **published GP meshes sit at 13.1° (7 of 9 within 30°), two random directions sit
>    at 60°.** I wrote it because 8 surfaces I grew on PHerc0813 looked completely healthy — right area, full
>    vertex validity, non-zero surface DN, sensible generation counts — and were sitting at a median **68.1°
>    to the lamellae**, i.e. indistinguishable from randomly oriented. I do not yet know why. The obvious
>    fix is refuted by experiment: I re-grew the same seeds with the released normal grids supplied and it
>    changed nothing (median |n_z| 0.97 before, 0.99 after; published meshes sit at 0.22). Curvature
>    averaging is excluded too — measuring only the vertices inside the sampled cube makes it slightly
>    worse (72.9°). **If you grow surfaces, check this before you spend GPU
>    time rendering them** — a surface oblique to the sheets averages away exactly the depth contrast an ink model
>    needs, and it passes every other health check. If anyone knows what makes `vc_grow_seg_from_seed` track or not
>    track lamellae, I would genuinely like to hear it.
>
> All three are checkpoint-agnostic. https://github.com/flummoxjr/measure-before-you-hunt
>
> Also released: a tifxyz → 21-slice surface-volume renderer that makes the flat-mode `ink_9um`
> checkpoints runnable on **any** 9 µm segment (validated end-to-end: re-rendering w035 from its
> mesh reproduces the published-surface-volume prediction at r = 0.81 and still passes the ruling
> test). Useful for any scroll whose segments have no published surface volumes.

---

## Post 3 — the corpus screen

> **Ran the `ink_9um` instrument over every published segment of the GP scrolls**
>
> 80 of 80 catalogue rows — PHerc1203 (22), PHerc1447 (52), PHerc0800 (6) — rendered from tifxyz,
> inferred in both z-directions, 416 cm² of surface, 160 inferences, 0 errors. ≈$5.5 of cloud
> compute on a 4-pod fleet.
>
> Method and controls are the battery above; the positive control (PHerc0139 w035) reproduces the
> published letters at pixel AUC 0.999, so the instrument demonstrably detects text at this
> resolution when text is present.
>
> **No segment tripped the letter-scale tripwire (0/160).** For line-ruling I'll quote the second
> pass, because the first one didn't survive its own verification. Hardened protocol — 200
> permutations per map, empirical p, and five pre-registered gates (significance, ≥6 cycles,
> positive autocorrelation at 2P and 3P, peak not at the band edge, |fwd/rev r| < 0.20):
> **0 of 71 scorable segments pass; the human-verified control passes 5 of 5** at z = +16.3,
> empirical p = 0.005, with 0 of 200 shuffles beating it. The single segment that cleared the first
> pass's threshold died twice — +0.54 (p = 0.18) under a dedicated 400-permutation verification, and
> +0.97 (p = 0.16, 0/5 gates) when the whole corpus was re-scored.
>
> The gate that does most of the work needs no periodicity machinery at all: **forward-vs-reverse
> map correlation, 0 of 71**. Ink sits on one face of a sheet, so reversing the render's z-order
> should destroy it. The lowest value anywhere in the corpus is 0.222; the control is 0.055–0.094.
>
> I'd rather say this plainly than bury it: my first pass used 16 permutations, and on a heavy-tailed
> statistic that underestimated the null's own standard deviation by 2.2×. Spearman ρ between the two
> passes is +0.37 and four of the first pass's top five collapsed — its ordering was substantially
> permutation noise. If you are screening prediction maps with a permutation null, this is the
> failure mode to check for first.
>
> Two things I'd flag rather than bury: 23 of the 65 unique surfaces are `z_dbg_gen_*` debug dumps,
> and 18 place part of their surface outside the reconstructed volume (all PHerc1447) — so the
> screen's honest scope is narrower than "the GP scrolls". A vertex-vs-volume check that rejects
> those before you spend GPU time runs in seconds at pyramid level 3, and is in the repo
> (`hunt/check_air.py`).
>
> Per-segment statistics, the downsampled prediction maps, and the scoring code are in the repo so
> the negative is checkable and re-scorable rather than something you have to take on trust.

---

## Notes for Ben

- Post as three separate messages over a few days, not one wall — the index is the hook.
- Every claim above has a number and a file behind it in the repo; if anyone challenges one, the
  primary artifact is the answer, not an argument.
- The honest framing that has served this work: *these are measurements of whether ink could be
  found, not claims that it was.* It costs nothing and buys credibility that a first-month
  contributor otherwise has to earn slowly.
- If the team asks for anything specific (a volume, a segment, a metric), say yes and run it —
  responsiveness is how tools get adopted, and adoption is what the prize actually rewards.
