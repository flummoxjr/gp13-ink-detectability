# Track F - the "unread" high-agreement PHerc0139 segments
Session 2026-08-25. All numbers below measured this session unless marked (quoted).
Repo-bankable; nothing here has been posted anywhere yet.

## Headline
Of the three high-cross-acquisition-agreement PHerc0139 segments believed unread
and unlabeled, TWO are not what the segment list says they are: they are
duplicate meshings of wraps already inside ink_9um's training set. The public
PHerc0139 segment inventory contains at least two same-wrap duplicate pairs
(w045/w046 and w041/w042). Only w059 survives premise verification; it is
genuinely unlabeled, unread, and 8+ wraps away from every annotated or read
surface.

## Step 0 - premise verification (the main result)
Method (new this session): duplication sweep - nearest-neighbor distances
between 9.362um tifxyz vertex grids (20k-40k point samples, full-cloud KDTree),
candidates vs every annotated or read PHerc0139 mesh; plus ds8 ink-map
correlation sampled at 3D-corresponding mesh points.

Calibration, measured:
- true duplicate pair w045/w046: mesh NN median 4.3-4.4 vox @9.362um (~40um),
  46% of points <4 vox, 84% of both meshes within 15 vox @2.399um; map corr at
  corresponding points r = 0.858.
- adjacent wraps in the same neighborhood (w039-w045 chain): 17-20 vox,
  ~0.5% <4 vox; map corr r = 0.074 (w040/w041), 0.195 (w043/w044).
- wrap spacing at w046/w047: 32.6 vox @9.362um (127.1 @2.399um).

Verdicts:
- 20260325000000-w046 (xacq r=0.6940, L99=37.7): DROPPED. Same physical wrap as
  20260126000000-w045 = ink_9um training segment pherc0139-w029 (README mapping
  table re-fetched 2026-08-25; corpus unchanged since the 08-16 scout: 5 native
  + 24 aligned, identical names). NN median 4.3 vox; map corr 0.858.
- 20260206000000-w042 (xacq r=0.6449, L99=36.1): DROPPED by the rule fixed in
  STEP0_W042_RULE.md before computing: NN median to training segment
  20260108000000-w041 (pherc0139-w041 + native w041) = 9.2 vox (~86um, i.e.
  the opposite face of the same sheet); map corr at 3D-corresponding points
  r = 0.591 > pre-stated threshold 0.527 = (0.858+0.195)/2, with controls
  separating (0.858 vs 0.074/0.195). Same wrap, re-meshed.
- 20250223000000-w059 (xacq r=0.6735, L99=20.8): SURVIVES all checks.
  (a) not in ink_9um by name (README table: no PHerc0139 corpus entry maps to
      it) nor by surface (nearest annotated mesh: w049 at 162.7 vox = 1.5mm =
      8+ wraps; all other references 201-575 vox);
  (b) sits in HF bucket ink/unused/0139 (mesh+preds, NO inklabels; the
      annotated set is still exactly 11 dirs, none new since 08-16, none w059);
  (c) not in any published reading: preprint readings cover w25/w34/w47/w49 +
      title; web check 2026-08-25 confirms PHerc0139 remains partially read
      (title/author only in the announcements; the June 2026 end-to-end scroll
      is PHerc1667, arXiv:2606.29085).
  Side finding: the ambiguous human-annotation dir ink/0139/w030_202601301912
  (GROUND_TRUTH_AUDIT could not resolve it between public w046 and w031) is
  decisively NOT w031 (mesh NN median 939 vox = 2.25mm away); it lies in the
  w045/w046-w047 neighborhood (55-70 vox). Its exact identity stays unresolved,
  but every wrap it could be is already annotated or read, and none is w059.
  The audit's timestamp heuristic (ann suffix = public mesh timestamp) is
  refuted for this batch.

## Step 1-2 - the pre-registered 4-gate battery on w059 (PREREG_F.md)
Protocol: v2 screen (PROTOCOL_V2.md), code imported unchanged from
analyze_survey_corpus_v2.py; 200 joint (map,mask) 64-tile permutations; gates
significance (Holm-3) / cycles>=6 / autocorr 2P&3P>0 / band bin>=2;
gate_fwd_rev NOT COMPUTABLE (no same-model reverse render exists for these
published maps) - reported N/A; flag requires all 4 computable gates on BOTH
acquisitions independently.

Inputs: published ds8 jpgs (8-bit, models trained on sibling segments of this
scroll - w059 itself is in no training set):
- A: 2.399um 78keV vol 20260102150214, recipe new_canon_autoresearch
  (2026-04-17), 38.384 um/px after 2x block-mean.
- B: 1.129um 59keV vol 20260413113053, mrg20736-1um-s1z2 (2026-07-09), native
  frame, 36.093 um/px after 2x block-mean. B covers 30.7% of the segment.
- fused-z: xacq 0.5(zA+zB) u8 map, A frame, with the xacq joint mask.

Equivalence proof (required before interpretation):
Both equivalence gates PASS (battery_E.json; runtimes 197/199 s at 10 workers):
- E1 (harness fidelity, control strided ds4): prominence 123.547 (protocol
  record: 123.55 - exact), null 18.84 +/- 5.80, z = +18.04 (CI +13.8..+24.4;
  prereg window 16.26 +/- 2.0 -> PASS), empirical p = 0.00498 (0/200), period
  4.678 mm @ theta 177, 11.0 cycles, bin 4/24, rho(1P/2P/3P) = +0.646/+0.544/
  +0.458, fwd/rev r 0.0943 -> 5/5 gates. The z difference vs the protocol's
  +16.26 is null-seed draw variance only (prominence and period are identical).
- E2 (ds8-jpg chain tolerance: full-res -> block-mean 2 -> uint8 -> JPEG q85 ->
  block-mean 2): prominence 123.689, z = +18.51 (prereg >= +13 -> PASS),
  p = 0.00498, period 4.678 mm, identical gate pattern 5/5. The 8-bit JPEG ds8
  product chain costs nothing measurable on the control.


Battery results:
VERDICT UNDER THE PRE-REGISTERED RULE: NOT FLAGGED. Map A passes all four
computable gates; map B fails significance and cycles; the rule required all
four on BOTH acquisitions. The rule is honored.

Per-map, per-gate (battery_W.json; 200 joint 64-tile perms each):

| map | prom | null mean+/-sd | z (CI95) | emp p | Holm-3 p | period | cycles | bin | rho2P/3P | gates |
|---|---|---|---|---|---|---|---|---|---|---|
| A (2.399um) | 93.44 | 30.5+/-11.1 | +5.66 (+3.8..+9.7) | 0.00995 (1/200) | 0.0299 | 4.925 mm @ 0deg | 13.0 (64.0mm) | 5/30 | +0.403/+0.356 | 4/4 PASS |
| B (1.129um) | 24.76 | 25.2+/-8.9 | -0.05 | 0.408 (81/200) | 0.816 | 4.620 mm @ 0deg | 5.0 (23.1mm) | 2/11 | +0.254/+0.149 | 2/4 |
| fused-z | 23.34 | 26.9+/-7.5 | -0.48 | 0.627 (125/200) | 0.816 | 4.591 mm @ 0deg | 5.0 (23.0mm) | 2/11 | +0.220/+0.117 | 2/4 |

Reading of the numbers, stated carefully:
- A is the FIRST map in this project other than the human-verified control to
  pass all four map-internal gates (the 71-segment corpus screen: 0/71; its
  best was 2/5 with a band-edge period). A's constrained search confirms the
  identical peak (z=+5.95, p=0.00995). A's period 4.925 mm is 5.3% from the
  control's independently measured 4.678 mm ruling on the same scroll, at a
  3-degree-adjacent orientation - consistent with one roll's ruling.
- B does not confirm - and structurally could not have: the published B-arm
  prediction covers a 23.1 mm strip, so at ~4.9 mm the profile can hold at
  most 5 cycles, and gate_cycles (>=6) is unpassable by construction (the
  protocol's documented small-segment blind spot, section 7.3, now on the B
  arm). B's own argmax period, 4.620 mm, agrees with A within 6.6%; its
  autocorr is positive at 2P and 3P. B is underpowered, not contradictory -
  but the prereg rule does not grade intent, and the flag is withheld.
- fused inherits B's strip (joint mask) and adds nothing (p=0.627) -
  consistent with the earlier xacq fusion kill (+0.0080 < 0.01).

Escalation that would settle it (NOT run this session; needs pods/GPU or a
community ask): a full-coverage B-arm render/inference of w059 from the
1.129um volume 20260413113053 (the volume covers the whole segment; only the
published prediction is a strip), then the identical battery. If a full B
passes, w059 clears the two-scanner rule at the next attempt with no rule
change.


## Step 3 - gallery (gallery_F/)
- w059_win1/2/3: top local-agreement 14.7mm crops, A | B(registered) | fused,
  2-98% stretch, local windowed r = 0.92 / 0.91 / 0.89 (256px ds8 windows,
  >=60% joint coverage). Both arms independently show letterform-scale
  connected components arranged in ruled rows. Labeled on the image:
  "cross-scanner agreement region - NOT verified text".
- DROPPED_w042_same_wrap_as_w041.png, DROPPED_w046_same_wrap_as_w045.png:
  candidate map vs training-segment map warped through the 3D mesh
  correspondence - the drop evidence, visually checkable.

## Limitations, stated plainly
- The maps are 8-bit, ds8-downsampled model predictions; both models were
  trained on OTHER segments of this same scroll (scroll-level leakage of style
  priors is possible; segment-level leakage is excluded for w059 by Step 0).
- The battery is a page-scale ruling test; it cannot certify letters. Ceiling
  language: "region worth human inspection".
- gate_fwd_rev - the single most decisive gate in the v2 corpus screen - is
  unavailable for these published maps.
- The two-scanner agreement r=0.67 was computed by the earlier xacq session
  (quoted); this session's new agreement numbers are the windowed local r
  values in the gallery.

## Addendum - verification pass, same day (2026-08-25, second agent)

1. **C2 modality control (battery_F.json) - omitted above, recorded now.** The
   pre-registered C2 ran and its numbers are in battery_F.json:
   - w035CTRL_A (control's own A-arm ds8 jpg): prom 96.39, z = +11.59, p = 0.00498,
     period 4.494 mm @ 177 deg, 11.0 cycles, bin 5/24, rho2P/3P +0.506/+0.460 ->
     ALL 4 map-internal gates PASS. Arm A is INFORMATIVE at ds8-jpg fidelity.
   - w035CTRL_B (control's own B-arm ds8 jpg, a 23.7 mm strip like every published
     B render): p = 0.0348 (PASS), autocorr PASS, band-bin PASS, but gate_cycles
     FAILS (5.0 < 6 - structurally unpassable on the strip) -> 3/4. Under
     PREREG_F's C2 clause ("must pass all 4"), the B arm is formally declared
     NON-INFORMATIVE at ds8-jpg fidelity; no flag and no null claim can issue
     from arm B. w035CTRL_F (fused): 3/4, p = 0.0498, cycles 5.0 - fused inherits
     the strip.
   - Consequence for the verdict: NOT FLAGGED stands unchanged (the flag needed
     both arms either way), but the correct characterization is twofold:
     (i) gate_cycles on arm B was never a fair test - the control fails it too;
     (ii) gate_significance on arm B WAS a fair test (the control's B strip passes
     it at p = 0.0348) and w059's B failed it (p = 0.408). So arm B is mostly
     censored, but not fully: on the one gate where the control's B succeeds,
     w059's B does not.

2. **Step 0 drop evidence independently recomputed this pass** (fresh code, fresh
   20k-point samples, scipy cKDTree over the sweep9 vertex grids):
   w046->w045 NN median 4.30 vox @9.362um, 46.7% < 4 vox (sweep: 4.34 / 46.5%);
   w042->w041 median 9.08 vox, 7.2% < 4 (sweep: 9.08 / 7.4%);
   w059->w049 median 161.9 vox, 0.0% < 4 (sweep: 162.7 / 0.0%). Reproduced.
   The w042 rule numbers verified against sweep9/w042_identity_test.json:
   r(w042,w041) = 0.5910 > 0.5266 = (0.8582 + 0.1954)/2, controls separated
   (same-wrap 0.8582 vs adjacent-wrap 0.0737/0.1954). Drop rule honored as written.

3. Gallery PNGs inspected: all three w059 crops and both DROPPED evidence panels
   carry the "cross-scanner agreement region - NOT verified text" labeling and
   5 mm scale bars as pre-registered.
