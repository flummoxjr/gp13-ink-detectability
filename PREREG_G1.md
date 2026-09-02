# Pre-registration — Pick 1 (Ink Mass Budget), G1 run

**Written 2026-08-20, before the G1 numbers exist.** Committed to git before the run so the
timestamp is checkable. Nothing in this file may be changed after the run; corrections go in a
separate section appended below, dated, with the original text left intact.

The reason this file exists: G1's verdict depends on a tile size the original pre-registration
never named, and the three candidate sizes give three different answers. Choosing the size after
seeing which one passes would be post-hoc. So it is fixed here, in advance.

---

## 1. What is being tested

Whether carbon ink is distinguishable from blank papyrus by **depth-integrated X-ray column
excess** on the detached Herculaneum fragments, which carry independent infrared ground truth.

Two outcomes are pre-specified and both are publishable:

- **DETECTION** — ink is not iso-attenuating; the measured difference becomes a matched-filter
  amplitude and the method converts into a detector.
- **CEILING** — no detection; publish a bounded upper limit on ink areal mass. The field has no
  such number: scrollprize's own open-problems page still describes ink's CT effect qualitatively.

## 2. Fixed analysis parameters

| parameter | value | note |
|---|---|---|
| tile size **T** | **32 px** | fixed here, in advance; see §5 for why |
| stride | **32 px (non-overlapping)** | overlapping tiles are correlated and buy no information |
| grid origin | **(0, 0)** of the fetched band, no edge clipping | |
| min pixels per class per tile | **64** | |
| positive class | `label == 1 AND mask == 1` | never `label == 1` alone — see §4 |
| negative class | `mask == 1 AND label == 0` | annotated blank, not assumed blank |
| tile weight | `min(n_ink, n_blank)` | |
| estimator | `S = sum over all 65 layers of (I(z) - I_air)` | |
| `I_air` | pooled median inside mask over the deepest 10 layers | cancels exactly in the paired estimator |
| null | **40 rigid translations** of the same label bitmap, `abs(dx) >= 128`, **seed 20260820** | preserves area, stroke geometry, autocorrelation; destroys only registration |
| z | `(obs - null_mean) / null_sd` | |

**TIFF constants, corrected this session and load-bearing:** pixel data begins at byte
**260** (`StripOffsets` parsed from the IFD), dtype **`>u2`** (big-endian uint16). The 262 used by
the pilot is wrong and shifted `vol.npy` by exactly one column. Any new fetch must parse
`StripOffsets` rather than assume a constant.

## 3. Decision rule

**G1 (sampling validity).** At `>= 1,700` admitted T=32 tiles the 40-translation null sd must be
`<= 2,300 DN*vox`. If it is not, **no bound is quoted**, and that failure is itself the reported
result — "here is why nobody can quote one" still ships.

**G0 (harness validity).** Each fragment must separate label-ink from label-blank on the infrared
plate at locally-paired AUC `>= 0.90` on the identical tiles. A fragment failing G0 is **dropped,
not explained**.

**DETECTION.** `z > 4.0` on an admitted fragment, Bonferroni across 6 fragments
(family alpha `1.9e-4`).

**CEILING.** If `abs(z) <= 4.0` on all admitted fragments, publish the pooled 2-sigma ceiling
`= abs(obs - null_mean) + 2 * null_sd`, expressed **primarily in papyrus-voxel-equivalents**
(`S / BULK`, `BULK = 6,434 DN` per 3.24 um voxel on Frag1) and **secondarily** in mg/cm² with the
linearity systematic stated explicitly. Paganin phase retrieval is a spatial filter, so DN is not
linear in mu; the dimensionless bound is the defensible headline and the mass figure is derived.

**Pooling requires `>= 3` admitted fragments.**

## 4. Known hazards, pre-committed handling

1. **Reference-class error.** Outside the supervision mask an absent label means *nobody looked*,
   not that there is no ink. On 500p2a, 625,970 label-ink pixels (4.87% of all ink) fall outside
   the mask, 621,303 of them in one window, which would inflate that window's positive class by
   28.8%. Positive class is `label AND mask`, everywhere, without exception.
2. **Tiling instability.** At the 86-tile pilot the excess point estimate ranged -805 to +6,244
   across conventions. Only the fixed convention in §2 is reported.
3. **The pilot's +3,249 excess is withdrawn.** Two independent implementations now agree it is
   statistically indistinguishable from zero (`z = +0.28` on the pilot's own numbers). It must not
   be quoted as a measured value anywhere.
4. **T=64 is structurally impossible on Frag1** — the whole fragment yields 1,477 qualifying tiles
   against a 1,700 requirement. This is measured, not a budget constraint.

## 5. Why T = 32, decided before the run

- **T=64 cannot reach the required tile count on Frag1 at all** (1,477 < 1,700). Ruled out on a
  structural fact, not on its projected sd.
- **T=16 projects a null sd of ~2,702**, above the 2,300 threshold, and its measured sqrt-n slope
  (-0.437) is the furthest from the ideal -0.500, so its scaling is the least trustworthy.
- **T=32 is the only size that both reaches >= 1,700 tiles on Frag1 and projects under the
  threshold** (~1,147), with a scaling slope of -0.513, close to ideal.

This reasoning uses only tile-count arithmetic and null-scaling behaviour measured **before** any
G1 excess value was computed at the target tile count. It does not use the excess itself.

## 6. What this run may NOT produce

- No ceiling, bound, or mass figure may leave this project unless G1 passes.
- No pick-2 (model-hallucination) result may be reported until P1 passes — the identical
  resample-and-depth-pool pipeline must yield `ink_9um` seed42 pixel AUC `>= 0.95` on the on-disk
  w035 crop, against an in-domain anchor of 0.9991.
- **P2 depth convention:** every surface is scored in both depth directions, the headline is
  `max(forward, reverse)`, and both are published. No post-hoc direction selection.

## 7. Honest prior, recorded now

The pilot showed no detection (`abs(z) <= 0.71`). The larger Stage-A band moved z to
+1.085 / +1.851 / +1.078 across the three tile sizes, positive everywhere, with the point estimate
stabilising. Naive sqrt-n scaling to 2,560 T=32 tiles projects `z ~ +3.4`, below the pre-registered
detection threshold of 4.0.

So the single most likely outcome is a **ceiling, not a detection** — and a z near 3 with a
threshold at 4 is precisely the situation in which it is tempting to move the threshold. It is
fixed at 4.0 and will not move.
