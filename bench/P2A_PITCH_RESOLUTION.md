# 500p2a source-volume ambiguity — RESOLVED (2026-09-01)

**Verdict: the `ink/unused/500p2a` surface volume is 2.215 µm/px, not 4.32 µm/px.
The 2026-08-25 transfer-ladder anchor (ink_9um pixel AUC 0.5382 fwd / 0.5055 rev on
win1, "chance on a clean foreign scroll") fed the model data at an effective pitch of
4.80 µm while presenting it as 9.36 µm, and is VOID as a transfer measurement.**

`GROUND_TRUTH_AUDIT.json` flagged this on Aug 18 ("bbox z-max 27616.26 exceeds the
4.317 µm volume's z extent of 15838 … resolve this before rendering"); the Aug-24/25
curve+audit runs (`out/transfer_ladder/`, commit d27affe) went ahead with 4.32 µm from
the meta.json `volume` string. SEPTEMBER_PLAN.md §5.3 listed the resolution as a debt.

## Evidence (three independent handles, all public data)

| handle | at 2.215 µm | at 4.32 µm | source |
|---|---|---|---|
| mesh bbox fits a public volume | x ≤ 16037, z ≤ 27616 fit the 2.215 µm volume `20250526151718` [28096, 18209, 18209] | do **not** fit the 4.317 µm volume `20250528085330` [15838, 9423, 9423] (x 16037 > 9423, z 27616 > 15838) | `bench/p2a/meta.json`, `x/y/z.tif` (max 16021 / 11978 / 27596), S3 `.zarray` of both volumes |
| canvas vs voxel extents | surface-volume canvas 26239 × 16182 is 1:1 with the mesh's z-extent (25033 vox) and x–y extent (≈14.5k vox) | would imply a 108 × 59 mm surface inside a 68 × 41 mm scan | `500p2a.zarr/.zattrs` canvas_size, `0/.zarray` shape [65, 26239, 16182] |
| letter geometry in the human labels | median component height **2.49 mm** (p25/p75 2.10/2.81), median stroke width **0.61 mm** — Herculaneum-typical | 4.86 mm letters, 1.19 mm strokes — implausible | `bench/p2a/500p2a_inklabels.tif`, measured 2026-09-01 (23 components > 20k px; EDT stroke width in win1) |

Supporting: meta.json `uuid` is `500p2a_2um`; the 65-layer stack is the 2 µm pipeline's
convention (`500p2a_max_22_42.tif`); `ink_canonical_2um` lists `500p2a` in its training
fragments. The meta.json `volume` string (`…4.32_bin_1m_090_4um_ds1`) names no public
volume and was the only basis for 4.32.

## What the fault did

v2 resampled 2.215 µm data by 4.32/9.36 = 0.4615, so the model received voxels of
2.215/0.4615 = **4.80 µm in-plane and in depth** (65 → 30 layers, centre-cropped to 17 =
a 82 µm window instead of 159 µm). That is the "wrong in-plane level" fake-null fault
class documented by nerln (#1648, PR #1580: level-0 instead of level-2 costs ~0.31 AUC)
and Jinhojeong (arm 6). A null without a positive control through the identical harness
is uninterpretable — the v2 prereg had data gates but no fault controls.

Frag1's 0.6925 / 0.4477 is unaffected (its 3.24 µm pitch is documented and correct).

## Corrected parameters

| quantity | wrong (Aug 25) | correct |
|---|---|---|
| pitch | 4.32 µm | **2.215 µm** |
| win size | 4438 px = 19.2 mm | 4438 px = **9.83 mm** |
| in-plane resample to 9.36 µm | 4438 → 2048 px | 4438 → **1050 px** |
| depth | 65 → 30 layers, cropped to 17 | 65 → **15** layers (iso; 144 µm of 159 µm, zero-padded by infer.py) or 17 layers at 8.47 µm (fit17) |
| translation-null annulus | [4.4 mm, 0.40·size] | [4.4 mm, 0.40·size] is **empty** (1987 px > 1775 px) → [3.75 mm, 0.60·size] |

## What happens next

`bench/p2a_v3/pod_p2a_v3.sh` (pre-registered in its header and `out/prereg.json`)
re-measures the anchor at 2.215 µm with both depth modes on win1/win2/win3, and runs
three positive-control arms on the in-domain w035 crop first: the released stack
(expect ≈ 0.999 fwd / ≈ 0.51 rev), the stack upsampled by exactly v2's factor 1.9504
(the fault reproduced on data the model reads), and a 0.5× arm. The harness gates are
fatal; the scale-fault arm's verdict says whether the Aug-25 number is explained by
the fault alone.

Corrections owed once the corrected anchor is in: SEPTEMBER_PLAN.md §1.2 row 1 and §0,
RESEARCH_PLAN_2026-27.md §0 ("0.54 on a clean foreign scroll"), the Road to First
Letters artifact, the 0358 screen prereg's expected-outcome text, `trackD/SESSION_STATE.md`,
the `out/transfer_ladder/` README/commit message, and the ADDENDUM's planned A5 entry.
None of the 0.54 figures were posted publicly.

## Upstream note

The `volume` field of `ink/unused/500p2a/meta.json` names a non-public volume at a pitch
the geometry contradicts. Worth a one-line note to the organizers once our corrected
number exists (Ben, Discord), so nobody else anchors on 4.32.
