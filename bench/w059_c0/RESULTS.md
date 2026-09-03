# w059 / C0 — results (2026-09-03)

Pre-registration: `parts/prereg.json` (locked in the pod script before any data; committed as 8c856b2 at 02:23 UTC,
before the first launch). Pod that produced the numbers: `f997q2ltr4ee4s` (RTX A6000, villa main 9daa477e,
script sha dbe2e6de → v1f), ALL DONE 11:03:59 UTC. Harvest: `experiments/w059_c0_k/w059c0k/`; ds16 maps, results
and status mirrored to `out/w059_c0/pod/`; battery in `out/w059_c0/battery_w059.json`.

## C0 — the inference contract

| quantity | value |
|---|---|
| model / code | `scrollprize/ink_canonical_2um` (resnet3d-152-3d-decoder), villa `optimized_inference`, tile 256 / stride 128, window [23,85) |
| our w035 arm-A map vs the published map | Pearson r = 0.9999997 on 24,009,865 joint ds4 px; identical canvas 22640×20400; identical nonzero fraction 0.831; max |Δ| = 2 gray levels |
| verdict | **PASS** (gate 0.90). The published w035 prediction is exactly this code + model + window. Reproduced on three separate pods. |

## Maps

| map | volume | canvas | nonzero | p99 (ds4) |
|---|---|---|---|---|
| c2_w035B (modality control) | w035 1.129 µm L1, window [26,88) | 24080×21700 | 0.388 | 237 |
| w059B_fwd | w059 1.129 µm L1, window [26,88) | 29860×41440 | 0.305 | 239 |
| w059B_rev | same, reversed depth order | 29860×41440 | 0.305 | 231 |
| w059 fwd/rev Pearson r (joint support, ds4) | | | **0.0039** on 23,646,426 px | |

## PROTOCOL_V2 five-gate battery (local, ds16 = 36.13 µm/px, 200 joint permutations, unchanged scoring code)

| unit | cycles ≥ 6 | autocorr | band bin | significance (Holm) | fwd/rev < 0.20 | gates | reading |
|---|---|---|---|---|---|---|---|
| w035_CONTROL_strided (ink_9um, validity control) | ✓ 11.0 | ✓ | ✓ | ✓ z 16.3, p 0.005 | ✓ 0.094 | **5/5** | battery valid |
| w035_LABELS (human labels, negative control) | ✓ | ✗ | ✓ | ✗ | – | 3/5 | as expected |
| c2_w035B (2 µm model, w035 arm B) | ✗ 5.0 | ✓ | ✓ | ✓ z +2.07, p 0.0498, Holm 0.0995 | n/a (forward only) | 3/5 | signal present but the L1 volume covers only 39 % of the sheet: too short for the cycles gate |
| **w059B_fwd** (2 µm model, w059 arm B) | ✓ 9.0 | ✓ | ✓ | **✗ z −1.10, p 0.94** | ✓ 0.0039 | 4/5 | **no periodicity at full coverage** |

## Reading against the pre-stated outcomes

- Escalation required C0 pass AND w035_B 4/4 map-internal AND w059_B 4/4 + |fwd/rev r| < 0.20. **Not met**: w059_B fails
  the significance gate outright (z −1.10, p 0.94) with 9 cycles of profile available. No escalation; nothing to post.
- The "downgrade and close" reading required the control to pass. The modality control is **marginal**, not clean:
  the ruling signal IS detected on w035 arm B by the 2 µm model (p 0.0498, z +2.07), but the cycles gate fails because
  the 1.129 µm L1 volume covers only 39 % of the sheet. So the w059 null is a null on the second scanner, but the
  battery's sensitivity at this modality is demonstrated only weakly (z 2 on w035-B vs z 16 for the ink_9um A-arm control).
- Consequence for Track F: the arm-A ruling lead on w059 is **not confirmed by the arm-B / 2 µm model at full
  coverage**. The lead is closed for escalation purposes; it may be reopened only by a new measurement (e.g. the ink_9um
  model on w059 arm A at full coverage, which is a different instrument and a different pre-registration).
- No letter language applies to any of this: these are ruling-periodicity and map-agreement statistics.

## Reproduction

    python bench/w059_c0/run_battery_w059.py --src experiments/w059_c0_k/w059c0k --nperm 200 --jobs 28

## Engineering notes (for the record; details in SESSION_STATE.md)

Eleven launches were needed: hosts that never boot, secure hosts with no PyPI / NVIDIA-index / GitHub egress, a torch
upgrade over the image's build, villa's inference/reduce split, two villa caches keyed without the volume identity
(every pass after the first read the first volume — the control had come out byte-identical to the A map),
OpenCV's 2^30-pixel imread limit, and the predictions directory sitting outside the served tree. All fixed in the
script and launcher; total w059 pod spend ≈ $5.
