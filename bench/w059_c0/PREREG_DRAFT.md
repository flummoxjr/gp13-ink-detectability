# PREREG DRAFT — one pod, two questions: Bet B's C0 control and Track F's w059 arm B (2026-09-02)

**Status: DRAFT** (commit as the prereg before launch; est. cost $2–4; ~1.5–2.5 h on a 5090).

## Why one run answers both

Both need the public canonical 2 µm model (`scrollprize/ink_canonical_2um`, ResNet-152 3D decoder,
recipe `new_canon_autoresearch_recipe`) run through villa's `ink-detection/optimized_inference`
(`entrypoint.py`, env-driven: `MODEL`, `MODEL_TYPE=resnet3d-152-3d-decoder`, `SURFACE_VOLUME_ZARR=s3://…`,
`START_LAYER`/`END_LAYER` (62-layer window), `TILE_SIZE=256`, `STRIDE=128`, `OUTPUT_PATH=<local png>`,
`FORCE_REVERSE=true` for the reverse pass). Everything it needs is public and already rendered:

| input | surface volume (public, S3, zarr v2 uint8 raw) | shape |
|---|---|---|
| w035 A-arm (control, 2.399 µm) | `PHerc0139/segments/20260317000000-w035_2026031718/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr` | [109, 22640, 20400] |
| w035 B-arm (control, 1.129 µm scan at L1 = 2.258 µm/px) | `…/1.129um-0.22m-59keV-volume-20260413113053-L1.zarr` | [116, 24080, 21700] |
| w059 A-arm (2.399 µm) | `PHerc0139/segments/20250223000000-w059_2025022312/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr` | [109, 28080, 38960] |
| w059 B-arm (1.129 µm scan at L1) | `…/1.129um-0.22m-59keV-volume-20260413113053-L1.zarr` | [116, 29860, 41440] |

Track F (`out/trackF/RESULTS_F.md`) found w059's published B-arm prediction covers only a 23 mm strip
(the `mrg20736-1um-s1z2` model is not public), which made gate_cycles unpassable. **The full-coverage
B-arm surface volume exists** (above); what is missing is an inference over all of it. Running the
public 2 µm model on the L1 (2.258 µm/px) representation gives a full-coverage second-acquisition map
from a public, reproducible model — a legitimate two-scanner test.

## Stages and gates (in order; each gate must pass before the next stage spends)

1. **C0 — reproduce the published A-arm prediction on w035** (Bet B §2.2 step 1). Run the model on the
   w035 2.399 SV with the centred 62-layer window (START 23, END 85) at tile 256 / stride 128; compare
   to the published `…new_canon_autoresearch_recipe-tile256-stride128.tif` on the same canvas.
   **Gate: pixel Pearson r ≥ 0.90 inside the published map's support.** If it fails, sweep the window
   (START ∈ {1, 12, 23, 34, 47}) once; if none reaches 0.90 the contract is wrong → stop, report,
   no further spend (≈ $0.5 lost). Also confirm the letters read on the human labels (pixel AUC ≥ 0.98 on
   the w035 native-9 labels upsampled to the 2.399 canvas; the labels are the plane the 0.9991 control used).
2. **C2 modality control — w035 B-arm.** Same model, same window, on the w035 L1 SV (2.258 µm/px).
   Battery (PROTOCOL_V2, ds8) on it: the pre-registered requirement from PREREG_F is *all four map-internal
   gates* on the control's B arm. Full coverage removes the strip's gate_cycles censoring; if the
   control's B arm still fails, arm B is declared non-informative for this model and the w059 B result is
   reported but cannot flag.
3. **w059 arm B, forward and reverse.** Same env, `FORCE_REVERSE=true` for the second pass. Outputs:
   full-res PNG (`OUTPUT_PATH`), ds8 for the battery, fwd/rev Pearson r on the common support (gate 5 —
   the gate Track F could not compute).
4. **Battery (local, unchanged code)** on w059 B fwd (ds8), with A's published map re-scored beside it:
   flag rule from PREREG_F — all four map-internal gates on BOTH acquisitions, plus gate 5 (|fwd/rev r|
   < 0.20 on arm B) now that it is computable. Escalation only ("region worth human inspection"), never
   letter language; any flag → ≥ 1000-permutation rerun + human eyes + Discord to the team.

Pre-stated readings: (i) C0 fails → the 2 µm contract is not reproduced; Bet B's C1/C2a wait. (ii) C0
passes, w059 B passes 4/4 + gate 5, A already 4/4 → **w059 clears the two-scanner rule** → escalation.
(iii) C0 passes, w059 B fails significance with full coverage (the control's B passing) → the A-arm
ruling signal is not confirmed on the second scanner; Track F's lead is downgraded, recorded, closed.

## Cost

Streaming reads: w035 SVs 50 + 60 GB raw, w059 SVs 119 + 144 GB raw (uncompressed zarr over S3; the
inference streams only the 62-layer window ≈ 57 % of each, chunked). Inference at tile 256 / stride
128: w059 L1 canvas 1.24 Gpx ≈ 76 k tiles, ×2 directions; w035 ≈ 0.5 Gpx each. Estimate 1.5–2.5 h on a
5090 ($1–2) + the C0 sweep contingency; **cap $4**, guard deadline 4 h, `--fetch-dirs` for the PNGs.

## Open before launch

- Which pyramid level and layer window the published A-arm run used (C0 decides empirically).
- Whether `optimized_inference` runs on the villa-pin image or needs its own `requirements.txt`
  (Lightning 2.0.9 checkpoint; the README's Docker image is the reference).
- Whether the canvas of the L1 SV matches the 1.129 µm tifxyz at 1/2 scale (for the battery's px/mm).
