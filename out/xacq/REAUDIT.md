# Cross-acquisition null re-audit (2026-09-03)

**Question.** Our corpus-wide confirmation lift is median L(0.99) = 25.1 across 112 verified-independent
acquisition pairs (nulls ~1). A pixel-level figure of 14.2x was reported on Discord. Is our number an
artefact of a scoring convention: the mask, the resolution, or the registration?

**Method.** Three addenda over all 112 pairs, each importing the frozen harness (`xacq_score.py`,
xacq-1.1: registration, `r_and_L`, constants) unchanged and varying exactly one convention:

| addendum | script | what varies | outputs |
|---|---|---|---|
| mask | `xacq_reaudit.py` | joint-valid (frozen) vs sheet footprint (11-px closing, holes < 64x64 filled); the roll null rolls B's FOOTPRINT with B's calls | `reaudit_masks.{json,md}` |
| resolution | `xacq_scale.py` | block-mean ds8 (as shipped) -> ds16 -> ds32 -> ds64; roll null at the same scale | `reaudit_scale.{json,md}` |
| registration | `xacq_shift.py` | B deliberately mis-shifted 0..4 ds8 px (0..32 native px) after phase registration | `reaudit_shift.{json,md}` |

**Result (all 112, medians).**

| convention | r | L99 | L99 roll null | pairs L99 >= 5 |
|---|---|---|---|---|
| joint-valid mask (frozen; reproduces shipped) | 0.559 | 25.09 | 1.11 | 103 |
| sheet footprint, footprint rolled with calls | 0.558 | 25.12 | 1.11 | 103 |
| ds16 / ds32 / ds64 | 0.561 / 0.566 / 0.580 | 24.69 / 25.03 / 26.61 | 1.10 / 1.07 / 1.04 | 103 / 103 / 103 |
| mis-shift 8 / 16 / 24 / 32 native px | 0.558 / 0.546 / 0.534 / 0.519 | 25.21 / 25.20 / 24.77 / 24.04 | (unchanged) | 103 / 103 / 103 / 104 |

Main stratum (75) and PHercParis4 (37) behave the same (25.6 / 23.6 at baseline; every variant within 3 units).
The footprint adds essentially no pixels (median zero-fraction 0.000 in A, 0.002 in B): the model does not
leave exact zeros inside the sheet, so a "sheet mask" and the joint-valid mask are the same support here.

**Reading.**
1. The 25x lift is not a mask convention, not a resolution convention, and not a registration artefact: it is
   invariant to all three within ~1-2 units, and survives a 32-native-px misregistration, i.e. it is a
   stroke/letter-scale agreement, not a pixel-coincidence effect.
2. The 14.2x figure is therefore not reproducible on these maps under any convention we can vary. Our
   L(0.98) is 17.1 and L(0.95) 9.4, so a quantile near 0.975 - or a different pair set / model outputs -
   would produce ~14. Either way the ordering "lift >> null" is unchanged.
3. The pre-registered kill floor (median L99 < 5) is cleared under every convention tested (minimum
   observed median 22.4, Paris4 at ds32); the survive-both-nulls count is 100-101 / 112 under both masks.

Nothing here touches ink readability: this is agreement between two scans of the same sheet, which the
addendum in `report/ADDENDUM.md` already frames as confirmation of surface signal, not of letters.
