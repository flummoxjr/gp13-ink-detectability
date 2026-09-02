# Discord draft — First Letters rules questions (for Ben to post in his own words)

_Context for Ben: these five questions gate Bets B and E of the year plan (RESEARCH_PLAN_2026-27.md
appendix). Post in the prize-rules / first-letters channel. Keep it short; the organizers answer fast
and publicly. Do not mention the 500p2a numbers._

---

Hi all — a few eligibility questions before we sink GPU time into the wrong approach. All of these are
about the First Letters rules for the thirteen ~9 µm scrolls.

1. **Pseudo-labels from a higher-resolution scan of the same scroll.** PHerc. 1203 has a 2.4 µm band
   alongside its 9.36 µm volume. If a 2 µm-pipeline model reads faint text in that band, may
   *model-generated* labels derived from it be used to train a model whose predictions on the 9.36 µm
   volume back a First Letters claim on 1203? The Grand Prize text says higher-resolution data of the
   submitted scroll cannot be used; does that clause apply to First Letters, and does it cover
   labels derived from it (as opposed to the scan itself)?

2. If the answer to (1) is no: may a model trained on such 1203-derived pseudo-labels be used for
   claims on the *other* eligible scrolls (whose claims derive nothing from their own higher-res scans)?

3. **Bootstrapping on the target scroll.** Are model-generated pseudo-labels on the target scroll
   itself — no human character annotation at any stage, the PHerc. 1667 iteration-0 playbook —
   allowed as training data for the claim model?

4. **Window size.** The guidance says model windows of at most 0.5 × 0.5 mm. Does that mean the input
   patch or the receptive field? The recommended `ink_9um` model takes 128 px patches, which is 1.2 mm
   at 9.36 µm.

5. **Legacy labels.** Please confirm the 2023 Scroll 1 human ink labels (7.91 µm, 54 keV) and the
   PHerc. 0172 labels are unrestricted training data for claims on other scrolls. We assume yes since
   they are public, but want it on record.

Thanks — happy to write any of these up as a doc PR if that helps future teams.
