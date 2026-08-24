# HUNT_PLAN — 13 days, $41, aimed at letters

_Written 2026-08-18. Inputs: investigations A–E (`geometry_compare.md`,
`depth_offset_plan.md`, `embedding_prospecting.md`, `two_micron_path.md`,
`grow_segments.md`). Every number below is measured or re-verified live today._

**Bottom line.** Two of the five routes are refuted and should not be funded as
discovery. The corpus null we have been treating as "the GP scrolls have no ink"
is actually "the three scrolls that happened to have meshes have no ink" — and
those are index ranks 8, 13 and 14. The six best-scanning GP scrolls have never
been looked at by anyone because no geometry exists on them. Route E builds that
geometry for ~$1 per scroll using the team's own tool. It is the highest-EV
action available and it runs tonight.

**But first: the budget is on fire.** See §0. Nothing else in this plan matters
until that is handled, and it is Ben's call.

---

## 0. THE BINDING CONSTRAINT (not compute — burn rate)

Measured live, `trackD/runpod/rp.py status`, 2026-08-18 03:5x UTC:

```
balance $131.65 | session spend $38.72 | burn $2.201/hr
 pod nopw1qunodm4wv  survey-0   RUNNING  $0.69/hr
 pod kdid9mdyqy44jc  c2-pod1c   RUNNING  $0.74/hr
 pod ylpu9luvgy71cf  c2-pod2    RUNNING  $0.74/hr
```

- Baseline $170.36 − balance $131.65 = **$38.72 spent of the $80 shared cap.**
- **$41.28 remains. Burn is $2.201/hr = $52.8/day.**
- **The cap is breached in ~18.7 hours if nothing is switched off.**

Thirteen days at the current rate is $687 — 16× the remaining cap. Every cost
estimate in the five investigations ($0.20 here, $1.90 there) is noise next to
this. The single highest-value action in this document is turning pods off.

I verified `survey-0` directly over SSH: **load average 0.11, zero python
processes, no job running.** The 80/80 corpus survey completed last night. It has
been idle-burning since.

It is, however, *fully provisioned*: `/workspace/villa`, `ink_9um` checkpoints at
`/workspace/ckpts/ink_9um/hybrid_3d2d-seed42/step-075000.pth`,
`render_tifxyz_sv.py` and `survey_segments.py` all present, 6.3 GB free. That is
~20 minutes of provisioning we do not have to repeat.

**Recommendation:** keep `survey-0` alive tonight only (it is Stage B for the
tonight experiment), then kill it. `c2-pod1c` and `c2-pod2` belong to the
parallel Track A session — **I have not touched them and will not.** Someone has
to decide their fate; see §6.

---

## 1. Route ranking by expected value

Value anchors: First Letters = $50K/scroll (Jun 25 2027). Monthly progress prize
= $1K–20K (next: Aug 31). Costs are marginal pod $ only; laptop time is free.

| # | Route | P(letters) | Cost to decide | Cost to complete | Status | Verdict |
|---|-------|-----------|----------------|------------------|--------|---------|
| **1** | **E — grow segments on unscreened high-SNR scrolls** | **0.15–0.22** (my discount of E's 0.20–0.30) | **$0.35** | $6–10 | all inputs published, tool verified, never run | **FUND — tonight** |
| **2** | **D — 2.4 µm canonical model on PHerc1203** | **0.06–0.10** | **$0.50** | $2 (stop after C2a) | registration SOLVED today; blocker moved to material | **FUND — after E gate** |
| 3 | C — dinovol embedding prospecting | 0.06–0.09, ~all in the 1203 2.4 µm arm | **$0** (local) | $3–10 | E0 passed on real weights | Run the $0 kill test only |
| 4 | B — depth-offset residual | 0.04 | **$0** (local) | $1.90 | motivating hypothesis refuted by its own sweep | Run the $0 gate to close it; do not fund the pod run |
| 5 | A — geometry fix | 0.02 | done | $0.20 | **REFUTED** — no geometric error left to correct | **DO NOT FUND** |

### Why E ranks first on evidence, not vibes

The three facts that moved it:

1. **The gap is real and I re-verified it on S3 today.** Listing
   `PHerc0813/` returns `photos/ representations/ volumes/` — **no `segments/`
   prefix.** Same for 0125/1545/0211/0191/0358. We screened ranks 8 (1203,
   SNR 87), 13 (0800, SNR 20) and 14 (1447, SNR 8.5). Ranks 1, 3, 4, 5, 6, 7 are
   untouched. PHerc0813 scores **159.6**, above the PHerc0139 calibrator (115.5)
   where `ink_9um` reads letters at AUC 0.9991 — and it is the same campaign
   (9.362 µm / 113 keV / BM18), so the model faces **no domain shift**.

2. **Investigations A and B independently destroyed the excuse for the null.**
   A measured GP mesh placement: worst displacement anywhere is PHerc0800 at
   +2.6 vox. B measured the model's tolerance: ±3 vox costs 2.4% of
   excess-over-chance, ±5 costs ~20%. So the corpus null is a statement about
   **the scrolls**, not our meshes. Combined with the fact that two of the three
   screened scrolls are in the degraded tier (SNR 20 and 8.5), only PHerc1203 was
   ever a fair test. "80 segments, zero hits" is much weaker than it sounded.

3. **The corpus was also smaller and dirtier than "80".** A and B agree
   independently: **66 unique meshes**, of which 23 are `z_dbg` debug dumps and
   **18 sit partly outside the reconstructed volume** (one at 91.1% of vertices
   on exactly-zero voxels). Some fraction of our null was measuring nothing at
   all.

Against that, the honest brake: **our index measures the released volume's
structural SNR, not ink content.** A scroll can scan beautifully and hold no
recoverable ink — carbon-ink contrast depends on ink chemistry and degradation,
which the index cannot see. Rank 1 tells us where we can best *look*, not what is
*there*. That is why I discount E's own 0.20–0.30 to 0.15–0.22 (§5).

---

## 2. Shared prerequisites — what gates what

This is the part that changes the ordering, and it runs the opposite way to
intuition.

```
A (geometry quality)  ──►  gated BOTH B and E
      │
      ├─ result: "GP meshes are fine, ±2.6 vox worst case"
      │
      ├──► B (depth-offset):  rationale DESTROYED.  If the meshes are
      │                       well-placed, a mis-placed window explains nothing.
      │                       B's own sweep then confirmed ±6 vox is survivable.
      │                       → B collapses to a $0 closing gate.
      │
      └──► E (grow segments): rationale STRENGTHENED.  If geometry is fine and
                              the scrolls we screened are still null, the null
                              is about those scrolls — so go to better scrolls.
                              → E is promoted.
```

Three more couplings that set the order:

- **E subsumes B's remaining diagnostic.** B's live residual is "a mesh sitting
  >6 vox off the recto would be invisible to A's centroid metric." Fresh patches
  grown on new data test that for free: record forward-vs-z-reversed `r` on them.
  Control = 0.076; corpus = 0.22–0.91. If our own fresh patches reproduce the
  corpus symmetry, the pipeline is implicated; if they land near 0.076, the
  scrolls are. **Costs nothing extra and resolves B's residual as a by-product.**
  → Do not fund B's $1.90 pod run; let E answer it.

- **D is a prerequisite for C's only valuable arm.** C's P(letters) is
  "essentially all in E3" — the PHerc1203 2.403 µm band. D *derived the missing
  9.362↔2.403 µm transform today* and validated the method against PHerc0139's
  published matrices (49–129 µm error). C-E3 needs that transform. → **D before
  C-E3, always.** Running C-E3 first would mean re-deriving D's work.

- **E and D share the render + `ink_9um` pipeline**, already warm on `survey-0`.
  Running E tonight keeps that pod earning instead of idling.

**Resulting order: E → D → (C-E1 kill test, free, in parallel) → B/A gates as
$0 cleanup.**

---

## 3. TONIGHT — the single highest-EV experiment

**Grow the first surface patches ever made on PHerc0813.**
~30 min wall-clock, **~$0.35**, one pod. Fully specified below; no design left.

### Why this one

It is the only experiment in the portfolio that opens data nobody has looked at.
Every input is published and verified. It is the cheapest item on the list. And
it is the gate for a $6–10 spend across six scrolls — so it is worth knowing
tonight whether the tool runs at all.

### Verified live today (re-checked, not taken on faith)

```
PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr              [CT — Stage B]
PHerc0813/representations/predictions/surfaces/
    20250821151723-surface-20260413222639-surface-m7-L0-th0.2.zarr           [TRACER INPUT]
    20250821151723-surface-20260413222639-surface-m7-L0-th0.2.normal-grids   [optional]
```

Image pull, verified anonymously today: `ghcr.io/v2/.../manifests/edge` →
**HTTP 200**, `linux/amd64` manifest present, anonymous token OK. No GitHub
login needed.

### Stage A — launch (do this first)

Use `rp.py create_pod` with the image overridden. It already parameterises
`imageName`, so this is a one-argument change — **do not** try to stand up a CPU
pod type tonight; the GPU pod path is proven and 30 minutes of a $0.69/hr GPU
costs $0.35. Growth is CPU-bound; the idle GPU is the price of not debugging a
new API surface at midnight.

```python
import rp
pod = rp.create_pod(
    name="grow-0813",
    gpu_type="NVIDIA GeForce RTX 4090",     # any cheap type; GPU is unused
    image="ghcr.io/scrollprize/villa/volume-cartographer:edge",
    disk_gb=40,
)
```

Then on the pod:

```bash
S=https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0813/representations/predictions/surfaces
B=20250821151723-surface-20260413222639-surface-m7-L0-th0.2

mkdir -p /work/paths /work/cache /work/ngrid
printf '{"url":"%s/%s.normal-grids"}' "$S" "$B" > /work/ngrid/normal-grids-remote.json

cat > /work/seed.json <<'JSON'
{
  "mode": "seed",
  "generations": 75,
  "step_size": 20,
  "min_area_cm": 0.3,
  "thread_limit": 1,
  "use_cuda": false,
  "voxelsize": 9.362,
  "cache_size": 4e9,
  "cache_root": "/work/cache",
  "normal_grid_path": "/work/ngrid"
}
JSON
```

**Use the 8 verified on-sheet seeds, not random_seed.** They are in
`trackD/hunt/seeds_0813.json`, each confirmed to sit on a sheet inside a 24³
neighbourhood with sheet fraction 0.18–0.42. Explicit seeds remove the
rejection-sampler as a failure mode on a first run — keep `random_seed` for
scale-up once the tool is proven.

```bash
while read -r X Y Z; do
  vc_grow_seg_from_seed \
    --volume "$S/$B.zarr" \
    --target-dir /work/paths \
    --params /work/seed.json \
    --seed $X $Y $Z &
done <<'SEEDS'
3968 3970 14878
5761 4513 11744
4512 4511 11744
1664 4519 4705
2494 4512 4705
5408 4480 4704
4160 1592 14973
3328 1594 14980
SEEDS
wait
```

### Sanity gates — abort on the first that fails

| # | Expected stdout | Proves | If it fails |
|---|-----------------|--------|-------------|
| 1 | `zarr dataset size for scale group 0 [16993, 7947, 7947]` | s3/https + anonymous auth | retry with the `s3://` form; if still failing, the S3 reader is the blocker — stop |
| 2 | `Loaded normal grid level …` | remote normal-grid streaming | **delete `normal_grid_path` from seed.json and rerun** — it is optional, only 13/36 published segments used one |
| 3 | `Found seed location … value: 255` | seeder is on a sheet | seeds are stale; fall back to `random_seed` (omit `--seed`) |

### Go/no-go

- **GO** if **≥3 patches with `area_cm2 ≥ 2`** in their `meta.json`.
  Benchmark against the measured PHerc1203 production rate (**3.89 cm² / 121 s**,
  from 22 published `meta.json`). PHerc0813's sheets are *better* separated than
  1203's (8-on/20-off, 262 µm period vs 10/35, 421 µm), so ≥1.9 cm²/min is the
  expectation.
- **NO-GO** if 0 patches clear 2 cm². Then try **PHerc0125** (next-best sheet
  separation) once before concluding the tool needs a traced scroll. Do not spend
  on rendering.
- **Cost ceiling tonight: $0.50.** Kill the pod the moment you have the meta.json
  numbers — growth is minutes, not hours.

### Stage B — only if Stage A passes (tomorrow morning, ~$0.70)

Copy the tifxyz dirs to `survey-0` (a whole segment is **~280 KB**), then run the
existing, validated pipeline unchanged:

```bash
python /workspace/scripts/render_tifxyz_sv.py /workspace/data/<seg>.tifxyz \
  "s3://vesuvius-challenge-open-data/PHerc0813/volumes/20250821151723-9.362um-1.2m-113keV-masked.zarr" \
  /workspace/data/<seg>.sv.zarr --num-slices 21
# then infer seed42 forward + reverse, exactly as survey_segments.py does
```

**Record `fwd_rev_r` on every patch.** Control 0.076, corpus 0.22–0.91. This is
the free B-residual test described in §2 and it costs one line.

---

## 4. Week plan with decision gates

Assumes §0 is resolved and burn is back under control. Laptop work is $0 and runs
in parallel with everything.

### Day 0 (tonight)
- **§0 budget triage.** Ben decides on `c2-*`. Kill `survey-0` after Stage B.
- **E Stage A** on PHerc0813 (§3). $0.35.
- Kick off, unattended on the laptop, free: `hunt/depth_profile.py` over the
  remaining **56 unprofiled unique meshes** (B's closing gate, 1–2 h CPU).

### Day 1
- **E Stage B**: render + `ink_9um` fwd/rev + the 4-test verdict battery on the
  0813 patches. $0.70.
- **GATE E1 — the big one.** Any tripwire hit, or ruling-periodicity z above the
  corpus ceiling (+5.94), on a fresh 0813 patch → **stop everything else and
  verify.** That is the First Letters path.
- **GATE B-close.** Read `tile_peak_off_med_vox` from the 56-mesh sweep.
  - No mesh beyond |6| vox → **route B is dead.** Bank the $1.90, write the
    refutation. (Most likely outcome.)
  - ≥5 meshes beyond |6| vox → keep B alive but still behind E and D.
- **C-E1 kill test**, local GPU, $0: AUC of the dinovol ink prototype vs matched
  blank on w035 at 2.399 µm. **Gate: all three scorers ≤0.65 → route C dead,
  do not run E2 or E3.**

### Days 2–4
- **If E1 GO:** replicate across **PHerc0125, 1545, 0211, 0191, 0358** — same
  command, one changed prefix. $4–6. This is the highest-EV use of the remaining
  budget by a wide margin.
- **If E1 NULL:** that is a *much* stronger negative than the 80-segment screen,
  because it is on the best-scanning scroll with fresh, purpose-grown geometry
  and no domain shift. Record it and move to D.
- **D-C0 in parallel** ($0.50): reproduce the *published* w035 2.399 µm ink
  prediction from published inputs, sweeping 12 `START_LAYER` values.
  **GATE D0: pixel correlation with the published tif > 0.9 and letters legible.**
  If we cannot reproduce a published prediction from published inputs, **stop the
  entire 2.4 µm route** — nothing downstream would be interpretable. $0.50 to
  learn that is the best-priced gate in the plan.

### Days 5–8
- **D-C1** ($0, laptop): contrast-rank the 24.1 cm² of PHerc1203 in-band surface.
  Do **not** render all of it — D measured in-band contrast at 0.138 vs w035's
  0.397 (Mann-Whitney p=2.6e-11), with only ~15% of patches looking w035-like.
- **D-C2a** ($1.50): render + infer the best 3–4 cm² only, fwd+rev.
  **GATE D2: verdict battery + tripwire.** Null → stop; do not fund the full
  24.1 cm² sweep ($5) on 20 cm² of measured mush.
- Fix `list_segments.py` (`has_tifxyz` picks `versions/0`, the *first* growth
  generation, for 9 of 22 PHerc1203 segments → we screened **54.0 of 130.2 cm²,
  41%**). One line. Re-survey the missing 59% (~$1–2). This is cheap, and it is
  the one place our existing negative is genuinely incomplete.

### Days 9–13
- Whichever route survived its gate gets the remaining budget.
- If all null: consolidate the fallback deliverable (§7). Reserve **$5** and
  **2 full days** for this — it is the outcome to plan for, and it is worth real
  money.

---

## 5. Honest ceiling

**P(a defensible First Letters claim by Jun 25 2027) ≈ 12–18%.**

Decomposition, and where I disagree with the investigations:

- **E across six scrolls.** E reports 0.20–0.30. I discount to **0.15–0.22** for
  one reason it flagged itself but did not price: **failure is correlated across
  scrolls.** If GP-13 ink is simply not recoverable at 9 µm — which the 1667
  precedent and our own 66/66 null both permit — all six fail together. They are
  not six independent 10% draws. The index measures scan quality; it is blind to
  ink chemistry.
- **D (1203 2.4 µm).** ~0.06–0.10, and genuinely partly independent of E
  (different resolution, different physics, above the resolution wall). D moved
  its own number *down* today after measuring the merged-laminae contrast — the
  registration got easier and the material got worse, and those do not offset.
- **C-E3.** Overlaps D almost entirely (same data). Adds maybe +0.02 independent.
- **A and B.** ~0.02 each. Effectively dead.

Union with correlation ≈ **0.20–0.28 if the plan runs to completion.** It will
not: at $41 and 13 days we execute E on two or three scrolls, D's control, and
maybe D's high-contrast subset. **Hence 12–18%.**

**P(First Letters by Aug 31, 2026 — 13 days) ≈ 4–7%.**
That requires, in sequence: growth works, the scroll holds ink, `ink_9um` sees
it, the 4-test battery + tripwire pass, *and* the letters are legible enough for
a human expert to verify. Five conjunctive gates in under two weeks. The Aug 31
monthly prize should be planned around §7, not around letters.

**Calibration note.** This project has refuted three of its own false positives.
Every apparent signal so far has died under a null with numbers attached. The
prior should be that these routes die too. I would rather state 12–18% and be
pleasantly surprised.

---

## 6. Needs Ben's decision

1. **URGENT — the burn.** $2.201/hr against $41.28 of cap. **Breach in ~18.7
   hours.** `survey-0` is confirmed idle and is mine to kill (recommend: after
   tonight's Stage B). **`c2-pod1c` and `c2-pod2` are the parallel session's and
   I have not touched them.** Two sessions share one cap and cannot see each
   other. This needs an explicit split — e.g. $20/$20 — or the first session to
   spend wins by default. *This is the decision that determines whether any of
   the rest of the plan is executable.*

2. **Cap raise?** The plan as scoped is ~$13–15 of pod spend and fits in $41
   *only if the burn stops today*. If E1 passes on PHerc0813, the natural
   follow-on is $6–10 to sweep all six scrolls — that is the best money in the
   portfolio and it may want headroom above $80.

3. **Claims in Ben's voice.** Per villa's CONTRIBUTING.md, PRs must arise from a
   *human* using the tools on real data. Four items are queued and all need his
   name and his sign-off:
   - the ~10× exact SDPA head-dim fix (C, verified on released weights);
   - the empty-volume QC gate (A, `check_air.py`);
   - the PHerc1447 data-release defect report (18/66 unique meshes partly outside
     the volume);
   - the `list_segments.py` `versions/0` bug.

4. **Risk on any First Letters claim.** The bar is human-legible,
   expert-verifiable text — not a detector response. A premature claim would
   burn credibility that this project has spent three refutations building. My
   recommendation: no claim goes out without the full battery, the blank control,
   *and* a second pair of eyes.

5. **New public artifacts.** Route E produces the first-ever surface meshes on
   six scrolls. Releasing them is likely prize-positive on its own — but it is a
   publication decision, not mine.

---

## 7. Fallback deliverable if every route returns null

Plan for this outcome; it is the modal one, and it is worth real money. Nothing
here depends on finding letters, and **all of it already exists or is nearly
free**:

1. **"Measure Before You Hunt"** — the built report, upgraded with a materially
   stronger negative: not "80 segments" but *66 unique meshes, geometry verified
   in-tolerance, on scrolls ranked 8/13/14*, **plus** fresh purpose-grown
   geometry on the rank-1 scroll. That converts a weak null into a calibrated
   statement about the scrolls.
2. **The missing catalogue asset** — the PHerc1203 9.362↔2.403 µm transform, in
   the team's own JSON format, method-validated against PHerc0139's published
   matrices. The open-data release does not have this.
3. **Three tool contributions** — the exact ~10× SDPA speedup on an official
   release; the empty-volume QC gate; the `list_segments.py` fix.
4. **A data-release defect report** — 18 of 66 unique PHerc1447 meshes place part
   of their surface outside the reconstructed volume; one has 91.1% of vertices
   on exactly-zero voxels.
5. **Two model-card bug reports** — 128 expert tokens, not 256; a Usage example
   importing a module that does not exist.
6. **The measured tolerance curve** for `ink_9um` window placement (±3 vox → 97.6%
   of excess retained), which nobody has published.

Calibration: the Jul 2026 monthly prizes paid **$1K each** for zarr read fixes, an
ink-detection validation harness, and self-intersection detection; **$2.5K** for
patch-aggregation notebooks. The list above is broader than any single one of
those. **Realistic $2.5–10K, and it lands whether or not we find a letter.**
