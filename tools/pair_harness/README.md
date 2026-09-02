# facing_pairs -- the facing-pair (fusion-site) harness from villa issue #191

This is the instrument behind the fusion numbers we quoted in the #191 thread,
packaged so you can run it on your own volumes -- in particular on your twin
dataset where you hold exact instance truth. It is a port of our internal
script `extract_fusion_sites.py` (included verbatim under `original/` for
diffing) with the hard-coded paths removed, arbitrary volume shapes allowed,
and two additions for exact-truth scoring: `--gt-instance` and
`--log-rejected`. On identical inputs the port and the original produce
byte-identical output rows; see "Equivalence proof" below.

## What it measures

Given a labelled sheet volume and one or more probability maps on the same
grid, it finds sites where two DISTINCT label sheets pass close to each other
(facing-pair sites), walks the straight line between the two facing surface
voxels, and records the shape of each probability map along that line.
Per site and per model, the question is: does the model resolve the two sheets
as two peaks with a dip between them, or does it smear them into one?

Readout at a threshold t (we report t = 0.5, and sweep):

    detected = (e1 >= t) and (e2 >= t)    both sheet crossings supra-threshold
    fused    = detected and (dip >= t)    no sub-threshold dip -> ONE peak
    split    = detected and (dip <  t)    dip drops below t    -> TWO peaks

Headline number: fused / detected.

## Package contents

    facing_pairs.py                  the instrument (single file, CLI)
    self_test.py                     synthetic tests with known answers
    verify_against_original.py       equivalence check against original/
    verify_report.json               result of that check on our machine
    original/extract_fusion_sites.py the unmodified original, for diffing
    heldout_cases_ds059_val.txt      the 171 held-out val case ids (see below)
    requirements.txt                 pinned versions we ran with
    README.md                        this file

## Quick start

    pip install -r requirements.txt        # numpy + scipy only
    python self_test.py                    # ~10 s, must print SELF-TEST PASS

Single case (arrays as .npy, or as keys inside an .npz given as `path.npz:key`):

    python facing_pairs.py --out results --case-id twin01 \
        --label case.npz:label \
        --prob  v0=case.npz:prob_v0 --prob v1=case.npz:prob_v1 \
        --gt-instance case.npz:instances --log-rejected

Batch mode takes `--manifest cases.json`, a JSON list of entries:

    [{"id": "twin01",
      "label": "d/twin01.npz:label",
      "probs": {"v0": "d/twin01.npz:prob_v0", "v1": "d/twin01.npz:prob_v1"},
      "gt_instance": "d/twin01.npz:instances"},
     ...]

Labels are any integer/bool 3-D array (nonzero = sheet); probability maps are
float arrays in [0, 1] of identical shape. Every case must carry the same set
of model names. Volumes must exceed 10 voxels per axis (a 5-voxel border is
excluded from site selection).

## Site definition (deterministic, seed 42)

1. **Reflect-pair candidates** from the EDT nearest-index field: for a
   background voxel g, p1 = nearest label voxel; reflect q = 2g - p1;
   p2 = nearest label voxel to q. Keep when the rays from g oppose,
   cos(p1-g, p2-g) < -0.3. Deduplicate on the unordered (p1, p2) pair.
   Tight sites: EDT in [0.9, 3.05], d12 = |p2-p1| in [2, 6] voxels.
   Loose sites: EDT in [3.9, 7.05], d12 in [8, 14] voxels (a distance null --
   sheets that far apart must not fuse; on by default, `--no-loose` disables).
2. **Clean interior**: no label voxel strictly between p1 and p2.
3. **Local two-sheet guard**: p1 and p2 must fall in DIFFERENT 26-connected
   components of the label volume inside a window of radius
   max(4, ceil(1.5 * d12)) centred on the midpoint.
4. **Spatial thinning**: greedy on midpoints, >= 3 voxels apart (4 for loose),
   lexicographic order.
5. **Seeded cap**: at most 300 tight / 100 loose sites per case,
   rng = default_rng([seed, case_index]).

Per site the profile runs from s = -1.0 to d12 + 1.0 in 0.25-voxel steps
(trilinear). Columns per model: `e1`/`e2` (max within 0.75 voxels of each
end), `dip` (min over the gap interior, margin-inset), `mx` (profile max),
`pk4/pk5/pk6` (find_peaks count at height 0.4/0.5/0.6, prominence 0.05).
One reimplementation trap, called out because it bit us: the peak counts have
a plateau fallback -- if find_peaks returns 0 but the profile max is >= t, the
count is recorded as 1 (a flat profile above t is one blob, not zero). Applied
identically to every model.

## Rejection-log semantics (`facing_pairs_case_diag.csv`)

One row per case, tight sites only, binned by d12 (2-3, 3-4, 4-5, 5-6.01 vox):

    n_pairs_tight_raw   deduplicated stage-1 candidates
    clean_<bin>         pairs surviving stage 2 (clean interior), per bin
    ccpass_<bin>        of those, pairs surviving the stage-3 CC guard
    n_tight_final       after thinning + cap        (same for n_loose_final)

`ccpass_<bin> / clean_<bin>` is the **local-CC survival rate**: the fraction of
clean facing pairs whose two endpoints the *labels themselves* keep as two
separate components locally. A stage-3 rejection means the label volume merges
the two sheets somewhere inside the local window -- with skeleton-thin labels
this is exactly where annotation merges tight contacts. The site CSV only ever
contains survivors, so fusion rates are conditioned on label separability;
that conditioning is why the rejection log matters and is the main thing we
want scored against exact truth (below).

## Known numbers: Dataset059 held-out val (measured with this pipeline)

Corpus: 171 held-out validation cases (deterministic 90/10 split by
sha1(case) of Dataset059; 1,583 train / 171 val; ids in
`heldout_cases_ds059_val.txt`; 160^3 central crops, skeleton-thin surface
labels). Local-CC survival through the stage-3 guard, summed over all 171
cases:

    d12 bin     clean    ccpass   survival
    2-3 vox      9319        78      0.8%   (0.84%)
    3-4 vox     10661       558      5.2%   (5.23%)
    4-5 vox     11504      1313     11.4%   (11.41%)
    5-6 vox    100936     94859     94.0%   (93.98%)

i.e. below ~4 voxels of separation, our labels merge nearly every real
contact, so almost no tight site survives to be measured -- consistent with
the fused-peak numbers discussed in the thread.

One correction to our own earlier wording: we had once said "171" and a later
results table said "n = 169". Both are traceable: the held-out split IS 171
cases, and the facing-pair extraction above ran on all 171. The 169 came from
a later training cycle (cycle 3) whose evaluation was missing two Scroll-5
cases (`s5_z6190_y3040_x3420`, `s5_z8500_y2090_x3230`); the logs we have do
not record why they were dropped there. For this instrument the number is
171; those two cases contribute 0 tight sites either way.

## What we would like scored: stage-3 rejections vs your exact truth

You hold a twin dataset with exact instance truth. The one thing this harness
cannot know by itself is whether a stage-3 rejection was a real two-sheet
contact that our style of labels merged, or genuinely one folded sheet. Your
truth can settle that pair by pair:

1. Run with your reconstruction-style labels as `--label`, your models as
   `--prob`, your exact instance volume as `--gt-instance`, and
   `--log-rejected`.
2. `facing_pairs_cc_rejected.csv` then holds every stage-2-surviving pair the
   CC guard rejected, with `gt_id1`, `gt_id2`, `gt_diff_sheet` looked up at
   p1/p2 in your truth. Per d12 bin, report the fraction of rejected pairs
   with `gt_diff_sheet == 1` -- that is the rate at which labels merge real
   contacts, as a function of gap. (On our data we can only infer it; on
   yours it is measurable.)
3. As the complementary check, the same GT columns appear on accepted sites in
   `facing_pairs_sites.csv`; there `gt_diff_sheet` should be ~always 1. Its
   failure rate is the guard's false-accept rate.

Practical note: GT ids are read at the label-surface voxels p1/p2. If your
truth volume does not mark those exact voxels (e.g. truth is thinner than the
labels, or offset), ids come back 0 -- in that case dilate your truth ids to
cover the label voxels first (nearest-id dilation), and please say so when
reporting. Rows with a 0 id should be reported as "unresolved", not folded
into either fraction.

Everything else you might want is in `facing_pairs_sites.csv`: fused/detected
at any threshold you like (compute from e1/e2/dip), the loose-pair distance
null (fusion must collapse at d12 in [8, 14]; if it does not, the measurement
is picking up something other than sheet separation), and `--mismatch` in
batch mode as a shuffle null (each case's site geometry re-evaluated on the
previous case's probability maps; xe1/xe2/xdip columns, blank for the first
case).

## Determinism and seeds

Default `--seed 42 --seed-mode index` reproduces our Dataset059 run exactly
(rng = default_rng([42, case_index]) -- depends on case order, which for us
was sorted filenames). `--seed-mode caseid` derives the rng from
sha256(case_id) instead and is stable under reordering; use it if your case
ordering is not fixed. Everything before the caps is deterministic; the caps
only subsample.

## Equivalence proof against the original

`verify_against_original.py` builds synthetic volumes that exercise every
stage (a tight pair, a merged pair with a connecting wall, a loose pair, 4
noisy models, 3 cases), runs the unmodified original (from `original/`,
sandboxed) and this port on identical inputs, and requires cell-for-cell
identical CSV output plus nonzero counts at every funnel stage.
`verify_report.json` is the run from our machine: 1200/1200 site rows and all
diagnostic rows identical, verdict PASS, original sha256
`209b0141e10768642d537fb88d4324c7f7c6fa248c5896c9608d6c54e4027ab9`.
Reproduce with:

    python verify_against_original.py

We additionally re-ran the port on the first 3 real Dataset059 val cases and
compared against the archived output of the original's full 171-case run
(seed and case order agree on a prefix): all 342 site rows and all diagnostic
rows identical, cell for cell.

Porting notes (the only intended behaviour differences): the original assumed
cubic volumes (it compared all three axes against `shape[0]`); the port does
per-axis bounds, so non-cubic volumes now work -- on cubic input the two are
identical, which is what the verifier proves. `--gt-instance` appends three
columns to the sites CSV; `--log-rejected` adds a third output file; both are
off by default and change nothing else. Output filenames gained a
`facing_pairs_` prefix.

## Pinned versions

We ran with Python 3.12.10, numpy 2.5.2, scipy 1.18.0 (see
`requirements.txt`). The instrument only uses numpy, scipy.ndimage
(EDT/labeling/map_coordinates) and scipy.signal.find_peaks; nearby versions
should be fine, but if your survival numbers differ from ours, match the
pinned versions before anything else -- EDT nearest-index tie-breaking is the
one place library internals could plausibly matter.

Questions, or anything ambiguous while running it: reply in the #191 thread.
