"""INVESTIGATION B, part 2: how fast does ink detectability collapse when the
17-slice model window is displaced along the surface normal?

Control = PHerc0139 w035, the ONE segment where we have (a) human-verified Greek
letters and (b) a published multi-slice surface volume.

  D:\vesuvius-data\trackD\w035_surf.npy   (28, 5820, 5240) uint8  surface volume
  D:\vesuvius-data\trackD\w035_ink.npy    human ink labels, populated on slice 14 only
  D:\vesuvius-data\trackD\w035_sup.npy    supervision mask,        slice 14 only

infer.py center-crops 28 -> 17 source slices (indices 6..22), i.e. the model window
is centred exactly on slice 14 = the label plane.  Here we slide that 17-slice
window to centres c = 8..19 and measure what a detector could still see.

Method (deliberately tile-local, to inherit the K1-v2 lessons):
  * 128x128 tiles = the model's Y/X crop. All statistics are computed WITHIN a tile
    and then aggregated across tiles, so global shading gradients cannot leak in.
  * AUC inside a tile is invariant to any per-tile affine rescaling -> the numbers
    below are exactly what a per-crop-normalised model (ink_9um uses robust_mad)
    would see from a linear depth projection.
  * Matched-filter weights are FIT on a random half of the tiles and EVALUATED on
    the held-out half (no circularity).
  * Shuffle control: ink mask rolled by half a tile -> AUC must return to 0.5.

Two filter families bracket the model:
  ANCHORED       weights indexed by ABSOLUTE depth -> a detector that finds the ink
                 wherever it sits inside the window (pure truncation loss).
  WINDOW-RELATIVE weights indexed by POSITION IN THE WINDOW -> a detector whose depth
                 prior is tied to the window centre (what a trained 3D net actually
                 has). This is the realistic case.
"""
import json
import os

import numpy as np

CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\hunt"
os.makedirs(OUT, exist_ok=True)

TILE = 128
DEPTH_WIN = 17          # model crop_z
LABEL_SLICE = 14        # where the human labels live == centre of the model's window
MIN_INK = 150
MIN_BG = 150
rng = np.random.default_rng(0)

print("loading...")
surf = np.load(os.path.join(CACHE, "w035_surf.npy"))          # (28,H,W) uint8
ink2d = np.load(os.path.join(CACHE, "w035_ink.npy"), mmap_mode="r")[LABEL_SLICE] > 0
sup2d = np.load(os.path.join(CACHE, "w035_sup.npy"), mmap_mode="r")[LABEL_SLICE] > 0
D, H, W = surf.shape
print(f"surf {surf.shape}  ink px {int(ink2d.sum())}  sup px {int(sup2d.sum())}")

# ---------------------------------------------------------------- tile inventory
tiles = []
for y0 in range(0, H - TILE + 1, TILE):
    for x0 in range(0, W - TILE + 1, TILE):
        s = sup2d[y0:y0 + TILE, x0:x0 + TILE]
        if not s.any():
            continue
        i = ink2d[y0:y0 + TILE, x0:x0 + TILE] & s
        b = s & ~i
        if i.sum() >= MIN_INK and b.sum() >= MIN_BG:
            tiles.append((y0, x0))
tiles = np.array(tiles)
print(f"{len(tiles)} qualifying 128px tiles (>= {MIN_INK} ink px and >= {MIN_BG} bg px)")

perm = rng.permutation(len(tiles))
fit_idx, eval_idx = perm[: len(tiles) // 2], perm[len(tiles) // 2:]
print(f"fit tiles {len(fit_idx)}  eval tiles {len(eval_idx)}")


def tile_data(t):
    y0, x0 = tiles[t]
    blk = surf[:, y0:y0 + TILE, x0:x0 + TILE].astype(np.float32)
    s = sup2d[y0:y0 + TILE, x0:x0 + TILE]
    i = ink2d[y0:y0 + TILE, x0:x0 + TILE] & s
    b = s & ~i
    # a valid-data guard: drop pixels where the render has holes anywhere in depth
    good = (blk > 0).all(axis=0)
    return blk, i & good, b & good


def auc(pos, neg):
    """Mann-Whitney AUC with tie handling."""
    n1, n2 = len(pos), len(neg)
    if n1 < 20 or n2 < 20:
        return np.nan
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(len(allv), np.float64)
    sv = allv[order]
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=np.float64)
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = r[i:j + 1].mean()
        i = j + 1
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


# ------------------------------------------------- pass 1: depth contrast profile
print("\npass 1: per-slice depth profile (fit tiles)")
mu_ink = np.zeros(D); mu_bg = np.zeros(D); var_bg = np.zeros(D); nT = 0
for t in fit_idx:
    blk, i, b = tile_data(t)
    if i.sum() < MIN_INK or b.sum() < MIN_BG:
        continue
    mu_ink += blk[:, i].mean(axis=1)
    mu_bg += blk[:, b].mean(axis=1)
    var_bg += blk[:, b].var(axis=1)
    nT += 1
mu_ink /= nT; mu_bg /= nT; var_bg /= nT
sd_bg = np.sqrt(var_bg)
contrast = mu_ink - mu_bg                    # signed DN contrast, tile-local
dprime = contrast / np.maximum(sd_bg, 1e-6)  # effect size per slice
print(f"  fit tiles used: {nT}")
print("  depth   mu_bg   mu_ink  contrast  d'")
for d in range(D):
    mark = "  <-- LABEL PLANE" if d == LABEL_SLICE else ""
    print(f"   {d:2d}  {mu_bg[d]:7.2f} {mu_ink[d]:7.2f}  {contrast[d]:+7.3f} {dprime[d]:+6.3f}{mark}")

peak = int(np.argmax(np.abs(dprime)))
half = np.abs(dprime).max() / 2
above = np.where(np.abs(dprime) >= half)[0]
print(f"  |d'| peaks at slice {peak} (d'={dprime[peak]:+.3f});"
      f" FWHM slices {above.min()}..{above.max()} (width {above.max()-above.min()+1})")
print(f"  bg-intensity landmark: argmax mu_bg = slice {int(np.argmax(mu_bg))}, "
      f"argmin mu_bg = slice {int(np.argmin(mu_bg))}")

# matched filter, zero-mean so it is invariant to a per-tile DC offset
w_full = dprime - dprime.mean()
w_full /= np.linalg.norm(w_full)

# ------------------------------------------------------- pass 2: window sweep
centres = list(range(8, 20))          # 17-slice windows fully inside 28 slices
print(f"\npass 2: window sweep, centres {centres[0]}..{centres[-1]} "
      f"(offset {centres[0]-LABEL_SLICE:+d}..{centres[-1]-LABEL_SLICE:+d} vox from the label plane)")

# window-relative filter: the aligned window's weight vector, re-anchored to the
# window centre. Aligned window = slices 6..22, so wr[k] applies to slice c-8+k.
wr = w_full[LABEL_SLICE - 8: LABEL_SLICE + 9].copy()
wr = wr - wr.mean(); wr /= np.linalg.norm(wr)

stats = {c: {k: [] for k in ("mean", "anch", "wrel", "wrel_shuf", "fr_r", "std")}
         for c in centres}
single = {d: [] for d in range(D)}

for n, t in enumerate(eval_idx):
    blk, i, b = tile_data(t)
    if i.sum() < MIN_INK or b.sum() < MIN_BG:
        continue
    # rolled ink mask -> shuffle control with the same spatial statistics
    ish = np.roll(np.roll(i, TILE // 2, 0), TILE // 2, 1) & b
    for d in range(D):
        single[d].append(auc(blk[d][i], blk[d][b]))
    for c in centres:
        lo, hi = c - 8, c + 9
        win = blk[lo:hi]                              # (17,h,w)
        # 1) plain depth mean
        m = win.mean(axis=0)
        stats[c]["mean"].append(auc(m[i], m[b]))
        # 2) anchored matched filter (weights at absolute depth, restricted to window)
        wa = w_full[lo:hi].copy(); wa -= wa.mean()
        nrm = np.linalg.norm(wa)
        ra = np.tensordot(wa / (nrm if nrm > 0 else 1), win, axes=(0, 0))
        stats[c]["anch"].append(auc(ra[i], ra[b]))
        # 3) window-relative matched filter (the realistic, depth-prior-tied case)
        rw = np.tensordot(wr, win, axes=(0, 0))
        stats[c]["wrel"].append(auc(rw[i], rw[b]))
        if ish.sum() >= MIN_INK:
            stats[c]["wrel_shuf"].append(auc(rw[ish], rw[b & ~ish]))
        # 4) forward/reverse symmetry proxy: same filter on depth-reversed input
        rr_ = np.tensordot(wr[::-1], win, axes=(0, 0))
        m2 = b | i
        if m2.sum() > 500:
            a1, a2 = rw[m2], rr_[m2]
            if a1.std() > 1e-6 and a2.std() > 1e-6:
                stats[c]["fr_r"].append(float(np.corrcoef(a1, a2)[0, 1]))
        stats[c]["std"].append(float(win.std()))
    if (n + 1) % 100 == 0:
        print(f"  {n+1}/{len(eval_idx)} eval tiles", flush=True)

print("\n=== per-slice tile-paired AUC (eval tiles, median) ===")
sing = {d: float(np.nanmedian(single[d])) for d in range(D)}
for d in range(D):
    mark = "  <-- LABEL PLANE" if d == LABEL_SLICE else ""
    print(f"   slice {d:2d}  AUC {sing[d]:.4f}{mark}")

print("\n=== 17-slice window sweep (median tile AUC over held-out tiles) ===")
print(" centre  offset   meanproj  anchored  win-relative  shuffle   fwd/rev r")
rows = {}
for c in centres:
    g = lambda k: float(np.nanmedian(stats[c][k])) if len(stats[c][k]) else float("nan")
    rows[c] = dict(offset=c - LABEL_SLICE, mean=g("mean"), anch=g("anch"),
                   wrel=g("wrel"), shuf=g("wrel_shuf"), fr_r=g("fr_r"),
                   n=len(stats[c]["wrel"]))
    r = rows[c]
    print(f"   {c:3d}   {r['offset']:+3d}     {r['mean']:.4f}    {r['anch']:.4f}"
          f"      {r['wrel']:.4f}     {r['shuf']:.4f}    {r['fr_r']:+.3f}")

# tolerance budget: |offset| where win-relative AUC falls to half of its aligned excess
aligned = rows[LABEL_SLICE]["wrel"]
excess0 = aligned - 0.5
print(f"\naligned (centre {LABEL_SLICE}) window-relative AUC = {aligned:.4f} "
       f"(excess over chance {excess0:+.4f})")
for c in centres:
    e = rows[c]["wrel"] - 0.5
    rows[c]["excess_retained"] = float(e / excess0) if excess0 else float("nan")
    print(f"  offset {rows[c]['offset']:+3d}: excess {e:+.4f}  "
          f"retained {100*e/excess0:5.1f}%")

json.dump(dict(
    tiles=int(len(tiles)), fit=int(len(fit_idx)), eval=int(len(eval_idx)),
    label_slice=LABEL_SLICE, depth_win=DEPTH_WIN,
    mu_bg=[round(float(v), 3) for v in mu_bg],
    mu_ink=[round(float(v), 3) for v in mu_ink],
    contrast=[round(float(v), 4) for v in contrast],
    dprime=[round(float(v), 4) for v in dprime],
    single_slice_auc={str(d): round(sing[d], 4) for d in range(D)},
    windows={str(c): {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in rows[c].items()} for c in centres},
), open(os.path.join(OUT, "depth_offset_control.json"), "w"), indent=1)
print("\nwrote", os.path.join(OUT, "depth_offset_control.json"))
