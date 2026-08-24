"""INVESTIGATION B, part 2 (v2): can ANY offline detector see w035's ink, and if so
how fast does it die when the 17-slice window is displaced?

v1 (depth_offset_control.py) returned a flat null (AUC ~0.5 at every offset INCLUDING
the aligned one).  That is consistent with K1 (raw z-mean AUC on w035 = 0.4956), so the
null is a property of the intensity channel, not a bug.  Before any tolerance budget can
be quoted we must first find a detector whose ALIGNED AUC is well above chance --
otherwise the sweep is measuring noise.

This script:
  1. crops to the supervision bbox (cheap: ~2090 x 2346 x 28)
  2. characterises the label + the released ink_9um prediction
  3. searches a family of offline detectors for one that separates ink from bg
  4. only if one works, sweeps the 17-slice window centre and reports decay
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi
import tifffile

CACHE = r"D:\vesuvius-data\trackD"
PRED = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_w035\w035_seed42-075000.tif"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\hunt"
os.makedirs(OUT, exist_ok=True)

LABEL_SLICE = 14
DEPTH_WIN = 17
TILE = 128
rng = np.random.default_rng(0)


def auc(pos, neg, cap=200_000):
    n1, n2 = len(pos), len(neg)
    if n1 < 30 or n2 < 30:
        return np.nan
    if n1 > cap:
        pos = rng.choice(pos, cap, replace=False); n1 = cap
    if n2 > cap:
        neg = rng.choice(neg, cap, replace=False); n2 = cap
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(len(allv), np.float64)
    sv = allv[order]
    r = np.arange(1, len(allv) + 1, dtype=np.float64)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = r[i:j + 1].mean()
        i = j + 1
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


print("loading label plane ...")
ink_full = np.load(os.path.join(CACHE, "w035_ink.npy"), mmap_mode="r")[LABEL_SLICE] > 0
sup_full = np.load(os.path.join(CACHE, "w035_sup.npy"), mmap_mode="r")[LABEL_SLICE] > 0
ys, xs = np.where(sup_full)
y0, y1 = int(ys.min()), int(ys.max()) + 1
x0, x1 = int(xs.min()), int(xs.max()) + 1
print(f"supervision bbox y[{y0}:{y1}] x[{x0}:{x1}]  ({y1-y0} x {x1-x0})")

ink = ink_full[y0:y1, x0:x1]
sup = sup_full[y0:y1, x0:x1]
del ink_full, sup_full

print("loading surface volume crop ...")
surf_mm = np.load(os.path.join(CACHE, "w035_surf.npy"), mmap_mode="r")
surf = np.ascontiguousarray(surf_mm[:, y0:y1, x0:x1]).astype(np.float32)
D, H, W = surf.shape
del surf_mm
print(f"surf crop {surf.shape}  mem {surf.nbytes/1e9:.2f} GB")

good = (surf > 0).all(axis=0)
valid = sup & good
inkv = ink & valid
bgv = valid & ~ink
print(f"valid {int(valid.sum())}  ink {int(inkv.sum())}  bg {int(bgv.sum())}"
      f"  ink frac {inkv.sum()/max(valid.sum(),1):.3f}")

# ------------------------------------------------------------------ the model's map
pred = None
if os.path.exists(PRED):
    p = tifffile.imread(PRED).astype(np.float32)
    print(f"pred tif {p.shape} range {p.min():.0f}..{p.max():.0f}")
    if p.shape[0] >= y1 and p.shape[1] >= x1:
        pred = p[y0:y1, x0:x1]
        gm = auc(pred[inkv], pred[bgv])
        print(f"  ink_9um GLOBAL pixel AUC on this crop: {gm:.4f}")
        # per-tile (removes any global gradient advantage) -- the fair comparison
        tl = []
        for a in range(0, H - TILE + 1, TILE):
            for b in range(0, W - TILE + 1, TILE):
                i_ = inkv[a:a+TILE, b:b+TILE]; g_ = bgv[a:a+TILE, b:b+TILE]
                if i_.sum() >= 150 and g_.sum() >= 150:
                    tl.append(auc(pred[a:a+TILE, b:b+TILE][i_],
                                  pred[a:a+TILE, b:b+TILE][g_]))
        print(f"  ink_9um PER-TILE median AUC: {np.nanmedian(tl):.4f}  (n={len(tl)} tiles)")
    del p

# ------------------------------------------------------------------ tile inventory
tiles = []
for a in range(0, H - TILE + 1, TILE // 2):
    for b in range(0, W - TILE + 1, TILE // 2):
        i_ = inkv[a:a+TILE, b:b+TILE]; g_ = bgv[a:a+TILE, b:b+TILE]
        if i_.sum() >= 300 and g_.sum() >= 300:
            tiles.append((a, b))
tiles = np.array(tiles)
perm = rng.permutation(len(tiles))
fit_idx, eval_idx = perm[:len(tiles)//2], perm[len(tiles)//2:]
print(f"{len(tiles)} tiles (stride {TILE//2}); fit {len(fit_idx)} eval {len(eval_idx)}")

# ------------------------------------------------------ fit the depth weight (fit half)
mu_i = np.zeros(D); mu_b = np.zeros(D); sd_b = np.zeros(D); n = 0
for t in fit_idx:
    a, b = tiles[t]
    blk = surf[:, a:a+TILE, b:b+TILE]
    i_ = inkv[a:a+TILE, b:b+TILE]; g_ = bgv[a:a+TILE, b:b+TILE]
    mu_i += blk[:, i_].mean(1); mu_b += blk[:, g_].mean(1); sd_b += blk[:, g_].std(1); n += 1
mu_i /= n; mu_b /= n; sd_b /= n
dprime = (mu_i - mu_b) / np.maximum(sd_b, 1e-6)
print("\ndepth d' profile (fit half):")
print("  " + " ".join(f"{d}:{dprime[d]:+.2f}" for d in range(D)))

# ------------------------------------------------------------- detector family search
def project(win, mode, wvec=None):
    if mode == "mean":
        return win.mean(0)
    if mode == "wfit":
        w = wvec - wvec.mean()
        nrm = np.linalg.norm(w)
        return np.tensordot(w / (nrm if nrm > 0 else 1), win, axes=(0, 0))
    if mode == "std":
        return win.std(0)
    if mode == "ptp":
        return win.max(0) - win.min(0)
    raise ValueError(mode)


def bandpass(img, s1, s2):
    if s1 <= 0:
        return img - ndi.gaussian_filter(img, s2)
    return ndi.gaussian_filter(img, s1) - ndi.gaussian_filter(img, s2)


CENTRE = LABEL_SLICE
win0 = surf[CENTRE - 8: CENTRE + 9]
wfit0 = dprime[CENTRE - 8: CENTRE + 9]

print("\n=== detector search at the ALIGNED window (centre 14), median per-tile AUC ===")
print(f"{'projection':>10s} {'bandpass':>14s}   AUC   |AUC-.5|")
cands = []
for mode in ("mean", "wfit", "std", "ptp"):
    base = project(win0, mode, wfit0)
    for (s1, s2) in ((0, 0), (0, 16), (0, 48), (2, 16), (4, 32), (8, 64), (16, 128)):
        img = base if (s1, s2) == (0, 0) else bandpass(base, s1, s2)
        vals = []
        for t in eval_idx:
            a, b = tiles[t]
            i_ = inkv[a:a+TILE, b:b+TILE]; g_ = bgv[a:a+TILE, b:b+TILE]
            vals.append(auc(img[a:a+TILE, b:b+TILE][i_], img[a:a+TILE, b:b+TILE][g_]))
        m = float(np.nanmedian(vals))
        tag = "none" if (s1, s2) == (0, 0) else f"DoG({s1},{s2})"
        print(f"{mode:>10s} {tag:>14s}  {m:.4f}   {abs(m-0.5):.4f}")
        cands.append((abs(m - 0.5), mode, s1, s2, m))

cands.sort(reverse=True)
best = cands[0]
print(f"\nBEST offline detector: {best[1]} + "
      f"{'none' if best[2]==0 and best[3]==0 else f'DoG({best[2]},{best[3]})'}"
      f"  aligned AUC {best[4]:.4f}  (excess {best[4]-0.5:+.4f})")

GATE = 0.60
result = dict(bbox=[y0, y1, x0, x1], n_tiles=int(len(tiles)),
              dprime=[round(float(v), 4) for v in dprime],
              detector_search=[dict(mode=c[1], s1=c[2], s2=c[3], auc=round(c[4], 4))
                               for c in cands],
              best=dict(mode=best[1], s1=best[2], s2=best[3], auc=round(best[4], 4)),
              gate=GATE)

if abs(best[4] - 0.5) < GATE - 0.5:
    print(f"\n*** GATE FAILED: no offline detector reaches |AUC-0.5| >= {GATE-0.5:.2f}.")
    print("*** A window-offset sweep on this channel would be measuring noise.")
    result["sweep"] = None
    result["verdict"] = "no offline detector sees the control ink; tolerance budget requires the real model"
else:
    _, mode, s1, s2, _ = best
    centres = list(range(8, 20))
    print(f"\n=== window sweep with the best detector, centres {centres[0]}..{centres[-1]} ===")
    print(" centre offset   AUC    excess  retained")
    rows = {}
    for c in centres:
        win = surf[c - 8: c + 9]
        base = project(win, mode, dprime[c - 8: c + 9])
        img = base if (s1, s2) == (0, 0) else bandpass(base, s1, s2)
        vals = []
        for t in eval_idx:
            a, b = tiles[t]
            i_ = inkv[a:a+TILE, b:b+TILE]; g_ = bgv[a:a+TILE, b:b+TILE]
            vals.append(auc(img[a:a+TILE, b:b+TILE][i_], img[a:a+TILE, b:b+TILE][g_]))
        rows[c] = float(np.nanmedian(vals))
    e0 = rows[LABEL_SLICE] - 0.5
    for c in centres:
        e = rows[c] - 0.5
        print(f"  {c:4d} {c-LABEL_SLICE:+4d}   {rows[c]:.4f} {e:+.4f}  "
              f"{100*e/e0 if e0 else float('nan'):6.1f}%")
    result["sweep"] = {str(c): round(rows[c], 4) for c in centres}
    result["verdict"] = "sweep valid"

json.dump(result, open(os.path.join(OUT, "depth_offset_v2.json"), "w"), indent=1)
print("\nwrote", os.path.join(OUT, "depth_offset_v2.json"))
