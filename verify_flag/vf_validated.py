"""The flagged map through the FULL validated battery protocol, 400 permutations.

salvage/verdict_periodicity.py is the protocol that scored the human-verified
control w035 at z = +25.6..+34.1 and PHerc1203 at z = -2.7..+1.0.  It differs
from the corpus screen in four ways the screen dropped for speed:
    ERODE 40 full-res px   (kills bright patch/tile rims)
    DETREND_SIGMA 90 ds4px (kills 1/f leakage)
    orientation grid 3 deg (vs 15)
    joint (map, mask) block permutation in 64 ds4-px tiles (vs map only)
Here it is run on the flag with 400 permutations instead of 5.
"""
import json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi

HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag"
sys.path.insert(0, HERE)
import vf_common as C

ERODE_DS4 = 10          # 40 full-res px
DETREND = 90            # ds4 px
TILE = 64               # ds4 px
THETAS = np.arange(0, 180, 3.0)
NPERM = 400
_G = {}


def valid_mask(a):
    m = ndi.binary_closing(a > 0, np.ones((5, 5), bool))
    m = ndi.binary_fill_holes(m)
    return ndi.binary_erosion(m, np.ones((3, 3), bool), ERODE_DS4)


def score(img, msk, px):
    cache = C.rot_cache(msk, THETAS)
    best = (0.0, None, None)
    for t in THETAS:
        pr = C.profile(img, t, cache)
        prom, _, per = C.band_prom(pr, px, DETREND)
        if prom > best[0]:
            best = (prom, float(t), per)
    return best


def joint_permute(img, msk, rng, tile=TILE):
    h, w = img.shape
    coords = [(y, x) for y in range(0, h - tile, tile) for x in range(0, w - tile, tile)]
    coords = [c for c in coords if msk[c[0]:c[0] + tile, c[1]:c[1] + tile].mean() > 0.5]
    if len(coords) < 8:
        return None, None
    perm = rng.permutation(len(coords))
    oi, om = img.copy(), msk.copy()
    si = [img[y:y + tile, x:x + tile].copy() for y, x in coords]
    sm = [msk[y:y + tile, x:x + tile].copy() for y, x in coords]
    for i, (y, x) in enumerate(coords):
        oi[y:y + tile, x:x + tile] = si[perm[i]]
        om[y:y + tile, x:x + tile] = sm[perm[i]]
    return oi, om


def _init(px):
    a = C.load(C.FLAG, "forward")
    m = valid_mask(a)
    _G.update(a=a * m, m=m, px=px)


def _one(i):
    if i < 0:
        return score(_G["a"], _G["m"], _G["px"])
    rng = np.random.default_rng(9000 + i)
    pi, pm = joint_permute(_G["a"], _G["m"], rng)
    if pi is None:
        return (np.nan, None, None)
    return score(pi, pm, _G["px"])


if __name__ == "__main__":
    px = C.PX_UM_DS4["PHerc1447"]
    a = C.load(C.FLAG, "forward"); m = valid_mask(a)
    print(f"validated protocol: mask frac {(a>0).mean():.3f} -> eroded {m.mean():.3f}")
    with Pool(16, initializer=_init, initargs=(px,)) as pool:
        out = pool.map(_one, [-1] + list(range(NPERM)), chunksize=2)
    obs = out[0]
    nulls = np.array([o[0] for o in out[1:]], float)
    nulls = nulls[np.isfinite(nulls)]
    mu, sd = nulls.mean(), nulls.std(ddof=1)
    z = (obs[0] - mu) / sd
    p = (1 + int((nulls >= obs[0]).sum())) / (len(nulls) + 1)
    rng = np.random.default_rng(3)
    bz = np.array([(obs[0] - b.mean()) / b.std(ddof=1) for b in
                   (rng.choice(nulls, len(nulls), replace=True) for _ in range(4000))])
    ci = np.percentile(bz, [2.5, 97.5])
    print(f"OBS prom={obs[0]:.2f} theta={obs[1]} period={obs[2]:.2f}mm")
    print(f"NULL n={len(nulls)} mean={mu:.2f} sd={sd:.2f} max={nulls.max():.2f} "
          f"p50={np.median(nulls):.2f} p95={np.percentile(nulls,95):.2f}")
    print(f"z = {z:+.2f}  CI95 [{ci[0]:+.2f},{ci[1]:+.2f}]   empirical p = {p:.4g} "
          f"({int((nulls>=obs[0]).sum())}/{len(nulls)})")
    print(f"control w035 through this protocol: z = +25.6..+34.1 at 4.68-4.73 mm")
    per = np.array([o[2] for o in out[1:] if o[2] is not None], float)
    print(f"null 'best period' distribution: p50={np.median(per):.2f}mm, "
          f"frac>6mm={np.mean(per>6):.2f}  (the band tops out at 8.4)")
    json.dump({"obs_prom": obs[0], "theta": obs[1], "period_mm": obs[2], "n_perm": int(len(nulls)),
               "null_mean": float(mu), "null_sd": float(sd), "null_max": float(nulls.max()),
               "z": float(z), "z_ci95": ci.tolist(), "empirical_p": p,
               "mask_frac_eroded": float(m.mean()),
               "null_period_p50_mm": float(np.median(per)),
               "null_period_frac_gt6mm": float(np.mean(per > 6))},
              open(os.path.join(HERE, "vf_validated.json"), "w"), indent=1)
