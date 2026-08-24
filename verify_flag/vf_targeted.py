"""Targeted nulls: is the flag nothing but between-patch intensity steps?

The screen's null shuffles blocks across the WHOLE mask, so it destroys the
segment's patchwork structure (3 disjoint inference-tile regions with different
mean response) as well as any real ruling.  Two sharper comparisons:

  A. within-component null - shuffle blocks only inside each connected mask
     component.  Preserves the patch-level mean steps, destroys long-range
     order within a patch.  If the flag is text, it survives; if the flag is
     the patch steps, z collapses to ~0.
  B. patch-demeaned observed - subtract each component's own mean before
     scoring.  Same logic, no permutation needed.
  C. rim removal - erode the mask (the validated battery erodes 40 px).
"""
import json, os, sys
import numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi

HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag"
sys.path.insert(0, HERE)
import vf_common as C

THETAS = list(range(0, 180, 15))
NPERM = 300
BLOCK = 64
_G = {}


def _init(px):
    a = C.load(C.FLAG, "forward")
    mask = a > 0
    lab, n = ndi.label(mask)
    _G.update(a=a, mask=mask, lab=lab, n=n, px=px, cache=C.rot_cache(mask, THETAS))


def _within_permute(a, lab, n, rng, block=BLOCK):
    out = a.copy()
    for cid in range(1, n + 1):
        m = lab == cid
        if m.sum() < 20 * block * block:
            continue
        h, w = a.shape
        coords = [(y, x) for y in range(0, h - block, block) for x in range(0, w - block, block)
                  if m[y:y + block, x:x + block].mean() > 0.8]
        if len(coords) < 4:
            continue
        perm = rng.permutation(len(coords))
        src = [a[y:y + block, x:x + block].copy() for y, x in coords]
        for i, (y, x) in enumerate(coords):
            out[y:y + block, x:x + block] = src[perm[i]]
    return out


def _one(i):
    if i < 0:
        return C.ruling_score(_G["a"], _G["cache"], THETAS, _G["px"])
    rng = np.random.default_rng(5000 + i)
    pa = _within_permute(_G["a"], _G["lab"], _G["n"], rng)
    return C.ruling_score(pa, _G["cache"], THETAS, _G["px"])


if __name__ == "__main__":
    px = C.PX_UM_DS4["PHerc1447"]
    res = {}
    with Pool(16, initializer=_init, initargs=(px,)) as pool:
        out = pool.map(_one, [-1] + list(range(NPERM)), chunksize=2)
    obs = out[0]
    nulls = np.array([o[0] for o in out[1:]], float)
    mu, sd = nulls.mean(), nulls.std(ddof=1)
    z = (obs[0] - mu) / sd
    p = (1 + int((nulls >= obs[0]).sum())) / (len(nulls) + 1)
    print(f"[A within-component null, n={len(nulls)}] obs prom={obs[0]:.2f} th={obs[1]} "
          f"per={obs[2]:.2f}mm | null {mu:.2f}+-{sd:.2f} (max {nulls.max():.2f}) "
          f"| z={z:+.2f} | empirical p={p:.4g}")
    res["within_component_null"] = {"obs_prom": obs[0], "theta": obs[1], "period_mm": obs[2],
                                    "n_perm": len(nulls), "null_mean": float(mu),
                                    "null_sd": float(sd), "null_max": float(nulls.max()),
                                    "z": float(z), "empirical_p": p}

    # ---- B: patch-demeaned observed ------------------------------------
    a = C.load(C.FLAG, "forward"); mask = a > 0
    lab, n = ndi.label(mask)
    cache = C.rot_cache(mask, THETAS)
    b = a.copy()
    gm = a[mask].mean()
    for cid in range(1, n + 1):
        m = lab == cid
        b[m] = a[m] - a[m].mean() + gm
    sc_raw = C.ruling_score(a, cache, THETAS, px)
    sc_dm = C.ruling_score(b, cache, THETAS, px)
    print(f"[B] raw   : prom={sc_raw[0]:.2f} th={sc_raw[1]} per={sc_raw[2]:.2f}mm")
    print(f"[B] patch-demeaned: prom={sc_dm[0]:.2f} th={sc_dm[1]} per={sc_dm[2]:.2f}mm "
          f"({100*(1-sc_dm[0]/sc_raw[0]):.0f}% of the prominence removed by 3 constants)")
    res["patch_demeaned"] = {"raw_prom": sc_raw[0], "raw_period_mm": sc_raw[2],
                             "demeaned_prom": sc_dm[0], "demeaned_theta": sc_dm[1],
                             "demeaned_period_mm": sc_dm[2],
                             "frac_prom_removed": float(1 - sc_dm[0] / sc_raw[0])}
    # and the same at theta=15
    for tag, img in (("raw", a), ("demeaned", b)):
        pr = C.profile(img, 15, cache)
        pp = C.band_prom(pr, px)
        print(f"    theta=15 {tag}: prom={pp[0]:.2f} per={pp[2]:.2f}mm")
        res[f"theta15_{tag}"] = {"prom": pp[0], "period_mm": pp[2]}

    # ---- C: which Fourier bin is the band edge? ------------------------
    pr = C.profile(a, 15, cache)
    L = len(pr)
    lo_px = C.BAND_MM[0] * 1000 / px; hi_px = C.BAND_MM[1] * 1000 / px
    kmin, kmax = L / hi_px, L / lo_px
    print(f"\n[C] profile length {L} px; ruling band {C.BAND_MM} mm = periods "
          f"{lo_px:.1f}-{hi_px:.1f} px = Fourier bins k={kmin:.2f}..{kmax:.2f}")
    print(f"    -> only {int(np.floor(kmax))-int(np.ceil(kmin))+1} integer bins in the band; "
          f"the flagged 7.24 mm is k={L*px/1000/7.24:.2f}, the LOWEST bin in the band.")
    res["band_bins"] = {"profile_len": L, "k_min": kmin, "k_max": kmax,
                        "n_integer_bins": int(np.floor(kmax)) - int(np.ceil(kmin)) + 1,
                        "flag_k": L * px / 1000 / 7.24}
    json.dump(res, open(os.path.join(HERE, "vf_targeted.json"), "w"), indent=1, default=float)
