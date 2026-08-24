"""Shared machinery for the flag verification (PHerc1447 z_dbg_gen_00166_inp_hr).

Reimplements the corpus screen's ruling test faithfully, with switches for the
things the screen skipped (extra downsample, detrending, permutation count).
"""
import os
import numpy as np
from scipy.ndimage import rotate, gaussian_filter1d

SURVEY = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\survey"
OUTDIR = r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag"
FLAG = "z_dbg_gen_00166_inp_hr"
BAND_MM = (1.7, 8.4)
PX_UM_DS4 = {"PHerc1203": 9.362 * 4, "PHerc1447": 8.64 * 4, "PHerc0800": 8.64 * 4}


def map_path(name, direction="forward"):
    import glob
    g = glob.glob(os.path.join(SURVEY, "maps_shard*", f"{name}_{direction}_ds4.npy"))
    return g[0] if g else None


def load(name, direction="forward", ds=1):
    a = np.load(map_path(name, direction)).astype(np.float32)
    if ds > 1:
        a = a[::ds, ::ds]
    return a


def erode_mask(mask, px):
    if px <= 0:
        return mask
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, np.ones((3, 3), bool), iterations=int(px))


def rot_cache(mask, thetas):
    """Pre-rotate the mask once per orientation (it never changes under permutation)."""
    out = {}
    for t in thetas:
        rm = rotate(mask.astype(np.float32), t, order=1, reshape=True, mode="constant", cval=0)
        den = rm.sum(axis=1)
        ok = den > 0.25 * den.max() if den.max() > 0 else den > 0
        out[t] = (rm, den, ok)
    return out


def profile(img, t, cache):
    rm, den, ok = cache[t]
    ri = rotate(img, t, order=1, reshape=True, mode="constant", cval=0)
    if ri.shape != rm.shape:                      # numerical edge case
        h = min(ri.shape[0], rm.shape[0]); w = min(ri.shape[1], rm.shape[1])
        ri = ri[:h, :w]
    num = ri.sum(axis=1)
    if ok.sum() < 64:
        return None
    return num[ok] / den[ok]


def band_prom(prof, px_um, detrend_sigma=0.0, return_spec=False):
    """Peak/median power prominence inside the ruling band (screen's statistic)."""
    lo_px = BAND_MM[0] * 1000 / px_um
    hi_px = BAND_MM[1] * 1000 / px_um
    if prof is None or len(prof) < 4 * lo_px:
        return (0.0, None, None) if not return_spec else (0.0, None, None, None, None)
    p = prof - prof.mean()
    if detrend_sigma > 0:
        p = p - gaussian_filter1d(p, detrend_sigma, mode="nearest")
    w = np.hanning(len(p))
    ps = np.abs(np.fft.rfft(p * w)) ** 2
    freqs = np.fft.rfftfreq(len(p))
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-9), np.inf)
    sel = (periods >= lo_px) & (periods <= hi_px)
    if sel.sum() < 3:
        return (0.0, None, None) if not return_spec else (0.0, None, None, None, None)
    band = ps[sel]
    prom = float(band.max() / max(np.median(band), 1e-12))
    per_px = float(periods[sel][np.argmax(band)])
    if return_spec:
        return prom, per_px, per_px * px_um / 1000, periods[sel], band
    return prom, per_px, per_px * px_um / 1000


def ruling_score(img, cache, thetas, px_um, detrend_sigma=0.0):
    best = (0.0, None, None)
    for t in thetas:
        pr = profile(img, t, cache)
        prom, per_px, per_mm = band_prom(pr, px_um, detrend_sigma)
        if prom > best[0]:
            best = (prom, t, per_mm)
    return best


def block_permute(img, mask, rng, block):
    """Screen's null: shuffle fully-inside-mask blocks; leave edge blocks in place."""
    h, w = img.shape
    out = img.copy()
    coords = [(y, x) for y in range(0, h - block, block) for x in range(0, w - block, block)
              if mask[y:y + block, x:x + block].mean() > 0.8]
    if len(coords) < 8:
        return None, 0
    perm = rng.permutation(len(coords))
    src = [img[y:y + block, x:x + block].copy() for y, x in coords]
    for i, (y, x) in enumerate(coords):
        out[y:y + block, x:x + block] = src[perm[i]]
    return out, len(coords)
