"""Shared library for pod_curve_audit. Env: ROOT/OUT/DATA/PREDS/RESULTS/STATUS/SEED."""
import json, os, sys, time, hashlib
import numpy as np

ROOT = os.environ["ROOT"]; OUT = os.environ["OUT"]; DATA = os.environ["DATA"]
PREDS = os.environ["PREDS"]; RESULTS = os.environ["RESULTS"]
STATUS = os.environ["STATUS"]; SEED = int(os.environ.get("SEED", "20260824"))

def say(msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg
    with open(STATUS, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# ---------------------------------------------------------------- expected --
# Every number below was measured on 2026-08-24 against the live servers and
# the laptop ground-truth copies BEFORE this script was written. Parse, then
# assert -- never assume.
FRAG1_BASE = ("https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/"
              "working/54keV_exposed_surface")
FRAG1_H, FRAG1_W, FRAG1_NZ = 8181, 6330, 65
FRAG1_SBC = 103571460                      # StripByteCounts = H*W*2, every layer
FRAG1_LABEL_SHA = "bc36bd54e84b423faa44d3b62ca0d3a3c28403cb714b154acb382f94241d39b6"
FRAG1_MASK_SHA = "1fdc13f7dc149dc4e00272af597ba236b650e54bb2dca38865f211207cee069f"
FRAG1_INK_PX = 5339364
FRAG1_MASK_PX = 29142840
FRAG1_BLANK_PX = 23803476
FRAG1_INK_OUTSIDE_MASK = 0
NATIVE_PITCH = 3.24
MODEL_PITCH = 9.36

BUCKET = ("https://huggingface.co/buckets/scrollprize/datasets/resolve/"
          "ink/unused/500p2a")
P2A_SHAPE = (65, 26239, 16182)             # .zarray, verified live
P2A_CHUNK = (65, 128, 128)
P2A_PITCH = 4.32
P2A_LABEL_BYTES = 3009475
P2A_MASK_BYTES = 2983168
P2A_RASTER = dict(ink=12856732, mask=34871346, ink_and_mask=12230762,
                  ink_outside_mask=625970, annot_blank=22640584)
WINDOWS = {
    "win1": dict(y0=12416, x0=6912, size=4438, ink=4389424, ink_and_mask=4388955,
                 ink_outside_mask=469, annot_blank=8576280, mask=12965235,
                 vol_mean=76.114, vol_std=32.313, vol_zero_frac=0.0),
    "win2": dict(y0=18432, x0=7424, size=4438, ink=2862504, ink_and_mask=2862504,
                 ink_outside_mask=0, annot_blank=4778034, mask=7640538,
                 vol_mean=72.612, vol_std=32.042, vol_zero_frac=0.0),
    "win3": dict(y0=12160, x0=2432, size=4438, ink=2158076, ink_and_mask=1536773,
                 ink_outside_mask=621303, annot_blank=2237639, mask=3774412,
                 vol_mean=74.129, vol_std=31.922, vol_zero_frac=0.00023),
}

# ------------------------------------------------------------------ prereg --
GATE_BASELINE_AUC = 0.85
EXPA_ANCHOR = "win1"          # the curve anchor window (rank-1 pairing weight)
DETECT_AUC_MIN = 0.75
DETECT_RETAIN_MIN = 0.50
PITCHES = [5.5, 6.5, 8.0, 9.36, 12.0]
NOISE_KS = [1, 2, 4, 8]
BLUR_SIGMAS = [1.0, 2.0]
N_TRANSLATIONS = 40
MIN_SHIFT_MM = 4.4
MAX_SHIFT_FRAC = 0.40
MIN_PSEUDO_POS = 50000
CONFOUND_MEDIAN_AUC = 0.60
CONFOUND_FRAC = 0.20          # OR-clause: >= 20% of nulls at/above 0.60
GAP_MIN = 0.15

# --------------------------------------------------------------- resampling -
def _aa_sigma(f):
    """skimage-convention anti-aliasing sigma for downscale factor f (>1)."""
    return max(0.0, (f - 1.0) / 2.0)

def zoom_to(arr, out_shape, prefilter=True):
    """Area-aligned linear resample of a float32 array to an EXACT shape,
    with per-axis Gaussian anti-alias prefilter on downsampled axes."""
    from scipy import ndimage
    arr = np.asarray(arr, dtype=np.float32)
    in_shape = arr.shape
    if prefilter:
        sigmas = []
        for i, o in zip(in_shape, out_shape):
            f = i / o
            sigmas.append(_aa_sigma(f) if f > 1.0 else 0.0)
        if any(s > 0 for s in sigmas):
            arr = ndimage.gaussian_filter(arr, sigma=sigmas, mode="nearest")
    factors = [o / i for i, o in zip(in_shape, out_shape)]
    out = ndimage.zoom(arr, factors, order=1, mode="nearest", grid_mode=True)
    if out.shape != tuple(out_shape):
        raise AssertionError(f"zoom shape {out.shape} != target {tuple(out_shape)}")
    return out.astype(np.float32)

def resample_stack(layer_get, n_in, H, W, out_shape, tmp_path, band=384, tag=""):
    """(n_in,H,W) -> float32 memmap of out_shape via separable two-pass
    (z first in y-bands, then per-slice yx). layer_get(z) -> (H,W) float-able."""
    from scipy import ndimage
    nz_out, Ht, Wt = out_shape
    fz = n_in / nz_out
    sz = _aa_sigma(fz) if fz > 1.0 else 0.0
    ztmp = tmp_path + ".zpass.f32"
    zmm = np.memmap(ztmp, dtype=np.float32, mode="w+", shape=(nz_out, H, W))
    for y0 in range(0, H, band):
        y1 = min(H, y0 + band)
        buf = np.empty((n_in, y1 - y0, W), dtype=np.float32)
        for z in range(n_in):
            buf[z] = layer_get(z)[y0:y1, :]
        if sz > 0:
            buf = ndimage.gaussian_filter1d(buf, sigma=sz, axis=0, mode="nearest")
        zb = ndimage.zoom(buf, (nz_out / n_in, 1, 1), order=1, mode="nearest",
                          grid_mode=True)
        assert zb.shape[0] == nz_out, (zb.shape, nz_out)
        zmm[:, y0:y1, :] = zb
    zmm.flush()
    out = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(nz_out, Ht, Wt))
    for z in range(nz_out):
        out[z] = zoom_to(np.asarray(zmm[z]), (Ht, Wt))
        if z % 8 == 0:
            say(f"resample {tag} slice {z + 1}/{nz_out}")
    out.flush()
    del zmm
    os.remove(ztmp)
    return out

def rint_shape(n, p_in, p_out):
    """Half-up rounding with epsilon: deterministic at exact .5 boundaries
    (65*3.24/9.36 = 22.5 exactly -> 23; np.rint would be FP-jitter fragile)."""
    return max(1, int(np.floor(n * p_in / p_out + 0.5 + 1e-9)))

# ------------------------------------------------------------------- zarr ---
def write_group_zarr(path, vol):
    """Write a Zarr-v2 group with level '0' (the volume) and a binary
    occupancy level '3' (YX max-pool by 8), atomically via <path>.tmp."""
    import shutil
    from numcodecs import Blosc
    from vesuvius.label_zarr import open_v2_group, create_v2_array
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    n, H, W = vol.shape
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    group = open_v2_group(tmp)
    a0 = create_v2_array(group, "0", shape=(n, H, W), chunks=(n, 256, 256),
                         dtype=vol.dtype, compressor=comp, fill_value=0)
    for y0 in range(0, H, 1024):
        a0[:, y0:min(H, y0 + 1024), :] = vol[:, y0:min(H, y0 + 1024), :]
    p = 8
    Hp, Wp = (H + p - 1) // p, (W + p - 1) // p
    a3 = create_v2_array(group, "3", shape=(n, Hp, Wp), chunks=(n, 256, 256),
                         dtype=np.uint8, compressor=comp, fill_value=0)
    for y0 in range(0, H, 4096):
        y1 = min(H, y0 + 4096)
        blk = np.asarray(vol[:, y0:y1, :])
        h = blk.shape[1]
        ph, pw = (-h) % p, (-W) % p
        if ph or pw:
            blk = np.pad(blk, ((0, 0), (0, ph), (0, pw)))
        pooled = blk.reshape(n, (h + ph) // p, p, (W + pw) // p, p).max(axis=(2, 4))
        a3[:, y0 // p: y0 // p + pooled.shape[1], :] = \
            ((pooled > 0) * np.uint8(255))
    if os.path.exists(path):
        shutil.rmtree(path)
    os.rename(tmp, path)

def read_zarr0(path):
    import zarr
    g = zarr.open_group(path, mode="r")
    return g["0"]

def quant4(v8):
    """uint8 -> 4-bit (16 levels {0,17,...,255}); the v2 bit-depth stressor.
    The released 4.32um 500p2a volume is ALREADY uint8, so the v1
    uint16->uint8 rung is the baseline by construction here."""
    x = np.rint(np.asarray(v8, dtype=np.float32) / 17.0) * 17.0
    return np.clip(x, 0, 255).astype(np.uint8)

# -------------------------------------------------------------------- AUC ---
NBINS = 65536

def quantize_map(m):
    """Map (uint8 native or float 0..255 upsampled) -> uint16 bins 0..65535."""
    q = np.rint(np.asarray(m, dtype=np.float32) * 257.0)
    return np.clip(q, 0, NBINS - 1).astype(np.uint16)

def masked_hist(q, mask):
    return np.bincount(q[mask], minlength=NBINS).astype(np.float64)

def hist_auc(hpos, hneg):
    """Exact tie-corrected rank AUC from per-bin histograms."""
    P, N = hpos.sum(), hneg.sum()
    if P == 0 or N == 0:
        return float("nan")
    cneg_below = np.concatenate([[0.0], np.cumsum(hneg)[:-1]])
    return float((hpos * (cneg_below + 0.5 * hneg)).sum() / (P * N))

def upsample_pred(pred, out_hw):
    """Bilinear, area-aligned upsample of a prediction map to the native grid."""
    return zoom_to(np.asarray(pred, dtype=np.float32), out_hw, prefilter=False)

# ------------------------------------------------------------ translations --
def draw_translations(shapes, blank, pitch_um, seed, n=N_TRANSLATIONS):
    """40 rigid (dy,dx) shifts, |shift|_inf in [4.4mm, 0.40*min(H,W)],
    each leaving >= MIN_PSEUDO_POS pseudo-positive px inside blank."""
    H, W = shapes.shape
    min_px = int(np.ceil(MIN_SHIFT_MM * 1000.0 / pitch_um))
    max_px = int(np.floor(MAX_SHIFT_FRAC * min(H, W)))
    if max_px <= min_px:
        raise AssertionError(f"translation annulus empty: {min_px}..{max_px}")
    rng = np.random.default_rng(seed)
    out, tried = [], 0
    while len(out) < n:
        tried += 1
        if tried > 4000:
            raise AssertionError(
                f"could not draw {n} valid translations (got {len(out)})")
        dy, dx = (int(v) for v in rng.integers(-max_px, max_px + 1, size=2))
        if max(abs(dy), abs(dx)) < min_px or (dy, dx) in out:
            continue
        if shifted_count(shapes, blank, dy, dx) < MIN_PSEUDO_POS:
            continue
        out.append((dy, dx))
    return out, min_px, max_px

def _shift_slices(H, W, dy, dx):
    sy0, sy1 = max(0, -dy), min(H, H - dy)
    dy0, dy1 = max(0, dy), min(H, H + dy)
    sx0, sx1 = max(0, -dx), min(W, W - dx)
    dx0, dx1 = max(0, dx), min(W, W + dx)
    return (slice(sy0, sy1), slice(sx0, sx1)), (slice(dy0, dy1), slice(dx0, dx1))

def shifted_count(shapes, blank, dy, dx):
    (ssy, ssx), (dsy, dsx) = _shift_slices(*shapes.shape, dy, dx)
    return int(np.count_nonzero(shapes[ssy, ssx] & blank[dsy, dsx]))

def translated_hist(q, shapes, blank, dy, dx):
    """Histogram of map values on T(shapes) & blank (the pseudo-positives)."""
    (ssy, ssx), (dsy, dsx) = _shift_slices(*shapes.shape, dy, dx)
    sel = shapes[ssy, ssx] & blank[dsy, dsx]
    return np.bincount(q[dsy, dsx][sel], minlength=NBINS).astype(np.float64)

# ----------------------------------------------------------------- preview --
def save_preview(arr2d, path, ds=4):
    from PIL import Image
    a = np.asarray(arr2d, dtype=np.float32)[::ds, ::ds]
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)
