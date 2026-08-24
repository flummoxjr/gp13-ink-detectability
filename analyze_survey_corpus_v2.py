"""Corpus screen v2 -- hardened re-score of every surveyed GP-scroll segment.

WHY v2 EXISTS
-------------
`analyze_survey_corpus.py` (v1) flagged exactly one segment out of 80,
PHerc1447/z_dbg_gen_00166_inp_hr, at ruling_z = +5.94.  The verification in
`verify_flag/FLAG_VERDICT.md` refuted it and showed the screen itself was
under-powered and under-specified.  It prescribed four fixes; all four are
implemented here.

  (1) EMPIRICAL p, N_PERM >= 200.  v1 used 16 block permutations and reported
      only z.  The prominence statistic is heavy-tailed: with 16 draws its sd
      was underestimated 2.2x, so z=+5.94 actually meant p ~ 0.01, and
      E[# segments with z>=5 among 80] = 1.07 under the screen's own noise.
      v2 reports empirical p = (1 + #{null >= obs}) / (n_null + 1) with
      N_PERM = 200 (min attainable p = 1/201 = 0.00498) plus a bootstrap CI on
      z, and Holm-corrects across the corpus.

  (2) VALIDATED PREPROCESSING.  v1 dropped, for speed, the two steps that the
      human-verified control protocol (salvage/verdict_periodicity.py) used:
      40 full-res px rim erosion (= 10 ds4 px; removes the ink_9um inference
      patch-boundary halo, where 90.8% of the flag's above-blank-p99 pixels
      lived) and sigma = 90 ds4-px profile detrending (removes 1/f leakage
      into the band).  v1 also coarsened the orientation grid to 15 deg and
      permuted the map but not the mask.  v2 restores all four: erosion 10 ds4
      px, detrend 90, 3 deg orientation grid, joint (map, mask) block
      permutation on 64 ds4-px tiles.

  (3) PERIODICITY SANITY GATES.  A peak in a power spectrum is not a period.
      v2 requires, at the winning orientation:
        - >= 6 cycles of the claimed period inside the profile
          (the flag had exactly 4.00),
        - positive autocorrelation of the detrended profile at 2P and 3P
          (the flag: rho(1P) = -0.269),
        - the peak bin not among the two lowest Fourier bins of the search
          band (the flag's peak was k = 4, the band's lowest bin).

  (4) FWD/REV SYMMETRY GATE.  Ink sits on one face of the sheet, so reversing
      the z-render destroys it: the control map has fwd/rev r = 0.076.
      Fibre/geometry texture survives reversal: every v1 top hit had
      r = 0.37-0.83.  v2 requires map-scale r < 0.20.

A screen that cannot detect the known letters is worthless, so the same code
path is run on the human-verified control (PHerc0139 w035, ink_9um
hybrid_3d2d-seed42 step-075000) and on its human ink labels.

Usage
-----
  python analyze_survey_corpus_v2.py              # full run (long; use --jobs)
  python analyze_survey_corpus_v2.py --nperm 8 --only z_dbg_gen_00166_inp_hr
  python analyze_survey_corpus_v2.py --figure-only

Outputs: out/survey/corpus_analysis_v2.json, out/survey/corpus_screen_v2.png
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

TRACKD = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
sys.path.insert(0, os.path.join(TRACKD, "verify_flag"))
import vf_common as C  # noqa: E402  (shared scorer from the flag verification)

OUT = os.path.join(TRACKD, "out", "survey")
CACHE = os.path.join(OUT, "v2cache")
CTRL_DIR = os.path.join(TRACKD, "out", "ink9um_w035")
CTRL_FWD = os.path.join(CTRL_DIR, "w035_seed42-075000.tif")
CTRL_REV = os.path.join(CTRL_DIR, "w035_seed42-075000_reverse.tif")
CTRL_LAB = r"D:\vesuvius-data\trackD\w035_ink.npy"

# ---- protocol constants (all inherited from the validated battery) ----------
PX_UM_DS4 = {"PHerc1203": 9.362 * 4, "PHerc1447": 8.64 * 4,
             "PHerc0800": 8.64 * 4, "PHerc0139": 9.362 * 4}
ERODE_DS4 = 10            # 40 full-res px rim erosion
DETREND = 90              # ds4 px, gaussian high-pass of the profile
TILE = 64                 # ds4 px, joint (map, mask) permutation tile
TILE_FALLBACK = 32        # for maps too small to give >= 8 tiles at 64
THETAS = np.arange(0, 180, 3.0)
N_PERM = 200
BAND_MM = C.BAND_MM       # (1.7, 8.4) mm
# ---- gate thresholds -------------------------------------------------------
MIN_CYCLES = 6.0          # profile must contain >= 6 cycles of the period
MIN_BAND_BIN = 2          # peak may not be one of the band's 2 lowest bins
MAX_FWD_REV_R = 0.20      # control 0.076; v1 top hits 0.37-0.83
ALPHA = 0.05              # on the Holm-corrected empirical p

_G = {}


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def valid_mask(a, erode=ERODE_DS4):
    """Validated-battery valid region: close, fill, erode 40 full-res px."""
    m = ndi.binary_closing(a > 0, np.ones((5, 5), bool))
    m = ndi.binary_fill_holes(m)
    return ndi.binary_erosion(m, np.ones((3, 3), bool), erode)


def prep(img_path, mask_path=None):
    """Load a ds4 map, build the eroded valid mask, crop to its bbox.

    Cropping to the mask bounding box is a pure speed optimisation: rows
    outside carry mask weight 0 and are already dropped by the profile's
    coverage test, so the retained profile is the same up to interpolation
    edge effects (measured: 0.4% on the flag's prominence).
    """
    a = np.load(img_path).astype(np.float32)
    src = a if mask_path is None else np.load(mask_path).astype(np.float32)
    if src.shape != a.shape:
        h, w = min(a.shape[0], src.shape[0]), min(a.shape[1], src.shape[1])
        a, src = a[:h, :w], src[:h, :w]
    m = valid_mask(src)
    if m.sum() < 4096:
        return None, None, float(m.mean())
    ys, xs = np.nonzero(m)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return (a * m)[sl].astype(np.float32), m[sl], float(m.mean())


# ---------------------------------------------------------------------------
# scoring (vf_common's scorer = the validated battery's statistic)
# ---------------------------------------------------------------------------
def band_stats(prof, px):
    """Two prominences from ONE spectrum, sharing the band-median normaliser.

    `full`  : the validated battery's statistic -- peak power anywhere in the
              1.7-8.4 mm band over the band median.  Reported as the primary,
              so v2 stays comparable with the control's published numbers.
    `gated` : the same peak restricted to bins that could physically be a
              ruling -- not among the band's MIN_BAND_BIN lowest Fourier bins
              and repeating at least MIN_CYCLES times inside the profile.
              This exists so the sanity gates cannot cause a false negative:
              a genuine 4.7 mm ruling that loses the argmax to a band-edge
              artifact still gets its own score and its own null.
    """
    out = C.band_prom(prof, px, DETREND, return_spec=True)
    prom, per_px, per_mm, per_axis, band = out
    if per_axis is None or band is None:
        return None
    n = len(prof)
    med = max(float(np.median(band)), 1e-12)
    idx = int(np.argmax(band))
    ok = (np.arange(band.size) >= MIN_BAND_BIN) & ((n / per_axis) >= MIN_CYCLES)
    if ok.any():
        sub = np.where(ok, band, -np.inf)
        gidx = int(np.argmax(sub))
        gprom, gper = float(band[gidx] / med), float(per_axis[gidx])
    else:
        gidx, gprom, gper = -1, 0.0, float("nan")
    return {"prom": float(prom), "per_px": float(per_px), "idx": idx,
            "gprom": gprom, "gper_px": gper, "gidx": gidx,
            "nbins": int(band.size), "n": int(n)}


def _detail(pr, px, per_px, idx, nbins):
    """Periodicity sanity measurements for one (profile, claimed period)."""
    from scipy.ndimage import gaussian_filter1d
    p = pr - pr.mean()
    p = p - gaussian_filter1d(p, DETREND, mode="nearest")   # same detrend as band_prom
    n = len(p)
    ac = np.correlate(p, p, "full")[n - 1:]
    ac = ac / max(ac[0], 1e-12)
    acs = [float(ac[L]) if 0 < L < n else None
           for L in (int(round(j * per_px)) for j in (1, 2, 3))]
    return {"n_profile": int(n), "profile_len_mm": float(n * px / 1000.0),
            "period_px": float(per_px), "n_cycles": float(n / per_px),
            "band_bins": int(nbins), "peak_bin_index": int(idx),
            "autocorr_1P": acs[0], "autocorr_2P": acs[1], "autocorr_3P": acs[2]}


def rot_cache_light(mask, thetas):
    """vf_common.rot_cache without keeping the 60 rotated mask images.

    `vf_common.profile` uses the rotated mask only for its column sums (`den`),
    the coverage test (`ok`) and a shape check, so storing the full rotated
    mask costs ~1 GB per worker on the larger maps for nothing. Identical
    arithmetic, ~60x less memory.
    """
    from scipy.ndimage import rotate
    out = {}
    for t in thetas:
        rm = rotate(mask.astype(np.float32), t, order=1, reshape=True,
                    mode="constant", cval=0)
        den = rm.sum(axis=1)
        ok = den > 0.25 * den.max() if den.max() > 0 else den > 0
        out[t] = (rm.shape, den, ok)
        del rm
    return out


def profile_light(img, t, cache):
    from scipy.ndimage import rotate
    shape, den, ok = cache[t]
    ri = rotate(img, t, order=1, reshape=True, mode="constant", cval=0)
    if ri.shape != shape:                      # numerical edge case
        ri = ri[:min(ri.shape[0], shape[0]), :min(ri.shape[1], shape[1])]
    num = ri.sum(axis=1)
    if ok.sum() < 64:
        return None
    if num.size != ok.size:
        k = min(num.size, ok.size)
        return num[:k][ok[:k]] / den[:k][ok[:k]]
    return num[ok] / den[ok]


def score(img, msk, px, want_detail=False):
    """Best full-band and best gated prominence over the orientation grid.

    `periods[sel]` runs from long to short (frequency increasing), so band
    index 0 is the band's LOWEST Fourier bin -- the band-edge/leakage failure
    mode that produced the v1 flag.
    """
    cache = rot_cache_light(msk, THETAS)
    bf = {"prom": 0.0, "theta": None, "per_px": None, "idx": None, "nbins": None}
    bg = {"prom": 0.0, "theta": None, "per_px": None, "idx": None, "nbins": None}
    for t in THETAS:
        pr = profile_light(img, t, cache)
        if pr is None:
            continue
        s = band_stats(pr, px)
        if s is None:
            continue
        if s["prom"] > bf["prom"]:
            bf = {"prom": s["prom"], "theta": float(t), "per_px": s["per_px"],
                  "idx": s["idx"], "nbins": s["nbins"]}
        if s["gprom"] > bg["prom"]:
            bg = {"prom": s["gprom"], "theta": float(t), "per_px": s["gper_px"],
                  "idx": s["gidx"], "nbins": s["nbins"]}
    for b in (bf, bg):
        b["period_mm"] = None if b["per_px"] is None else b["per_px"] * px / 1000.0
    if not want_detail:
        return bf, bg, None, None
    df = dg = None
    if bf["theta"] is not None:
        df = _detail(profile_light(img, bf["theta"], cache), px, bf["per_px"],
                     bf["idx"], bf["nbins"])
    if bg["theta"] is not None:
        dg = _detail(profile_light(img, bg["theta"], cache), px, bg["per_px"],
                     bg["idx"], bg["nbins"])
    return bf, bg, df, dg


def joint_permute(img, msk, rng, tile):
    """Validated-battery null: permute (map, mask) together in `tile` blocks.

    Preserves stroke-scale texture and the mask's own block statistics while
    destroying long-range row alignment.
    """
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


def n_tiles(msk, tile):
    h, w = msk.shape
    return sum(1 for y in range(0, h - tile, tile) for x in range(0, w - tile, tile)
               if msk[y:y + tile, x:x + tile].mean() > 0.5)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
def _load_unit(u):
    if _G.get("key") == u["key"]:
        return
    img, msk, mf = prep(u["img"], u.get("mask"))
    _G.clear()
    _G.update(key=u["key"], img=img, msk=msk, mask_frac=mf, px=u["px"], tile=u["tile"])


def _task(arg):
    """arg = (unit, draw). draw < 0 -> observed map; else permutation index."""
    u, draw = arg
    _load_unit(u)
    if _G["img"] is None:
        return (u["key"], draw, None)
    if draw < 0:
        bf, bg, df, dg = score(_G["img"], _G["msk"], _G["px"], want_detail=True)
        return (u["key"], draw, {"full": bf, "gated": bg, "detail_full": df,
                                 "detail_gated": dg})
    import zlib
    seed = 20260818 + 1000 * draw + (zlib.crc32(u["key"].encode()) % 997)
    rng = np.random.default_rng(seed)
    pi, pm = joint_permute(_G["img"], _G["msk"], rng, _G["tile"])
    if pi is None:
        return (u["key"], draw, None)
    bf, bg, _, _ = score(pi, pm, _G["px"])
    return (u["key"], draw, {"full": bf, "gated": bg})


# ---------------------------------------------------------------------------
# unit construction
# ---------------------------------------------------------------------------
def block_mean(a, f):
    a = a[: a.shape[0] // f * f, : a.shape[1] // f * f].astype(np.float32)
    return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).mean(axis=(1, 3))


def build_control_cache():
    """Write the control's ds4 arrays so the control uses the identical loader.

    Survey maps are strided decimations (`pred[::4, ::4]`, uint8) of the
    full-res prediction, so the control's strided variant is the
    like-for-like comparison; the block-mean variant is what the original
    validated battery used and is kept as a cross-check.
    """
    import tifffile
    os.makedirs(CACHE, exist_ok=True)
    paths = {}
    fwd = tifffile.imread(CTRL_FWD).astype(np.float32)
    paths["w035_CONTROL_strided"] = os.path.join(CACHE, "w035_fwd_strided_ds4.npy")
    if not os.path.exists(paths["w035_CONTROL_strided"]):
        np.save(paths["w035_CONTROL_strided"], fwd[::4, ::4])
    paths["w035_CONTROL_blockmean"] = os.path.join(CACHE, "w035_fwd_blockmean_ds4.npy")
    if not os.path.exists(paths["w035_CONTROL_blockmean"]):
        np.save(paths["w035_CONTROL_blockmean"], block_mean(fwd, 4))
    # human ink labels, max-projected over z, on the control's canvas
    paths["w035_LABELS"] = os.path.join(CACHE, "w035_labels_strided_ds4.npy")
    if not os.path.exists(paths["w035_LABELS"]):
        lab3 = np.load(CTRL_LAB, mmap_mode="r")
        lab2d = np.zeros(lab3.shape[1:], np.uint8)
        for z in range(lab3.shape[0]):
            np.maximum(lab2d, np.asarray(lab3[z]), out=lab2d)
        lab2d = lab2d[: fwd.shape[0], : fwd.shape[1]] > 0
        np.save(paths["w035_LABELS"], (lab2d.astype(np.float32) * 255.0)[::4, ::4])
    return paths


def control_fwd_rev_r():
    """Map-scale fwd/rev correlation for the control, computed exactly as the
    survey computed it for every segment (full-res, common nonzero support),
    plus the eroded-mask variant used in the flag battery."""
    import tifffile
    a = tifffile.imread(CTRL_FWD).astype(np.float32)
    b = tifffile.imread(CTRL_REV).astype(np.float32)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    mm = (a > 0) & (b > 0)
    r_survey = float(np.corrcoef(a[mm], b[mm])[0, 1])
    me = ndi.binary_erosion(ndi.binary_fill_holes(
        ndi.binary_closing(a > 0, np.ones((5, 5), bool))), np.ones((3, 3), bool), 40)
    r_eroded = float(np.corrcoef(a[me], b[me])[0, 1])
    return r_survey, r_eroded


def build_units(only=None):
    survey = {}
    p = os.path.join(OUT, "survey_all.json")
    if os.path.exists(p):
        for r in json.load(open(p)):
            survey[r["name"]] = r
    v1 = {}
    p1 = os.path.join(OUT, "corpus_analysis.json")
    if os.path.exists(p1):
        for r in json.load(open(p1))["results"]:
            v1[r["name"]] = r

    units, meta = [], {}
    for m in sorted(glob.glob(os.path.join(OUT, "maps_shard*", "*_forward_ds4.npy"))):
        name = os.path.basename(m).replace("_forward_ds4.npy", "")
        if only and name not in only:
            continue
        rec = survey.get(name, {})
        scroll = rec.get("scroll", "PHerc1203")
        units.append({"key": name, "img": m, "px": PX_UM_DS4.get(scroll, 37.448),
                      "scroll": scroll, "kind": "segment"})
        meta[name] = {"name": name, "scroll": scroll, "kind": "segment",
                      "fwd_rev_r": rec.get("fwd_rev_r"),
                      "tripwire": len((rec.get("forward") or {}).get("tripwire_hits") or []),
                      "v1_ruling_z": (v1.get(name) or {}).get("ruling_z"),
                      "v1_period_mm": (v1.get(name) or {}).get("period_mm"),
                      "v1_prominence": (v1.get(name) or {}).get("ruling_prominence")}

    if not only or any(k.startswith("w035") for k in (only or [])):
        cp = build_control_cache()
        r_survey, r_eroded = control_fwd_rev_r()
        for key, path in cp.items():
            if only and key not in only:
                continue
            mask_src = cp["w035_CONTROL_strided"] if key == "w035_LABELS" else None
            units.append({"key": key, "img": path, "mask": mask_src,
                          "px": PX_UM_DS4["PHerc0139"], "scroll": "PHerc0139",
                          "kind": "control"})
            meta[key] = {"name": key, "scroll": "PHerc0139", "kind": "control",
                         "fwd_rev_r": r_survey, "fwd_rev_r_eroded_mask": r_eroded,
                         "tripwire": None, "v1_ruling_z": None}
    return units, meta


# ---------------------------------------------------------------------------
# gates + stats
# ---------------------------------------------------------------------------
def _perm_stats(obs_prom, nulls):
    nulls = np.asarray([x for x in nulls if x is not None and np.isfinite(x)], float)
    n = len(nulls)
    if n < 2:
        return {"n_perm": n, "empirical_p": float("nan")}
    mu, sd = float(nulls.mean()), float(nulls.std(ddof=1))
    z = (obs_prom - mu) / sd if sd > 0 else float("nan")
    n_ge = int((nulls >= obs_prom).sum())
    rng = np.random.default_rng(3)
    bz = np.array([(obs_prom - b.mean()) / max(b.std(ddof=1), 1e-12) for b in
                   (rng.choice(nulls, n, replace=True) for _ in range(2000))])
    return {"n_perm": n, "null_mean": round(mu, 3), "null_sd": round(sd, 3),
            "null_max": round(float(nulls.max()), 3),
            "null_p95": round(float(np.percentile(nulls, 95)), 3),
            "z_corrected": None if not np.isfinite(z) else round(float(z), 3),
            "z_ci95": [round(float(np.percentile(bz, 2.5)), 3),
                       round(float(np.percentile(bz, 97.5)), 3)],
            "empirical_p": round((1 + n_ge) / (n + 1), 5), "n_null_ge_obs": n_ge}


def summarize(obs, nulls_full, nulls_gated, meta, mask_frac, tile, px):
    d, bf, bg = obs["detail_full"], obs["full"], obs["gated"]
    dg = obs["detail_gated"]
    out = dict(meta)
    out.update({"mask_frac_eroded": round(mask_frac, 4), "perm_tile_ds4": tile,
                "px_um_ds4": round(px, 3),
                "obs_prominence": round(float(bf["prom"]), 3),
                "theta_deg": bf["theta"],
                "period_mm": None if bf["period_mm"] is None else round(bf["period_mm"], 3)})
    out.update(_perm_stats(bf["prom"], nulls_full))
    out.update({
        "profile_len_mm": round(d["profile_len_mm"], 2),
        "n_cycles": round(d["n_cycles"], 2),
        "band_bins": d["band_bins"], "peak_bin_index": d["peak_bin_index"],
        "autocorr_1P": None if d["autocorr_1P"] is None else round(d["autocorr_1P"], 4),
        "autocorr_2P": None if d["autocorr_2P"] is None else round(d["autocorr_2P"], 4),
        "autocorr_3P": None if d["autocorr_3P"] is None else round(d["autocorr_3P"], 4),
    })
    ac2, ac3 = d["autocorr_2P"], d["autocorr_3P"]
    r = meta.get("fwd_rev_r")
    out["gate_cycles"] = bool(d["n_cycles"] >= MIN_CYCLES)
    out["gate_autocorr"] = bool(ac2 is not None and ac3 is not None and ac2 > 0 and ac3 > 0)
    out["gate_band_bin"] = bool(d["peak_bin_index"] >= MIN_BAND_BIN)
    out["gate_fwd_rev"] = bool(r is not None and abs(r) < MAX_FWD_REV_R)

    # secondary, band-constrained search: guards against the gates hiding a
    # real ruling that lost the argmax to a band-edge artifact
    g = {"obs_prominence": round(float(bg["prom"]), 3), "theta_deg": bg["theta"],
         "period_mm": None if bg["period_mm"] is None or not np.isfinite(bg["period_mm"])
         else round(bg["period_mm"], 3)}
    if bg["prom"] > 0:
        g.update(_perm_stats(bg["prom"], nulls_gated))
        if dg:
            g.update({"n_cycles": round(dg["n_cycles"], 2),
                      "peak_bin_index": dg["peak_bin_index"],
                      "autocorr_1P": None if dg["autocorr_1P"] is None else round(dg["autocorr_1P"], 4),
                      "autocorr_2P": None if dg["autocorr_2P"] is None else round(dg["autocorr_2P"], 4),
                      "autocorr_3P": None if dg["autocorr_3P"] is None else round(dg["autocorr_3P"], 4)})
            g["gate_autocorr"] = bool(dg["autocorr_2P"] is not None and dg["autocorr_3P"] is not None
                                      and dg["autocorr_2P"] > 0 and dg["autocorr_3P"] > 0)
        g["gate_significance"] = bool(g.get("empirical_p", 1.0) <= ALPHA)
        g["passes_all_gates"] = bool(g.get("gate_significance") and g.get("gate_autocorr")
                                     and out["gate_fwd_rev"])
    out["constrained_search"] = g
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (monotone, step-down)."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=N_PERM)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(OUT, "corpus_analysis_v2.json"))
    ap.add_argument("--figure-only", action="store_true")
    args = ap.parse_args()
    if args.figure_only:
        import corpus_v2_figure
        corpus_v2_figure.main(args.out)
        return

    units, meta = build_units(args.only)
    print(f"v2 screen: {len(units)} units "
          f"({sum(1 for u in units if u['kind']=='segment')} segments + "
          f"{sum(1 for u in units if u['kind']=='control')} control), "
          f"N_PERM={args.nperm}, jobs={args.jobs}", flush=True)

    # decide permutation tile + drop units whose mask is unusable; order the
    # task list biggest-first so the longest map is not the tail of the run.
    sized = []
    for u in units:
        img, msk, mf = prep(u["img"], u.get("mask"))
        if img is None:
            meta[u["key"]]["status"] = f"skipped: eroded mask too small ({mf:.3f})"
            print(f"  SKIP {u['key']}: eroded mask {mf:.3f}", flush=True)
            continue
        t = TILE if n_tiles(msk, TILE) >= 8 else TILE_FALLBACK
        if n_tiles(msk, t) < 8:
            meta[u["key"]]["status"] = "skipped: <8 permutation tiles"
            print(f"  SKIP {u['key']}: <8 tiles", flush=True)
            continue
        if float(img[msk].std()) < 1.0:
            meta[u["key"]]["status"] = "skipped: near-constant response"
            print(f"  SKIP {u['key']}: near-constant", flush=True)
            continue
        u["tile"] = t
        sized.append((img.size, u, mf))
    sized.sort(key=lambda s: -s[0])
    print(f"  {len(sized)} units scoreable; largest {sized[0][0]/1e6:.1f} Mpx", flush=True)

    tasks = []
    for _, u, _ in sized:
        tasks.append((u, -1))
        tasks += [(u, i) for i in range(args.nperm)]

    from multiprocessing import Pool
    t0 = time.time()
    raw = {}
    done = 0
    with Pool(args.jobs) as pool:
        for key, draw, res in pool.imap_unordered(_task, tasks, chunksize=1):
            raw.setdefault(key, {})[draw] = res
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(tasks)} tasks  {el/60:.1f} min  "
                      f"eta {el/done*(len(tasks)-done)/60:.1f} min", flush=True)

    results = []
    for _, u, mf in sized:
        r = raw.get(u["key"], {})
        obs = r.get(-1)
        if obs is None or obs.get("detail_full") is None:
            meta[u["key"]]["status"] = "skipped: observed score failed"
            continue
        nf = [v["full"]["prom"] for k, v in r.items() if k >= 0 and v is not None]
        ng = [v["gated"]["prom"] for k, v in r.items() if k >= 0 and v is not None]
        rec = summarize(obs, nf, ng, meta[u["key"]], mf, u["tile"], u["px"])
        rec["status"] = "scored"
        results.append(rec)

    seg = [r for r in results if r["kind"] == "segment"]
    if seg:
        adj = holm(np.array([r["empirical_p"] for r in seg]))
        for r, a in zip(seg, adj):
            r["holm_p"] = round(float(a), 5)
    for r in results:
        # Gate on the RAW empirical p. With 200 permutations the smallest
        # attainable p is 1/201 = 0.00498, while Holm across ~70 tests would
        # demand p < 7.1e-4 -- unattainable at this permutation count, which
        # would make the gate vacuously unpassable. Holm-adjusted p is
        # reported alongside; any segment clearing all five gates is escalated
        # to a dedicated high-permutation run.
        r["gate_significance"] = bool(r["empirical_p"] <= ALPHA)
        r["gates_passed"] = int(sum(bool(r.get(g)) for g in
                                    ("gate_significance", "gate_cycles", "gate_autocorr",
                                     "gate_band_bin", "gate_fwd_rev")))
        r["passes_all_gates"] = bool(r["gates_passed"] == 5)

    seg.sort(key=lambda r: (r["empirical_p"], -(r["z_corrected"] or -99)))
    ctrl = [r for r in results if r["kind"] == "control"]
    skipped = [{"name": k, "status": v["status"]} for k, v in meta.items()
               if v.get("status", "").startswith("skipped")]

    payload = {
        "protocol": "v2",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"n_perm": args.nperm, "erode_ds4_px": ERODE_DS4,
                   "erode_fullres_px": ERODE_DS4 * 4, "detrend_sigma_ds4_px": DETREND,
                   "theta_step_deg": float(THETAS[1] - THETAS[0]),
                   "perm_tile_ds4_px": TILE, "perm_tile_fallback_ds4_px": TILE_FALLBACK,
                   "band_mm": list(BAND_MM), "min_cycles": MIN_CYCLES,
                   "min_band_bin_index": MIN_BAND_BIN,
                   "max_fwd_rev_r": MAX_FWD_REV_R, "alpha_holm": ALPHA,
                   "null": "joint (map, mask) block permutation",
                   "statistic": "band peak power / band median power (prominence)"},
        "gates": [f"gate_significance (raw empirical p <= {ALPHA})",
                  f"gate_cycles (>= {MIN_CYCLES} cycles in profile)",
                  "gate_autocorr (rho(2P) > 0 and rho(3P) > 0)",
                  f"gate_band_bin (peak not in band's {MIN_BAND_BIN} lowest Fourier bins)",
                  f"gate_fwd_rev (|fwd/rev r| < {MAX_FWD_REV_R})"],
        "multiplicity_note":
            f"min attainable p at N_PERM={args.nperm} is {1/(args.nperm+1):.5f}; "
            f"Holm across {len(seg)} segments would require p < {ALPHA/max(len(seg),1):.2e}, "
            "so gate_significance uses the raw p and holm_p is reported for context. "
            f"Under the null, E[# segments with p <= {ALPHA}] = {ALPHA*len(seg):.1f}.",
        "n_segments_scored": len(seg), "n_segments_skipped": len(skipped),
        "n_segments_p_le_alpha": sum(1 for r in seg if r["gate_significance"]),
        "expected_n_p_le_alpha_under_null": round(ALPHA * len(seg), 2),
        "n_segments_passing_all_gates": sum(1 for r in seg if r["passes_all_gates"]),
        "n_segments_passing_constrained_search":
            sum(1 for r in seg if r["constrained_search"].get("passes_all_gates")),
        "gate_pass_counts": {g: sum(1 for r in seg if r[g]) for g in
                             ("gate_significance", "gate_cycles", "gate_autocorr",
                              "gate_band_bin", "gate_fwd_rev")},
        "control": ctrl, "results": seg, "skipped": skipped,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {args.out}  [{(time.time()-t0)/60:.1f} min]")

    print("\n=== CONTROL ===")
    for r in ctrl:
        print(f"  {r['name']}: prom={r['obs_prominence']:.1f} z={r['z_corrected']} "
              f"p={r['empirical_p']} period={r['period_mm']}mm cycles={r['n_cycles']} "
              f"bin={r['peak_bin_index']}/{r['band_bins']} "
              f"ac={r['autocorr_1P']}/{r['autocorr_2P']}/{r['autocorr_3P']} "
              f"r={r['fwd_rev_r']} gates={r['gates_passed']}/5 PASS={r['passes_all_gates']}")
    print("\n=== CORPUS (top 12 by empirical p) ===")
    for r in seg[:12]:
        print(f"  {r['scroll']}/{r['name'][:42]:42s} p={r['empirical_p']:.4f} "
              f"holm={r['holm_p']:.3f} z={r['z_corrected']:+6.2f} "
              f"per={r['period_mm']}mm cyc={r['n_cycles']:.1f} bin={r['peak_bin_index']} "
              f"ac2={r['autocorr_2P']} r={r['fwd_rev_r']} gates={r['gates_passed']}/5")
    print(f"\nsegments passing ALL FIVE gates: {payload['n_segments_passing_all_gates']}"
          f" of {len(seg)}")
    for g, c in payload["gate_pass_counts"].items():
        print(f"  {g}: {c}/{len(seg)} pass")
    print(f"segments with p <= {ALPHA}: {payload['n_segments_p_le_alpha']} "
          f"(expected under null {payload['expected_n_p_le_alpha_under_null']})")
    print(f"constrained (band-restricted) search survivors: "
          f"{payload['n_segments_passing_constrained_search']}")
    zs = [r["z_corrected"] for r in seg if r["z_corrected"] is not None]
    if zs:
        print(f"corrected z: min {min(zs):+.2f}  median {np.median(zs):+.2f}  "
              f"max {max(zs):+.2f}")


if __name__ == "__main__":
    main()
