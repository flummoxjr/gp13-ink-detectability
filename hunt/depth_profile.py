#!/usr/bin/env python
"""Part B: sample the CT volume along each mesh normal at offsets -16..+16 voxels
and characterise the depth profile (is the mesh centred on a papyrus sheet?)."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\villa\vesuvius\src")
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from vesuvius.tifxyz import read_tifxyz
from vesuvius.ink_detection.volume_io import open_volume, read_bbox_with_padding

CACHE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\meshcache"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\out"
S3 = "s3://vesuvius-challenge-open-data/"
os.makedirs(OUT, exist_ok=True)

OFF = np.arange(-16.0, 16.01, 0.5)          # 65 samples, 0.5-voxel steps
ZERO = int(np.argmin(np.abs(OFF)))
PAD = 18
TILE = 112          # full-res grid points per tile side
SUB = 4             # subsample within tile
NTILES = 18


def pick_tiles(surf_stored_valid, scale, full_shape, ntiles, seed=0):
    """Pick tile origins (full-res row/col) inside fully-valid stored blocks."""
    h, w = surf_stored_valid.shape
    blk = max(2, int(round(TILE * scale)) + 1)   # stored cells spanned by a tile
    ok = []
    for i in range(0, h - blk):
        for j in range(0, w - blk):
            if surf_stored_valid[i:i + blk + 1, j:j + blk + 1].all():
                ok.append((i, j))
    if not ok:
        # fall back: any stored cell whose 3x3 nbhd is valid
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if surf_stored_valid[i - 1:i + 2, j - 1:j + 2].all():
                    ok.append((i, j))
    if not ok:
        return []
    ok = np.array(ok)
    rng = np.random.default_rng(seed)
    # greedy farthest-point spread for coverage
    sel = [int(rng.integers(len(ok)))]
    d = np.abs(ok - ok[sel[0]]).max(1).astype(float)
    while len(sel) < min(ntiles, len(ok)):
        k = int(np.argmax(d))
        if d[k] <= 0:
            break
        sel.append(k)
        d = np.minimum(d, np.abs(ok - ok[k]).max(1).astype(float))
    H, W = full_shape
    out = []
    for k in sel:
        r0 = min(int(ok[k][0] / scale), H - TILE - 1)
        c0 = min(int(ok[k][1] / scale), W - TILE - 1)
        out.append((max(r0, 0), max(c0, 0)))
    return out


def profile_mesh(seg, ntiles=NTILES):
    d = os.path.join(CACHE, seg["key"])
    surf = read_tifxyz(d, load_mask=False, validate=False)
    stored_valid = surf._valid_mask.copy()
    scale = float(surf._scale[0])
    surf = surf.use_full_resolution()
    H, W = surf.shape
    tiles = pick_tiles(stored_valid, scale, (H, W), ntiles)
    vol = open_volume(S3 + seg["volume"], 0)

    def do_tile(rc):
        r0, c0 = rc
        r1, c1 = min(r0 + TILE, H), min(c0 + TILE, W)
        x, y, z, valid = surf[r0:r1, c0:c1]
        nx, ny, nz = surf.get_normals(r0, r1, c0, c1)
        ok = valid & np.isfinite(nx) & np.isfinite(ny) & np.isfinite(nz)
        ok[::SUB, ::SUB] &= True
        m = np.zeros_like(ok); m[::SUB, ::SUB] = True
        ok = ok & m
        if ok.sum() < 20:
            return None
        z0 = int(np.floor(z[ok].min())) - PAD; z1 = int(np.ceil(z[ok].max())) + PAD + 1
        y0 = int(np.floor(y[ok].min())) - PAD; y1 = int(np.ceil(y[ok].max())) + PAD + 1
        x0 = int(np.floor(x[ok].min())) - PAD; x1 = int(np.ceil(x[ok].max())) + PAD + 1
        if (z1 - z0) * (y1 - y0) * (x1 - x0) > 4e8:
            return None
        crop, _ = read_bbox_with_padding(vol, (z0, y0, x0, z1, y1, x1), fill_value=0)
        crop = crop.astype(np.float32, copy=False)
        zi, yi, xi = z[ok] - z0, y[ok] - y0, x[ok] - x0
        nzo, nyo, nxo = nz[ok], ny[ok], nx[ok]
        prof = np.empty((zi.size, OFF.size), np.float32)
        for si, off in enumerate(OFF):
            coords = np.stack([zi + off * nzo, yi + off * nyo, xi + off * nxo])
            prof[:, si] = map_coordinates(crop, coords, order=1, mode="constant", cval=0.0)
        return prof

    profs = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        for p in ex.map(do_tile, tiles):
            if p is not None:
                profs.append(p)
    if not profs:
        return {"key": seg["key"], "error": "no tiles"}
    P = np.concatenate(profs, 0)
    secs = time.time() - t0
    return analyze_profiles(seg, P, len(tiles), secs, profs)


def _peak_and_centroid(p, win, cwin):
    """Peak offset (within win) and intensity-weighted centroid (within cwin)."""
    idx = np.nonzero(win)[0]
    j = idx[int(np.argmax(p[win]))]
    base = np.percentile(p, 10)
    w = np.clip(p - base, 0, None) * cwin
    c = float((w * OFF).sum() / max(w.sum(), 1e-9))
    return float(OFF[j]), c


def analyze_profiles(seg, P, ntiles, secs, profs=None):
    vox = seg["vox_um"]
    r = {"key": seg["key"], "scroll": seg["scroll"], "role": seg["role"],
         "fwd_rev_r": seg["fwd_rev_r"], "n_tiles": ntiles, "n_profiles": int(P.shape[0]),
         "secs": round(secs, 1)}
    # drop profiles that are entirely padding / air
    live = P.max(1) > 20
    r["frac_air_profiles"] = round(float(1 - live.mean()), 4)
    P = P[live]
    if P.shape[0] < 50:
        r["error"] = "too few live profiles"
        return r
    S = gaussian_filter1d(P, sigma=2.0, axis=1)    # 1.0-voxel smoothing

    r["mean_profile"] = [round(float(v), 2) for v in S.mean(0)]
    r["median_profile"] = [round(float(v), 2) for v in np.median(S, 0)]
    r["offsets_vox"] = [round(float(v), 2) for v in OFF]

    # --- where is the brightest material relative to offset 0? ------------
    win = np.abs(OFF) <= 10.0                      # the render window (+/-10 vox)
    idx_win = np.nonzero(win)[0]
    pk = idx_win[np.argmax(S[:, win], axis=1)]
    pk_off = OFF[pk]
    r["peak_off_med_vox"] = round(float(np.median(pk_off)), 3)
    r["peak_off_mad_vox"] = round(float(np.median(np.abs(pk_off - np.median(pk_off)))), 3)
    r["frac_peak_within_1vox"] = round(float((np.abs(pk_off) <= 1.0).mean()), 4)
    r["frac_peak_within_2vox"] = round(float((np.abs(pk_off) <= 2.0).mean()), 4)
    r["frac_peak_at_window_edge"] = round(float((np.abs(pk_off) >= 9.5).mean()), 4)

    # --- intensity-weighted centroid within +/-8 --------------------------
    cwin = np.abs(OFF) <= 8.0
    base = np.percentile(S, 10, axis=1, keepdims=True)
    Wt = np.clip(S - base, 0, None) * cwin
    sw = Wt.sum(1)
    cen = np.where(sw > 1e-6, (Wt * OFF).sum(1) / np.maximum(sw, 1e-6), np.nan)
    cen = cen[np.isfinite(cen)]
    r["centroid_off_med_vox"] = round(float(np.median(cen)), 3)
    r["centroid_off_iqr_vox"] = round(float(np.percentile(cen, 75) - np.percentile(cen, 25)), 3)
    r["frac_centroid_within_2vox"] = round(float((np.abs(cen) <= 2.0).mean()), 4)

    # --- contrast / sheet strength ----------------------------------------
    mx = S[:, cwin].max(1); mn = S[:, cwin].min(1)
    r["profile_contrast_med"] = round(float(np.median((mx - mn) / 255.0)), 4)
    r["intensity_at_0_med"] = round(float(np.median(S[:, ZERO])), 2)
    r["intensity_peak_med"] = round(float(np.median(mx)), 2)
    r["intensity_gain_peak_over_0"] = round(float(np.median(mx) / max(np.median(S[:, ZERO]), 1e-6)), 4)

    # --- sheet width (FWHM around the peak nearest 0) ---------------------
    fw = []
    for i in range(min(P.shape[0], 4000)):
        p = S[i]
        j = pk[i]
        half = 0.5 * (p[j] + np.percentile(p, 10))
        a = j
        while a > 0 and p[a] > half:
            a -= 1
        b = j
        while b < len(p) - 1 and p[b] > half:
            b += 1
        fw.append(OFF[b] - OFF[a])
    r["sheet_fwhm_med_vox"] = round(float(np.median(fw)), 3)
    r["sheet_fwhm_med_um"] = round(float(np.median(fw)) * vox, 1)

    # --- number of sheets crossed in +/-16 and their spacing --------------
    npk, spac = [], []
    for i in range(min(P.shape[0], 4000)):
        p = S[i]
        thr = p.min() + 0.4 * (p.max() - p.min())
        loc = np.nonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] > thr))[0] + 1
        npk.append(len(loc))
        if len(loc) > 1:
            spac.extend(np.diff(OFF[loc]))
    r["n_sheets_in_32vox_med"] = float(np.median(npk))
    if spac:
        r["sheet_spacing_med_vox"] = round(float(np.median(spac)), 2)
        r["sheet_spacing_med_um"] = round(float(np.median(spac)) * vox, 1)

    # --- symmetry of the profile about offset 0 ---------------------------
    Sm = S[:, np.abs(OFF) <= 10.0]
    Sm = Sm - Sm.mean(1, keepdims=True)
    num = (Sm * Sm[:, ::-1]).sum(1)
    den = (Sm ** 2).sum(1)
    sym = num / np.maximum(den, 1e-9)
    r["profile_symmetry_med"] = round(float(np.median(sym)), 4)      # 1 = perfectly symmetric about 0
    mp = np.asarray(r["mean_profile"])[np.abs(OFF) <= 10.0]
    mp = mp - mp.mean()
    r["mean_profile_symmetry"] = round(float((mp * mp[::-1]).sum() / max((mp ** 2).sum(), 1e-9)), 4)

    # ================= tile-level (texture-averaged) analysis =============
    # This is the scale the 21-slice render actually sees: a patch of sheet,
    # not a single fibre. Averaging ~800 points kills fibre noise.
    if profs:
        cwin8 = np.abs(OFF) <= 8.0
        tm = np.stack([gaussian_filter1d(p[p.max(1) > 20].mean(0), 2.0) for p in profs
                       if (p.max(1) > 20).sum() > 20])
        r["tile_mean_profiles"] = [[round(float(v), 2) for v in t] for t in tm]
        pk, cen = [], []
        for t in tm:
            a, b = _peak_and_centroid(t, win, cwin8)
            pk.append(a); cen.append(b)
        pk, cen = np.array(pk), np.array(cen)
        r["tile_peak_off_med_vox"] = round(float(np.median(pk)), 3)
        r["tile_peak_off_spread_vox"] = round(float(np.percentile(pk, 75) - np.percentile(pk, 25)), 3)
        r["tile_frac_peak_within_2vox"] = round(float((np.abs(pk) <= 2.0).mean()), 4)
        r["tile_centroid_off_med_vox"] = round(float(np.median(cen)), 3)
        r["tile_centroid_abs_med_vox"] = round(float(np.median(np.abs(cen))), 3)
        r["tile_centroid_off_spread_vox"] = round(float(np.percentile(cen, 75) - np.percentile(cen, 25)), 3)
        # modulation depth of the tile-mean profile = how sheet-like it is
        md = [(t[win].max() - t[win].min()) / max(t[win].mean(), 1e-6) for t in tm]
        r["tile_modulation_med"] = round(float(np.median(md)), 4)
        # symmetry of tile-mean profiles about offset 0
        tw = tm[:, win] - tm[:, win].mean(1, keepdims=True)
        ts = (tw * tw[:, ::-1]).sum(1) / np.maximum((tw ** 2).sum(1), 1e-9)
        r["tile_symmetry_med"] = round(float(np.median(ts)), 4)
        # ---- LOCAL SHEET-TRACKING ERROR --------------------------------
        # Within ONE tile (~1 mm of surface) how far does the sheet centre
        # wander away from the mesh?  A mesh locked onto a lamella has a small
        # spread; a mesh cutting obliquely across lamellae has a large one.
        # Restricted to tiles where the sheet is actually resolvable, otherwise
        # the centroid is noise and regresses to 0.
        track, tilt = [], []
        for p in profs:
            live = p.max(1) > 20
            if live.sum() < 50:
                continue
            q = gaussian_filter1d(p[live], 2.0, axis=1)
            tmn = q.mean(0)
            if (tmn[win].max() - tmn[win].min()) / max(tmn[win].mean(), 1e-6) < 0.15:
                continue
            b = np.percentile(q, 10, axis=1, keepdims=True)
            wq = np.clip(q - b, 0, None) * cwin8
            sw = wq.sum(1)
            c = np.where(sw > 1e-6, (wq * OFF).sum(1) / np.maximum(sw, 1e-6), np.nan)
            c = c[np.isfinite(c)]
            if c.size < 50:
                continue
            track.append(float(np.percentile(c, 75) - np.percentile(c, 25)))
            tilt.append(float(np.percentile(c, 90) - np.percentile(c, 10)))
        if track:
            r["n_tiles_sheet_resolvable"] = len(track)
            r["local_tracking_iqr_vox"] = round(float(np.median(track)), 3)
            r["local_tracking_p10p90_vox"] = round(float(np.median(tilt)), 3)
        else:
            r["n_tiles_sheet_resolvable"] = 0

        # is there a SECOND sheet inside the render window?  (straddle test)
        second = []
        for t in tm:
            tt = t[win]
            oo = OFF[win]
            thr = tt.min() + 0.5 * (tt.max() - tt.min())
            loc = np.nonzero((tt[1:-1] > tt[:-2]) & (tt[1:-1] >= tt[2:]) & (tt[1:-1] > thr))[0] + 1
            second.append(len(loc) >= 2)
        r["tile_frac_two_sheets_in_window"] = round(float(np.mean(second)), 4)
    return r


if __name__ == "__main__":
    segs = json.load(open(os.path.join(CACHE, "segs.json")))
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    res = []
    outp = os.path.join(OUT, "depth_profiles.json")
    if os.path.exists(outp):
        res = json.load(open(outp))
    done = {x["key"] for x in res if "error" not in x}
    for s in segs:
        if only and s["key"] not in only:
            continue
        if s["key"] in done:
            print("skip", s["key"]); continue
        t0 = time.time()
        try:
            r = profile_mesh(s)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"key": s["key"], "error": f"{type(e).__name__}: {e}"}
        res = [x for x in res if x["key"] != s["key"]] + [r]
        print(json.dumps({k: v for k, v in r.items() if not k.endswith("profile") and k != "offsets_vox"}), flush=True)
        json.dump(res, open(outp, "w"), indent=1)
    print("wrote", outp)
