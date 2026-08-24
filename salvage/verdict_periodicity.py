"""Test 1 (line periodicity) + Test 3 (quiet bands / anisotropy) for ink9um 1203 verdict.

Method: block-mean ds4 (37.4 um/px). For each map, search orientation theta in
[0,180) by rotating map+mask, taking the masked row-mean projection profile,
detrending, Hann-windowing, and scoring the max power in the text-ruling band
(period 40-250 full-res px = 0.37-2.34 mm) normalized by the band's median power
("prominence"). Null: joint block-permutation of (map, mask) in 256-px (full-res)
tiles -- preserves stroke-scale texture, destroys long-range row alignment --
run through the IDENTICAL search. Report z-score of real prominence vs null.
"""
import sys, time
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import (load_map, valid_mask, load_w035_label2d, downsample,
                            save_json, SALVAGE, MAPS)

DS = 4                      # analysis scale
# Band calibrated from w035 human labels (verdict_labelgeom.json):
# letter height ~242 full-res px, row spacing ~490 full-res px (4.6 mm).
PERIOD_LO = 180 / DS        # ds4 px  (180 full-res px = 1.7 mm)
PERIOD_HI = 900 / DS        # ds4 px  (900 full-res px = 8.4 mm)
DETREND_SIGMA = 90          # ds4 px
MIN_ROWFRAC = 0.25
NULL_TILE = 64              # ds4 px = 256 full-res px (< half the ruling period)
N_NULL = 5
ERODE = 40                  # full-res px; kills the bright rim on 1203 patch edges
UM_PER_PX = 9.362


def profile_at(img, msk, theta, min_rowfrac=MIN_ROWFRAC):
    """Masked row-mean projection after rotating by theta (degrees)."""
    stack = np.stack([img * msk, msk.astype(np.float32)])
    rot = ndi.rotate(stack, theta, axes=(1, 2), reshape=True, order=1, prefilter=False)
    num, den = rot[0], rot[1]
    cnt = den.sum(axis=1)
    keep = cnt >= min_rowfrac * cnt.max()
    if keep.sum() < 64:
        return None, None
    i0, i1 = np.flatnonzero(keep)[[0, -1]]
    num_s = num[i0:i1 + 1].sum(axis=1)
    cnt_s = cnt[i0:i1 + 1]
    bad = cnt_s < min_rowfrac * cnt.max()
    prof = np.where(~bad, num_s / np.maximum(cnt_s, 1e-6), np.nan)
    # interpolate interior gaps
    if np.isnan(prof).any():
        x = np.arange(prof.size)
        ok = ~np.isnan(prof)
        prof = np.interp(x, x[ok], prof[ok])
    # column profile for anisotropy
    ccnt = den[i0:i1 + 1].sum(axis=0)
    ckeep = ccnt >= min_rowfrac * ccnt.max()
    cprof = None
    if ckeep.sum() >= 64:
        j0, j1 = np.flatnonzero(ckeep)[[0, -1]]
        cnum = num[i0:i1 + 1, j0:j1 + 1].sum(axis=0)
        cden = den[i0:i1 + 1, j0:j1 + 1].sum(axis=0)
        cprof = cnum / np.maximum(cden, 1e-6)
    return prof.astype(np.float64), (None if cprof is None else cprof.astype(np.float64))


def band_spectrum(prof):
    """Detrend, window, rfft power; return freqs(period ds4 px), power, band mask."""
    p = prof - gaussian_filter1d(prof, DETREND_SIGMA, mode="reflect")
    p = p - p.mean()
    n = p.size
    w = np.hanning(n)
    P = np.abs(np.fft.rfft(p * w)) ** 2
    freq = np.fft.rfftfreq(n)          # cycles / ds4 px
    with np.errstate(divide="ignore"):
        period = np.where(freq > 0, 1.0 / freq, np.inf)
    band = (period >= PERIOD_LO) & (period <= PERIOD_HI)
    return period, P, band, p


def score_profile(prof):
    period, P, band, pd = band_spectrum(prof)
    Pb = P[band]
    if Pb.size < 8:
        return None
    k = np.argmax(Pb)
    prom = Pb[k] / np.median(Pb)
    return {"prom": float(prom), "period_ds": float(period[band][k]),
            "detrended": pd, "period_axis": period[band], "power_band": Pb}


def orientation_search(img, msk, thetas):
    best = None
    curve = []
    for th in thetas:
        prof, _ = profile_at(img, msk, th)
        if prof is None:
            curve.append((th, np.nan, np.nan))
            continue
        s = score_profile(prof)
        if s is None:
            curve.append((th, np.nan, np.nan))
            continue
        curve.append((th, s["prom"], s["period_ds"]))
        if best is None or s["prom"] > best[1]["prom"]:
            best = (th, s)
    return best, np.array(curve)


def block_permute(img, msk, tile, rng):
    H, W = img.shape
    h, w = H // tile, W // tile
    im = img[:h * tile, :w * tile].reshape(h, tile, w, tile).transpose(0, 2, 1, 3).reshape(h * w, tile, tile)
    mk = msk[:h * tile, :w * tile].reshape(h, tile, w, tile).transpose(0, 2, 1, 3).reshape(h * w, tile, tile)
    perm = rng.permutation(h * w)
    im2 = im[perm].reshape(h, w, tile, tile).transpose(0, 2, 1, 3).reshape(h * tile, w * tile)
    mk2 = mk[perm].reshape(h, w, tile, tile).transpose(0, 2, 1, 3).reshape(h * tile, w * tile)
    return im2, mk2


def analyze(name, img_full, mask_full, do_null=True, save_rot=None):
    t0 = time.time()
    img = downsample(img_full, DS)
    msk = (downsample(mask_full.astype(np.float32), DS) > 0.6).astype(np.float32)
    img = img * msk

    coarse = np.arange(0, 180, 2.0)
    best, curve = orientation_search(img, msk, coarse)
    fine = np.arange(best[0] - 3, best[0] + 3.01, 0.5)
    bestf, _ = orientation_search(img, msk, fine)
    th, s = bestf

    # full metrics at best theta
    prof, cprof = profile_at(img, msk, th)
    period, P, band, pdet = band_spectrum(prof)
    Pb, perb = P[band], period[band]
    k = int(np.argmax(Pb))
    prom = float(Pb[k] / np.median(Pb))
    per_full = float(perb[k] * DS)                     # full-res px
    per_mm = per_full * UM_PER_PX / 1000.0

    # quiet bands + anisotropy (test 3)
    psm = gaussian_filter1d(prof, 4.0)
    med, std = np.median(psm), psm.std()
    quiet_frac = float((psm < med - 0.5 * std).mean())
    p10, p90 = np.percentile(psm, [10, 90])
    modulation = float((p90 - p10) / max(med, 1.0))
    aniso = np.nan
    if cprof is not None:
        rd = prof - gaussian_filter1d(prof, DETREND_SIGMA)
        cd = cprof - gaussian_filter1d(cprof, DETREND_SIGMA)
        aniso = float(rd.std() / max(cd.std(), 1e-9))

    res = {"theta_deg": float(th), "prominence": prom,
           "period_px_fullres": per_full, "period_mm": per_mm,
           "quiet_frac": quiet_frac, "modulation": modulation,
           "row_col_anisotropy": aniso, "n_profile": int(prof.size)}

    if do_null:
        rng = np.random.default_rng(1234)
        null_proms = []
        nth = np.arange(0, 180, 3.0)
        for i in range(N_NULL):
            im2, mk2 = block_permute(img, msk, NULL_TILE, rng)
            nb, _ = orientation_search(im2, mk2, nth)
            if nb is not None:
                null_proms.append(nb[1]["prom"])
        null_proms = np.array(null_proms)
        res["null_prom_mean"] = float(null_proms.mean())
        res["null_prom_std"] = float(null_proms.std(ddof=1))
        res["null_prom_max"] = float(null_proms.max())
        res["z_vs_null"] = float((prom - null_proms.mean()) / max(null_proms.std(ddof=1), 1e-9))

    if save_rot is not None:
        stack = np.stack([img, msk])
        rot = ndi.rotate(stack, th, axes=(1, 2), reshape=True, order=1, prefilter=False)
        np.savez_compressed(save_rot, img=rot[0].astype(np.float32),
                            msk=rot[1].astype(np.float32), theta=th,
                            profile=prof, period_band=perb, power_band=Pb,
                            k_peak=k)
    res["seconds"] = round(time.time() - t0, 1)
    print(f"{name}: theta={th:.1f}  prom={prom:.1f}  period={per_full:.0f}px "
          f"({per_mm:.2f}mm)  z={res.get('z_vs_null', float('nan')):.1f}  "
          f"quiet={quiet_frac:.2f} mod={modulation:.3f} aniso={aniso:.2f} "
          f"[{res['seconds']}s]", flush=True)
    return res


def main():
    results = {}
    # ground-truth labels first (defines the true ruling)
    lab = load_w035_label2d().astype(np.float32) * 255.0
    w035 = load_map("w035_s42").astype(np.float32)
    m035 = valid_mask(w035, erode=ERODE)
    results["w035_LABELS"] = analyze("w035_LABELS", lab, m035, do_null=True,
                                     save_rot=SALVAGE / "rot_w035_LABELS.npz")

    todo = ["w035_s42", "w035_s43", "w035_rend42",
            "1203A_s42", "1203A_s42r", "1203A_s43", "1203A_s43r",
            "1203B_s42", "1203B_s42r", "1203B_s43", "1203B_s43r"]
    for key in todo:
        a = load_map(key).astype(np.float32)
        m = valid_mask(a, erode=ERODE)
        save_rot = SALVAGE / f"rot_{key}.npz" if key in ("w035_s42", "1203A_s42", "1203B_s42") else None
        results[key] = analyze(key, a, m, do_null=True, save_rot=save_rot)

    save_json(SALVAGE / "verdict_periodicity.json", results)
    print("saved verdict_periodicity.json")


if __name__ == "__main__":
    main()
