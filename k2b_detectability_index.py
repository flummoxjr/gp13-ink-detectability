"""K2b — per-scroll ink-detectability index over the 13 GP scrolls (+ calibrators).

QC-hardened per qc/k1_k2_review.md: multiple well-separated papyrus ROIs per
scroll, air-noise reference PSDs from the same volume, and only the three
metrics that survived review:
  (a) structural bandwidth: max q (cycles/px) where papyrus PSD >= 2x air PSD
  (b) structural SNR at q=0.25 cyc/px (papyrus/air power)
  (c) DN headroom: in-mask p99.5-p0.5 spread in DN (window utilization)
Reported as median +/- IQR over ROIs. PHerc0139 (letters found at native 9um)
anchors the readable-class reference line.

Air reference: primary picker (central-band percentile gate) or, when none of
its windows exist or survive validation, a full-z search with smaller windows
(128^3 at L0) and a 0.85 fill gate. Every candidate must pass validation
(mean in-mask DN < 0.5x papyrus
mean, PSD flat-ish at high q); hull-clipped pockets are salvaged via a fully
in-mask sub-cube (>= 64^3). A scroll with no genuine air window gets a
SECONDARY noise reference — white floor from the papyrus ROIs' own 3x3x3
high-pass residuals — and "noise_ref": "residual" in its JSON (vs "air") so
downstream reporting can caveat it (never a dark-papyrus stand-in).

Run pattern: resumable — writes per-scroll JSON as it goes.
Outputs: trackD/out/k2b_index/<scroll>.json, k2b_index_summary.{json,png}
"""
import json
import os
import sys

import numpy as np

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
META_DIR = r"C:\Users\benbl\Desktop\Vsuvious\trackD\meta"
OUT = os.path.join(r"C:\Users\benbl\Desktop\Vsuvious\trackD\out", "k2b_index")
CACHE = r"D:\vesuvius-data\trackD\k2b"
ROI = 256
AIR_ROI = 128  # smaller fetch for fallback (full-z) air windows
N_PAP = 5
N_AIR = 2
MIN_SEP_L3 = 64  # voxels at level 3 between ROI centers
# air validation gates (see qc note in module docstring):
AIR_DN_FRAC = 0.5    # air mean in-mask DN must be < this x papyrus mean DN
AIR_FLAT_MAX = 40.0  # PSD mid/high-q ratio; genuine air ~14-27, papyrus ~80-200

SCROLLS = [
    ("PHerc0125", "20250821151825"), ("PHerc0191", "20250821151635"),
    ("PHerc0211", "20250821151803"), ("PHerc0257", "20250821151750"),
    ("PHerc0268", "20251110183117"), ("PHerc0358", "20250821151737"),
    ("PHerc0800", "20250521135224"), ("PHerc0813", "20250821151723"),
    ("PHerc0826", "20250821151701"), ("PHerc1218", "20250521120456"),
    ("PHerc1447", "20250521151220"), ("PHerc1545", "20250821151648"),
    ("PHerc1203", "20250820131727"),
    # calibrator: letters found at native 9.362um
    ("PHerc0139", "20250728140407"),
]


def vol_info(sample, vid):
    with open(os.path.join(META_DIR, f"{sample}.json"), encoding="utf-8") as f:
        s = json.load(f)
    v = s["volumes"][vid]
    c = v["creation"]["metadata"]
    return v["long_id"], v["properties"]["shape"], (c["target_window_f32_min"], c["target_window_f32_max"])


def open_level(sample, long_id, level):
    import zarr, fsspec
    return zarr.open(fsspec.get_mapper(f"{BUCKET}/{sample}/volumes/{long_id}"), mode="r")[str(level)]


def radial_psd(vol):
    n = vol.shape[0]
    w = np.hanning(n)
    W = w[:, None, None] * w[None, :, None] * w[None, None, :]
    v = (vol - vol.mean()) * W
    P = np.abs(np.fft.fftn(v)) ** 2 / W.sum()
    fz = np.fft.fftfreq(n)
    q = np.sqrt(fz[:, None, None] ** 2 + fz[None, :, None] ** 2 + fz[None, None, :] ** 2)
    bins = np.linspace(0, 0.5, 60)  # per-axis band only
    idx = np.digitize(q.ravel(), bins)
    psd = np.bincount(idx, weights=P.ravel(), minlength=len(bins) + 1)
    cnt = np.bincount(idx, minlength=len(bins) + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = psd / cnt
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, prof[1:len(bins)]


def psd_flatness(q, p):
    """Mid-q / high-q radial-PSD power ratio. Genuine air noise is near-white at
    high q (ratio ~14-27 empirically); papyrus structure falls much more steeply
    (~80-200), so this separates real air from dark papyrus."""
    lo = np.nanmedian(p[(q >= 0.18) & (q <= 0.30)])
    hi = np.nanmedian(p[(q >= 0.36) & (q <= 0.48)])
    return float(lo / hi)


def extract_air_subcube(a):
    """Largest fully in-mask cube (darkest placement) inside a partially masked
    air candidate — hull-clipped air pockets are genuine air, but their hard-zero
    mask voxels inject broadband edge power into the PSD, so the PSD must come
    from a fully covered sub-cube. Returns (cube, origin) or None if nothing
    >= 64^3 fits. radial_psd bins to a fixed q grid and is normalized per voxel,
    so PSD levels from smaller cubes remain comparable."""
    from scipy.ndimage import minimum_filter, uniform_filter
    n = a.shape[0]
    nz = a > 0
    for k in (3 * n // 4, 5 * n // 8, n // 2):
        if k < 64:
            break
        ok = minimum_filter(nz, size=k, mode="constant")
        if ok.any():
            inten = uniform_filter(a.astype(np.float32), size=k, mode="constant")
            inten[~ok] = np.inf
            c = np.unravel_index(np.argmin(inten), inten.shape)
            o = [int(np.clip(ci - k // 2, 0, n - k)) for ci in c]
            return a[o[0]:o[0] + k, o[1]:o[1] + k, o[2]:o[2] + k], o
    return None


def validate_air(a, pap_mean_dn):
    """Gate an air-candidate ROI. Returns (dict|None, reason)."""
    mask = a > 0
    sub_origin = None
    if mask.mean() < 0.85:
        sub = extract_air_subcube(a)
        if sub is None:
            return None, f"coverage {mask.mean():.2f}, no clean sub-cube"
        a, sub_origin = sub
        mask = a > 0
    mean_dn = float(a[mask].mean())
    if not mean_dn < AIR_DN_FRAC * pap_mean_dn:
        return None, f"too bright ({mean_dn:.1f} DN vs papyrus {pap_mean_dn:.1f})"
    q, p = radial_psd(a.astype(np.float32))
    flat = psd_flatness(q, p)
    if not flat < AIR_FLAT_MAX:
        return None, f"PSD not white-ish (flatness {flat:.1f})"
    return {"mean_dn": mean_dn, "flatness": flat, "psd": p,
            "sub_origin": sub_origin, "sub_n": int(a.shape[0])}, ""


def _highpass_gain_profile(n):
    """Radial profile of |1-U|^2 for the 3x3x3 uniform filter on an n^3 grid,
    binned exactly like radial_psd."""
    f = np.fft.fftfreq(n)
    u1 = (1.0 + 2.0 * np.cos(2 * np.pi * f)) / 3.0
    U = u1[:, None, None] * u1[None, :, None] * u1[None, None, :]
    hp = (1.0 - U) ** 2
    q = np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2)
    bins = np.linspace(0, 0.5, 60)
    idx = np.digitize(q.ravel(), bins)
    num = np.bincount(idx, weights=hp.ravel(), minlength=len(bins) + 1)
    cnt = np.bincount(idx, minlength=len(bins) + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = num / cnt
    return prof[1:len(bins)]


def residual_noise_level(a):
    """SECONDARY noise reference for scrolls with no genuine air window: white
    noise floor estimated from the ROI's own high-pass residual. The PSD of
    (ROI - its 3x3x3 uniform smooth) equals |1-U|^2 x (signal+noise); dividing
    out the |1-U|^2 gain and reading the 0.35-0.48 cyc/px band (where papyrus
    structure is weakest) gives the noise level, used as a flat reference PSD.
    Bias caveat: any true structure left at high q inflates the estimate, so
    SNR/bandwidth from this reference are conservative (lower bounds)."""
    from scipy.ndimage import uniform_filter
    af = a.astype(np.float32)
    r = af - uniform_filter(af, size=3, mode="nearest")
    q, pr = radial_psd(r)
    g = _highpass_gain_profile(a.shape[0])
    band = (q >= 0.35) & (q <= 0.48)
    return float(np.nanmedian(pr[band] / g[band]))


def nms_pick(score, n, min_sep):
    picks = []
    s = score.copy()
    for _ in range(n * 8):
        if not np.isfinite(s).any() or s.max() <= 0:
            break
        zyx = np.unravel_index(np.nanargmax(s), s.shape)
        if all(max(abs(a - b) for a, b in zip(zyx, p)) >= min_sep for p in picks):
            picks.append(zyx)
            if len(picks) >= n:
                break
        z, y, x = zyx
        s[max(z - 8, 0):z + 8, max(y - 8, 0):y + 8, max(x - 8, 0):x + 8] = 0
    return picks


def _probe_level(sample, long_id):
    """Level 3, or 4 when the central slab would exceed ~8e8 voxels (the
    float32 filters would OOM) — same rule the original picker used."""
    zl = open_level(sample, long_id, 3)
    level = 3
    zlo, zhi = zl.shape[0] // 6, 5 * zl.shape[0] // 6
    if (zhi - zlo) * zl.shape[1] * zl.shape[2] > 8e8:
        zl = open_level(sample, long_id, 4)
        level = 4
        zlo, zhi = zl.shape[0] // 6, 5 * zl.shape[0] // 6
    return zl, level, zlo, zhi


def pick_rois(sample, long_id, shape0):
    """From level 3 (or 4 for oversized volumes): papyrus + interior-air windows."""
    from scipy.ndimage import uniform_filter
    z3, level_used, zlo, zhi = _probe_level(sample, long_id)
    arr = np.asarray(z3[zlo:zhi])
    nz = arr > 0
    fill = uniform_filter(nz.astype(np.float32), size=32, mode="constant")
    inten = uniform_filter(arr.astype(np.float32), size=32, mode="constant")
    inmask = arr[nz]
    air_hi = np.percentile(inmask, 12)

    pap_score = np.where(fill > 0.98, inten, 0)
    pap = nms_pick(pap_score, N_PAP, MIN_SEP_L3 // 2)
    air_score = np.where((fill > 0.95) & (inten < air_hi), 1.0 + (air_hi - inten), 0)
    air = nms_pick(air_score, N_AIR, MIN_SEP_L3 // 2)

    scale = 2 ** level_used
    def to_l0(zyx):
        z, y, x = zyx
        z0 = int(np.clip((z + zlo) * scale - ROI // 2, 0, shape0[0] - ROI))
        y0 = int(np.clip(y * scale - ROI // 2, 0, shape0[1] - ROI))
        x0 = int(np.clip(x * scale - ROI // 2, 0, shape0[2] - ROI))
        return z0, y0, x0

    return [to_l0(p) for p in pap], [to_l0(p) for p in air]


def pick_air_fallback(sample, long_id, shape0):
    """Full-z air-candidate search, used when the primary picker's central-band
    air windows are absent or fail validation: smaller windows (AIR_ROI^3 at
    L0) and a relaxed 0.85 fill gate, ranked darkest-first. Returns candidate
    L0 origins for AIR_ROI^3 fetches; these are CANDIDATES only — each must
    still pass validate_air(), and a scroll where none passes gets the
    residual noise reference (never a dark-papyrus stand-in)."""
    from scipy.ndimage import uniform_filter
    zl, level, _, _ = _probe_level(sample, long_id)
    arr = np.asarray(zl)  # FULL z-range
    scale = 2 ** level
    size = max(AIR_ROI // scale, 4)
    nz = arr > 0
    fill = uniform_filter(nz.astype(np.float32), size=size, mode="constant")
    del nz
    inten = uniform_filter(arr.astype(np.float32), size=size, mode="constant")
    del arr
    dark = np.where(fill > 0.85, 1e6 - inten, 0)
    cands = nms_pick(dark, 4 * N_AIR, MIN_SEP_L3 // 2)

    def to_l0(zyx):
        z, y, x = zyx
        z0 = int(np.clip(z * scale - AIR_ROI // 2, 0, shape0[0] - AIR_ROI))
        y0 = int(np.clip(y * scale - AIR_ROI // 2, 0, shape0[1] - AIR_ROI))
        x0 = int(np.clip(x * scale - AIR_ROI // 2, 0, shape0[2] - AIR_ROI))
        return z0, y0, x0

    return [to_l0(p) for p in cands]


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    summary = {}
    for sample, vid in SCROLLS:
        out_json = os.path.join(OUT, f"{sample}.json")
        if os.path.exists(out_json):
            with open(out_json) as f:
                summary[sample] = json.load(f)
            print(f"{sample}: cached", flush=True)
            continue
        try:
            long_id, shape0, win = vol_info(sample, vid)
            lo, hi = win
            step_dn = 1.0  # work in DN units (uint8 counts) directly
            pap_rois, air_rois = pick_rois(sample, long_id, shape0)
            print(f"{sample}: {len(pap_rois)} papyrus + {len(air_rois)} air ROIs", flush=True)
            z0l = open_level(sample, long_id, 0)

            def fetch(origin, tag, i, roi=ROI):
                cpath = os.path.join(CACHE, f"{sample}_{tag}{i}.npy")
                if os.path.exists(cpath):
                    return np.load(cpath)
                a = np.asarray(z0l[origin[0]:origin[0] + roi,
                                   origin[1]:origin[1] + roi,
                                   origin[2]:origin[2] + roi])
                np.save(cpath, a)
                return a

            # papyrus ROIs first — air validation needs their mean in-mask DN
            pap_data = []
            for i, o in enumerate(pap_rois):
                a = fetch(o, "pap", i)
                if (a > 0).mean() < 0.9:
                    continue
                pap_data.append((o, a))
            if not pap_data:
                raise RuntimeError("no valid papyrus ROI")
            pap_mean_dn = float(np.median([a[a > 0].mean() for _, a in pap_data]))

            # air reference: every candidate must pass the DN + PSD-whiteness
            # gates; a scroll with no genuine air gets the residual reference
            # instead of a dark-papyrus stand-in.
            air_ok = []
            for i, o in enumerate(air_rois):
                got, why = validate_air(fetch(o, "air", i), pap_mean_dn)
                if got:
                    air_ok.append({"origin": list(o), "roi": ROI, **got})
                else:
                    print(f"{sample}: air ROI {o} rejected — {why}", flush=True)
            if not air_ok:
                # no primary air window survived validation (or none existed):
                # full-z search with smaller windows and relaxed fill gate
                air2_rois = pick_air_fallback(sample, long_id, shape0)
                print(f"{sample}: full-z fallback search — "
                      f"{len(air2_rois)} air candidates", flush=True)
                for o in air2_rois:
                    if len(air_ok) >= N_AIR:
                        break
                    got, why = validate_air(
                        fetch(o, "airs", "_".join(map(str, o)), AIR_ROI), pap_mean_dn)
                    if got:
                        air_ok.append({"origin": list(o), "roi": AIR_ROI, **got})
                    else:
                        print(f"{sample}: air candidate {o} rejected — {why}", flush=True)

            if air_ok:
                noise_ref = "air"
                air_psd = np.nanmedian(np.stack([d["psd"] for d in air_ok]), axis=0)
            else:
                noise_ref = "residual"
                print(f"{sample}: no genuine air window — using papyrus high-pass "
                      "residual noise reference", flush=True)

            rows = []
            for o, a in pap_data:
                mask = a > 0
                q, p = radial_psd(a.astype(np.float32))
                npsd = air_psd if noise_ref == "air" else residual_noise_level(a)
                with np.errstate(invalid="ignore", divide="ignore"):
                    snr = p / npsd
                valid = q > 0.02
                above = valid & (snr >= 2.0)
                bw = float(q[above].max()) if above.any() else 0.0
                i25 = int(np.argmin(np.abs(q - 0.25)))
                snr25 = float(snr[i25])
                dn = a[mask].astype(np.float32)
                headroom = float(np.percentile(dn, 99.5) - np.percentile(dn, 0.5))
                rows.append({"origin": list(o), "bandwidth_cyc_px": round(bw, 4),
                             "snr_q025": round(snr25, 2), "dn_headroom": round(headroom, 1)})
            if not rows:
                raise RuntimeError("no valid papyrus ROI")

            def med_iqr(key):
                v = np.array([r[key] for r in rows])
                return [round(float(np.median(v)), 4),
                        round(float(np.percentile(v, 25)), 4),
                        round(float(np.percentile(v, 75)), 4)]

            summary[sample] = {
                "n_pap_rois": len(rows), "rois": rows,
                "bandwidth_med_iqr": med_iqr("bandwidth_cyc_px"),
                "snr_q025_med_iqr": med_iqr("snr_q025"),
                "dn_headroom_med_iqr": med_iqr("dn_headroom"),
                "noise_ref": noise_ref,
                "pap_mean_dn": round(pap_mean_dn, 1),
            }
            if noise_ref == "air":
                summary[sample]["air_rois"] = [
                    {"origin": d["origin"], "roi": d["roi"],
                     "sub_origin": d["sub_origin"], "sub_n": d["sub_n"],
                     "mean_dn": round(d["mean_dn"], 1),
                     "psd_flatness": round(d["flatness"], 1)} for d in air_ok]
            else:
                summary[sample]["noise_note"] = (
                    "no genuine air window passed validation; noise ref is the PSD "
                    "of (papyrus ROI - its 3x3x3 uniform smooth), |1-U|^2-corrected "
                    "white floor — SNR and bandwidth are conservative estimates")
            with open(out_json, "w") as f:
                json.dump(summary[sample], f, indent=1)
            print(f"{sample}: bw={summary[sample]['bandwidth_med_iqr']} "
                  f"snr25={summary[sample]['snr_q025_med_iqr']} "
                  f"dn={summary[sample]['dn_headroom_med_iqr']}", flush=True)
        except Exception as e:
            print(f"{sample}: FAILED — {e}", flush=True)
            summary[sample] = {"error": str(e)[:300]}

    with open(os.path.join(OUT, "k2b_index_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ranking figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [s for s, _ in SCROLLS if s in summary and "bandwidth_med_iqr" in summary[s]]
    if names:
        bw = [summary[s]["bandwidth_med_iqr"] for s in names]
        sn = [summary[s]["snr_q025_med_iqr"] for s in names]
        order = np.argsort([b[0] for b in bw])[::-1]
        labels = [names[i] + ("*" if summary[names[i]].get("noise_ref") == "residual"
                              else "") for i in order]
        fig, axes = plt.subplots(1, 2, figsize=(15, 0.45 * len(names) + 2), squeeze=False)
        ypos = np.arange(len(names))
        for ax, vals, title in [(axes[0][0], bw, "structural bandwidth (cyc/px, 2x air)"),
                                (axes[0][1], sn, "structural SNR @ q=0.25")]:
            med = [vals[i][0] for i in order]
            lo_ = [vals[i][0] - vals[i][1] for i in order]
            hi_ = [vals[i][2] - vals[i][0] for i in order]
            colors = ["#C8791E" if names[i] == "PHerc0139" else "#3A6B8C" for i in order]
            ax.barh(ypos, med, xerr=[lo_, hi_], color=colors, height=0.6)
            ax.set_yticks(ypos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=10)
        fig.tight_layout()
        if any("*" in l for l in labels):
            fig.text(0.01, 0.005, "* noise ref = papyrus high-pass residual "
                     "(no genuine air window found)", fontsize=7)
        fig.savefig(os.path.join(OUT, "k2b_index_summary.png"), dpi=110)
        print("wrote k2b_index_summary.png", flush=True)


if __name__ == "__main__":
    main()
