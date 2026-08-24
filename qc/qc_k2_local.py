"""QC audit of K2 (spectral ceiling) — local part, uses cached ROIs only.

  A. Validate radial_psd on synthetic white noise (flatness, bin alignment,
     DC leakage into the first reported bin, normalization convention).
  B. Quantization-floor model: quantize a synthetic red-spectrum field with
     the same DN step; compare PSD(quantization error) against the uniform
     (-step/2, step/2) model used by K2.  Also: additivity (is
     PSD(quantized) ~= PSD(clean) + floor, i.e. is the data-vs-floor
     comparison double-counting?).
  C. Recompute the three cached-ROI PSDs; report the data/floor ratio across
     the per-axis band (0..0.5 cyc/px), the minimum ratio, any sub-Nyquist
     crossing of the 2x-floor criterion, and the log-slope profile (where the
     'cliff' actually is).
  D. Verify the transfer-function overlay numerically (Paganin lambda, filter
     magnitude at key frequencies, unsharp gaussian) and quantify how much a
     2D-only (in-plane) unsharp would differ from the isotropic 3D model in
     the radial average.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD")
from k2_spectral_ceiling import radial_psd, transfer_function, vol_node  # noqa: E402

CACHE = r"D:\vesuvius-data\trackD"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc"
TARGETS = [
    ("PHerc0813", "20250821151723"),
    ("PHerc1203", "20250820131727"),
    ("PHerc0139", "20250728140407"),
]


def main():
    res = {}
    rng = np.random.default_rng(1)

    # ---------- A: synthetic white noise ----------
    print("[A] radial_psd on white gaussian noise, sigma=1, 256^3")
    wn = rng.normal(0, 1, (256, 256, 256)).astype(np.float32)
    q, p = radial_psd(wn)
    n = 256
    w = np.hanning(n)
    W3sum = w.sum() ** 3
    W3sq = (w ** 2).sum() ** 3
    # standard periodogram convention: E[|F|^2] = sigma^2 * sum(W^2); K2 divides by sum(W)
    expect_k2 = W3sq / W3sum  # expected flat level under K2's normalization
    inband = (q > 0.02) & (q < 0.5)
    lvl = p[inband]
    print(f"  mean level in 0.02-0.5: {lvl.mean():.4e}, expected {expect_k2:.4e} "
          f"(ratio {lvl.mean()/expect_k2:.3f}), flatness max/min={lvl.max()/lvl.min():.3f}")
    print(f"  first reported bin q={q[0]:.4f} level={p[0]:.4e} "
          f"(DC leaks here; {p[0]/lvl.mean():.2f}x flat level)")
    res["A_white_noise"] = {"flat_ratio_to_expected": float(lvl.mean() / expect_k2),
                            "flatness_maxmin": float(lvl.max() / lvl.min()),
                            "first_bin_over_flat": float(p[0] / lvl.mean())}

    # ---------- B: quantization model ----------
    print("\n[B] quantization-error spectrum vs uniform model (red-spectrum field)")
    # synth field with a steep red spectrum + small white noise, matching papyrus-ish DN stats
    f3 = np.fft.fftfreq(256)
    qq = np.sqrt(f3[:, None, None] ** 2 + f3[None, :, None] ** 2 + f3[None, None, :] ** 2)
    amp = 1.0 / (0.01 + qq) ** 2.2
    amp[0, 0, 0] = 0
    field = np.fft.ifftn(amp * np.exp(2j * np.pi * rng.random(qq.shape))).real
    field /= field.std()
    step_dn = 1.0
    for noise_dn in (0.0, 1.5):
        sig = field * 40 + 120 + rng.normal(0, noise_dn, field.shape)  # in DN units
        qz = np.round(sig).clip(0, 255)
        err = qz - sig
        _, p_err = radial_psd(err.astype(np.float32))
        u = rng.uniform(-step_dn / 2, step_dn / 2, field.shape).astype(np.float32)
        _, p_u = radial_psd(u)
        band = (q > 0.05) & (q < 0.5)
        r = p_err[band] / p_u[band]
        print(f"  scan-noise {noise_dn} DN: err-PSD / uniform-model in 0.05-0.5: "
              f"median {np.median(r):.3f}, min {r.min():.3f}, max {r.max():.3f}")
        res[f"B_err_vs_model_noise{noise_dn}"] = {"median": float(np.median(r)),
                                                  "min": float(r.min()), "max": float(r.max())}
    # additivity: PSD(quantized) vs PSD(clean)+floor at high q
    _, p_clean = radial_psd((field * 40 + 120).astype(np.float32))
    _, p_qz = radial_psd(np.round(field * 40 + 120).clip(0, 255).astype(np.float32))
    hi = (q > 0.3) & (q < 0.5)
    print(f"  additivity at 0.3-0.5: PSD(quantized)/[PSD(clean)+floor] median "
          f"{np.median(p_qz[hi]/(p_clean[hi]+p_u[hi])):.3f}")

    # ---------- C: cached ROI recompute ----------
    print("\n[C] cached ROIs: data/floor ratio across per-axis band")
    for sample, vid in TARGETS:
        meta = vol_node(sample, vid)
        roi = np.load(os.path.join(CACHE, f"k2_roi_{sample}_{vid}.npy"))
        lo, hi_w = meta["win"]
        f = roi.astype(np.float32) / 255.0 * (hi_w - lo) + lo
        q, psd = radial_psd(f)
        rng0 = np.random.default_rng(0)
        step = (hi_w - lo) / 255.0
        _, psd_q = radial_psd(rng0.uniform(-step / 2, step / 2, roi.shape).astype(np.float32))
        band = (q > 0.005) & (q <= 0.5)
        ratio = psd[band] / psd_q[band]
        qb = q[band]
        # log-slope profile to locate the cliff
        ls = np.gradient(np.log10(psd[band]), np.log10(qb))
        icliff = int(np.argmin(ls))
        # 2x-floor crossings inside the per-axis band
        cross = qb[ratio < 2]
        r05 = float(ratio[-1])
        stats = {
            "ratio_at_0.05": float(np.interp(0.05, qb, ratio)),
            "ratio_at_0.25": float(np.interp(0.25, qb, ratio)),
            "ratio_at_0.35": float(np.interp(0.35, qb, ratio)),
            "ratio_at_0.5": r05,
            "min_ratio_below_0.5": float(ratio.min()),
            "crossings_below_0.5": [float(c) for c in cross],
            "steepest_logslope": float(ls[icliff]),
            "steepest_logslope_at_q": float(qb[icliff]),
            "roi_dn_mean": float(roi.mean()), "roi_dn_std": float(roi.std()),
            "roi_dn_minmax": [int(roi.min()), int(roi.max())],
            "dn_at_255_frac": float((roi == 255).mean()),
            "dn_at_0_frac": float((roi == 0).mean()),
        }
        res[f"C_{sample}"] = stats
        print(f"  {sample}: ratio@0.05={stats['ratio_at_0.05']:.1f} @0.25={stats['ratio_at_0.25']:.1f} "
              f"@0.35={stats['ratio_at_0.35']:.1f} @0.5={stats['ratio_at_0.5']:.1f} "
              f"min={stats['min_ratio_below_0.5']:.1f} crossings<0.5: {len(cross)}")
        print(f"           steepest log-slope {stats['steepest_logslope']:.1f} at q={stats['steepest_logslope_at_q']:.3f}; "
              f"DN mean/std {stats['roi_dn_mean']:.0f}/{stats['roi_dn_std']:.0f}, "
              f"clip frac 0:{stats['dn_at_0_frac']:.4f} 255:{stats['dn_at_255_frac']:.4f}")

    # ---------- D: transfer function ----------
    print("\n[D] transfer-function verification (PHerc0813 meta)")
    meta = vol_node("PHerc0813", "20250821151723")
    lam = 1.23984193e-9 / meta["energy_keV"]
    print(f"  lambda = {lam*1e12:.4f} pm (113 keV -> expect ~10.97 pm)")
    qs = np.array([0.05, 0.1, 0.25, 0.5])
    H = transfer_function(qs, meta)
    px = meta["px_um"] * 1e-6
    zdist = meta["dist_mm"] * 1e-3
    pag_manual = 1 / (1 + np.pi * lam * zdist * meta["delta_beta"] * (qs / px) ** 2)
    g = np.exp(-2 * np.pi ** 2 * meta["unsharp_sigma"] ** 2 * qs ** 2)
    uns_manual = 1 + meta["unsharp_coeff"] * (1 - g)
    print(f"  q={qs}")
    print(f"  paganin  = {pag_manual}")
    print(f"  unsharp  = {uns_manual}")
    print(f"  product  = {pag_manual*uns_manual}")
    print(f"  script H = {H}   match: {np.allclose(H, pag_manual*uns_manual)}")
    res["D_transfer"] = {"lambda_pm": lam * 1e12, "match": bool(np.allclose(H, pag_manual * uns_manual)),
                         "H_at_q": {float(a): float(b) for a, b in zip(qs, H)}}
    # 2D-in-plane unsharp vs isotropic-3D model, sphere-averaged |H|^2
    print("  sphere-averaged |unsharp|^2: in-plane-only (xy) vs isotropic model")
    for qv in (0.1, 0.25, 0.4):
        ct = rng.random(200000)  # cos(theta) uniform on sphere
        qxy = qv * np.sqrt(1 - ct ** 2)
        g2d = np.exp(-2 * np.pi ** 2 * meta["unsharp_sigma"] ** 2 * qxy ** 2)
        u2d = (1 + meta["unsharp_coeff"] * (1 - g2d)) ** 2
        giso = np.exp(-2 * np.pi ** 2 * meta["unsharp_sigma"] ** 2 * qv ** 2)
        uiso = (1 + meta["unsharp_coeff"] * (1 - giso)) ** 2
        print(f"    q={qv}: <|H2d|^2>={u2d.mean():.3f} vs |Hiso|^2={uiso:.3f} "
              f"(ratio {u2d.mean()/uiso:.3f})")

    with open(os.path.join(OUT, "qc_k2_local.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("\nwrote qc_k2_local.json")


if __name__ == "__main__":
    main()
