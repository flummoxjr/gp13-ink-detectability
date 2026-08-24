"""K2 — spectral information-ceiling test for 9um-class GP volumes.

For each target volume: find a papyrus-rich 256^3 ROI (probed at level 3),
stream it at level 0, invert the recorded uint8 window, and compare its
radially-averaged 3D power spectrum against
  (a) the recorded Paganin transfer function x unsharp boost, and
  (b) the uint8 quantization noise floor (computed empirically through the
      identical PSD code path, so normalization conventions cancel).

Question: at which spatial frequencies does papyrus signal sink below the
quantization floor -> what detail can NO downstream method recover from the
released volumes.

Outputs: trackD/out/k2_spectra.png, k2_spectra.json
"""
import json
import os

import numpy as np

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
META_DIR = r"C:\Users\benbl\Desktop\Vsuvious\trackD\meta"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out"
CACHE = r"D:\vesuvius-data\trackD"
ROI = 256

# sample, volume_id, label; processing params read from the meta dumps
TARGETS = [
    ("PHerc0813", "20250821151723", "GP 9.362um 113keV"),
    ("PHerc1203", "20250820131727", "GP 9.362um 113keV (has 2.4um sibling)"),
    ("PHerc0139", "20250728140407", "letters-bearing 9.362um 113keV"),
]

HC_KEVM = 1.23984193e-9  # keV*m


def vol_node(sample, vid):
    with open(os.path.join(META_DIR, f"{sample}.json"), encoding="utf-8") as f:
        s = json.load(f)
    v = s["volumes"][vid]
    scan = s["scans"][v["scan_id"]]
    phase = (
        scan.get("creation", {}).get("metadata", {}).get("metadata", {})
        .get("tomo", {}).get("processing", {}).get("preprocessing", {}).get("phase", {})
    )
    c = v["creation"]["metadata"]
    return {
        "long_id": v["long_id"],
        "px_um": v["properties"]["pixel_size_um"],
        "energy_keV": v["properties"]["energy_keV"],
        "dist_mm": v["properties"]["detector_distance_mm"],
        "win": (c["target_window_f32_min"], c["target_window_f32_max"]),
        "delta_beta": phase.get("delta_beta"),
        "unsharp_coeff": phase.get("unsharp_coeff"),
        "unsharp_sigma": phase.get("unsharp_sigma"),
        "shape": v["properties"].get("shape"),
    }


def open_level(sample, long_id, level):
    import zarr, fsspec
    url = f"{BUCKET}/{sample}/volumes/{long_id}"
    return zarr.open(fsspec.get_mapper(url), mode="r")[str(level)]


def find_papyrus_roi(sample, long_id, shape0):
    """Probe level 3 for the 32x32x32 L3-block (=256^3 L0) with max papyrus fill."""
    z3 = open_level(sample, long_id, 3)
    # search central band only (avoid ends)
    arr = z3[z3.shape[0] // 4: 3 * z3.shape[0] // 4]
    fill = (arr > 0).astype(np.float32)
    # box-sum via cumulative sums, box=32
    from scipy.ndimage import uniform_filter
    sm = uniform_filter(fill, size=32, mode="constant")
    # also prefer mid-intensity (papyrus, not dense blobs): weight by mean value in window
    zi, yi, xi = np.unravel_index(np.argmax(sm), sm.shape)
    zi += z3.shape[0] // 4
    frac = float(sm.max())
    z0, y0, x0 = zi * 8 - ROI // 2, yi * 8 - ROI // 2, xi * 8 - ROI // 2
    z0 = int(np.clip(z0, 0, shape0[0] - ROI))
    y0 = int(np.clip(y0, 0, shape0[1] - ROI))
    x0 = int(np.clip(x0, 0, shape0[2] - ROI))
    return (z0, y0, x0), frac


def radial_psd(vol):
    """Radially averaged 3D PSD, Hann-windowed. Returns (freq cycles/px, psd)."""
    n = vol.shape[0]
    w = np.hanning(n)
    W = w[:, None, None] * w[None, :, None] * w[None, None, :]
    v = (vol - vol.mean()) * W
    F = np.fft.fftn(v)
    P = np.abs(F) ** 2 / W.sum()
    fz = np.fft.fftfreq(n)
    q = np.sqrt(fz[:, None, None] ** 2 + fz[None, :, None] ** 2 + fz[None, None, :] ** 2)
    bins = np.linspace(0, 0.5 * np.sqrt(3), 80)
    idx = np.digitize(q.ravel(), bins)
    psd = np.bincount(idx, weights=P.ravel(), minlength=len(bins) + 1)
    cnt = np.bincount(idx, minlength=len(bins) + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = psd / cnt
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, prof[1:len(bins)]


def transfer_function(q_cyc_px, meta):
    """|Paganin low-pass x unsharp boost| as a function of frequency (cycles/px)."""
    lam = HC_KEVM / meta["energy_keV"]
    z = meta["dist_mm"] * 1e-3
    px = meta["px_um"] * 1e-6
    q_m = q_cyc_px / px  # cycles/m
    pag = 1.0 / (1.0 + np.pi * lam * z * meta["delta_beta"] * q_m ** 2)
    s = meta["unsharp_sigma"] or 0.0
    c = meta["unsharp_coeff"] or 0.0
    gauss = np.exp(-2 * (np.pi ** 2) * (s ** 2) * q_cyc_px ** 2)
    unsharp = 1.0 + c * (1.0 - gauss)
    return pag * unsharp


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6.2 * len(TARGETS), 5.2), squeeze=False)

    for ax, (sample, vid, label) in zip(axes[0], TARGETS):
        meta = vol_node(sample, vid)
        print(f"\n=== {sample} {vid} ({label}) ===")
        print(" ", {k: meta[k] for k in ("px_um", "energy_keV", "dist_mm", "win", "delta_beta", "unsharp_coeff", "unsharp_sigma")})

        cache = os.path.join(CACHE, f"k2_roi_{sample}_{vid}.npy")
        if os.path.exists(cache):
            roi = np.load(cache)
            origin = None
        else:
            origin, frac = find_papyrus_roi(sample, meta["long_id"], meta["shape"])
            print(f"  ROI origin L0 {origin}, papyrus fill {frac:.2f}")
            z0 = open_level(sample, meta["long_id"], 0)
            roi = z0[origin[0]:origin[0] + ROI, origin[1]:origin[1] + ROI, origin[2]:origin[2] + ROI]
            np.save(cache, roi)

        lo, hi = meta["win"]
        f = roi.astype(np.float32) / 255.0 * (hi - lo) + lo
        pap_frac = float((roi > 0).mean())
        print(f"  ROI papyrus fraction: {pap_frac:.2f}, mean f={f[roi > 0].mean():.4f}")

        q, psd = radial_psd(f)

        # Empirical quantization floor: uniform noise of one DN step through same path
        rng = np.random.default_rng(0)
        step = (hi - lo) / 255.0
        qn = rng.uniform(-step / 2, step / 2, size=roi.shape).astype(np.float32)
        _, psd_q = radial_psd(qn)

        # Transfer function scaled to match the data PSD at low frequency
        H = transfer_function(q, meta)
        # unfiltered-signal estimate: data / |H|^2 (what the pre-Paganin spectrum implies)
        valid = q > 0.005
        results[f"{sample}"] = {
            "meta": {k: meta[k] for k in ("px_um", "energy_keV", "delta_beta", "unsharp_coeff", "unsharp_sigma", "win")},
            "roi_papyrus_frac": pap_frac,
            "crossover_cyc_per_px": None,
        }
        # frequency where data PSD falls below 2x quantization floor
        below = np.where((psd < 2 * psd_q) & valid)[0]
        if len(below):
            fx = float(q[below[0]])
            results[f"{sample}"]["crossover_cyc_per_px"] = fx
            um_scale = meta["px_um"] / fx / 2  # half-period in um
            results[f"{sample}"]["smallest_recoverable_halfperiod_um"] = um_scale
            print(f"  PSD crosses 2x quantization floor at {fx:.3f} cyc/px -> half-period {um_scale:.1f} um")

        ax.loglog(q[valid], psd[valid], label="papyrus ROI PSD")
        ax.loglog(q[valid], psd_q[valid], "--", label="uint8 quantization floor")
        ax.loglog(q[valid], psd[valid][0] * (H[valid] / H[valid][0]) ** 2, ":", label="|Paganin x unsharp|^2 (shape)")
        ax.axvline(0.5, color="gray", lw=0.5)
        ax.set_title(f"{sample} ({meta['px_um']}um)")
        ax.set_xlabel("frequency (cycles/px)")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("power")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "k2_spectra.png"), dpi=110)
    with open(os.path.join(OUT, "k2_spectra.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("\nwrote k2_spectra.png / .json")


if __name__ == "__main__":
    main()
