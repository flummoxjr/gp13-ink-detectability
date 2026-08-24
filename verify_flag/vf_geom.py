"""Where does the 7.24 mm / theta=15 peak live in 2D, and is it z-symmetric?

Ink sits on one surface: reversing the render destroys it (control r=0.076).
A mesh/render/tile artifact is identical in both z directions.
"""
import json, sys, numpy as np
sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag")
import vf_common as C

px = C.PX_UM_DS4["PHerc1447"]     # um per ds4 px
res = {}

fwd = C.load(C.FLAG, "forward")
rev = C.load(C.FLAG, "reverse")
mask = fwd > 0
print("mask identical fwd/rev:", np.array_equal(mask, rev > 0))

# ---- 1. same-orientation, same-period test on the reverse map -------------
thetas = list(range(0, 180, 15))
cache = C.rot_cache(mask, thetas)
for tag, img in (("forward", fwd), ("reverse", rev)):
    b = C.ruling_score(img, cache, thetas, px)
    pr = C.profile(img, 15, cache)
    p15 = C.band_prom(pr, px)
    print(f"{tag}: best prom={b[0]:.2f} theta={b[1]} period={b[2]:.3f}mm | "
          f"at theta=15: prom={p15[0]:.2f} period={p15[2]:.3f}mm")
    res[tag] = {"best_prom": b[0], "best_theta": b[1], "best_period_mm": b[2],
                "theta15_prom": p15[0], "theta15_period_mm": p15[2]}
    np.save(rf"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag\vf_prof_{tag}.npy", pr)

# ---- 2. mask-only control: is the periodicity in the MASK or the VALUES? --
# Replace all in-mask values by a constant -> profile becomes flat by
# construction, so instead use the mask's own row-support profile.
rm, den, ok = cache[15]
res["mask_support_profile_len"] = int(ok.sum())
np.save(r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag\vf_den15.npy", den[ok])
pm = C.band_prom(den[ok].astype(np.float32), px)
print(f"MASK support profile at theta=15: prom={pm[0]:.2f} period={pm[2]}mm")
res["mask_support_prom"] = pm[0]; res["mask_support_period_mm"] = pm[2]

# ---- 3. 2D power spectrum: true wavevector, orientation, pitch -----------
a = fwd.astype(np.float64).copy()
a[~mask] = a[mask].mean()
a -= a.mean()
h, w = a.shape
win = np.outer(np.hanning(h), np.hanning(w))
F = np.fft.fftshift(np.abs(np.fft.fft2(a * win)) ** 2)
fy = np.fft.fftshift(np.fft.fftfreq(h)); fx = np.fft.fftshift(np.fft.fftfreq(w))
FY, FX = np.meshgrid(fy, fx, indexing="ij")
R = np.hypot(FY, FX)
band = (R > px / (8400.0)) & (R < px / 1700.0)   # 1.7-8.4 mm
Fb = np.where(band, F, 0)
# radial-median normalisation so a big low-f pedestal doesn't win by default
rr = (R * max(h, w)).astype(int)
med = np.zeros_like(F)
for k in np.unique(rr[band]):
    s = rr == k
    med[s] = np.median(F[s]) + 1e-30
S = np.where(band, F / med, 0)
peaks = []
idx = np.argsort(-S.ravel())[:4000]
used = []
for i in idx:
    y, x = np.unravel_index(i, S.shape)
    if any((y - u[0]) ** 2 + (x - u[1]) ** 2 < 36 for u in used):
        continue
    used.append((y, x))
    per_mm = px / 1000.0 / R[y, x]
    ang = np.degrees(np.arctan2(FY[y, x], FX[y, x])) % 180
    peaks.append({"period_mm": round(per_mm, 3), "angle_deg": round(ang, 2),
                  "snr_vs_radial_median": round(float(S[y, x]), 2)})
    if len(peaks) >= 8:
        break
print("\n2D spectral peaks (fwd, band-limited, radial-median normalised):")
for p in peaks:
    print("  ", p)
res["fwd_2d_peaks"] = peaks

b = rev.astype(np.float64).copy(); b[~mask] = b[mask].mean(); b -= b.mean()
Fr = np.fft.fftshift(np.abs(np.fft.fft2(b * win)) ** 2)
Sr = np.where(band, Fr / med.clip(1e-30), 0)
# read the reverse map's SNR at the forward peak's exact bin
y0, x0 = used[0]
print(f"\nforward top bin: period {px/1000/R[y0,x0]:.3f} mm, angle {np.degrees(np.arctan2(FY[y0,x0],FX[y0,x0]))%180:.1f} deg")
medr = np.zeros_like(Fr)
for k in np.unique(rr[band]):
    s = rr == k
    medr[s] = np.median(Fr[s]) + 1e-30
print(f"  fwd SNR at that bin = {F[y0,x0]/med[y0,x0]:.2f}")
print(f"  rev SNR at same bin = {Fr[y0,x0]/medr[y0,x0]:.2f}")
res["peak_bin_snr_fwd"] = float(F[y0, x0] / med[y0, x0])
res["peak_bin_snr_rev"] = float(Fr[y0, x0] / medr[y0, x0])

json.dump(res, open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag\vf_geom.json", "w"), indent=1)
