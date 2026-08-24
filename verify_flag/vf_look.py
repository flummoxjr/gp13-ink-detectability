"""Visual + structural dissection of the theta=15 / 7.24 mm peak."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag")
import vf_common as C

px = C.PX_UM_DS4["PHerc1447"]
fwd = C.load(C.FLAG, "forward"); rev = C.load(C.FLAG, "reverse")
mask = fwd > 0
thetas = list(range(0, 180, 15))
cache = C.rot_cache(mask, thetas)
res = {}

rm, den, ok = cache[15]
prof = C.profile(fwd, 15, cache)
profr = C.profile(rev, 15, cache)
cov = den[ok]
n = len(prof)
print(f"profile n={n} ({n*px/1000:.1f} mm), coverage min/med/max = "
      f"{cov.min():.0f}/{np.median(cov):.0f}/{cov.max():.0f}")
res["n"] = n; res["len_mm"] = n * px / 1000
res["cycles_at_7.24mm"] = n * px / 1000 / 7.24

# --- coverage coupling -----------------------------------------------------
r_pc = float(np.corrcoef(prof, cov)[0, 1])
print(f"corr(value profile, coverage profile) = {r_pc:+.3f}")
res["corr_profile_coverage"] = r_pc

# --- restrict to near-fully-covered rows (coverage held ~constant) ---------
for frac in (0.5, 0.75, 0.9):
    sel = cov > frac * cov.max()
    # take the longest contiguous run so the FFT stays meaningful
    idx = np.flatnonzero(sel)
    if len(idx) < 100:
        continue
    splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    run = max(splits, key=len)
    sub = prof[run]
    p = C.band_prom(sub.astype(np.float32), px)
    print(f"  coverage>{frac:.2f}*max: longest run {len(run)} px ({len(run)*px/1000:.1f} mm) "
          f"prom={p[0]:.2f} period={p[2]}mm")
    res[f"cov_gt_{frac}"] = {"n": int(len(run)), "prom": p[0], "period_mm": p[2]}

# --- eroded mask (validated battery erodes 40 px) --------------------------
for er in (0, 10, 20, 40):
    m2 = C.erode_mask(mask, er)
    c2 = C.rot_cache(m2, thetas)
    a2 = fwd * m2
    b = C.ruling_score(a2, c2, thetas, px)
    p15 = C.band_prom(C.profile(a2, 15, c2), px)
    print(f"  erode {er:2d}px: maskfrac={m2.mean():.3f} best prom={b[0]:.2f} th={b[1]} "
          f"per={b[2]}mm | th15 prom={p15[0]:.2f} per={p15[2]}mm")
    res[f"erode{er}"] = {"maskfrac": float(m2.mean()), "prom": b[0], "theta": b[1],
                         "period_mm": b[2], "th15_prom": p15[0], "th15_period_mm": p15[2]}

# --- half-split: is the "periodicity" spatially coherent? -----------------
for tag, sub in (("first half", prof[:n // 2]), ("second half", prof[n // 2:])):
    p = C.band_prom(sub.astype(np.float32), px)
    print(f"  {tag}: prom={p[0]:.2f} period={p[2]}mm")
    res[tag.replace(" ", "_")] = {"prom": p[0], "period_mm": p[2]}

# --- autocorrelation: a real comb repeats at 2P, 3P ----------------------
q = prof - prof.mean()
ac = np.correlate(q, q, "full")[n - 1:]
ac /= ac[0]
P = 7.24 * 1000 / px
print(f"  autocorr at 1P/2P/3P ({P:.0f}px): "
      f"{ac[int(round(P))]:+.3f} {ac[int(round(2*P))]:+.3f} {ac[int(round(3*P))]:+.3f}")
res["autocorr_1P_2P_3P"] = [float(ac[int(round(k * P))]) for k in (1, 2, 3) if int(round(k * P)) < n]

json.dump(res, open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag\vf_look.json", "w"),
          indent=1, default=float)

# --- figure ---------------------------------------------------------------
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(2, 3, 1); ax.imshow(fwd, cmap="magma", vmin=0, vmax=220)
ax.set_title(f"{C.FLAG}\nforward ds4 (34.56 um/px)", fontsize=9); ax.axis("off")
ax = fig.add_subplot(2, 3, 2); ax.imshow(rev, cmap="magma", vmin=0, vmax=220)
ax.set_title("reverse ds4  (fwd/rev r=0.754)", fontsize=9); ax.axis("off")
ax = fig.add_subplot(2, 3, 3); ax.imshow(mask, cmap="gray")
ax.set_title(f"valid mask (frac {mask.mean():.3f})", fontsize=9); ax.axis("off")
ax = fig.add_subplot(2, 3, 4)
x = np.arange(n) * px / 1000
ax.plot(x, prof, lw=0.8, label="forward"); ax.plot(x, profr, lw=0.8, alpha=.7, label="reverse")
for k in range(1, 6):
    ax.axvline(k * 7.24, color="r", ls=":", lw=.8)
ax.set_xlabel("mm along projection (theta=15)"); ax.set_ylabel("masked row mean")
ax.legend(fontsize=7); ax.set_title("projection profile, claimed period 7.24 mm (red)", fontsize=9)
ax = fig.add_subplot(2, 3, 5)
ax.plot(x, cov / cov.max(), lw=.8, color="k")
for k in range(1, 6):
    ax.axvline(k * 7.24, color="r", ls=":", lw=.8)
ax.set_title(f"MASK coverage profile (same axis), r={r_pc:+.2f}", fontsize=9)
ax.set_xlabel("mm")
ax = fig.add_subplot(2, 3, 6)
ax.plot(np.arange(len(ac)) * px / 1000, ac, lw=.8)
for k in range(1, 5):
    ax.axvline(k * 7.24, color="r", ls=":", lw=.8)
ax.axhline(0, color="gray", lw=.5); ax.set_xlim(0, 25)
ax.set_title("profile autocorrelation", fontsize=9); ax.set_xlabel("lag (mm)")
fig.tight_layout()
fig.savefig(r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag\vf_look.png", dpi=110)
print("wrote vf_look.png")
