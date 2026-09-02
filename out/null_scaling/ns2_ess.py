"""NS2 -- effective sample size of the single-global-shift null.

Runs the committed null generator far past 40 draws (400 global rigid shifts,
same seed stream: draws 0..39 ARE the committed 40), storing the per-tile
excess field of every draw.  From that ensemble:

  * the converged null sd of the whole-plate statistic (what 40 draws was
    trying to estimate),
  * the design effect DEFF = Var_ensemble(T) / Var_if_tiles_were_independent,
    and the effective sample size n_eff = n_kish / DEFF,
  * the spatial autocorrelation of the per-tile null field (correlation
    length in 32-px tile units) -> feeds the NS3 block size,
  * per-slab converged references, from BOTH the global ensemble restricted
    to the slab's tiles and 400 slab-local rolls per slab.
"""
import sys, os, json, time
import numpy as np
MET = r"C:\Users\benbl\AppData\Local\Temp\claude\C--Users-benbl-Desktop-Vsuvious\35d67aba-4ea6-4b13-a3c0-2f3fc87bbe13\scratchpad\metrology"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MET)
import m1_lib as G
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

t0 = time.time()
T = 32; H, W = 8181, 6330; NDRAW = 400
L = (np.array(Image.open(os.path.join(MET, "f1_inklabels.png"))) > 0)
M = (np.array(Image.open(os.path.join(MET, "f1_mask.png"))) > 0)
P = np.load(os.path.join(MET, "f1_P.npy"))
ny, nx = H // T, W // T
Mv = G.tileview(M, ny, nx, T); Pv = G.tileview(P, ny, nx, T)

cache = os.path.join(HERE, "ns2_draws.npz")
if os.path.exists(cache):
    z = np.load(cache)
    Dm, Wm = z["Dm"], z["Wm"]
    print("loaded cached draws", Dm.shape, flush=True)
else:
    sh = G.shifts_for(H, W, n=NDRAW)
    Dm = np.zeros((NDRAW, ny * nx), np.float32)
    Wm = np.zeros((NDRAW, ny * nx), np.float32)
    for i, (dy, dx) in enumerate(sh):
        d, w, ok = G.excess_full(Pv, G.roll(L, dy, dx), Mv, ny, nx, T)
        Dm[i] = np.nan_to_num(d); Wm[i] = w
        if i % 50 == 0: print("draw %d %.1fs" % (i, time.time() - t0), flush=True)
    np.savez_compressed(cache, Dm=Dm, Wm=Wm)

sw = Wm.sum(1)
Tt = (Dm * Wm).sum(1) / sw                       # T(s), the null statistic
nadm = (Wm > 0).sum(1)
nkish = sw**2 / (Wm**2).sum(1)

out = {}
# ---- converged whole-plate null ----
sd400 = float(Tt.std(ddof=1))
sd40 = float(Tt[:40].std(ddof=1))                 # committed draws
zc = Tt - Tt.mean()
kurt = float((zc**4).mean() / (zc**2).mean()**2 - 3.0)
# SE of an sd estimate from 40 draws of this distribution (delta method)
se_sd40 = sd400 * np.sqrt((kurt + 2.0) / (4 * 40))
out["whole_plate"] = dict(
    n_draws=NDRAW, sd_400=round(sd400, 1), sd_first40_committed=round(sd40, 1),
    excess_kurtosis_400=round(kurt, 2),
    se_of_a_40draw_sd_estimate=round(float(se_sd40), 1),
    admitted_tiles_per_draw=dict(min=int(nadm.min()), median=int(np.median(nadm)),
                                 max=int(nadm.max())),
    obs_config_tiles=2560,
    sd_sqrt_n_at_2560=round(sd400 * np.sqrt(2560), 0))
print("converged whole-plate null sd (400 draws) = %.1f   (40-draw committed estimate 2755.3)"
      % sd400, flush=True)
print("excess kurtosis %.2f -> SE of a 40-draw sd estimate = %.0f" % (kurt, se_sd40), flush=True)

# ---- design effect / effective sample size ----
# per-tile null variance, over draws where the tile was admitted
adm = Wm > 0
cnt = adm.sum(0)
d1 = np.where(adm, Dm, 0).sum(0)
d2 = np.where(adm, Dm.astype(np.float64)**2, 0).sum(0)
with np.errstate(invalid="ignore", divide="ignore"):
    mu_t = d1 / cnt
    v_t = d2 / cnt - mu_t**2
v_t = np.where(cnt >= 8, v_t, np.nan)   # median admission count per tile is ~11/400
v_fill = np.nanmedian(v_t)
v_use = np.where(np.isfinite(v_t), v_t, v_fill)
# hypothetical independent-tile variance of T(s), per draw, then averaged
Vind = ((Wm.astype(np.float64)**2) * v_use).sum(1) / sw.astype(np.float64)**2
DEFF = sd400**2 / Vind.mean()
neff = float(np.median(nkish) / DEFF)
out["ess"] = dict(
    per_tile_null_var_median=round(float(np.nanmedian(v_t)), 1),
    mean_independent_tiles_variance=round(float(Vind.mean()), 1),
    implied_sd_if_tiles_independent=round(float(np.sqrt(Vind.mean())), 1),
    DEFF=round(float(DEFF), 1),
    n_kish_median=round(float(np.median(nkish)), 1),
    n_admitted_median=int(np.median(nadm)),
    n_nominal_obs=2560,
    n_eff=round(neff, 1))
print("independent-tiles sd would be %.1f; actual %.1f -> DEFF %.1f, n_eff %.1f "
      "(vs nominal 2560 obs tiles, median %d admitted per draw)"
      % (np.sqrt(Vind.mean()), sd400, DEFF, neff, np.median(nadm)), flush=True)

# ---- spatial autocorrelation of the per-tile null field ----
# mean over draws of the admitted-pair autocorrelation, via FFT on the tile grid
def field_autocorr(D, A, nyg, nxg, ndr=200):
    acc = None; accN = None
    for i in range(min(ndr, D.shape[0])):
        x = np.where(A[i], D[i] - D[i][A[i]].mean(), 0.0).reshape(nyg, nxg)
        a = A[i].reshape(nyg, nxg).astype(np.float64)
        FY, FX = 2 * nyg, 2 * nxg
        Xf = np.fft.rfft2(x, s=(FY, FX)); Af = np.fft.rfft2(a, s=(FY, FX))
        cxx = np.fft.irfft2(Xf * np.conj(Xf), s=(FY, FX))
        cnn = np.fft.irfft2(Af * np.conj(Af), s=(FY, FX))
        acc = cxx if acc is None else acc + cxx
        accN = cnn if accN is None else accN + cnn
    with np.errstate(invalid="ignore", divide="ignore"):
        r = acc / np.maximum(accN, 1e-9)
    r = r / r[0, 0]
    return r, accN

r2d, npairs = field_autocorr(Dm, adm, ny, nx)
lag_x = [float(r2d[0, k]) for k in range(0, 13)]
lag_y = [float(r2d[k, 0]) for k in range(0, 13)]
def clen(vals, thr):
    for k, v in enumerate(vals):
        if v < thr: return k
    return len(vals)
out["null_field_autocorr"] = dict(
    lags_tiles=list(range(13)),
    corr_x=[round(v, 3) for v in lag_x], corr_y=[round(v, 3) for v in lag_y],
    L_x_below_1_over_e=clen(lag_x, 1/np.e), L_y_below_1_over_e=clen(lag_y, 1/np.e),
    L_x_below_0p1=clen(lag_x, 0.1), L_y_below_0p1=clen(lag_y, 0.1))
print("null-field autocorr  x:", [round(v, 2) for v in lag_x], flush=True)
print("null-field autocorr  y:", [round(v, 2) for v in lag_y], flush=True)

# ---- observed per-tile field (for NS3) + its autocorrelation ----
d_obs, w_obs, ok_obs = G.excess_full(Pv, L, Mv, ny, nx, T)
np.savez_compressed(os.path.join(HERE, "ns2_obsfield.npz"),
                    d=np.nan_to_num(d_obs).astype(np.float32),
                    w=w_obs.astype(np.float32))
ro, _ = field_autocorr(np.nan_to_num(d_obs)[None, :].astype(np.float32),
                       (w_obs > 0)[None, :], ny, nx, 1)
out["obs_field_autocorr"] = dict(
    corr_x=[round(float(ro[0, k]), 3) for k in range(13)],
    corr_y=[round(float(ro[k, 0]), 3) for k in range(13)])
print("obs-field  autocorr  x:", [round(float(ro[0, k]), 2) for k in range(13)], flush=True)
print("obs-field  autocorr  y:", [round(float(ro[k, 0]), 2) for k in range(13)], flush=True)

# ---- per-slab references ----
slabs = []
for si, a in enumerate(range(0, H - 1023, 1024)):
    trows = slice((a // T) * nx, ((a + 1024) // T) * nx)   # tile-row block of this slab
    Dw = Dm[:, trows]; Ww = Wm[:, trows]
    sws = Ww.sum(1); okd = sws > 0
    Ts = (Dw * Ww).sum(1)[okd] / sws[okd]
    sd_g = float(Ts.std(ddof=1))
    # slab-local ensemble: 400 rolls of the slab's own label strip
    hh = 1024; nyy = hh // T
    Mv2 = G.tileview(M[a:a+hh], nyy, nx, T); Pv2 = G.tileview(P[a:a+hh], nyy, nx, T)
    Lb2 = L[a:a+hh]
    sh2 = G.shifts_for(hh, W, n=NDRAW)
    Dl = np.zeros((NDRAW, nyy * nx), np.float32); Wl = np.zeros((NDRAW, nyy * nx), np.float32)
    for i, (dy, dx) in enumerate(sh2):
        d, w, _ = G.excess_full(Pv2, G.roll(Lb2, dy, dx), Mv2, nyy, nx, T)
        Dl[i] = np.nan_to_num(d); Wl[i] = w
    swl = Wl.sum(1); okl = swl > 0
    Tl = (Dl * Wl).sum(1)[okl] / swl[okl]
    sd_l = float(Tl.std(ddof=1))
    sd_l40 = float(Tl[:40].std(ddof=1))
    zl = Tl - Tl.mean()
    kl = float((zl**4).mean() / (zl**2).mean()**2 - 3.0)
    oo, nn, _, _ = G.excess(Pv2, Lb2, Mv2, nyy, nx, T)
    np.savez_compressed(os.path.join(HERE, "ns2_slab%d.npz" % si),
                        Dl=Dl, Wl=Wl)
    slabs.append(dict(rows=[a, a + 1023], n_tiles=int(nn),
                      sd_40draw_committed=round(sd_l40, 1),
                      sd_400_local=round(sd_l, 1), sd_400_globalrestrict=round(sd_g, 1),
                      excess_kurtosis_local=round(kl, 2),
                      sd_sqrt_n_40draw=round(sd_l40 * np.sqrt(nn), 0),
                      sd_sqrt_n_400=round(sd_l * np.sqrt(nn), 0)))
    print("slab %d rows %4d..%4d tiles %4d  sd40 %7.1f  sd400_local %7.1f  "
          "sd400_globalview %7.1f  kurt %+5.2f" %
          (si, a, a + 1023, nn, sd_l40, sd_l, sd_g, kl), flush=True)
out["slabs"] = slabs
c40 = [s["sd_sqrt_n_40draw"] for s in slabs]
c400 = [s["sd_sqrt_n_400"] for s in slabs]
out["slab_constant_spread"] = dict(
    ratio_40draw=round(max(c40) / min(c40), 2),
    ratio_400draw=round(max(c400) / min(c400), 2),
    values_40=c40, values_400=c400)
print("sd*sqrt(n) spread across slabs:  40-draw %.2fx  ->  400-draw %.2fx" %
      (max(c40) / min(c40), max(c400) / min(c400)), flush=True)

out["elapsed_s"] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(HERE, "ns2_ess.json"), "w"), indent=1)
print("done %.1fs" % out["elapsed_s"])
