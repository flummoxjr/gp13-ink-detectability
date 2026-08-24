"""Diagnostic: reproduce the screen's z=5.94 and dissect what drives the peak."""
import json, sys, numpy as np
sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag")
import vf_common as C

name = C.FLAG
px4 = C.PX_UM_DS4["PHerc1447"]      # 34.56 um/px at ds4
out = {}

for DS in (2, 1):
    a = C.load(name, "forward", ds=DS)
    mask = a > 0
    px = px4 * DS
    thetas = list(range(0, 180, 15))
    cache = C.rot_cache(mask, thetas)
    obs = C.ruling_score(a, cache, thetas, px)
    print(f"DS={DS} px={px:.1f}um shape={a.shape} maskfrac={mask.mean():.3f} "
          f"obs prom={obs[0]:.2f} theta={obs[1]} period={obs[2]:.2f}mm")
    # screen's own null: 16 perms, block 32 analysis-px
    rng = np.random.default_rng(0)
    nulls = []
    blk = 32 if DS == 2 else 64
    for _ in range(16):
        pa, nb = C.block_permute(a, mask, rng, blk)
        if pa is None: break
        nulls.append(C.ruling_score(pa, cache, thetas, px)[0])
    mu, sd = float(np.mean(nulls)), float(np.std(nulls))
    print(f"   null16 mean={mu:.2f} sd={sd:.2f}  z={(obs[0]-mu)/sd:+.2f}  nblocks={nb}")
    out[f"ds{DS}"] = {"prom": obs[0], "theta": obs[1], "period_mm": obs[2],
                      "null_mean": mu, "null_sd": sd, "z": (obs[0]-mu)/sd, "nblocks": nb}

# --- dissect the winning profile at ds4 ---
a = C.load(name, "forward", ds=1)
mask = a > 0
thetas = list(range(0, 180, 15))
cache = C.rot_cache(mask, thetas)
best_t = out["ds1"]["theta"]
for t in thetas:
    pr = C.profile(a, t, cache)
    p0 = C.band_prom(pr, px4, 0.0)
    p90 = C.band_prom(pr, px4, 90.0)
    print(f"theta {t:3d}: len={len(pr):4d} prom_raw={p0[0]:7.2f} @{p0[2]}mm | "
          f"prom_detrend={p90[0]:7.2f} @{p90[2]}mm")

# spectrum at the winning theta, raw vs detrended
pr = C.profile(a, best_t, cache)
for sig, tag in ((0.0, "raw"), (90.0, "detrend90")):
    prom, per_px, per_mm, periods, band = C.band_prom(pr, px4, sig, return_spec=True)
    order = np.argsort(-band)[:6]
    print(f"\n[{tag}] top band peaks (period mm, power/median):")
    for i in order:
        print(f"   {periods[i]*px4/1000:6.2f} mm   {band[i]/np.median(band):8.2f}")
np.save(r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag\vf_prof_best.npy", pr)
json.dump(out, open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag\vf_diag.json", "w"), indent=1)
