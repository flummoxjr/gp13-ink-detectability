"""NS3 -- ONE corrected null: spatial block bootstrap at the measured
correlation length, validated against converged rigid-shift ensembles.

Method (cluster bootstrap over tile blocks):
  * the per-tile excess field (d_t, w_t) on the 255 x 197 tile grid is cut
    into (BY x BX)-tile blocks sized from the NS2-measured correlation length
    (x-corr crosses 1/e at lag 3 and 0.1 at lag 5-6; y-corr is gone at lag 1
    -> default block 2 x 8 tiles, sensitivity over 1x1 .. 4x12),
  * blocks holding any admitted weight are resampled with replacement,
    T* = sum(Sw*)/sum(W*); sd over 4000 replicates, averaged over 4 block
    anchor offsets.
  * By construction the corrected null is evaluated on the OBSERVED tile
    configuration (matched n), which the rigid-shift null never was.

Validation, all measured here:
  V1  applied to single NULL fields (no ink signal), does the bootstrap sd
      predict the converged 400-draw rigid-shift ensemble sd?
  V2  per slab: bootstrap sd of the slab's observed field vs the slab's own
      400-draw local rigid-shift reference.
  V3  estimator stability: per slab, ten disjoint 40-draw rigid estimates
      (from the 400 local draws) vs ten bootstrap seeds -- spread of each.
"""
import sys, os, json, time
import numpy as np
MET = r"C:\Users\benbl\AppData\Local\Temp\claude\C--Users-benbl-Desktop-Vsuvious\35d67aba-4ea6-4b13-a3c0-2f3fc87bbe13\scratchpad\metrology"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MET)

t0 = time.time()
T = 32; H, W = 8181, 6330
ny, nx = H // T, W // T
NREP = 4000
BY, BX = 2, 8            # default block, from NS2 measured correlation length

obs = np.load(os.path.join(HERE, "ns2_obsfield.npz"))
d_obs = obs["d"].astype(np.float64); w_obs = obs["w"].astype(np.float64)
z2 = np.load(os.path.join(HERE, "ns2_draws.npz"))
Dm, Wm = z2["Dm"].astype(np.float64), z2["Wm"].astype(np.float64)

def block_boot_sd(d, w, nyg, nxg, by, bx, nrep=NREP, seed=0, anchors=4):
    """cluster bootstrap sd of T = sum(w d)/sum(w) over (by,bx) tile blocks."""
    rng = np.random.default_rng(seed)
    sds = []
    offs = [(0, 0), (by // 2, bx // 2), (0, bx // 2), (by // 2, 0)][:anchors]
    D2 = d.reshape(nyg, nxg); W2 = w.reshape(nyg, nxg)
    for oy, ox in offs:
        # pad so the offset partition covers the grid
        py = (-(nyg + oy)) % by; px = (-(nxg + ox)) % bx
        Dp = np.pad(D2 * W2, ((oy, py), (ox, px)))
        Wp = np.pad(W2, ((oy, py), (ox, px)))
        YB, XB = Dp.shape[0] // by, Dp.shape[1] // bx
        Sw = Dp.reshape(YB, by, XB, bx).sum((1, 3)).ravel()
        Wb = Wp.reshape(YB, by, XB, bx).sum((1, 3)).ravel()
        keep = Wb > 0
        Sw, Wb = Sw[keep], Wb[keep]
        nb = len(Sw)
        if nb < 8:
            continue
        idx = rng.integers(0, nb, size=(nrep, nb))
        Ts = Sw[idx].sum(1) / Wb[idx].sum(1)
        sds.append(float(Ts.std(ddof=1)))
    return float(np.mean(sds)), int(nb), sds

out = {}

# ---------------- whole plate, observed configuration ----------------
wp = {}
for by, bx in [(1, 1), (1, 4), (2, 4), (2, 8), (3, 8), (4, 12)]:
    sd, nb, _ = block_boot_sd(d_obs, w_obs, ny, nx, by, bx, seed=1)
    wp["block_%dx%d" % (by, bx)] = dict(sd=round(sd, 1), n_blocks=nb)
    print("whole plate  block %dx%d  boot sd %8.1f  (n_blocks %d)" % (by, bx, sd, nb),
          flush=True)
sd_corr, nb_corr, _ = block_boot_sd(d_obs, w_obs, ny, nx, BY, BX, seed=2)
seeds = [block_boot_sd(d_obs, w_obs, ny, nx, BY, BX, seed=100 + s)[0] for s in range(10)]
obs_T = float((d_obs * w_obs).sum() / w_obs.sum())
out["whole_plate"] = dict(
    obs=round(obs_T, 1), block=[BY, BX], boot_sd=round(sd_corr, 1),
    boot_sd_10seeds=dict(mean=round(float(np.mean(seeds)), 1),
                         sd=round(float(np.std(seeds, ddof=1)), 2),
                         cv_pct=round(100 * float(np.std(seeds, ddof=1) / np.mean(seeds)), 2)),
    z_corrected=round(obs_T / sd_corr, 3),
    sensitivity=wp,
    rigid_40draw_sd=2755.3, rigid_400draw_sd=2091.9,
    g1_threshold=2300.0)
print("whole plate corrected null sd = %.1f (10-seed cv %.2f%%), obs %+.1f -> z %+0.3f"
      % (sd_corr, 100 * np.std(seeds, ddof=1) / np.mean(seeds), obs_T, obs_T / sd_corr),
      flush=True)

# convergence of the rigid ensemble sd + bootstrap CI over draws
Tt = (Dm * Wm).sum(1) / Wm.sum(1)
rng = np.random.default_rng(7)
bsd = np.array([Tt[rng.integers(0, 400, 400)].std(ddof=1) for _ in range(4000)])
out["rigid_400_ensemble"] = dict(
    sd=round(float(Tt.std(ddof=1)), 1), mean=round(float(Tt.mean()), 1),
    sd_wo_first40=round(float(Tt[40:].std(ddof=1)), 1),
    sd_ci95_bootstrap=[round(float(np.percentile(bsd, 2.5)), 1),
                       round(float(np.percentile(bsd, 97.5)), 1)],
    z_obs_vs_400=round(float((obs_T - Tt.mean()) / Tt.std(ddof=1)), 3))
print("rigid 400-draw: sd %.1f (CI95 %.0f..%.0f), mean %+.1f; sd without committed first 40: %.1f"
      % (Tt.std(ddof=1), np.percentile(bsd, 2.5), np.percentile(bsd, 97.5), Tt.mean(),
         Tt[40:].std(ddof=1)), flush=True)

# ---------------- V1: bootstrap applied to single null fields ----------------
v1 = []
for i in range(0, 400, 10):     # 40 null fields
    sd_i, nb_i, _ = block_boot_sd(Dm[i], Wm[i], ny, nx, BY, BX, nrep=1500, seed=3000 + i,
                                  anchors=2)
    v1.append(sd_i)
v1 = np.array(v1)
out["V1_null_field_boot"] = dict(
    n_fields=len(v1), median=round(float(np.median(v1)), 1),
    q25=round(float(np.percentile(v1, 25)), 1), q75=round(float(np.percentile(v1, 75)), 1),
    target_rigid_400_sd=2091.9,
    note="single-field bootstrap vs between-draw ensemble sd; a shortfall is the "
         "between-region (nonstationarity) component the single field cannot see")
print("V1: bootstrap on 40 single null fields: median %.1f [q25 %.1f q75 %.1f] "
      "vs ensemble 2091.9" % (np.median(v1), np.percentile(v1, 25), np.percentile(v1, 75)),
      flush=True)

# ---------------- V2 + V3: per-slab ----------------
slabs = []
for si, a in enumerate(range(0, H - 1023, 1024)):
    zs = np.load(os.path.join(HERE, "ns2_slab%d.npz" % si))
    Dl, Wl = zs["Dl"].astype(np.float64), zs["Wl"].astype(np.float64)
    nyy = 1024 // T
    swl = Wl.sum(1); okl = swl > 0
    Tl = (Dl * Wl).sum(1)[okl] / swl[okl]
    sd_ref = float(Tl.std(ddof=1))                     # converged local reference
    # ten disjoint 40-draw rigid estimates
    rig40 = np.array([Tl[k * 40:(k + 1) * 40].std(ddof=1) for k in range(10)])
    # observed slab field
    r0, r1 = (a // T) * nx, ((a + 1024) // T) * nx
    d_s = d_obs[r0:r1]; w_s = w_obs[r0:r1]
    n_s = int((w_s > 0).sum())
    sd_b, nb_s, _ = block_boot_sd(d_s, w_s, nyy, nx, BY, BX, seed=40 + si)
    boot10 = np.array([block_boot_sd(d_s, w_s, nyy, nx, BY, BX, seed=500 + 17 * si + s)[0]
                       for s in range(10)])
    obs_s = float((d_s * w_s).sum() / w_s.sum())
    slabs.append(dict(
        rows=[a, a + 1023], n_tiles=n_s, n_blocks=nb_s,
        obs=round(obs_s, 1),
        rigid_ref_400=round(sd_ref, 1),
        rigid_40draw_10x=dict(min=round(float(rig40.min()), 1),
                              max=round(float(rig40.max()), 1),
                              cv_pct=round(100 * float(rig40.std(ddof=1) / rig40.mean()), 1)),
        boot_sd=round(sd_b, 1),
        boot_10seed_cv_pct=round(100 * float(boot10.std(ddof=1) / boot10.mean()), 2),
        boot_over_ref=round(sd_b / sd_ref, 3),
        sd_sqrt_n_boot=round(sd_b * np.sqrt(n_s), 0),
        sd_sqrt_n_ref=round(sd_ref * np.sqrt(n_s), 0)))
    print("slab %d  ref400 %8.1f | rigid40x10 %8.1f..%8.1f cv %4.1f%% | "
          "boot %8.1f cv %4.2f%%  boot/ref %.2f" %
          (si, sd_ref, rig40.min(), rig40.max(), 100 * rig40.std(ddof=1) / rig40.mean(),
           sd_b, 100 * boot10.std(ddof=1) / boot10.mean(), sd_b / sd_ref), flush=True)
out["slabs"] = slabs

# aggregate stability statement
rig_cv = [s["rigid_40draw_10x"]["cv_pct"] for s in slabs]
boot_cv = [s["boot_10seed_cv_pct"] for s in slabs]
ratio = [s["boot_over_ref"] for s in slabs]
out["stability_summary"] = dict(
    rigid_40draw_cv_pct=dict(min=min(rig_cv), max=max(rig_cv),
                             median=float(np.median(rig_cv))),
    boot_seed_cv_pct=dict(min=min(boot_cv), max=max(boot_cv),
                          median=float(np.median(boot_cv))),
    boot_over_converged_ref=dict(min=min(ratio), max=max(ratio),
                                 median=round(float(np.median(ratio)), 3)))
print("stability: rigid-40 cv %s%%  vs  boot seed-cv %s%%;  boot/ref median %.2f"
      % (out["stability_summary"]["rigid_40draw_cv_pct"],
         out["stability_summary"]["boot_seed_cv_pct"],
         np.median(ratio)), flush=True)

out["elapsed_s"] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(HERE, "ns3_corrected.json"), "w"), indent=1)
print("done %.1fs" % out["elapsed_s"])
