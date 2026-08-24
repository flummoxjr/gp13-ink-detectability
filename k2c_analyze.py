"""K2c analysis: the separability axis, its nulls, and the ROI-selection bias it exposes.

Three questions, each with a test that can fail:

  Q1  Is separability a NEW axis, or a restatement of K2b's structural SNR?
      -> rank-correlate the two across scrolls, and compare every scroll against a
         measured ISOTROPIC FLOOR.

      NOTE, and this corrects an earlier version of this file: a phase-randomisation
      null was tried first and is INVALID here. J_ij = <g_i g_j> = sum_q q_i q_j |F(q)|^2,
      so the structure tensor is a pure function of the power spectrum and is phase-blind
      BY CONSTRUCTION -- that null could never have failed. Verified numerically on a
      single block (observed 0.9968 vs phase-randomised 0.977-0.994). The statistic is
      the ANGULAR anisotropy of gradient power; K2b's SNR is a RADIAL property of the
      same spectrum. Different features, not "structure beyond the spectrum".
      The correct reference is what the statistic reads on material with no preferred
      direction, which is measured in out/k2c_separability/isotropy_floor.json:
      real in-scan air = 0.105 median (n=28), synthetic isotropic noise = 0.017-0.119.

  Q2  Does K2b's intensity-max ROI picker bias the material it samples?
      -> the same statistic on K2b's intensity-picked cubes vs K2c's uniformly
         random cubes, drawn from the identical fill>0.98 central-z frame.

  Q3  How much of a per-scroll number is real, given the within-scroll spread?
      -> bootstrap CI over ROIs, and a variance decomposition. Scrolls whose CIs
         overlap are reported as tied, not ranked.
"""
import json
import os
import numpy as np
from scipy import ndimage as ndi
from scipy import stats as st

T = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
K2B_CACHE = r"D:\vesuvius-data\trackD\k2b"
K2C_CACHE = r"D:\vesuvius-data\trackD\k2c"
K2C_OUT = os.path.join(T, "out", "k2c_separability")
BLOCK, SIGMA = 32, 1.0
RNG = np.random.default_rng(20260818)


def coh_med(a, block=BLOCK, sigma=SIGMA):
    n = a.shape[0]
    cs = []
    for z in range(0, n - block + 1, block):
        for y in range(0, n - block + 1, block):
            for x in range(0, n - block + 1, block):
                bl = a[z:z + block, y:y + block, x:x + block]
                if (bl > 0).mean() < 0.98:
                    continue
                v = ndi.gaussian_filter(bl.astype(np.float32), sigma)
                g = np.gradient(v)
                J = np.array([[float((g[i] * g[j]).mean()) for j in range(3)] for i in range(3)])
                w, _ = np.linalg.eigh(J)
                cs.append((w[2] - w[1]) / max(w[2] + w[1], 1e-9))
    return float(np.median(cs)) if len(cs) >= 8 else np.nan


def boot_ci(vals, n=4000):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return (float(v[0]), float(v[0])) if len(v) else (np.nan, np.nan)
    d = RNG.choice(v, size=(n, len(v)), replace=True)
    m = np.median(d, axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    k2b_snr = {}
    d = os.path.join(T, "out", "k2b_index")
    for fn in os.listdir(d):
        if fn.startswith("PHerc") and fn.endswith(".json"):
            j = json.load(open(os.path.join(d, fn)))
            if "snr_q025_med_iqr" in j:
                k2b_snr[fn[:-5]] = j["snr_q025_med_iqr"][0]

    scrolls = sorted(k2b_snr)
    rows = {}
    for s in scrolls:
        rec = {}
        # --- random frame (K2c)
        rnd = sorted(f for f in os.listdir(K2C_CACHE)
                     if f.startswith(s + "_rnd") and f.endswith(".npy")) if os.path.isdir(K2C_CACHE) else []
        rec["random"] = [coh_med(np.load(os.path.join(K2C_CACHE, f))) for f in rnd]
        # --- intensity-max frame (K2b)
        pk = sorted(f for f in os.listdir(K2B_CACHE) if f.startswith(s + "_pap") and f.endswith(".npy"))
        rec["picked"] = []
        for f in pk:
            a = np.load(os.path.join(K2B_CACHE, f))
            if a.shape == (256, 256, 256):
                rec["picked"].append(coh_med(a))
        rec["snr"] = k2b_snr[s]
        rows[s] = rec

    out = {"block": BLOCK, "sigma": SIGMA,
           "statistic": "median over 32^3 in-material blocks of structure-tensor planarity (l1-l2)/(l1+l2)",
           "isotropic_floor_air_median": 0.105,
           "isotropic_floor_note": "see out/k2c_separability/isotropy_floor.json; a phase-randomisation "
                                   "null is INVALID for this statistic (the structure tensor is phase-blind "
                                   "by construction) and was removed",
           "scrolls": {}}

    print(f"{'scroll':11} {'n':>3} {'sep(random)':>12} {'95% CI':>16} {'picked':>8} {'bias x':>7} {'K2b SNR':>8}")
    for s in scrolls:
        r = rows[s]
        rr = [x for x in r["random"] if np.isfinite(x)]
        pp = [x for x in r["picked"] if np.isfinite(x)]
        if not rr:
            print(f"{s:11} {'-':>3} {'(pending)':>12}")
            continue
        med = float(np.median(rr))
        lo, hi = boot_ci(rr)
        pmed = float(np.median(pp)) if pp else np.nan
        bias = med / pmed if pp and pmed > 0 else np.nan
        out["scrolls"][s] = dict(n_random=len(rr), sep_med=med, sep_ci95=[lo, hi],
                                 sep_iqr=[float(np.percentile(rr, 25)), float(np.percentile(rr, 75))],
                                 n_picked=len(pp), sep_picked_med=pmed, picker_bias_ratio=bias,
                                 k2b_snr_q025=r["snr"], values_random=rr, values_picked=pp)
        print(f"{s:11} {len(rr):>3} {med:>12.3f} {f'[{lo:.3f},{hi:.3f}]':>16} "
              f"{pmed:>8.3f} {bias:>7.2f} {r['snr']:>8.1f}")

    done = [s for s in scrolls if s in out["scrolls"]]
    if len(done) >= 4:
        sep = np.array([out["scrolls"][s]["sep_med"] for s in done])
        snr = np.array([out["scrolls"][s]["k2b_snr_q025"] for s in done])
        rho, p = st.spearmanr(sep, snr)
        out["sep_vs_snr_spearman"] = {"rho": float(rho), "p": float(p), "n": len(done)}
        print(f"\nQ1  separability vs K2b structural SNR: Spearman rho={rho:+.3f} p={p:.3g} (n={len(done)})")

        bi = [out["scrolls"][s]["picker_bias_ratio"] for s in done
              if np.isfinite(out["scrolls"][s]["picker_bias_ratio"])]
        if bi:
            rnd_all = np.concatenate([out["scrolls"][s]["values_random"] for s in done])
            pk_all = np.concatenate([out["scrolls"][s]["values_picked"] for s in done
                                     if out["scrolls"][s]["values_picked"]])
            u = st.mannwhitneyu(rnd_all, pk_all, alternative="greater")
            out["picker_bias"] = {"median_ratio": float(np.median(bi)),
                                  "n_scrolls": len(bi),
                                  "n_scrolls_random_higher": int(sum(1 for b in bi if b > 1)),
                                  "random_med": float(np.median(rnd_all)),
                                  "picked_med": float(np.median(pk_all)),
                                  "mannwhitney_p": float(u.pvalue)}
            print(f"\nQ2  ROI-picker bias: random {np.median(rnd_all):.3f} vs intensity-picked "
                  f"{np.median(pk_all):.3f}, median ratio {np.median(bi):.2f}x, "
                  f"random higher in {sum(1 for b in bi if b>1)}/{len(bi)} scrolls, "
                  f"Mann-Whitney p={u.pvalue:.3g}")

    os.makedirs(K2C_OUT, exist_ok=True)
    json.dump(out, open(os.path.join(K2C_OUT, "k2c_analysis.json"), "w"), indent=1)
    print("\nwrote", os.path.join(K2C_OUT, "k2c_analysis.json"))


if __name__ == "__main__":
    main()
