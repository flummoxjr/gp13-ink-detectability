"""Step 1+2: high-N permutation test on the flag, and the corpus-wide max-z null.

Step 1  : >=400 block permutations at full ds4 (and at the screen's ds8) ->
          corrected z, bootstrap CI on z, empirical p.
Step 2  : same procedure on 9 other randomly chosen segments; for every segment
          a leave-one-out z is computed for each null draw, giving the null
          distribution of the screen's own statistic.  From it: the familywise
          max-z expectation over 69-80 tests, both for N_PERM=200+ and for the
          screen's N_PERM=16 estimator.
"""
import json, os, sys, glob
import numpy as np
from multiprocessing import Pool

HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag"
sys.path.insert(0, HERE)
import vf_common as C

NPERM = 400
THETAS = list(range(0, 180, 15))
_G = {}


def _init(name, ds, block, seed0):
    a = C.load(name, "forward", ds=ds)
    mask = a > 0
    rec = _G["rec"] = {}
    rec["a"] = a
    rec["mask"] = mask
    rec["cache"] = C.rot_cache(mask, THETAS)
    rec["px"] = _G["px"]
    rec["block"] = block
    rec["seed0"] = seed0


def _one(i):
    r = _G["rec"]
    if i < 0:
        return C.ruling_score(r["a"], r["cache"], THETAS, r["px"])
    rng = np.random.default_rng(r["seed0"] + i)
    pa, nb = C.block_permute(r["a"], r["mask"], rng, r["block"])
    if pa is None:
        return (np.nan, None, None)
    return C.ruling_score(pa, r["cache"], THETAS, r["px"])


def run(name, scroll, ds, block, nperm=NPERM, seed0=1000):
    _G["px"] = C.PX_UM_DS4.get(scroll, 37.4) * ds
    with Pool(16, initializer=_init2, initargs=(name, ds, block, seed0, _G["px"])) as pool:
        out = pool.map(_one, [-1] + list(range(nperm)), chunksize=2)
    obs = out[0]
    nulls = np.array([o[0] for o in out[1:]], float)
    nulls = nulls[np.isfinite(nulls)]
    per = [o[2] for o in out[1:] if o[2] is not None]
    return obs, nulls, per


def _init2(name, ds, block, seed0, px):
    _G["px"] = px
    _init(name, ds, block, seed0)


def loo_z(nulls):
    """z of each null draw against the other draws -> null distribution of z."""
    n = len(nulls); s = nulls.sum(); s2 = (nulls ** 2).sum()
    m = (s - nulls) / (n - 1)
    v = (s2 - nulls ** 2) / (n - 1) - m ** 2
    return (nulls - m) / np.sqrt(np.maximum(v, 1e-18))


def screen16_z(nulls, rng, reps=20000):
    """Null distribution of the SCREEN's z estimator (obs vs 16 permutations)."""
    idx = np.array([rng.permutation(len(nulls))[:17] for _ in range(reps)])
    s = nulls[idx]
    return (s[:, 0] - s[:, 1:].mean(1)) / (s[:, 1:].std(1) + 1e-9)


def summarize(tag, obs, nulls, rng):
    mu, sd = nulls.mean(), nulls.std(ddof=1)
    z = (obs[0] - mu) / sd
    emp_p = (1 + int((nulls >= obs[0]).sum())) / (len(nulls) + 1)
    bz = np.empty(4000)
    for i in range(4000):
        b = rng.choice(nulls, len(nulls), replace=True)
        bz[i] = (obs[0] - b.mean()) / (b.std(ddof=1) + 1e-12)
    ci = np.percentile(bz, [2.5, 97.5])
    d = {"prom": float(obs[0]), "theta": obs[1], "period_mm": obs[2], "n_perm": int(len(nulls)),
         "null_mean": float(mu), "null_sd": float(sd), "null_max": float(nulls.max()),
         "z": float(z), "z_ci95": [float(ci[0]), float(ci[1])],
         "empirical_p": emp_p, "n_null_ge_obs": int((nulls >= obs[0]).sum())}
    print(f"[{tag}] prom={obs[0]:.2f} th={obs[1]} per={obs[2]:.2f}mm | null {mu:.2f}+-{sd:.2f} "
          f"(max {nulls.max():.2f}, n={len(nulls)}) | z={z:+.2f} CI95 [{ci[0]:+.2f},{ci[1]:+.2f}] "
          f"| empirical p={emp_p:.4g} ({int((nulls>=obs[0]).sum())}/{len(nulls)})")
    return d


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    res = {"flag": {}, "others": {}}

    # ---------------- step 1: the flag -----------------------------------
    for ds, block, tag in ((2, 32, "flag ds8 (screen geometry)"), (1, 64, "flag ds4 (full res)")):
        obs, nulls, per = run(C.FLAG, "PHerc1447", ds, block)
        d = summarize(tag, obs, nulls, rng)
        d["null_periods_mm_hist"] = np.histogram(np.array(per, float), bins=np.arange(1.5, 9.0, .5))[0].tolist()
        d["loo_z"] = loo_z(nulls).tolist()
        res["flag"][f"ds{ds}"] = d

    # ---------------- step 2: 9 other segments ---------------------------
    surv = {r["name"]: r for r in json.load(open(os.path.join(C.SURVEY, "survey_all.json")))}
    corpus = json.load(open(os.path.join(C.SURVEY, "corpus_analysis.json")))["results"]
    pool_names = [r["name"] for r in corpus if r["name"] != C.FLAG]
    pick = list(np.random.default_rng(11).choice(pool_names, 9, replace=False))
    # plus the two next-highest screen hits, to see whether the top is one family
    for extra in ("z_dbg_gen_00260", "z_dbg_gen_00320"):
        if extra not in pick and extra in pool_names:
            pick.append(extra)
    print("\nstep-2 segments:", pick)
    for nm in pick:
        sc = surv.get(nm, {}).get("scroll", "PHerc1203")
        try:
            obs, nulls, per = run(nm, sc, 1, 64, seed0=2000)
        except Exception as e:
            print(f"{nm}: FAILED {e}")
            continue
        if len(nulls) < 50:
            print(f"{nm}: too few valid perms ({len(nulls)})"); continue
        d = summarize(f"{sc}/{nm} ds4", obs, nulls, rng)
        d["scroll"] = sc
        d["screen_z"] = next((r["ruling_z"] for r in corpus if r["name"] == nm), None)
        d["loo_z"] = loo_z(nulls).tolist()
        d["nulls"] = nulls.tolist()
        res["others"][nm] = d

    # ---------------- familywise max-z ------------------------------------
    allz = np.concatenate([np.array(d["loo_z"]) for d in res["others"].values()])
    z16 = np.concatenate([screen16_z(np.array(d["nulls"]), rng, 20000)
                          for d in res["others"].values()])
    fam = {"n_segments_nulled": len(res["others"]),
           "loo": {"n": int(len(allz)), "mean": float(allz.mean()), "sd": float(allz.std()),
                   "pcts": {str(p): float(np.percentile(allz, p)) for p in (50, 90, 95, 99, 99.9)},
                   "max": float(allz.max())},
           "screen16": {"n": int(len(z16)), "mean": float(z16.mean()), "sd": float(z16.std()),
                        "pcts": {str(p): float(np.percentile(z16, p)) for p in (50, 90, 95, 99, 99.9)},
                        "max": float(z16.max())}}
    for tag, zz in (("loo", allz), ("screen16", z16)):
        f = float((zz >= 5.94).mean())
        fam[tag]["frac_ge_5.94"] = f
        for N in (69, 80):
            fam[tag][f"P_max_of_{N}_ge_5.94"] = float(1 - (1 - f) ** N)
            fam[tag][f"expected_max_z_over_{N}"] = float(np.percentile(zz, 100 * (1 - 1 / N)))
    # per-segment observed z (real maps, high-N): the empirical corpus max-z picture
    fam["per_segment_observed_z_ds4"] = {k: round(v["z"], 2) for k, v in res["others"].items()}
    fam["per_segment_observed_z_ds4"][C.FLAG] = round(res["flag"]["ds1"]["z"], 2)
    res["familywise"] = fam
    json.dump(res, open(os.path.join(HERE, "vf_perm.json"), "w"), indent=1, default=float)
    print("\n=== familywise ===")
    print(json.dumps({k: v for k, v in fam.items()}, indent=1))
