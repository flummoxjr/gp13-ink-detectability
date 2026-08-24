"""H1/H5 analysis: geography structure tests, nuisance regression, residual
clustering with permutation nulls. Writes analysis_results.json + npz for figs.

Nulls (DISCIPLINE): every clustering/structure claim carries >=200-permutation
null; block permutation (4^3-tile blocks ~ 2.5 mm) used where spatial
autocorrelation would inflate naive p-values.
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats as sps

SAL = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"
rng = np.random.default_rng(42)
NPERM = 200

df = pd.read_parquet(SAL + r"\tiles.parquet")
sc = df[~df.skipped].copy()
sc["ti"], sc["tj"], sc["tk"] = sc.z // 256, sc.y // 256, sc.x // 256
# analysis frame: need meanct_mat + rnorm
n0 = len(sc)
sc = sc.dropna(subset=["meanct_mat", "rnorm", "f05", "f08", "fill"]).reset_index(drop=True)
print(f"scored {n0}, analysis rows {len(sc)} (dropped {n0-len(sc)} NaN covar)")

pos = sc[["ti", "tj", "tk"]].values.astype(np.int64)
f05 = sc.f05.values
f08 = sc.f08.values
res_out = {"n_scored": int(n0), "n_analysis": int(len(sc)), "nperm": NPERM}

# ---------------- neighbor structure (6-connectivity) ----------------
key = {tuple(p): i for i, p in enumerate(pos)}
pairs = []
for dz, dy, dx in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
    for i, p in enumerate(pos):
        q = (p[0] + dz, p[1] + dy, p[2] + dx)
        j = key.get(q)
        if j is not None:
            pairs.append((i, j))
pairs = np.array(pairs)
print("undirected 6-conn neighbor pairs:", len(pairs))
res_out["n_pairs_6conn"] = int(len(pairs))
W = 2 * len(pairs)
n = len(sc)

def morans_I(v):
    z = v - v.mean()
    num = 2 * (z[pairs[:, 0]] * z[pairs[:, 1]]).sum()
    return (n / W) * num / (z ** 2).sum()

# ---------------- block permutation machinery (4^3 tile blocks) -------------
bid = (pos[:, 0] // 4) * 10**6 + (pos[:, 1] // 4) * 10**3 + (pos[:, 2] // 4)
order_local = np.lexsort((pos[:, 2] % 4, pos[:, 1] % 4, pos[:, 0] % 4))
# group tile indices per block, each sorted by local offset
from collections import defaultdict
blocks = defaultdict(list)
for i in order_local:
    blocks[bid[i]].append(i)
blocks = list(blocks.values())
sizes = np.array([len(b) for b in blocks])
by_size = defaultdict(list)
for k, b in enumerate(blocks):
    by_size[len(b)].append(k)
n_swappable = sum(len(v) * s for s, v in by_size.items() if len(v) > 1)
print(f"blocks: {len(blocks)}, sizes p50={np.median(sizes):.0f}, "
      f"tiles in swappable blocks: {n_swappable}/{n}")
res_out["n_blocks"] = int(len(blocks))
res_out["block_tiles_swappable_frac"] = float(n_swappable / n)

def block_perm(v, rg):
    """Permute value-vectors between equal-size blocks (local order preserved)."""
    out = v.copy()
    for s, ks in by_size.items():
        if len(ks) < 2:
            continue
        perm = rg.permutation(ks)
        for k_t, k_s in zip(ks, perm):
            out[np.array(blocks[k_t])] = v[np.array(blocks[k_s])]
    return out

# ---------------- 2. geography: profiles + eta^2 nulls ----------------
def eta2(v, g):
    gt = pd.Series(v).groupby(g)
    ssb = (gt.mean().values - v.mean()) ** 2 @ gt.size().values
    return ssb / ((v - v.mean()) ** 2).sum()

g_z = sc.ti.values
g_r = np.clip((sc.rnorm.values / 1.25 * 12).astype(int), 0, 11)
g_a = np.clip(((sc.theta.values + np.pi) / (2 * np.pi) * 16).astype(int), 0, 15)

geo = {}
for name, g in [("z", g_z), ("radial", g_r), ("angular", g_a)]:
    obs = eta2(f05, g)
    null_naive = np.array([eta2(rng.permutation(f05), g) for _ in range(NPERM)])
    rg = np.random.default_rng(7)
    null_block = np.array([eta2(block_perm(f05, rg), g) for _ in range(NPERM)])
    geo[name] = {
        "eta2_obs": float(obs),
        "null_naive_mean": float(null_naive.mean()), "null_naive_sd": float(null_naive.std()),
        "null_naive_max": float(null_naive.max()),
        "p_naive": float((null_naive >= obs).mean()),
        "null_block_mean": float(null_block.mean()), "null_block_sd": float(null_block.std()),
        "null_block_max": float(null_block.max()),
        "p_block": float((null_block >= obs).mean()),
    }
    print(f"eta2[{name}] obs={obs:.4f} naive null {null_naive.mean():.4f}"
          f"±{null_naive.std():.4f} (p={geo[name]['p_naive']:.3f}) "
          f"block null {null_block.mean():.4f}±{null_block.std():.4f} "
          f"(p={geo[name]['p_block']:.3f})")
res_out["geography_eta2"] = geo

# spearman refs
for a, b in [("fill", "f05"), ("meanct_mat", "f05"), ("rnorm", "f05"),
             ("z", "f05"), ("fill", "f08"), ("meanct_mat", "f08")]:
    r = sps.spearmanr(sc[a], sc[b]).statistic
    res_out[f"spearman_{a}_{b}"] = float(r)
    print(f"spearman {a} vs {b}: {r:.3f}")

# ---------------- 3. nuisance regression ----------------
def ols(X, yv):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, yv, rcond=None)
    fit = X1 @ beta
    r2 = 1 - ((yv - fit) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return beta, fit, r2

def zs(a):
    return (a - a.mean()) / a.std()

reg = {}
resid = {}
for yname, yv in [("f05", f05), ("f08", f08)]:
    # M1: spec model, linear
    Xm1 = np.column_stack([zs(sc.fill.values), zs(sc.meanct_mat.values),
                           zs(sc.z.values.astype(float)), zs(sc.rnorm.values)])
    b1, fit1, r2_1 = ols(Xm1, yv)
    # incremental R2
    incr = {}
    cols = ["fill", "meanct_mat", "z", "rnorm"]
    for kc in range(1, 5):
        _, _, r2k = ols(Xm1[:, :kc], yv)
        incr[cols[kc - 1]] = float(r2k)
    # M2: flexible nuisance (slab dummies + quadratics)
    dz = pd.get_dummies(sc.ti).values.astype(float)[:, 1:]
    Xm2 = np.column_stack([dz,
                           zs(sc.fill.values), zs(sc.fill.values) ** 2,
                           zs(sc.meanct_mat.values), zs(sc.meanct_mat.values) ** 2,
                           zs(sc.rnorm.values), zs(sc.rnorm.values) ** 2])
    b2, fit2, r2_2 = ols(Xm2, yv)
    reg[yname] = {"R2_M1_spec": float(r2_1), "R2_M1_incremental": incr,
                  "betas_M1_std": {c: float(v) for c, v in zip(cols, b1[1:])},
                  "R2_M2_flexible": float(r2_2)}
    resid[yname] = {"M1": yv - fit1, "M2": yv - fit2}
    print(f"[{yname}] M1 R2={r2_1:.4f} betas(std) "
          f"{dict(zip(cols, np.round(b1[1:], 5)))} | M2 R2={r2_2:.4f}")
res_out["regression"] = reg

# ---------------- 4. clustering: top-1% NN + adjacency + Moran ----------------
def top_idx(v, frac):
    k = max(1, int(round(len(v) * frac)))
    return np.argpartition(v, -k)[-k:]

def mean_nn(idx):
    P = pos[idx].astype(float)
    D = cdist(P, P)
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1).mean()

def adj_frac(idx):
    s = set(map(tuple, pos[idx]))
    c = 0
    for p in pos[idx]:
        for d in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            if (p[0]+d[0], p[1]+d[1], p[2]+d[2]) in s:
                c += 1
                break
    return c / len(idx)

def cluster_test(v, frac, tag, block_null=False):
    idx = top_idx(v, frac)
    k = len(idx)
    obs_nn, obs_af = mean_nn(idx), adj_frac(idx)
    nn_null, af_null = [], []
    for _ in range(NPERM):
        ridx = rng.choice(n, size=k, replace=False)
        nn_null.append(mean_nn(ridx)); af_null.append(adj_frac(ridx))
    nn_null, af_null = np.array(nn_null), np.array(af_null)
    out = {"n_top": int(k),
           "mean_nn_obs": float(obs_nn),
           "mean_nn_null_mean": float(nn_null.mean()), "mean_nn_null_sd": float(nn_null.std()),
           "p_clustered_nn": float((nn_null <= obs_nn).mean()),   # small NN = clustered
           "p_anticlustered_nn": float((nn_null >= obs_nn).mean()),
           "z_nn": float((obs_nn - nn_null.mean()) / nn_null.std()),
           "adjfrac_obs": float(obs_af),
           "adjfrac_null_mean": float(af_null.mean()), "adjfrac_null_sd": float(af_null.std()),
           "p_clustered_adj": float((af_null >= obs_af).mean()),  # high adj = clustered
           }
    if block_null:
        rg = np.random.default_rng(13)
        nnb = []
        for _ in range(NPERM):
            vb = block_perm(v, rg)
            nnb.append(mean_nn(top_idx(vb, frac)))
        nnb = np.array(nnb)
        out["mean_nn_blocknull_mean"] = float(nnb.mean())
        out["mean_nn_blocknull_sd"] = float(nnb.std())
        out["p_clustered_nn_block"] = float((nnb <= obs_nn).mean())
    print(f"[{tag}] top{frac*100:.0f}% n={k} NN obs={obs_nn:.3f} "
          f"null={nn_null.mean():.3f}±{nn_null.std():.3f} z={out['z_nn']:+.2f} "
          f"adj obs={obs_af:.3f} null={af_null.mean():.3f}"
          + (f" blockNN={out['mean_nn_blocknull_mean']:.3f}±{out['mean_nn_blocknull_sd']:.3f}"
             if block_null else ""))
    return out

clus = {}
for tag, v in [("raw_f05", f05), ("raw_f08", f08),
               ("resid_f05_M1", resid["f05"]["M1"]),
               ("resid_f05_M2", resid["f05"]["M2"]),
               ("resid_f08_M1", resid["f08"]["M1"])]:
    clus[tag] = {}
    for frac in (0.01, 0.05):
        clus[tag][f"top{int(frac*100)}"] = cluster_test(
            v, frac, tag, block_null=(tag in ("resid_f05_M1", "raw_f05") and frac == 0.01))
res_out["clustering"] = clus

# Moran's I
moran = {}
for tag, v in [("raw_f05", f05), ("resid_f05_M1", resid["f05"]["M1"]),
               ("resid_f05_M2", resid["f05"]["M2"]),
               ("resid_f08_M1", resid["f08"]["M1"])]:
    obs = morans_I(v)
    null_naive = np.array([morans_I(rng.permutation(v)) for _ in range(NPERM)])
    rg = np.random.default_rng(23)
    null_block = np.array([morans_I(block_perm(v, rg)) for _ in range(NPERM)])
    moran[tag] = {
        "I_obs": float(obs),
        "null_naive_mean": float(null_naive.mean()), "null_naive_sd": float(null_naive.std()),
        "p_naive": float((null_naive >= obs).mean()),
        "null_block_mean": float(null_block.mean()), "null_block_sd": float(null_block.std()),
        "null_block_max": float(null_block.max()),
        "p_block": float((null_block >= obs).mean()),
        "z_block": float((obs - null_block.mean()) / null_block.std()),
    }
    print(f"Moran[{tag}] I={obs:.4f} naive null {null_naive.mean():+.5f}"
          f"±{null_naive.std():.5f} block null {null_block.mean():+.4f}"
          f"±{null_block.std():.4f} p_block={moran[tag]['p_block']:.3f}")
res_out["moran"] = moran

json.dump(res_out, open(SAL + r"\analysis_results.json", "w"), indent=1)

# save arrays for figures
np.savez_compressed(SAL + r"\fig_data.npz",
                    pos=pos, f05=f05, f08=f08, fill=sc.fill.values,
                    meanct=sc.meanct_mat.values, rnorm=sc.rnorm.values,
                    theta=sc.theta.values, z=sc.z.values,
                    resid_f05_M1=resid["f05"]["M1"], resid_f05_M2=resid["f05"]["M2"])
print("saved analysis_results.json + fig_data.npz")
