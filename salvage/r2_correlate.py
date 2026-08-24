"""r2_correlate.py — reframe analyst, H6 steps 1+3.

(a) Spearman of f05 vs each condition proxy, each with a naive AND a 4^3-tile
    block-permutation null (200 perms; f05 field permuted, proxies fixed).
(b) Incremental-R2 decomposition: what do texture/surface add beyond density?
(c) Community-value test: does f05 predict m7 surface-prediction holes
    (low sheet recovery per material voxel) BEYOND cheap CT covariates?
    - partial Spearman with block-perm null
    - leave-slab-out CV R2: cheap-covariate model vs +f05
    - AUC for bottom-decile-recovery ("hole") tiles among interior tiles
Writes reframe_correlations.json.
"""
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats as sps

SAL = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"
rng = np.random.default_rng(42)
NPERM = 200

sc = pd.read_parquet(SAL + r"\proxies.parquet").reset_index(drop=True)
sc["surf_recovery"] = np.where(sc.mat_frac_l4 >= 0.05,
                               sc.surf_frac / sc.mat_frac_l4, np.nan)
n_all = len(sc)
print(f"{n_all} scored tiles; surf_recovery defined for "
      f"{(~sc.surf_recovery.isna()).sum()}")

pos = sc[["ti", "tj", "tk"]].values.astype(np.int64)
out = {"nperm": NPERM, "n": n_all}

# ---- block permutation machinery (4^3 tile blocks), as in analysis.py ----
bid = (pos[:, 0] // 4) * 10**6 + (pos[:, 1] // 4) * 10**3 + (pos[:, 2] // 4)
order_local = np.lexsort((pos[:, 2] % 4, pos[:, 1] % 4, pos[:, 0] % 4))
blocks = defaultdict(list)
for i in order_local:
    blocks[bid[i]].append(i)
blocks = list(blocks.values())
by_size = defaultdict(list)
for k, b in enumerate(blocks):
    by_size[len(b)].append(k)

def block_perm(v, rg):
    outv = v.copy()
    for s, ks in by_size.items():
        if len(ks) < 2:
            continue
        perm = rg.permutation(ks)
        for k_t, k_s in zip(ks, perm):
            outv[np.array(blocks[k_t])] = v[np.array(blocks[k_s])]
    return outv

# =========== (a) Spearman f05 vs proxies, naive + block nulls ===========
f05 = sc.f05.values
proxies = ["fill", "meanct_mat", "ct_std_all", "ct_std_mat",
           "surf_frac", "surf_recovery"]
corr = {}
for p in proxies:
    x = sc[p].values
    m = ~np.isnan(x)
    obs = sps.spearmanr(f05[m], x[m]).statistic
    null_n, null_b = [], []
    rg = np.random.default_rng(7)
    for _ in range(NPERM):
        null_n.append(sps.spearmanr(rng.permutation(f05)[m], x[m]).statistic)
        null_b.append(sps.spearmanr(block_perm(f05, rg)[m], x[m]).statistic)
    null_n, null_b = np.array(null_n), np.array(null_b)
    p2_n = float((np.abs(null_n) >= abs(obs)).mean())
    p2_b = float((np.abs(null_b) >= abs(obs)).mean())
    corr[p] = {"rho_obs": float(obs), "n": int(m.sum()),
               "null_naive_sd": float(null_n.std()), "p_naive": p2_n,
               "null_block_mean": float(null_b.mean()),
               "null_block_sd": float(null_b.std()), "p_block": p2_b,
               "z_block": float((obs - null_b.mean()) / null_b.std())}
    print(f"spearman f05~{p:13s} rho={obs:+.3f}  block null "
          f"{null_b.mean():+.3f}±{null_b.std():.3f}  p_block={p2_b:.3f}")
out["spearman_f05"] = corr

# inter-proxy structure (context for mediation claims)
inter = {}
for a, b in [("meanct_mat", "ct_std_all"), ("meanct_mat", "surf_frac"),
             ("ct_std_all", "surf_frac"), ("meanct_mat", "surf_recovery"),
             ("fill", "surf_frac"), ("ct_std_mat", "meanct_mat")]:
    m = ~(np.isnan(sc[a].values) | np.isnan(sc[b].values))
    inter[f"{a}~{b}"] = float(sps.spearmanr(sc[a][m], sc[b][m]).statistic)
print("inter-proxy spearman:", {k: round(v, 3) for k, v in inter.items()})
out["spearman_interproxy"] = inter

# =========== (b) incremental R2 decompositions ===========
def zs(a):
    return (a - a.mean()) / a.std()

def ols_r2(X, yv):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, yv, rcond=None)
    fit = X1 @ beta
    return 1 - ((yv - fit) ** 2).sum() / ((yv - yv.mean()) ** 2).sum(), fit

mfull = ~(sc[proxies].isna().any(axis=1).values)
sub = sc[mfull].reset_index(drop=True)
y = sub.f05.values
orders = {
    "density_first": ["meanct_mat", "fill", "ct_std_all", "surf_frac", "surf_recovery"],
    "texture_first": ["ct_std_all", "surf_frac", "surf_recovery", "meanct_mat", "fill"],
    "surface_first": ["surf_frac", "surf_recovery", "ct_std_all", "meanct_mat", "fill"],
}
incr = {}
for oname, cols in orders.items():
    seq = {}
    for k in range(1, len(cols) + 1):
        X = np.column_stack([zs(sub[c].values) for c in cols[:k]])
        r2, _ = ols_r2(X, y)
        seq[cols[k - 1]] = float(r2)
    incr[oname] = seq
    print(f"incremental R2 [{oname}]: "
          + " -> ".join(f"{c}:{v:.3f}" for c, v in seq.items()))
out["incremental_R2_f05"] = incr
X_all = np.column_stack([zs(sub[c].values) for c in
                         ["meanct_mat", "fill", "ct_std_all", "surf_frac",
                          "surf_recovery"]])
r2_all, fit_all = ols_r2(X_all, y)
out["R2_f05_all_proxies"] = float(r2_all)
print(f"R2 f05 ~ all condition proxies: {r2_all:.4f}")

# =========== (c) community value: predicting m7 surface holes ===========
# interior tiles only (mask-edge holes are trivial)
INT = (sc["fill"].values >= 0.9) & ~np.isnan(sc.surf_recovery.values) \
      & ~np.isnan(sc.ct_std_all.values)
si = sc[INT].reset_index(drop=True)
print(f"\ninterior tiles for hole analysis: {len(si)}")
rec = si.surf_recovery.values
out["hole_analysis_n"] = int(len(si))
out["surf_recovery_interior_quantiles"] = {
    q: float(np.quantile(rec, float(q))) for q in ["0.05", "0.25", "0.5", "0.75", "0.95"]}

# raw + partial Spearman f05 ~ recovery, block null recomputing the statistic
f05i = si.f05.values
obs_raw = sps.spearmanr(f05i, rec).statistic

cheap_cols = ["meanct_mat", "ct_std_all", "fill", "rnorm"]
Xc = np.column_stack([zs(si[c].values) for c in cheap_cols])

def partial_rho(f05v):
    rf = sps.rankdata(f05v).astype(float)
    rr = sps.rankdata(rec).astype(float)
    _, fit_f = ols_r2(Xc, rf)
    _, fit_r = ols_r2(Xc, rr)
    return sps.spearmanr(rf - fit_f, rr - fit_r).statistic

obs_part = partial_rho(f05i)
posi = si[["ti", "tj", "tk"]].values.astype(np.int64)
bidi = (posi[:, 0] // 4) * 10**6 + (posi[:, 1] // 4) * 10**3 + (posi[:, 2] // 4)
oli = np.lexsort((posi[:, 2] % 4, posi[:, 1] % 4, posi[:, 0] % 4))
blocks_i = defaultdict(list)
for i in oli:
    blocks_i[bidi[i]].append(i)
blocks_i = list(blocks_i.values())
by_size_i = defaultdict(list)
for k, b in enumerate(blocks_i):
    by_size_i[len(b)].append(k)

def block_perm_i(v, rg):
    outv = v.copy()
    for s, ks in by_size_i.items():
        if len(ks) < 2:
            continue
        perm = rg.permutation(ks)
        for k_t, k_s in zip(ks, perm):
            outv[np.array(blocks_i[k_t])] = v[np.array(blocks_i[k_s])]
    return outv

rg = np.random.default_rng(11)
null_raw, null_part = [], []
for _ in range(NPERM):
    fp = block_perm_i(f05i, rg)
    null_raw.append(sps.spearmanr(fp, rec).statistic)
    null_part.append(partial_rho(fp))
null_raw, null_part = np.array(null_raw), np.array(null_part)
out["hole_spearman_raw"] = {
    "rho_obs": float(obs_raw),
    "null_block_mean": float(null_raw.mean()), "null_block_sd": float(null_raw.std()),
    "p_block_2sided": float((np.abs(null_raw) >= abs(obs_raw)).mean())}
out["hole_spearman_partial"] = {
    "rho_obs": float(obs_part), "controls": cheap_cols,
    "null_block_mean": float(null_part.mean()), "null_block_sd": float(null_part.std()),
    "p_block_2sided": float((np.abs(null_part) >= abs(obs_part)).mean())}
print(f"f05~surf_recovery raw rho={obs_raw:+.3f} "
      f"(block null {null_raw.mean():+.3f}±{null_raw.std():.3f})")
print(f"f05~surf_recovery partial rho={obs_part:+.3f} | {cheap_cols} "
      f"(block null {null_part.mean():+.3f}±{null_part.std():.3f})")

# leave-slab-out CV: cheap model vs cheap+f05 predicting recovery
slabs = si.ti.values
y_rec = rec.copy()
def cv_r2(cols_extra):
    Xcols = cheap_cols + cols_extra
    X = np.column_stack([zs(si[c].values) for c in Xcols])
    press, sstot = 0.0, 0.0
    per_slab = {}
    for s in np.unique(slabs):
        tr, te = slabs != s, slabs == s
        X1 = np.column_stack([np.ones(tr.sum()), X[tr]])
        beta, *_ = np.linalg.lstsq(X1, y_rec[tr], rcond=None)
        pred = np.column_stack([np.ones(te.sum()), X[te]]) @ beta
        press += ((y_rec[te] - pred) ** 2).sum()
        sstot += ((y_rec[te] - y_rec[tr].mean()) ** 2).sum()
        per_slab[int(s)] = float(1 - ((y_rec[te] - pred) ** 2).sum()
                                 / ((y_rec[te] - y_rec[tr].mean()) ** 2).sum())
    return 1 - press / sstot, per_slab

r2_cheap, ps_cheap = cv_r2([])
r2_plus, ps_plus = cv_r2(["f05"])
out["hole_cv"] = {"R2_cheap": float(r2_cheap), "R2_cheap_plus_f05": float(r2_plus),
                  "delta": float(r2_plus - r2_cheap),
                  "per_slab_cheap": ps_cheap, "per_slab_plus": ps_plus}
print(f"leave-slab-out CV R2 predicting surf_recovery: cheap={r2_cheap:.4f} "
      f"+f05={r2_plus:.4f} (delta {r2_plus-r2_cheap:+.4f})")

# AUC: hole tile = bottom decile of recovery among interior tiles
thr = np.quantile(rec, 0.10)
hole = rec <= thr
def auc(score):
    r = sps.rankdata(score)
    return (r[hole].mean() - (hole.sum() + 1) / 2) / (len(r) - hole.sum())
aucs = {"f05": float(auc(-f05i)), "neg_f05": float(auc(f05i)),
        "neg_meanct": float(auc(-si.meanct_mat.values)),
        "meanct": float(auc(si.meanct_mat.values)),
        "ct_std_all": float(auc(si.ct_std_all.values)),
        "neg_ct_std_all": float(auc(-si.ct_std_all.values))}
out["hole_auc_bottom_decile"] = aucs
print("AUC for flagging hole tiles:", {k: round(v, 3) for k, v in aucs.items()})

json.dump(out, open(SAL + r"\reframe_correlations.json", "w"), indent=1)
print("saved reframe_correlations.json")
