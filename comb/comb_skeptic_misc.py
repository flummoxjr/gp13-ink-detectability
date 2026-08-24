r"""SKEPTIC PASS — flags 7 (uint8 saturation caveat) and 8 (residual families).

F7: recompute sat_frac (DN>=250, in-mask = nonzero voxels, matching
    comb_dense_scan's dense-phase mask definition) directly from three K2b
    cubes spanning the claimed range; compare to dense_scan.json.
F8: reproduce M1 (f05 ~ fill + meanct_mat + z + rnorm, standardized OLS,
    R2 0.6207) from salvage/tiles.parquet, then test the hunter's family-A
    mundane explanation quantitatively: add a low-density nonlinearity
    (quadratic + cubic in meanct_mat). If the 7 outer-rim picks' residuals
    are a linear-model-can't-bend-to-zero artifact, the nonlinear terms
    should absorb a large share of their residual. Report per-family
    residual shrinkage + overall R2 gain.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")
K2B = Path(r"D:\vesuvius-data\trackD\k2b")
SAL = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")

out = {}

# ---------- F7 ----------
ds = json.loads((COMB / "dense_scan.json").read_text())["per_cube"]
f7 = {}
for cube in ("PHerc0191_pap0", "PHerc1203_pap0", "PHerc0139_pap0"):
    a = np.load(K2B / f"{cube}.npy", mmap_mode="r")
    a = np.asarray(a)
    inmask = a > 0
    sat = float((a >= 250).sum() / inmask.sum())
    f7[cube] = {"recomputed_sat_frac_ge250": round(sat, 5),
                "hunter_sat_frac": ds[cube]["sat_frac"],
                "shape": list(a.shape)}
    print(cube, f7[cube])
out["F7_sat_recompute"] = f7

# ---------- F8 ----------
df = pd.read_parquet(SAL / "tiles.parquet")
sc = df[~df.skipped].copy()
sc = sc.dropna(subset=["meanct_mat", "rnorm", "f05", "f08", "fill"]).reset_index(drop=True)

def zs(a):
    return (a - a.mean()) / a.std()

y = sc.f05.values
base_cols = [zs(sc.fill.values), zs(sc.meanct_mat.values),
             zs(sc.z.values.astype(float)), zs(sc.rnorm.values)]
X1 = np.column_stack([np.ones(len(sc))] + base_cols)
beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
fit1 = X1 @ beta
r2_1 = 1 - ((y - fit1) ** 2).sum() / ((y - y.mean()) ** 2).sum()

mc = zs(sc.meanct_mat.values)
X2 = np.column_stack([np.ones(len(sc))] + base_cols + [mc ** 2, mc ** 3])
beta2, *_ = np.linalg.lstsq(X2, y, rcond=None)
fit2 = X2 @ beta2
r2_2 = 1 - ((y - fit2) ** 2).sum() / ((y - y.mean()) ** 2).sum()

sc["resid1"] = y - fit1
sc["resid2"] = y - fit2

picks = json.loads((COMB / "residual_picks.json").read_text())
famA = [(p["z"], p["y"], p["x"]) for p in picks if p["meanct"] < 60]
famB = [(p["z"], p["y"], p["x"]) for p in picks if p["meanct"] >= 60]
sc_idx = {(int(r.z), int(r.y), int(r.x)): i for i, r in sc.iterrows()}

def fam_stats(fam):
    idx = [sc_idx[t] for t in fam if t in sc_idx]
    r1 = sc.resid1.values[idx]; r2v = sc.resid2.values[idx]
    return {"n": len(idx),
            "resid_M1_med": float(np.median(r1)),
            "resid_M1nl_med": float(np.median(r2v)),
            "shrink_frac_med": float(1 - np.median(r2v) / np.median(r1))}

out["F8_residuals"] = {
    "R2_M1": round(float(r2_1), 4),
    "R2_M1_plus_meanct_sq_cube": round(float(r2_2), 4),
    "familyA_rim": fam_stats(famA),
    "familyB_damage": fam_stats(famB),
    "note": ("familyA = meanct<60 picks (near-empty rim), familyB = rest "
             "(frothy damage). Shrink_frac = how much of the median pick "
             "residual a meanct nonlinearity alone absorbs."),
}
# also: what does M1 predict vs observed for family A (the 7-8x claim)
idxA = [sc_idx[t] for t in famA if t in sc_idx]
out["F8_residuals"]["familyA_f05_obs_med"] = float(np.median(sc.f05.values[idxA]))
out["F8_residuals"]["familyA_f05_M1pred_med"] = float(np.median(fit1[idxA]))
out["F8_residuals"]["familyA_f05_M1nlpred_med"] = float(np.median(fit2[idxA]))
print(json.dumps(out["F8_residuals"], indent=1))

(COMB / "comb_skeptic_misc.json").write_text(json.dumps(out, indent=1))
print("saved comb_skeptic_misc.json")
