"""E0 — geometry of the released ink prototype (Investigation C).

Reproduces Findings C-1/C-2/C-3 of trackD/hunt/embedding_prospecting.md.
No GPU, no network (assets already in trackD/hunt/assets/). Runs in <1 s.

    .venv/Scripts/python.exe trackD/hunt/e0_prototype_geometry.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

ref = np.load(os.path.join(ASSETS, "avg_ref_embedding.npy"))
e1 = np.load(os.path.join(ASSETS, "recorded_embeddings.npy"))
e2 = np.load(os.path.join(ASSETS, "recorded_embeddings_2.npy"))

res = {}

# --- C-1: are the two "128-token" files distinct? -------------------------
res["ref_shape"] = list(ref.shape)
res["ref_dtype"] = str(ref.dtype)
res["ref_norm"] = float(np.linalg.norm(ref))
res["files_byte_identical"] = bool(np.array_equal(e1, e2))
stacked = np.concatenate([e1, e2], 0)
res["n_rows_total"] = int(len(stacked))
res["n_unique_rows"] = int(len(np.unique(np.round(stacked, 5), axis=0)))
res["readme_claims_256"] = True

# --- provenance: is avg_ref exactly the normalised mean of the 128? -------
E = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
mean_unit = E.mean(0)
mean_unit_n = mean_unit / np.linalg.norm(mean_unit)
refn = ref / np.linalg.norm(ref)
res["cos_mean128_vs_avgref"] = float(mean_unit_n @ refn)

# --- C-2: cluster geometry ------------------------------------------------
C = E @ E.T
iu = np.triu_indices(len(E), 1)
pw = C[iu]
res["pairwise_cos"] = {
    "min": float(pw.min()),
    "p05": float(np.percentile(pw, 5)),
    "median": float(np.median(pw)),
    "p95": float(np.percentile(pw, 95)),
    "max": float(pw.max()),
}
cr = E @ refn
res["cos_token_vs_ref"] = {
    "min": float(cr.min()),
    "p05": float(np.percentile(cr, 5)),
    "median": float(np.median(cr)),
    "p95": float(np.percentile(cr, 95)),
    "max": float(cr.max()),
}
res["norm_of_mean_unit_token"] = float(np.linalg.norm(mean_unit))
res["energy_in_mean_direction"] = float(np.linalg.norm(mean_unit) ** 2)

X = E - E.mean(0)
s = np.linalg.svd(X, compute_uv=False)
v = s**2 / (s**2).sum()
res["pca"] = {
    "pc1": float(v[0]),
    "pc1_3": float(v[:3].sum()),
    "pc1_10": float(v[:10].sum()),
    "n_pcs_for_90pct": int(np.searchsorted(np.cumsum(v), 0.90) + 1),
}
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
res["centred_pairwise_cos_median"] = float(np.median((Xn @ Xn.T)[iu]))

# --- C-3: self-recall of the published tau=0.5 rule -----------------------
res["self_recall_vs_tau"] = {
    f"{t:.2f}": float((cr >= t).mean()) for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
}

# --- the three scorers frozen for E1 --------------------------------------
# S_mean: cosine to avg_ref.  S_nn: max cosine over the 128.
# S_maha: shrinkage-whitened score (128 samples, 864 dims -> shrinkage is mandatory).
cov = np.cov(X.T)
lam = 0.5
cov_s = (1 - lam) * cov + lam * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
prec = np.linalg.inv(cov_s).astype(np.float32)
np.savez(
    os.path.join(OUT, "e0_scorers.npz"),
    ref=refn.astype(np.float32),
    tokens=E.astype(np.float32),
    token_mean=E.mean(0).astype(np.float32),
    precision=prec,
    shrinkage=np.float32(lam),
)
res["scorers_saved"] = "out/e0_scorers.npz (ref, tokens[128,864], token_mean, precision, shrinkage)"

with open(os.path.join(OUT, "e0_prototype.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
