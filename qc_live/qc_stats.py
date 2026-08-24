"""QC test 3: is there RELATIVE signal in the per-tile stats despite inflated probs?"""
import glob
import json
import os

import numpy as np
from scipy import stats as sps

RND = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\round_1"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\qc_stats_result.json"

rows = []
for p in glob.glob(os.path.join(RND, "w*_stats.jsonl")):
    w = os.path.basename(p).split("_")[0]
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["w"] = w
            rows.append(d)

run = [r for r in rows if "pmax" in r]
skip = [r for r in rows if r.get("skipped")]
print(f"rows={len(rows)} run={len(run)} skipped={len(skip)}")

z = np.array([r["tile"][0] for r in run])
y = np.array([r["tile"][1] for r in run])
x = np.array([r["tile"][2] for r in run])
fill = np.array([r["fill"] for r in run])
pmax = np.array([r["pmax"] for r in run])
f05 = np.array([r["f05"] for r in run])
f08 = np.array([r["f08"] for r in run])

res = {"n_run": len(run), "n_skipped": len(skip),
       "z_slabs_run": sorted(set(int(v) for v in z))}


def pct(a, q=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    return {f"p{qq}": round(float(np.percentile(a, qq)), 5) for qq in q}


res["dist"] = {"pmax": pct(pmax), "f05": pct(f05), "f08": pct(f08), "fill": pct(fill)}

# how many tiles fire at various thresholds on f08 / f05
res["frac_tiles"] = {}
for name, a in (("f05", f05), ("f08", f08)):
    res["frac_tiles"][name] = {str(t): round(float((a > t).mean()), 4)
                               for t in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2)}
# threshold to keep <5% / <1% of tiles
res["f08_thresh_top5pct"] = round(float(np.percentile(f08, 95)), 5)
res["f08_thresh_top1pct"] = round(float(np.percentile(f08, 99)), 5)

# correlations
cx = 26493 / 2
r_xy = np.sqrt((y - cx) ** 2 + (x - cx) ** 2)  # distance from axis (winding radius proxy)
res["spearman"] = {}
for name, a in (("fill", fill), ("z", z), ("radius_xy", r_xy)):
    for mname, m in (("pmax", pmax), ("f05", f05), ("f08", f08)):
        rho, pv = sps.spearmanr(a, m)
        res["spearman"][f"{mname}_vs_{name}"] = round(float(rho), 3)

# full tiles only: spread of f08 (is there discriminative variance left?)
full = fill >= 0.95
res["full_tiles"] = {"n": int(full.sum()), "f08": pct(f08[full]), "f05": pct(f05[full]),
                     "pmax": pct(pmax[full])}
edge = fill < 0.5
res["edge_tiles"] = {"n": int(edge.sum()), "f08": pct(f08[edge]),
                     "pmax": pct(pmax[edge])}

# per-z-slab medians (domain drift along z?)
res["by_z"] = {}
for zv in res["z_slabs_run"]:
    m = z == zv
    if m.sum() < 30:
        continue
    res["by_z"][str(zv)] = {"n": int(m.sum()),
                            "f05_med": round(float(np.median(f05[m])), 4),
                            "f08_med": round(float(np.median(f08[m])), 4),
                            "fill_med": round(float(np.median(fill[m])), 3)}

# spatial clustering of top 1% f08 among FULL tiles, within each z slab
# metric: fraction of top tiles having >=1 top neighbor (8-neighborhood in y,x)
# vs expected for a random subset of the same size drawn from run tiles in slab
top_frac = 0.01
clust = {}
rng = np.random.default_rng(0)
for zv in res["z_slabs_run"]:
    m = (z == zv) & full
    if m.sum() < 200:
        continue
    ff = f08[m]
    yy, xx = y[m] // 256, x[m] // 256
    k = max(5, int(len(ff) * top_frac))
    top_idx = np.argsort(ff)[-k:]
    cells = set(zip(yy, xx))
    top_cells = set(zip(yy[top_idx], xx[top_idx]))

    def frac_with_neighbor(cset):
        n = 0
        for (a, b) in cset:
            if any((a + da, b + db) in cset
                   for da in (-1, 0, 1) for db in (-1, 0, 1) if (da, db) != (0, 0)):
                n += 1
        return n / max(len(cset), 1)

    obs = frac_with_neighbor(top_cells)
    exp = []
    all_cells = list(cells)
    for _ in range(20):
        samp = rng.choice(len(all_cells), size=k, replace=False)
        exp.append(frac_with_neighbor(set(all_cells[i] for i in samp)))
    clust[str(zv)] = {"n_full": int(m.sum()), "k_top": k,
                      "obs_frac_w_neighbor": round(obs, 3),
                      "rand_frac_w_neighbor": round(float(np.mean(exp)), 3)}
res["top1pct_clustering"] = clust

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1))
