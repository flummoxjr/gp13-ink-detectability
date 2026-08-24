"""INVESTIGATION B, part 1: distribution of forward/reverse prediction correlation
across the 80-segment corpus survey, and what it covaries with.

Reads out/survey/survey_all.json (+ corpus_analysis.json for ruling stats).
Writes out/hunt/symmetry_stats.json and a figure.
"""
import json
import os

import numpy as np

ROOT = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
SURVEY = os.path.join(ROOT, "out", "survey", "survey_all.json")
CORPUS = os.path.join(ROOT, "out", "survey", "corpus_analysis.json")
OUT = os.path.join(ROOT, "out", "hunt")
os.makedirs(OUT, exist_ok=True)

rows = json.load(open(SURVEY))
corp = json.load(open(CORPUS))
cmap = {r["name"]: r for r in corp["results"]}

recs = []
for r in rows:
    f = r.get("forward") or {}
    rev = r.get("reverse") or {}
    if "fwd_rev_r" not in r or not f:
        continue
    H, W = f.get("canvas", [0, 0])
    c = cmap.get(r["name"], {})
    recs.append(dict(
        name=r["name"], scroll=r["scroll"], r=r["fwd_rev_r"],
        area_px=H * W, area_valid=H * W * f.get("nonzero_frac", 0),
        nonzero_frac=f.get("nonzero_frac", 0),
        p50=f.get("p50", np.nan), p99=f.get("p99", np.nan), max=f.get("max", np.nan),
        frac_gt_half=f.get("frac_gt_half", np.nan),
        frac_gt_b99=f.get("frac_gt_blankp99", np.nan),
        n_hot=f.get("n_hot_components", np.nan),
        secs=r.get("secs", np.nan),
        ruling_z=c.get("ruling_z", np.nan), mask_frac=c.get("mask_frac", np.nan),
        period_mm=c.get("period_mm", np.nan),
        rev_p50=rev.get("p50", np.nan), rev_frac_gt_half=rev.get("frac_gt_half", np.nan),
    ))

n_err = sum(1 for r in rows if "error" in r or not r.get("forward"))
print(f"{len(rows)} survey rows, {len(recs)} with fwd+rev, {n_err} errored/incomplete")

rr = np.array([x["r"] for x in recs])
print("\n=== fwd/rev correlation distribution (corpus) ===")
for q in (0, 5, 10, 25, 50, 75, 90, 95, 100):
    print(f"  p{q:<3d} {np.percentile(rr, q):.3f}")
print(f"  mean {rr.mean():.3f}  sd {rr.std(ddof=1):.3f}  n={len(rr)}")
print(f"  CONTROL (w035, ink-bearing): {corp['control_reference']['fwd_rev_r']}")
print(f"  fraction of corpus below control+0.1 (=0.176): {(rr < 0.176).mean():.3f}")

print("\n=== by scroll ===")
by = {}
for x in recs:
    by.setdefault(x["scroll"], []).append(x["r"])
for s, v in sorted(by.items()):
    v = np.array(v)
    print(f"  {s:12s} n={len(v):3d}  median {np.median(v):.3f}  "
          f"IQR {np.percentile(v,25):.3f}-{np.percentile(v,75):.3f}  min {v.min():.3f} max {v.max():.3f}")


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return np.nan, 0
    from scipy.stats import rankdata
    ra, rb = rankdata(a[m]), rankdata(b[m])
    return float(np.corrcoef(ra, rb)[0, 1]), int(m.sum())


print("\n=== what does fwd/rev r covary with? (Spearman, whole corpus) ===")
cov = {}
for key in ("area_px", "area_valid", "nonzero_frac", "p50", "p99", "max", "frac_gt_half",
            "frac_gt_b99", "n_hot", "secs", "ruling_z", "mask_frac", "period_mm"):
    v = np.array([x[key] for x in recs], dtype=float)
    rho, n = spearman(rr, v)
    cov[key] = dict(rho=None if not np.isfinite(rho) else round(rho, 3), n=n)
    print(f"  {key:14s} rho={rho:+.3f}  (n={n})")

# within-scroll (removes the scroll-level confound)
print("\n=== within PHerc1447 only (n=52, the big block) ===")
sub = [x for x in recs if x["scroll"] == "PHerc1447"]
rs = np.array([x["r"] for x in sub])
cov_1447 = {}
for key in ("area_px", "nonzero_frac", "p50", "p99", "frac_gt_half", "frac_gt_b99",
            "n_hot", "secs", "ruling_z", "mask_frac"):
    v = np.array([x[key] for x in sub], dtype=float)
    rho, n = spearman(rs, v)
    cov_1447[key] = dict(rho=None if not np.isfinite(rho) else round(rho, 3), n=n)
    print(f"  {key:14s} rho={rho:+.3f}  (n={n})")

# the lowest-symmetry segments = the most control-like = best sweep candidates
print("\n=== 15 LOWEST fwd/rev r (most asymmetric = most control-like) ===")
for x in sorted(recs, key=lambda z: z["r"])[:15]:
    print(f"  {x['r']:.3f}  {x['scroll']:10s} {x['name'][:44]:44s} "
          f"area={x['area_px']/1e6:6.1f}Mpx nz={x['nonzero_frac']:.2f} "
          f"p99={x['p99']:.0f} hot={x['n_hot']:.0f} rz={x['ruling_z']}")

print("\n=== 10 HIGHEST fwd/rev r ===")
for x in sorted(recs, key=lambda z: -z["r"])[:10]:
    print(f"  {x['r']:.3f}  {x['scroll']:10s} {x['name'][:44]:44s} "
          f"area={x['area_px']/1e6:6.1f}Mpx nz={x['nonzero_frac']:.2f} p99={x['p99']:.0f}")

json.dump(dict(n=len(recs), r_percentiles={str(q): round(float(np.percentile(rr, q)), 4)
                                           for q in (0, 5, 10, 25, 50, 75, 90, 95, 100)},
               r_mean=round(float(rr.mean()), 4), r_sd=round(float(rr.std(ddof=1)), 4),
               control_r=corp["control_reference"]["fwd_rev_r"],
               by_scroll={s: dict(n=len(v), median=round(float(np.median(v)), 4),
                                  min=round(float(np.min(v)), 4), max=round(float(np.max(v)), 4))
                          for s, v in by.items()},
               spearman_all=cov, spearman_1447=cov_1447,
               records=recs),
          open(os.path.join(OUT, "symmetry_stats.json"), "w"), indent=1, default=float)
print("\nwrote", os.path.join(OUT, "symmetry_stats.json"))
