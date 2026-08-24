"""HUNTER 2 / stage 1b — exploratory numbers before fixing the flag rule.
Print-only: cross-seed reproducibility of tile-r, letter/blank separations on
every tile feature, and the joint tail structure on 1203.
"""
import numpy as np
from pathlib import Path

COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")


def load(key):
    return dict(np.load(COMB / f"sym_tiles_{key}.npz"))


def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


for seg in ["w035", "1203A", "1203B"]:
    s42, s43 = load(f"{seg}_s42"), load(f"{seg}_s43")
    ok = np.isfinite(s42["r"]) & np.isfinite(s43["r"])
    print(f"[{seg}] tile-r cross-seed reproducibility r={pear(s42['r'][ok], s43['r'][ok]):.3f}"
          f"  (n={ok.sum()})")
    # does tile-r track tile std (mundane driver)?
    print(f"    r vs stdf (s42): {pear(s42['r'][ok], s42['stdf'][ok]):.3f}"
          f"   r vs meanf: {pear(s42['r'][ok], s42['meanf'][ok]):.3f}"
          f"   r vs validf: {pear(s42['r'][ok], s42['validf'][ok]):.3f}")

# control: letter vs blank on all features
s42, s43 = load("w035_s42"), load("w035_s43")
ok = np.isfinite(s42["r"]) & np.isfinite(s43["r"])
lf = s42["labfrac"]
letter = ok & (lf >= 0.02)
blank = ok & (lf == 0.0)
print(f"\n[w035] letter tiles {letter.sum()}, blank {blank.sum()}")
rmean = (s42["r"] + s43["r"]) / 2
for name, arr in [("r_mean2seed", rmean), ("frac195_s42", s42["frac195"]),
                  ("p99_s42", s42["p99f"]), ("stdf_s42", s42["stdf"])]:
    a, b = arr[letter], arr[blank]
    print(f"    {name:>13}: letter p10/50/90 = {np.percentile(a,10):.3f}/{np.percentile(a,50):.3f}/{np.percentile(a,90):.3f}"
          f"   blank = {np.percentile(b,10):.3f}/{np.percentile(b,50):.3f}/{np.percentile(b,90):.3f}")

# candidate low-r thresholds and what they select on control letter/blank + 1203
print("\nthreshold sweep on mean-of-two-seeds tile r (both-seed AND is implied by mean? no —"
      " use explicit AND):")
for thr in [0.15, 0.10, 0.05, 0.0, -0.05, -0.10, -0.15]:
    sel_ctrl = (s42["r"] <= thr) & (s43["r"] <= thr)
    n_l = (sel_ctrl & letter).sum(); n_b = (sel_ctrl & blank).sum()
    line = f"  thr {thr:+.2f}: ctrl letter {n_l}/{letter.sum()}  blank {n_b}/{blank.sum()}"
    for seg in ["1203A", "1203B"]:
        a42, a43 = load(f"{seg}_s42"), load(f"{seg}_s43")
        okk = np.isfinite(a42["r"]) & np.isfinite(a43["r"])
        sel = okk & (a42["r"] <= thr) & (a43["r"] <= thr)
        line += f"   {seg} {sel.sum()}/{okk.sum()}"
    print(line)

# 1203 low-r tail: what are those tiles like (frac195, std)?
print("\n1203 lowest-r tiles (both-seed mean r, bottom 10):")
for seg in ["1203A", "1203B"]:
    a42, a43 = load(f"{seg}_s42"), load(f"{seg}_s43")
    okk = np.isfinite(a42["r"]) & np.isfinite(a43["r"])
    rm = (a42["r"] + a43["r"]) / 2
    idx = np.argsort(np.where(okk, rm, np.inf), axis=None)[:10]
    for i in idx:
        iy, ix = np.unravel_index(i, rm.shape)
        print(f"  {seg} tile({iy},{ix}): r42={a42['r'][iy,ix]:+.3f} r43={a43['r'][iy,ix]:+.3f}"
              f" frac195={a42['frac195'][iy,ix]:.4f}/{a43['frac195'][iy,ix]:.4f}"
              f" p99={a42['p99f'][iy,ix]:.0f} std={a42['stdf'][iy,ix]:.1f} valid={a42['validf'][iy,ix]:.2f}")
