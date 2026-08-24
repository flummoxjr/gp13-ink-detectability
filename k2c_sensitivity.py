"""Parameter sensitivity: does the separability ordering survive block size and
pre-smoothing, or is it an artefact of the two numbers I happened to pick?
Run on the K2C random-frame cubes (the frame the axis will actually ship on)."""
import os
import numpy as np, os, sys, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k2c_analyze import coh_med
from scipy import stats as st
C=r"D:\vesuvius-data\trackD\k2c"
scrolls=sorted({f.split("_")[0] for f in os.listdir(C) if f.endswith(".npy")})
print("scrolls available:", " ".join(scrolls))
grid=[(16,1.0),(32,0.5),(32,1.0),(32,2.0),(64,1.0)]
res={}
for B,S in grid:
    per={}
    for s in scrolls:
        fs=sorted(f for f in os.listdir(C) if f.startswith(s+"_rnd"))[:8]
        v=[coh_med(np.load(os.path.join(C,f)),block=B,sigma=S) for f in fs]
        v=[x for x in v if np.isfinite(x)]
        if v: per[s]=float(np.median(v))
    res[f"B{B}_s{S}"]=per
    print(f"B={B:2} sigma={S}: " + "  ".join(f"{k[5:]}={v:.3f}" for k,v in sorted(per.items())),flush=True)

ref=res["B32_s1.0"]
common=sorted(ref)
print("\nrank correlation vs the shipped setting (B=32, sigma=1.0):")
for k,per in res.items():
    if k=="B32_s1.0": continue
    a=[ref[s] for s in common if s in per]; b=[per[s] for s in common if s in per]
    if len(a)>2:
        rho,p=st.spearmanr(a,b)
        print(f"  {k:12} Spearman rho={rho:+.3f} (n={len(a)})  p={p:.3g}")
json.dump(res,open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\k2c_separability\sensitivity.json","w"),indent=1)
print("\nwrote sensitivity.json")
