"""The correct null for an ANGULAR-anisotropy statistic: what does it read on
material with no preferred direction? Finite-sample eigenvalue repulsion puts
that floor above zero, so it must be measured, not assumed.

Two isotropic references:
  (a) real in-scan noise  -- the K2b air cubes from the same volumes
  (b) synthetic isotropic Gaussian noise, matched in block size and smoothing
"""
import os
import numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k2c_analyze import coh_med
rng=np.random.default_rng(11)
CB=r"D:\vesuvius-data\trackD\k2b"

# (b) synthetic isotropic noise, several correlation lengths
print("SYNTHETIC ISOTROPIC NOISE (no preferred direction by construction)")
from scipy import ndimage as ndi
syn=[]
for sm in [0.0,1.0,2.0,4.0]:
    v=rng.normal(128,25,(128,128,128))
    if sm: v=ndi.gaussian_filter(v,sm)
    v=np.clip((v-v.mean())*(25/v.std())+128,0,255).astype(np.uint8)
    c=coh_med(v); syn.append(c)
    print(f"  smoothing sigma={sm:>3}  coherence={c:.3f}")

# (a) real air cubes
print("\nREAL IN-SCAN AIR (K2b air windows, same volumes)")
air=[]
for fn in sorted(os.listdir(CB)):
    if "air" not in fn or not fn.endswith(".npy"): continue
    a=np.load(os.path.join(CB,fn))
    if a.ndim!=3 or min(a.shape)<64: continue
    c=coh_med(a)
    if np.isfinite(c):
        air.append((fn[:-4],c))
print(f"  n={len(air)}  median={np.median([c for _,c in air]):.3f}  "
      f"range {min(c for _,c in air):.3f}-{max(c for _,c in air):.3f}")
for n,c in sorted(air,key=lambda t:-t[1])[:5]: print(f"    high: {n:34} {c:.3f}")

json.dump({"synthetic_isotropic":syn,"air":[{"cube":n,"coh":c} for n,c in air],
           "air_median":float(np.median([c for _,c in air])) if air else None},
          open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\k2c_separability\isotropy_floor.json","w"),indent=1)
print("\nwrote isotropy_floor.json")
