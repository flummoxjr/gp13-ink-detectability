"""Inspect w035 supervision mask extent, label extent, prediction stats.
Groundwork for the beyond-labels component catalog."""
import numpy as np
import tifffile
from pathlib import Path
from scipy import ndimage as ndi

DDATA = Path(r"D:\vesuvius-data\trackD")
OUT_W035 = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_w035")

# supervision mask -> 2D max
sup = np.load(DDATA / "w035_sup.npy", mmap_mode="r")
sup2d = np.zeros(sup.shape[1:], dtype=np.uint8)
for z in range(sup.shape[0]):
    np.maximum(sup2d, np.asarray(sup[z]), out=sup2d)
print("sup2d nonzero frac:", (sup2d > 0).mean())
ys, xs = np.nonzero(sup2d)
if len(ys):
    print("sup extent y:", ys.min(), ys.max(), " x:", xs.min(), xs.max())
    print("sup unique vals:", np.unique(sup2d)[:10])
# per-z fill
for z in range(sup.shape[0]):
    f = (np.asarray(sup[z]) > 0).mean()
    if f > 0:
        print(f"  sup z={z} fill={f:.4f}")

# labels -> 2D max
lab = np.load(DDATA / "w035_ink.npy", mmap_mode="r")
lab2d = np.zeros(lab.shape[1:], dtype=np.uint8)
for z in range(lab.shape[0]):
    np.maximum(lab2d, np.asarray(lab[z]), out=lab2d)
lab2d = lab2d > 0
ys, xs = np.nonzero(lab2d)
print("label extent y:", ys.min(), ys.max(), " x:", xs.min(), xs.max(),
      " fill:", lab2d.mean())

# prediction valid mask + value stats
pred = tifffile.imread(str(OUT_W035 / "w035_seed42-075000.tif"))
m = pred > 0
m = ndi.binary_closing(m, structure=np.ones((5, 5), bool))
m = ndi.binary_fill_holes(m)
m = ndi.binary_erosion(m, structure=np.ones((3, 3), bool), iterations=40)
print("valid mask frac:", m.mean())
inm = pred[m]
print("pred in-mask percentiles:",
      {p: int(np.percentile(inm, p)) for p in (50, 80, 90, 95, 99)})
print("frac >=195 in-mask:", (inm >= 195).mean())
np.save(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb\_sup2d.npy", (sup2d > 0))
np.save(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb\_lab2d.npy", lab2d)
np.save(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb\_valid42.npy", m)
