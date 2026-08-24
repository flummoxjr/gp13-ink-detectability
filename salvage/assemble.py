"""Assemble six worker stats files into one tile table, join L5-derived mean CT
and per-z scroll geometry (centroid/radius/angle). Output: tiles.parquet.

Tile coords are L0 origins, stride 256. L5 = L0/32, so one tile = 8^3 L5 block.
"""
import json
import numpy as np
import pandas as pd

STATS_DIR = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\final_stats"
L5_PATH = r"D:\vesuvius-data\trackD\ct1203_L5.npy"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage\tiles.parquet"

# ---------- 1. read jsonl ----------
rows = []
for w in range(6):
    p = rf"{STATS_DIR}\w{w}_stats_final.jsonl"
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            z, y, x = d["tile"]
            rows.append((w, z, y, x, d.get("fill", np.nan),
                         bool(d.get("skipped", False)),
                         d.get("pmax", np.nan), d.get("f05", np.nan),
                         d.get("f08", np.nan)))
df = pd.DataFrame(rows, columns=["worker", "z", "y", "x", "fill", "skipped",
                                 "pmax", "f05", "f08"])
print("total rows:", len(df))
print("scored:", (~df.skipped).sum(), "skipped:", df.skipped.sum())
print("per worker scored:")
print(df[~df.skipped].groupby("worker").size())

# duplicates across workers?
dup = df.duplicated(subset=["z", "y", "x"], keep=False)
print("rows sharing a tile coord:", dup.sum())
if dup.any():
    # keep scored over skipped, then first
    df = df.sort_values("skipped").drop_duplicates(subset=["z", "y", "x"],
                                                   keep="first")
    print("after dedupe:", len(df), "scored:", (~df.skipped).sum())

print("z range", df.z.min(), df.z.max(), "y", df.y.min(), df.y.max(),
      "x", df.x.min(), df.x.max())
print("scored z slabs:", sorted(df[~df.skipped].z.unique())[:100])

# ---------- 2. L5 block means -> tile mean CT ----------
L5 = np.load(L5_PATH, mmap_mode="r")
Z5, Y5, X5 = L5.shape          # (474, 828, 828)
B = 8                          # 256/32
nz, ny, nx = -(-Z5 // B), -(-Y5 // B), -(-X5 // B)
zb = np.arange(0, Z5, B); yb = np.arange(0, Y5, B); xb = np.arange(0, X5, B)
arr = np.asarray(L5, dtype=np.float64)
s = np.add.reduceat(arr, zb, axis=0)
s = np.add.reduceat(s, yb, axis=1)
s = np.add.reduceat(s, xb, axis=2)
lz = np.diff(np.append(zb, Z5)); ly = np.diff(np.append(yb, Y5))
lx = np.diff(np.append(xb, X5))
cnt = lz[:, None, None] * ly[None, :, None] * lx[None, None, :]
meanct_grid = s / cnt          # (60,104,104) mean CT DN incl. background zeros
print("meanCT grid:", meanct_grid.shape)

ti, tj, tk = (df.z.values // 256, df.y.values // 256, df.x.values // 256)
inb = (ti < nz) & (tj < ny) & (tk < nx)
print("tiles outside L5 grid:", (~inb).sum())
mc = np.full(len(df), np.nan)
mc[inb] = meanct_grid[ti[inb], tj[inb], tk[inb]]
df["meanct"] = mc
# material-only mean CT (background is 0 in the masked volume)
df["meanct_mat"] = np.where(df["fill"] > 0.02, df["meanct"] / df["fill"], np.nan)

# ---------- 3. per-z-slab mask centroid + radius scale ----------
# mask at L5: CT > 5 (same interior criterion as QC round)
mask = arr > 5
yy = np.arange(Y5)[:, None]
xx = np.arange(X5)[None, :]
recs = []
for zi in range(nz):
    m = mask[zi * B:(zi + 1) * B].any(axis=0)  # collapse the 8 L5 slices
    n = m.sum()
    if n < 50:
        recs.append((zi, np.nan, np.nan, np.nan, np.nan, 0))
        continue
    cy = (yy * m).sum() / n
    cx = (xx * m).sum() / n
    ys, xs = np.nonzero(m)
    r = np.hypot(ys - cy, xs - cx)
    recs.append((zi, cy * 32, cx * 32, np.median(r) * 32,
                 np.percentile(r, 95) * 32, int(n)))
cen = pd.DataFrame(recs, columns=["zi", "cy", "cx", "r50", "r95", "npix"])
print(cen.describe())

cy = cen.set_index("zi").cy.reindex(ti).values
cx = cen.set_index("zi").cx.reindex(ti).values
r95 = cen.set_index("zi").r95.reindex(ti).values
dy = (df.y.values + 128) - cy
dx = (df.x.values + 128) - cx
df["radius"] = np.hypot(dy, dx)
df["rnorm"] = df["radius"] / r95
df["theta"] = np.arctan2(dy, dx)

df.to_parquet(OUT, index=False)
cen.to_parquet(OUT.replace("tiles.parquet", "zslab_geometry.parquet"),
               index=False)
print("saved", OUT)
sc = df[~df.skipped]
print(sc[["fill", "pmax", "f05", "f08", "meanct", "meanct_mat",
          "rnorm"]].describe().T)
