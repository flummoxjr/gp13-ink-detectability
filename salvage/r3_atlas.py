"""r3_atlas.py — reframe analyst, H6 step 2 deliverable.

Builds:
  atlas_1203.npy — (60,104,104) float32 tile-grid field of f05 (NaN = unscored),
                   axis order (z,y,x), 1 cell = 256 L0 voxels = 0.6152 mm.
  atlas_1203.png — montage of the 14 scored z-slabs, mm axes, mask outline,
                   sequential colorbar. Honest labeling: this is the Paris4-
                   trained ink3d RESPONSE field (texture resemblance under
                   domain shift), which tracks CT density — NOT ink, NOT damage.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SAL = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"
VOX_MM = 2.403e-3          # L0 voxel in mm
TILE_MM = 256 * VOX_MM     # 0.6152 mm

# ---- palette (dataviz reference instance, light mode) ----
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
cmap.set_bad(SURFACE)

sc = pd.read_parquet(SAL + r"\proxies.parquet")
NZ, NY, NX = 60, 104, 104
atlas = np.full((NZ, NY, NX), np.nan, dtype=np.float32)
atlas[sc.ti, sc.tj, sc.tk] = sc.f05.astype(np.float32)
np.save(SAL + r"\atlas_1203.npy", atlas)
print("atlas grid saved:", atlas.shape, "scored cells:", np.isfinite(atlas).sum())

# ---- material mask outline at tile resolution from cached CT L5 ----
ct5 = np.load(r"D:\vesuvius-data\trackD\ct1203_L5.npy", mmap_mode="r")
slabs = sorted(sc.ti.unique())
mask_frac = {}
for ti in slabs:
    blk = np.asarray(ct5[ti * 8:(ti + 1) * 8])          # (8, 828, 828)
    mat = np.zeros((8, NY * 8, NX * 8), dtype=np.float32)
    mat[:, :blk.shape[1], :blk.shape[2]] = blk > 5
    m = mat.reshape(8, NY, 8, NX, 8).mean(axis=(0, 2, 4))
    mask_frac[ti] = m                                    # (104,104) per-tile material frac
print("mask fractions computed for", len(mask_frac), "slabs")

# common crop: bounding box of material across scored slabs, small margin
anym = np.any([mask_frac[t] > 0.5 for t in slabs], axis=0)
jj, kk = np.where(anym)
j0, j1 = max(jj.min() - 2, 0), min(jj.max() + 3, NY)
k0, k1 = max(kk.min() - 2, 0), min(kk.max() + 3, NX)
print(f"crop tj [{j0},{j1}) tk [{k0},{k1})")

counts = sc.groupby("ti").size().to_dict()
vmax = float(np.nanquantile(atlas, 0.995))

fig, axes = plt.subplots(3, 5, figsize=(16.5, 11.6), facecolor=SURFACE)
extent = [k0 * TILE_MM, k1 * TILE_MM, j0 * TILE_MM, j1 * TILE_MM]
im = None
for ax, ti in zip(axes.flat, slabs):
    sl = atlas[ti, j0:j1, k0:k1]
    im = ax.imshow(sl, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                   extent=extent, interpolation="nearest")
    ax.contour(mask_frac[ti][j0:j1, k0:k1], levels=[0.5], colors=MUTED,
               linewidths=0.7, extent=extent, origin="lower")
    zmid = (ti + 0.5) * TILE_MM
    ax.set_title(f"z \u2248 {zmid:.1f} mm   (slab {ti})", fontsize=10,
                 color=INK2, pad=4)
    ax.text(0.03, 0.03, f"n = {counts[ti]:,}", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)
    if ax not in axes[:, 0]:
        ax.set_yticklabels([])
    if ax not in axes[-1, :] and ti not in slabs[-5:]:
        pass
axes[1, 0].set_ylabel("y (mm)", fontsize=9, color=INK2)
axes[-1, 2].set_xlabel("x (mm)", fontsize=9, color=INK2)
# unused panel: reading guide
guide = axes.flat[len(slabs)]
guide.axis("off")
guide.text(0.02, 0.95, "How to read this",
           fontsize=10.5, color=INK, weight="bold", va="top")
guide.text(0.02, 0.82,
           "Cell = 0.62 mm tile; value = f05, the fraction of\n"
           "voxels the Paris4-trained ink3d model scores\n"
           "P(ink) > 0.5. On this out-of-domain volume the\n"
           "field is a MATERIAL-CONDITION read-out, not ink:\n"
           "\u03c1(f05, mean CT density) = +0.73\n"
           "\u03c1(f05, crack/texture std) = \u22120.54\n"
           "\u03c1(f05, m7 sheet recovery) = +0.41\n"
           "Dark = dense, well-preserved, well-segmentable.\n"
           "Light = sparse, cracked, damaged, or scroll rim.\n"
           "Gray outline = material mask. Blank = not scored\n"
           "(fleet stopped at ~45%) or air.",
           fontsize=8.6, color=INK2, va="top", linespacing=1.45)
fig.suptitle("PHerc1203 2.4 \u00b5m band \u2014 ink3d response atlas (f05 per 0.62 mm tile)",
             fontsize=14, color=INK, y=0.985)
fig.text(0.5, 0.955,
         "What a Paris4-trained ink model fires on when there is no ink to find: "
         "a proxy map of material density / preservation \u2014 NOT ink, NOT damage density",
         fontsize=10, color=INK2, ha="center")
cax = fig.add_axes([0.33, 0.045, 0.34, 0.013])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("f05 \u2014 fraction of tile voxels with P(ink) > 0.5", fontsize=9, color=INK2)
cb.ax.tick_params(labelsize=8, colors=MUTED)
cb.outline.set_color(GRID)
fig.subplots_adjust(left=0.045, right=0.985, top=0.925, bottom=0.075,
                    wspace=0.06, hspace=0.22)
fig.savefig(SAL + r"\atlas_1203.png", dpi=170, facecolor=SURFACE)
print("saved atlas_1203.png")
