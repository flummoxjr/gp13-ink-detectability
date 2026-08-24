"""Render geography.png and residuals.png (light mode, reference dataviz palette)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SAL = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"
d = np.load(SAL + r"\fig_data.npz")
R = json.load(open(SAL + r"\analysis_results.json"))
pos, f05, resid = d["pos"], d["f05"], d["resid_f05_M1"]
meanct, rnorm, theta = d["meanct"], d["rnorm"], d["theta"]

# ---- palette (reference instance, light mode) ----
SURF, INK, INK2, MUT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
seq_blue = LinearSegmentedColormap.from_list("sb", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
    "#1c5cab", "#104281", "#0d366b"])
seq_blue.set_bad("#ecebe8")
seq_orange = LinearSegmentedColormap.from_list("so", [
    "#fdeee6", "#f8c8ae", "#f29a70", "#eb6834", "#c94e1d", "#93370f"])
seq_orange.set_bad("#ecebe8")
div_br = LinearSegmentedColormap.from_list("dv", [
    "#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2b1ae", "#e34948", "#8f2726"])
div_br.set_bad("#ecebe8")

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUT, "ytick.color": MUT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlecolor": INK,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
})
VOX_MM = 2.403e-3  # L0 voxel in mm

def profile(ax, xs, groups, vals, color, xlabel, title, note):
    import pandas as pd
    s = pd.DataFrame({"g": groups, "v": vals}).groupby("g")["v"]
    med, q1, q3 = s.median(), s.quantile(.25), s.quantile(.75)
    x = np.asarray([xs[g] for g in med.index], dtype=float)
    o = np.argsort(x)
    ax.fill_between(x[o], q1.values[o], q3.values[o], color=color, alpha=0.16,
                    lw=0, label="IQR")
    ax.plot(x[o], med.values[o], color=color, lw=2, marker="o", ms=4,
            label="median f05")
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left", pad=8)
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=8.5,
            color=INK2)
    ax.set_ylim(0, 0.115)

fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6))
fig.subplots_adjust(hspace=0.42, wspace=0.34, top=0.87, bottom=0.07,
                    left=0.06, right=0.94)
fig.suptitle("PHerc1203 2.4 µm ink screen — f05 geography over the tile grid",
             x=0.06, ha="left", fontsize=14, fontweight="bold", y=0.965)
fig.text(0.06, 0.925, "29,748 scored 256³ tiles, 14 z-slabs (lower 40% of band). "
         "η² = share of f05 variance explained by binning; p from 200 block permutations (4³-tile blocks ≈ 2.5 mm).",
         fontsize=9, color=INK2)

g = R["geography_eta2"]
# z profile
zi = pos[:, 0]
profile(axes[0, 0], {t: t * 256 * VOX_MM for t in np.unique(zi)}, zi, f05, BLUE,
        "z (mm, band frame)", "Firing vs z",
        f"η²={g['z']['eta2_obs']:.3f}  block-null {g['z']['null_block_mean']:.3f}"
        f"±{g['z']['null_block_sd']:.3f}  p={g['z']['p_block']:.3f}")
# radial
rb = np.clip((rnorm / 1.25 * 12).astype(int), 0, 11)
profile(axes[0, 1], {b: (b + .5) * 1.25 / 12 for b in range(12)}, rb, f05, BLUE,
        "radius / R95 (per-z mask centroid)", "Firing vs winding radius",
        f"η²={g['radial']['eta2_obs']:.3f}  block-null {g['radial']['null_block_mean']:.3f}"
        f"±{g['radial']['null_block_sd']:.3f}  p<0.005")
# angular
ab = np.clip(((theta + np.pi) / (2 * np.pi) * 16).astype(int), 0, 15)
profile(axes[0, 2], {b: -180 + (b + .5) * 22.5 for b in range(16)}, ab, f05,
        BLUE, "angle around centroid (°)", "Firing vs angle",
        f"η²={g['angular']['eta2_obs']:.3f}  block-null {g['angular']['null_block_mean']:.3f}"
        f"±{g['angular']['null_block_sd']:.3f}  p<0.005")
for ax in axes[0]:
    ax.set_ylabel("f05")

# ---- slab maps ----
def slab_map(ax, slab, vals, cmap, vmin, vmax, title):
    m = np.full((104, 104), np.nan)
    sel = pos[:, 0] == slab
    m[pos[sel, 1], pos[sel, 2]] = vals[sel]
    yy, xx = np.nonzero(~np.isnan(m))
    y0, y1, x0, x1 = yy.min() - 2, yy.max() + 2, xx.min() - 2, xx.max() + 2
    im = ax.imshow(np.ma.masked_invalid(m[y0:y1 + 1, x0:x1 + 1]), cmap=cmap,
                   vmin=vmin, vmax=vmax, origin="lower",
                   extent=[x0 - .5, x1 + .5, y0 - .5, y1 + .5])
    ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel("tile x"); ax.set_ylabel("tile y")
    ax.grid(False)
    ax.set_aspect("equal")
    return im

v1, v2 = np.nanpercentile(f05, [2, 98])
im = slab_map(axes[1, 0], 5, f05, seq_blue, v1, v2,
              "f05 map — slab z=1280 (3.1 mm)")
im2 = slab_map(axes[1, 1], 11, f05, seq_blue, v1, v2,
               "f05 map — slab z=2816 (6.8 mm)")
c1, c2 = np.nanpercentile(meanct, [2, 98])
im3 = slab_map(axes[1, 2], 11, meanct, seq_orange, c1, c2,
               "mean CT (material) — slab z=2816")
for ax, imx, lab in [(axes[1, 0], im, "f05"), (axes[1, 1], im2, "f05"),
                     (axes[1, 2], im3, "mean CT DN")]:
    cb = fig.colorbar(imx, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(lab, color=INK2)
    cb.ax.tick_params(color=MUT, labelcolor=MUT)
    cb.outline.set_edgecolor(GRID)
fig.text(0.06, 0.015, "Gray = unscreened / masked-out tiles. Slabs 16 & 23 have sparse coverage (321 / 118 tiles); all others ≥1,700.   "
         f"Spearman(meanCT, f05) = {R['spearman_meanct_mat_f05']:.2f} — compare panels E and F.",
         fontsize=8.5, color=MUT)
fig.savefig(SAL + r"\geography.png", dpi=160)
print("geography.png saved")

# ================= residuals figure =================
fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.6))
fig.subplots_adjust(hspace=0.40, wspace=0.26, top=0.87, bottom=0.07,
                    left=0.07, right=0.97)
fig.suptitle("Residual field (f05 − nuisance fit) — spatial structure",
             x=0.07, ha="left", fontsize=14, fontweight="bold", y=0.965)
fig.text(0.07, 0.925, "Nuisance model M1: f05 ~ fill + meanCT + z + radius (R²=0.62). "
         "Outlined tiles = top-1% residuals (n=297 overall).", fontsize=9, color=INK2)

k = 297
tidx = np.argpartition(resid, -k)[-k:]
tset = set(map(tuple, pos[tidx]))
rmax = np.nanpercentile(np.abs(resid), 99)
for ax, slab in [(axes[0, 0], 5), (axes[0, 1], 11)]:
    im = slab_map(ax, slab, resid, div_br, -rmax, rmax,
                  f"residual map — slab z={slab*256} ({slab*256*VOX_MM:.1f} mm)")
    tt = [p for p in tset if p[0] == slab]
    if tt:
        tt = np.array(tt)
        ax.scatter(tt[:, 2], tt[:, 1], marker="s", s=26, facecolors="none",
                   edgecolors=INK, linewidths=0.9, label="top-1% residual")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9,
                  facecolor=SURF, edgecolor=GRID)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("residual f05", color=INK2)
    cb.ax.tick_params(color=MUT, labelcolor=MUT)
    cb.outline.set_edgecolor(GRID)

# ---- NN null panel ----
ax = axes[1, 0]
c = R["clustering"]
rng = np.random.default_rng(5)
# rebuild null histograms quickly for display: use stored summary stats as normal approx
xs = np.linspace(1.8, 3.6, 400)
def normpdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
r1 = c["resid_f05_M1"]["top1"]
ax.fill_between(xs, normpdf(xs, r1["mean_nn_null_mean"], r1["mean_nn_null_sd"]),
                color=MUT, alpha=0.25, lw=0, label="random-subset null (200)")
ax.fill_between(xs, normpdf(xs, r1["mean_nn_blocknull_mean"], r1["mean_nn_blocknull_sd"]),
                color=AQUA, alpha=0.30, lw=0, label="block-perm null (200)")
ax.axvline(c["raw_f05"]["top1"]["mean_nn_obs"], color=ORANGE, lw=2)
ax.axvline(r1["mean_nn_obs"], color=BLUE, lw=2)
ax.text(c["raw_f05"]["top1"]["mean_nn_obs"] + 0.03, ax.get_ylim()[1] * 0.30,
        f"raw f05\n{c['raw_f05']['top1']['mean_nn_obs']:.2f}", color=ORANGE,
        ha="left", fontsize=8.5)
ax.text(r1["mean_nn_obs"] - 0.03, ax.get_ylim()[1] * 0.30,
        f"residual\n{r1['mean_nn_obs']:.2f}", color=BLUE, ha="right", fontsize=8.5)
ax.set_xlabel("mean nearest-neighbour distance of top-1% tiles (tile units, 1 = 0.61 mm)")
ax.set_ylabel("null density")
ax.set_title("Top-1% clustering: observed vs nulls", loc="left", pad=8)
ax.text(0.30, 0.72, f"z = {r1['z_nn']:+.1f} vs random subsets;\n"
        f"p<0.005 even vs block null", transform=ax.transAxes, va="top",
        fontsize=8.5, color=INK2)
ax.legend(loc="upper right", fontsize=8, facecolor=SURF, edgecolor=GRID)

# ---- Moran panel ----
ax = axes[1, 1]
m = R["moran"]
xs = np.linspace(-0.02, 0.58, 600)
ax.fill_between(xs, normpdf(xs, m["resid_f05_M1"]["null_naive_mean"],
                            m["resid_f05_M1"]["null_naive_sd"]),
                color=MUT, alpha=0.25, lw=0, label="naive perm null (200)")
ax.fill_between(xs, normpdf(xs, m["resid_f05_M1"]["null_block_mean"],
                            m["resid_f05_M1"]["null_block_sd"]),
                color=AQUA, alpha=0.30, lw=0, label="block-perm null (200)")
ax.axvline(m["raw_f05"]["I_obs"], color=ORANGE, lw=2)
ax.axvline(m["resid_f05_M1"]["I_obs"], color=BLUE, lw=2)
ax.text(m["raw_f05"]["I_obs"] - 0.008, ax.get_ylim()[1] * 0.18,
        f"raw f05\nI={m['raw_f05']['I_obs']:.2f}", color=ORANGE, fontsize=8.5,
        ha="right")
ax.text(m["resid_f05_M1"]["I_obs"] + 0.008, ax.get_ylim()[1] * 0.45,
        f"residual\nI={m['resid_f05_M1']['I_obs']:.2f}", color=BLUE, fontsize=8.5)
ax.set_xlabel("Moran's I (6-connectivity)")
ax.set_ylabel("null density")
ax.set_title("Spatial autocorrelation: observed vs nulls", loc="left", pad=8)
ax.text(0.02, 0.97, f"residual I z = {m['resid_f05_M1']['z_block']:+.0f} vs block null",
        transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
ax.legend(loc="center right", fontsize=8, facecolor=SURF, edgecolor=GRID)
ax.set_yscale("linear")

fig.savefig(SAL + r"\residuals.png", dpi=160)
print("residuals.png saved")
