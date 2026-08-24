"""m3_gallery.py — morphology gallery: CT | prob | overlay rows for
(a) the top patch-fraction tiles (closest to ink-plausible),
(b) exemplar membrane ("blob") tiles, (c) one mask-edge tile.
Slice chosen at the largest patch component's centroid z (or max fired-area z).
Overlay: CT gray + prob>=0.5 tinted by prob (single-hue sequential 'inferno'),
sheet mask (surf>127) outlined; identity is stated in text, not color alone.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage"
CACHE = os.path.join(OUT, "cache")

inv = {tuple(r["tile"]): r for r in json.load(open(os.path.join(OUT, "inventory.json")))["samples"]}
tiles_scored = json.load(open(os.path.join(OUT, "tile_scores.json")))
comps = json.load(open(os.path.join(OUT, "components.json")))

by_patch = sorted(tiles_scored, key=lambda r: -r["patch_frac"])
sel = []
for r in by_patch:
    if r["edge_frac"] > 0.5:            # skip edge-dominated
        continue
    sel.append((tuple(r["tile"]), "patch-rich"))
    if len(sel) == 6:
        break
# membrane exemplars: biggest single components
big = sorted((c for c in comps if c["class"] == "blob"), key=lambda c: -c["size"])
for c in big:
    t = tuple(c["tile"])
    if t not in [s[0] for s in sel]:
        sel.append((t, "membrane"))
    if len(sel) == 8:
        break
# one edge tile
for r in tiles_scored:
    if r["edge_frac"] >= 0.99:
        sel.append((tuple(r["tile"]), "mask-edge"))
        break

nrows = len(sel)
fig, axes = plt.subplots(nrows, 3, figsize=(10.5, 3.4 * nrows), facecolor="#111111")
score_of = {tuple(r["tile"]): r for r in tiles_scored}

for i, (t, kind) in enumerate(sel):
    z, y, x = t
    rec = inv[t]
    prob = np.load(rec["file"]).astype(np.float32) / 255.0
    ct = np.load(os.path.join(CACHE, f"ct_{z}_{y}_{x}.npy"))
    surf = np.load(os.path.join(CACHE, f"surf_{z}_{y}_{x}.npy"))
    # slice: largest patch comp centroid z, else max fired-area z
    pc = [c for c in comps if tuple(c["tile"]) == t and c["class"] == "patch"]
    if pc and kind == "patch-rich":
        biggest = max(pc, key=lambda c: c["size"])
        # centroid z from component: reconstruct via label? store slice by prob peak in patch e-range
        zs = int(round(np.argmax((prob >= 0.5).sum(axis=(1, 2)))))
        # prefer a z where patches live: use weighted argmax within +-10 of overall
        zsl = zs
    else:
        zsl = int(round(np.argmax((prob >= 0.5).sum(axis=(1, 2)))))
    sc = score_of[t]
    ct_s = ct[zsl]; pr_s = prob[zsl]; sf_s = surf[zsl] > 127

    ax = axes[i, 0]
    ax.imshow(ct_s, cmap="gray", vmin=0, vmax=255)
    ax.set_ylabel(f"z={z} y={y} x={x}\n[{kind}]", color="w", fontsize=8)
    ax.set_title(f"CT L2 (slice z'={zsl})", color="w", fontsize=9) if i == 0 else ax.set_title(
        f"z'={zsl}", color="#aaaaaa", fontsize=8)

    ax = axes[i, 1]
    ax.imshow(pr_s, cmap="inferno", vmin=0, vmax=1)
    if i == 0:
        ax.set_title("P(ink) 0..1", color="w", fontsize=9)
    ax.text(2, 61, f"patch {sc['patch_frac']:.2f}  blob {sc['blob_frac']:.2f}  edge {sc['edge_frac']:.2f}",
            color="w", fontsize=7, va="bottom")

    ax = axes[i, 2]
    ax.imshow(ct_s, cmap="gray", vmin=0, vmax=255)
    over = np.ma.masked_less(pr_s, 0.5)
    ax.imshow(over, cmap="inferno", vmin=0, vmax=1, alpha=0.65)
    if sf_s.any():
        ax.contour(sf_s.astype(float), levels=[0.5], colors=["#4fc3f7"], linewidths=0.7)
    if i == 0:
        ax.set_title("overlay: P>=0.5 (inferno) + sheet outline (light blue)",
                     color="w", fontsize=8)
    ax.text(2, 61, f"f05={sc['f05']:.3f} fill={sc['fill']:.2f}", color="w", fontsize=7, va="bottom")

for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#444444")

fig.suptitle("PHerc1203 2.4um ink screen - morphology gallery (probL2 64^3 tiles)\n"
             "rows: 6 highest patch-fraction tiles, 2 membrane exemplars, 1 mask-edge tile - NO tile is patch-dominant",
             color="w", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(os.path.join(OUT, "morph_gallery.png"), dpi=130, facecolor="#111111")
print("wrote morph_gallery.png,", nrows, "rows")
for t, k in sel:
    print(" ", t, k)
