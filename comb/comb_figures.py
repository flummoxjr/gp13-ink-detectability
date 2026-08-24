"""HUNTER 2 / stage 4 — deliverable figures.

1. comb\\sym_calibration.png — calibrated tile-symmetry distributions:
   ECDF of tile fwd-vs-rev r for control letter tiles / control blank tiles /
   1203A / 1203B (both seeds pooled), + the r-vs-frac195 scatter that shows why
   low r alone is uninterpretable (letters separate on VALUE, not on r).
2. comb\\sym_flags.png — gallery: every cross-seed low-symmetry patch on 1203
   (s42 forward crop, letter-strength >195 px overlaid) next to the control's
   letter template and the control patches the same rule flags.
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_map, valid_mask, load_w035_label2d

from pathlib import Path
COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")
TILE = 256

# palette (dataviz reference, light mode; aqua/yellow relief = direct labels)
SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"
C_LETTER = "#2a78d6"; C_BLANK = "#eb6834"; C_A = "#1baf7a"; C_B = "#eda100"
C_OVER = "#eb6834"; C_TILE = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans"})


def load_tiles(key):
    return dict(np.load(COMB / f"sym_tiles_{key}.npz"))


def ecdf(ax, v, color, label):
    v = np.sort(v)
    y = np.arange(1, v.size + 1) / v.size
    ax.step(v, y, where="post", color=color, lw=2, label=f"{label} (n={v.size})")
    return v, y


def fig_calibration(flags):
    c42, c43 = load_tiles("w035_s42"), load_tiles("w035_s43")
    ok = np.isfinite(c42["r"]) & np.isfinite(c43["r"])
    letter = ok & (c42["labfrac"] >= 0.02)
    blank = ok & (c42["labfrac"] == 0.0)
    thr = flags["thr"]

    groups = {
        "w035 letter tiles": (np.concatenate([c42["r"][letter], c43["r"][letter]]), C_LETTER),
        "w035 blank tiles": (np.concatenate([c42["r"][blank], c43["r"][blank]]), C_BLANK),
    }
    seg_tiles = {}
    for seg, col in [("1203A", C_A), ("1203B", C_B)]:
        t42, t43 = load_tiles(f"{seg}_s42"), load_tiles(f"{seg}_s43")
        okk = np.isfinite(t42["r"]) & np.isfinite(t43["r"])
        groups[f"{seg} tiles"] = (np.concatenate([t42["r"][okk], t43["r"][okk]]), col)
        seg_tiles[seg] = (t42, t43, okk)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.11, wspace=0.22)

    # Panel A — ECDFs
    for name, (v, col) in groups.items():
        ecdf(axA, v, col, name)
    axA.axvline(thr, color=MUTED, lw=1.2, ls="--")
    axA.text(thr + 0.01, 0.03, f"flag threshold r = {thr:.2f}\n(letter-tile median)",
             color=INK2, fontsize=8.5, ha="left")
    axA.set_xlabel("tile fwd-vs-rev Pearson r (256 px tiles)")
    axA.set_ylabel("cumulative fraction of tiles")
    axA.set_title("A — local z-symmetry: letter tiles do NOT separate from blank",
                  fontsize=11, loc="left", color=INK)
    axA.legend(loc="lower right", fontsize=8.5, frameon=False)
    # direct labels near curves
    axA.set_xlim(-0.5, 1.0)

    # Panel B — r vs frac195
    axB.scatter(*[np.array(x) for x in [( (c42["r"][blank] + c43["r"][blank]) / 2),
                 np.maximum(c42["frac195"][blank], c43["frac195"][blank])]],
                s=14, color=C_BLANK, alpha=0.55, lw=0, label="w035 blank")
    for seg, col, lab in [("1203A", C_A, "1203A"), ("1203B", C_B, "1203B")]:
        t42, t43, okk = seg_tiles[seg]
        axB.scatter((t42["r"][okk] + t43["r"][okk]) / 2,
                    np.maximum(t42["frac195"][okk], t43["frac195"][okk]),
                    s=14, color=col, alpha=0.7, lw=0, label=lab)
    axB.scatter((c42["r"][letter] + c43["r"][letter]) / 2,
                np.maximum(c42["frac195"][letter], c43["frac195"][letter]),
                s=26, color=C_LETTER, alpha=0.95, lw=0, label="w035 letter")
    axB.axhline(0.017, color=MUTED, lw=1.2, ls="--")
    axB.text(-0.47, 0.021, "letter-tile floor frac>195 = 0.017", color=INK2, fontsize=8.5)
    axB.axvline(thr, color=MUTED, lw=1.2, ls="--")
    axB.set_xlim(-0.5, 1.0)
    axB.set_xlabel("tile fwd-vs-rev r (mean of seeds)")
    axB.set_ylabel("tile frac of pixels > 195 (max of seeds)")
    axB.set_title("B — letters separate on VALUE, not on symmetry",
                  fontsize=11, loc="left", color=INK)
    axB.legend(loc="upper right", fontsize=8.5, frameon=False)
    # direct labels for sub-3:1 colors (relief rule)
    axB.text(0.86, 0.0035, "1203A/B", color=INK2, fontsize=8.5, ha="center")
    axB.text(0.06, 0.30, "w035 letter", color=C_LETTER, fontsize=9, weight="bold")

    fig.suptitle("PHerc1203 local z-symmetry calibration — tile fwd-vs-rev r, control (w035) "
                 "letter/blank vs 1203  ·  256 px tiles, both seeds pooled",
                 fontsize=11.5, x=0.06, ha="left", color=INK)
    fig.text(0.06, 0.895, "Low tile symmetry is NOT ink-specific: on the letter-bearing control, "
             "letter and blank tiles share the same r distribution; the letter signature lives in the "
             ">195 value tail (panel B, y-axis).", fontsize=9, color=INK2)
    fig.savefig(COMB / "sym_calibration.png", dpi=140)
    plt.close(fig)
    print("saved sym_calibration.png")


def crop_for_tiles(arr, tiles, margin=1, max_dim=560):
    ys = [t[0] for t in tiles]; xs = [t[1] for t in tiles]
    y0 = max(0, (min(ys) - margin) * TILE); y1 = min(arr.shape[0], (max(ys) + 1 + margin) * TILE)
    x0 = max(0, (min(xs) - margin) * TILE); x1 = min(arr.shape[1], (max(xs) + 1 + margin) * TILE)
    crop = arr[y0:y1, x0:x1]
    f = max(1, int(np.ceil(max(crop.shape) / max_dim)))
    c = crop[:crop.shape[0] // f * f, :crop.shape[1] // f * f].astype(np.float32)
    c = c.reshape(c.shape[0] // f, f, c.shape[1] // f, f)
    cm = c.mean(axis=(1, 3))
    hot = (c > 195).max(axis=(1, 3))  # preserve letter-strength pixels at any ds
    return cm, hot, (y0, x0, f)


def draw_cell(ax, arr, tiles, title, stats, margin=1):
    cm, hot, (y0, x0, f) = crop_for_tiles(arr, tiles, margin)
    ax.imshow(cm, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    # letter-strength overlay
    ov = np.zeros(cm.shape + (4,), np.float32)
    rgb = tuple(int(C_OVER[i:i + 2], 16) / 255 for i in (1, 3, 5))
    ov[hot > 0] = (*rgb, 1.0)
    ax.imshow(ov, interpolation="nearest")
    # flagged-tile outlines
    for (iy, ix) in tiles:
        ax.add_patch(Rectangle(((ix * TILE - x0) / f, (iy * TILE - y0) / f),
                               TILE / f, TILE / f, fill=False, ls=(0, (3, 2)),
                               ec=C_TILE, lw=1.1))
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=4)
    ax.text(0, -0.025, stats, transform=ax.transAxes, fontsize=7.2, color=INK2,
            va="top", ha="left")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(AXIS); s.set_linewidth(0.6)


def patch_stats_text(p):
    s42, s43 = p["s42"], p["s43"]
    wmax = max(s42["width_max"] or 0, s43["width_max"] or 0)
    return (f"r42 {min(p['r42']):+.2f}..{max(p['r42']):+.2f}  "
            f"r43 {min(p['r43']):+.2f}..{max(p['r43']):+.2f}\n"
            f"frac>195 {max(s42['frac195'], s43['frac195']):.4f}  "
            f"p99 {max(s42['p99'], s43['p99']):.0f}  wmax {wmax:.0f}px")


def fig_gallery(flags):
    thr = flags["thr"]
    maps = {"w035": load_map("w035_s42"), "1203A": load_map("1203A_s42"),
            "1203B": load_map("1203B_s42")}
    lab2d = load_w035_label2d(maps["w035"].shape)

    # letter-template location: tile with max labfrac
    c42 = load_tiles("w035_s42")
    iy, ix = np.unravel_index(np.nanargmax(c42["labfrac"]), c42["labfrac"].shape)
    template_tiles = [(iy + dy, ix + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]

    wp = {p["id"]: p for p in flags["segments"]["w035"]["patches"]}
    pA = flags["segments"]["1203A"]["patches"]
    pB = flags["segments"]["1203B"]["patches"]

    ncol, nrow = 4, 5
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.5, 20.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02,
                        hspace=0.36, wspace=0.08)
    axf = axes.ravel()

    # header row
    draw_cell(axf[0], maps["w035"], template_tiles,
              "w035 LETTER TEMPLATE (labelled glyphs)",
              "what real letters look like: 0.5 mm strokes,\nsolid >195 cores (orange)", margin=0)
    p2, p3 = wp.get(2), wp.get(3)
    draw_cell(axf[1], maps["w035"], p2["tiles"],
              f"w035 flagged patch 2 — {p2['n_tiles']} tiles (HAS letters)",
              patch_stats_text(p2) + "  · LV+LM PASS", margin=0)
    draw_cell(axf[2], maps["w035"], [t for t in p3["tiles"]][:40],
              f"w035 flagged patch 3 — {p3['n_tiles']} tiles (HAS letters)",
              patch_stats_text(p3) + "  · LM PASS", margin=0)
    axf[3].axis("off")
    axf[3].text(0, 0.95,
        "READING GUIDE\n\n"
        f"Flag rule: tile fwd-vs-rev r \u2264 {thr:.2f}\n"
        "in BOTH seeds; 8-connected patches.\n\n"
        "Crops: seed42 forward map, shared\n"
        "0\u2013255 gray scale. Orange = pixels > 195\n"
        "(letter-strength, control blank p99).\n"
        "Dashed blue = flagged tiles.\n\n"
        "Letter template: frac>195 \u2265 0.017,\n"
        "p99 \u2265 200, stroke width ~58 px.\n\n"
        "VERDICT: all 16 x 1203 patches fail\n"
        "every letter criterion (max frac>195\n"
        "= 0.0044, no >30 px-wide >195 mark).",
        fontsize=9.5, color=INK, va="top", family="DejaVu Sans")

    cells = [("1203A", p) for p in pA] + [("1203B", p) for p in pB]
    for i, (seg, p) in enumerate(cells):
        ax = axf[4 + i]
        draw_cell(ax, maps[seg], p["tiles"],
                  f"{seg} patch {p['id']} — {p['n_tiles']} tile{'s' if p['n_tiles']>1 else ''}",
                  patch_stats_text(p) + "  · not letter-like")
    for j in range(4 + len(cells), nrow * ncol):
        axf[j].axis("off")

    fig.suptitle("PHerc1203 cross-seed low-z-symmetry patches vs the w035 letter template — "
                 "all flagged patches, seed42 forward crops", fontsize=13, x=0.02,
                 ha="left", color=INK, y=0.975)
    fig.text(0.02, 0.9585, "Top row = control calibration: the same rule applied to w035 flags its "
             "letter regions, and those PASS the value/morphology template (orange letter cores).",
             fontsize=10, color=INK2)
    fig.text(0.02, 0.947, "The 16 x 1203 patches below carry no letter-strength component "
             "(max frac>195 = 0.0044 vs letter-tile floor 0.017).",
             fontsize=10, color=INK2)
    fig.savefig(COMB / "sym_flags.png", dpi=110)
    plt.close(fig)
    print("saved sym_flags.png")


def main():
    flags = json.load(open(COMB / "comb_flags.json"))
    fig_calibration(flags)
    fig_gallery(flags)


if __name__ == "__main__":
    main()
