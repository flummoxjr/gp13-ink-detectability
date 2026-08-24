"""Verdict gallery: control vs 1203, rotated to detected orientation, with
projection profiles, ruling spectra (null-calibrated), and full-res crops."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_map, SALVAGE

UM = 9.362
DS = 4
INK = "#334155"; ACCENT = "#2563EB"; MUTED = "#94A3B8"; GRID = "#E2E8F0"

res = json.loads((SALVAGE / "verdict_periodicity.json").read_text())

ROWS = [
    ("w035_s42", "CONTROL w035 seed42 (letters proven, AUC 0.96)",
     (1050, 1500), None),
    ("1203A_s42", "PHerc1203 segment A seed42", None, (2000, 2400)),
    ("1203B_s42", "PHerc1203 segment B seed42", None, (2300, 2300)),
]

fig = plt.figure(figsize=(21, 14.5), facecolor="white")
gs = GridSpec(3, 4, figure=fig, width_ratios=[1.25, 1.0, 1.0, 0.85],
              hspace=0.32, wspace=0.25)

for i, (key, title, crop_ctrl, crop_1203) in enumerate(ROWS):
    d = np.load(SALVAGE / f"rot_{key}.npz")
    img, msk, th = d["img"], d["msk"], float(d["theta"])
    prof, perb, Pb, kpk = d["profile"], d["period_band"], d["power_band"], int(d["k_peak"])
    r = res[key]

    # --- panel 1: rotated map ---
    ax = fig.add_subplot(gs[i, 0])
    show = np.where(msk > 0.5, img, np.nan)
    ax.imshow(show, cmap="inferno", vmin=40, vmax=200, interpolation="nearest")
    ax.set_title(f"{title}\nrotated to theta={th:.1f} deg", fontsize=11, loc="left")
    ax.axis("off")
    if r["z_vs_null"] > 3:
        # draw ruling lines at the detected period, phased to profile minima
        per_ds = r["period_px_fullres"] / DS
        sm = gaussian_filter1d(prof, 4)
        y0 = np.argmin(sm[: int(per_ds)])
        for y in np.arange(y0, img.shape[0], per_ds):
            ax.axhline(y, color="#22D3EE", lw=0.9, alpha=0.85)
        ax.text(0.02, 0.02, f"ruling {r['period_mm']:.2f} mm",
                transform=ax.transAxes, color="#22D3EE", fontsize=10, weight="bold")

    # --- panel 2: projection profile ---
    ax = fig.add_subplot(gs[i, 1])
    y_mm = np.arange(prof.size) * DS * UM / 1000.0
    ax.plot(y_mm, prof, color=INK, lw=1.0)
    ax.plot(y_mm, gaussian_filter1d(prof, 90), color=MUTED, lw=1.2, ls="--")
    ax.set_xlabel("position across rows (mm)", fontsize=9)
    ax.set_ylabel("mean prediction", fontsize=9)
    ax.set_title(f"projection profile  (quiet-band frac {r['quiet_frac']:.2f}, "
                 f"aniso {r['row_col_anisotropy']:.2f})", fontsize=10, loc="left")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- panel 3: ruling-band spectrum, prominence units ---
    ax = fig.add_subplot(gs[i, 2])
    per_mm = perb * DS * UM / 1000.0
    prom_units = Pb / np.median(Pb)
    order = np.argsort(per_mm)
    ax.plot(per_mm[order], prom_units[order], color=ACCENT, lw=1.4)
    thr = r["null_prom_mean"] + 2 * r["null_prom_std"]
    ax.axhline(thr, color=MUTED, lw=1.0, ls=":")
    ax.text(per_mm.max() * 0.98, thr * 1.05, "null mean+2sd", ha="right",
            color=MUTED, fontsize=8)
    pk_mm = r["period_mm"]; pk_p = r["prominence"]
    ax.plot([pk_mm], [pk_p], "o", color=ACCENT, ms=7)
    right_half = pk_mm > 0.55 * per_mm.max()
    ax.annotate(f"{pk_mm:.2f} mm\nprom {pk_p:.0f}, z={r['z_vs_null']:.1f}",
                (pk_mm, pk_p), textcoords="offset points",
                xytext=(-10, -4) if right_half else (10, -4),
                ha="right" if right_half else "left",
                fontsize=9, color=INK)
    ax.set_xlabel("period (mm)", fontsize=9)
    ax.set_ylabel("power / band median", fontsize=9)
    ax.set_title("ruling-band spectrum (1.7-8.4 mm)", fontsize=10, loc="left")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- panel 4: full-res crop (same physical scale, 8.4 mm across) ---
    ax = fig.add_subplot(gs[i, 3])
    full = load_map(key)
    cy, cx = crop_ctrl if crop_ctrl else crop_1203
    half = 450
    crop = full[cy - half: cy + half, cx - half: cx + half]
    ax.imshow(crop, cmap="inferno", vmin=40, vmax=200, interpolation="nearest")
    ax.axis("off")
    ax.set_title("full-res crop (8.4 mm sq)", fontsize=10, loc="left")
    # scale bar: 1 mm
    px_mm = 1000 / UM
    ax.plot([40, 40 + px_mm], [crop.shape[0] - 45] * 2, color="white", lw=3)
    ax.text(40, crop.shape[0] - 65, "1 mm", color="white", fontsize=9)

fig.suptitle("ink_9um on PHerc1203 vs validated w035 control -- line-ruling periodicity test",
             fontsize=14, y=0.995)
plt.savefig(SALVAGE / "verdict_gallery.png", dpi=110, bbox_inches="tight",
            facecolor="white")
print("saved verdict_gallery.png")
