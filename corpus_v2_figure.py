"""Publication figure for the hardened corpus screen (protocol v2).

Six panels, one argument: the v2 screen finds nothing, and it finds nothing
while still detecting the human-verified control by an order of magnitude.

  A  gate cascade -- how many of the N scored segments survive each gate
  B  empirical-p ECDF against the uniform null (what "no signal" looks like)
  C  the two decisive axes: fwd/rev symmetry vs corrected z
  D  before/after -- v1's 16-permutation z against v2's corrected z
  E  is the peak a period? cycles-in-profile vs claimed period
  F  control and the refuted flag, same protocol, side by side

Colors are the validated categorical palette (slots 1-3 for the three scrolls,
which is the all-pairs-safe cap), status-critical for the refuted flag, and
primary ink for the control reference. Run via
`python analyze_survey_corpus_v2.py --figure-only`.
"""
import json
import os

import numpy as np

# ---- validated palette -----------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SCROLL_COLOR = {"PHerc1203": "#2a78d6", "PHerc1447": "#eb6834", "PHerc0800": "#1baf7a"}
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"
BLUE_500 = "#256abf"
BLUE_250 = "#86b6ef"

FLAG = "z_dbg_gen_00166_inp_hr"
CTRL = "w035_CONTROL_strided"


def _style(ax, ylabel=None, xlabel=None, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8, fontweight="bold")


def main(json_path, png_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401  (ScalarFormatter on panel E)

    d = json.load(open(json_path))
    seg = d["results"]
    ctrl = {c["name"]: c for c in d["control"]}
    n = len(seg)
    png_path = png_path or os.path.join(os.path.dirname(json_path), "corpus_screen_v2.png")

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.grid": False})
    fig = plt.figure(figsize=(16.0, 9.4))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          left=0.105, right=0.985, top=0.855, bottom=0.075)

    cfg = d["config"]
    fig.suptitle("Hardened corpus screen (protocol v2): no published GP-scroll segment "
                 "shows text-like line ruling", x=0.055, y=0.965, ha="left",
                 fontsize=15, color=INK, fontweight="bold")
    fig.text(0.055, 0.917,
             f"{n} segments scored from the saved ink_9um prediction maps  ·  "
             f"{cfg['n_perm']} joint (map, mask) block permutations each  ·  "
             f"{cfg['erode_fullres_px']}-px rim erosion, sigma={cfg['detrend_sigma_ds4_px']} "
             f"detrending, {cfg['theta_step_deg']:.0f}-deg orientation grid  ·  "
             f"band {cfg['band_mm'][0]}-{cfg['band_mm'][1]} mm",
             ha="left", fontsize=9.5, color=INK2)
    fig.text(0.055, 0.892,
             "Gates: empirical p <= 0.05  ·  >= 6 cycles in profile  ·  positive "
             "autocorrelation at 2P and 3P  ·  peak not in the band's 2 lowest Fourier "
             "bins  ·  |fwd/rev r| < 0.20",
             ha="left", fontsize=9.5, color=MUTED)

    # ---------------- A: gate cascade --------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    gates = [("gate_significance", "empirical p \u2264 0.05"),
             ("gate_cycles", "\u2265 6 cycles"),
             ("gate_autocorr", "\u03c1(2P), \u03c1(3P) > 0"),
             ("gate_band_bin", "not band-edge bin"),
             ("gate_fwd_rev", "|fwd/rev r| < 0.20")]
    counts = [sum(1 for r in seg if r[g]) for g, _ in gates]
    labels = [lbl for _, lbl in gates]
    inner = ("gate_significance", "gate_cycles", "gate_autocorr", "gate_band_bin")
    counts.append(sum(1 for r in seg if all(r[g] for g in inner)))
    labels.append("4 map-internal gates")
    counts.append(sum(1 for r in seg if r["passes_all_gates"]))
    labels.append("ALL FIVE GATES")
    y = np.arange(len(counts))[::-1]
    colors = [BLUE_250] * 5 + [BLUE_500, CRITICAL]
    ax.barh(y, counts, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.axvline(n, color=BASELINE, lw=1, ls=":")
    for yy, c in zip(y, counts):
        ax.text(c + n * 0.02, yy, f"{c}", va="center", ha="left", fontsize=10,
                color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, n * 1.18)
    _style(ax, xlabel=f"segments passing (of {n})",
           title="A   Gate cascade: nothing survives")
    ax.text(n, len(counts) - 0.35, f"all {n}", fontsize=8, color=MUTED, ha="right")

    # ---------------- B: empirical-p ECDF ----------------------------------
    ax = fig.add_subplot(gs[0, 1])
    p = np.sort([r["empirical_p"] for r in seg])
    ax.step(np.concatenate([[0], p, [1]]),
            np.concatenate([[0], np.arange(1, n + 1) / n, [1]]),
            where="post", color=BLUE_500, lw=2, label="corpus (v2 empirical p)")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls="--",
            label="uniform null (no signal anywhere)")
    ax.axvline(0.05, color=BASELINE, lw=1, ls=":")
    n_sig = sum(1 for r in seg if r["gate_significance"])
    exp_sig = d["expected_n_p_le_alpha_under_null"]
    ax.text(0.30, 0.94, f"p \u2264 0.05:  {n_sig} observed,\n{exp_sig:.1f} expected under "
            f"the null", fontsize=8.5, color=INK2, va="top",
            transform=ax.transAxes)
    ctrl_p = ctrl[CTRL]["empirical_p"]
    ax.annotate(f"control: p = {ctrl_p:.4f}\n(the floor, 1/{cfg['n_perm'] + 1})",
                xy=(ctrl_p, 0.0), xytext=(0.12, 0.52), fontsize=8.5, color=INK,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    _style(ax, xlabel="empirical p", ylabel="fraction of segments",
           title="B   The corpus is indistinguishable from its own null")

    # ---------------- C: symmetry vs corrected z ---------------------------
    ax = fig.add_subplot(gs[0, 2])
    for sc, col in SCROLL_COLOR.items():
        pts = [(r["fwd_rev_r"], r["z_corrected"]) for r in seg
               if r["scroll"] == sc and r["fwd_rev_r"] is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=34, color=col, alpha=0.85, linewidths=0.8,
                       edgecolors=SURFACE, label=f"{sc} (n={len(pts)})", zorder=3)
    c = ctrl[CTRL]
    ax.scatter([c["fwd_rev_r"]], [c["z_corrected"]], marker="*", s=340, color=INK,
               zorder=5, label="control: known Greek letters")
    ax.annotate(f"control\nz = {c['z_corrected']:+.1f}", xy=(c["fwd_rev_r"], c["z_corrected"]),
                textcoords="offset points", xytext=(16, -6), fontsize=9,
                color=INK, fontweight="bold", annotation_clip=True)
    fl = next((r for r in seg if r["name"] == FLAG), None)
    if fl:
        ax.scatter([fl["fwd_rev_r"]], [fl["z_corrected"]], s=90, facecolors="none",
                   edgecolors=CRITICAL, linewidths=2, zorder=6)
        ax.annotate(f"v1 flag\n(v1 z was +{fl['v1_ruling_z']:.2f})",
                    xy=(fl["fwd_rev_r"], fl["z_corrected"]), textcoords="offset points",
                    xytext=(-46, 46), fontsize=8.5, ha="center", annotation_clip=True,
                    color=CRITICAL, arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.1))
    ax.axvspan(-0.02, 0.20, color=GRID, alpha=0.55, zorder=0)
    ax.text(0.008, 0.02, "ink-like\n(one-sided)", fontsize=8, color=MUTED, va="bottom",
            transform=ax.transAxes)
    ax.axhline(0, color=BASELINE, lw=0.9)
    ax.legend(loc="center left", fontsize=8, frameon=False,
              bbox_to_anchor=(0.02, 0.55))
    _style(ax, xlabel="map-scale forward/reverse correlation r",
           ylabel="corrected z vs permutation null",
           title="C   The two axes that separate ink from texture")

    # ---------------- D: v1 vs v2 ------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    pts = [(r["v1_ruling_z"], r["z_corrected"], r) for r in seg
           if r.get("v1_ruling_z") is not None and r["z_corrected"] is not None]
    lim = [min(-3, min(x for x, _, _ in pts) - 0.5), max(7, max(x for x, _, _ in pts) + 0.8)]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", label="no change")
    for x, yv, r in pts:
        ax.scatter([x], [yv], s=30, color=SCROLL_COLOR.get(r["scroll"], MUTED),
                   alpha=0.85, linewidths=0.8, edgecolors=SURFACE, zorder=3)
    ax.axhline(0, color=BASELINE, lw=0.9)
    ax.axvline(5, color=BASELINE, lw=1, ls=":")
    ax.text(5.1, lim[0] + 0.4, "v1 flag threshold", fontsize=8, color=MUTED, rotation=90,
            va="bottom")
    if fl and fl.get("v1_ruling_z") is not None:
        ax.scatter([fl["v1_ruling_z"]], [fl["z_corrected"]], s=95, facecolors="none",
                   edgecolors=CRITICAL, linewidths=2, zorder=6)
        ax.annotate(f"{FLAG}\n+{fl['v1_ruling_z']:.2f}  \u2192  {fl['z_corrected']:+.2f}",
                    xy=(fl["v1_ruling_z"], fl["z_corrected"]), textcoords="offset points",
                    xytext=(-70, 52), fontsize=8.5, ha="center", annotation_clip=True,
                    color=CRITICAL, arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.1))
    ax.set_xlim(*lim)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _style(ax, xlabel="v1 ruling z  (16 permutations, no erosion, no detrending)",
           ylabel="v2 corrected z", title="D   What the four fixes cost the v1 scores")

    # ---------------- E: is the peak a period? -----------------------------
    ax = fig.add_subplot(gs[1, 1])
    for sc, col in SCROLL_COLOR.items():
        pts = [(r["period_mm"], r["n_cycles"]) for r in seg
               if r["scroll"] == sc and r["period_mm"]]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=34, color=col, alpha=0.85, linewidths=0.8,
                       edgecolors=SURFACE, zorder=3)
    ax.axhline(6, color=CRITICAL, lw=1.3, ls="--")
    n_below = sum(1 for r in seg if r["period_mm"] and r["n_cycles"] < 6)
    ax.text(cfg["band_mm"][0] + 0.05, 6.3, f"6-cycle gate — {n_below} of {n} segments below",
            color=CRITICAL, fontsize=8.5, ha="left", va="bottom")
    ax.scatter([c["period_mm"]], [c["n_cycles"]], marker="*", s=340, color=INK, zorder=5)
    ax.annotate(f"control\n{c['period_mm']:.2f} mm, {c['n_cycles']:.1f} cycles",
                xy=(c["period_mm"], c["n_cycles"]), textcoords="offset points",
                xytext=(16, 4), fontsize=9, color=INK, fontweight="bold",
                annotation_clip=True)
    if fl:
        ax.scatter([fl["period_mm"]], [fl["n_cycles"]], s=95, facecolors="none",
                   edgecolors=CRITICAL, linewidths=2, zorder=6)
        ax.annotate(f"v1 flag: {fl['n_cycles']:.1f} cycles",
                    xy=(fl["period_mm"], fl["n_cycles"]), textcoords="offset points",
                    xytext=(-30, -40), fontsize=8.5, ha="center", annotation_clip=True,
                    color=CRITICAL, arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.1))
    ax.set_yscale("log")
    ax.set_yticks([2, 3, 4, 6, 10, 20, 40])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(ax, xlabel="claimed period at the winning orientation (mm)",
           ylabel="cycles of that period inside the profile",
           title="E   A spectral peak is not a period")

    # ---------------- F: control vs flag table -----------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_title("F   Same protocol, opposite verdicts", fontsize=10, color=INK,
                 loc="left", pad=8, fontweight="bold")
    rows = [
        ("prominence (obs)", f"{c['obs_prominence']:.1f}",
         f"{fl['obs_prominence']:.1f}" if fl else "-"),
        ("null mean \u00b1 sd", f"{c['null_mean']:.1f} \u00b1 {c['null_sd']:.1f}",
         f"{fl['null_mean']:.1f} \u00b1 {fl['null_sd']:.1f}" if fl else "-"),
        ("corrected z", f"{c['z_corrected']:+.1f}",
         f"{fl['z_corrected']:+.2f}" if fl else "-"),
        ("empirical p", f"{c['empirical_p']:.4f}", f"{fl['empirical_p']:.3f}" if fl else "-"),
        ("period", f"{c['period_mm']:.2f} mm", f"{fl['period_mm']:.2f} mm" if fl else "-"),
        ("cycles in profile", f"{c['n_cycles']:.1f}", f"{fl['n_cycles']:.1f}" if fl else "-"),
        ("band bin of peak", f"{c['peak_bin_index']} of {c['band_bins']}",
         f"{fl['peak_bin_index']} of {fl['band_bins']}" if fl else "-"),
        ("\u03c1(1P) / \u03c1(2P) / \u03c1(3P)",
         f"{c['autocorr_1P']:+.2f} / {c['autocorr_2P']:+.2f} / {c['autocorr_3P']:+.2f}",
         f"{fl['autocorr_1P']:+.2f} / {fl['autocorr_2P']:+.2f} / {fl['autocorr_3P']:+.2f}"
         if fl else "-"),
        ("fwd/rev r", f"{c['fwd_rev_r']:.3f}", f"{fl['fwd_rev_r']:.3f}" if fl else "-"),
        ("gates passed", f"{c['gates_passed']} of 5",
         f"{fl['gates_passed']} of 5" if fl else "-"),
    ]
    ax.text(0.0, 1.0, "measurement", fontsize=8.5, color=MUTED, transform=ax.transAxes)
    ax.text(0.52, 1.0, "CONTROL\nw035, known letters", fontsize=8.5, color=INK,
            fontweight="bold", ha="center", transform=ax.transAxes, va="top")
    ax.text(0.87, 1.0, "v1 FLAG\nz_dbg_gen_00166", fontsize=8.5, color=CRITICAL,
            fontweight="bold", ha="center", transform=ax.transAxes, va="top")
    y0, dy = 0.855, 0.083
    for i, (lab, a, b) in enumerate(rows):
        yy = y0 - i * dy
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((-0.02, yy - 0.028), 1.04, dy * 0.86,
                                       transform=ax.transAxes, color=GRID, alpha=0.5,
                                       zorder=0, lw=0))
        bold = lab in ("corrected z", "empirical p", "fwd/rev r", "gates passed")
        ax.text(0.0, yy, lab, fontsize=8.6, color=INK2, transform=ax.transAxes,
                va="center")
        ax.text(0.52, yy, a, fontsize=8.9, color=INK, ha="center", va="center",
                transform=ax.transAxes, fontweight="bold" if bold else "normal")
        ax.text(0.87, yy, b, fontsize=8.9, color=CRITICAL if bold else INK2, ha="center",
                va="center", transform=ax.transAxes, fontweight="bold" if bold else "normal")
    ax.text(0.0, y0 - len(rows) * dy - 0.02,
            "The screen still detects the human-verified letters by an order of\n"
            "magnitude; the one v1 flag fails every gate under the same code path.",
            fontsize=8.4, color=INK2, transform=ax.transAxes, va="top")

    fig.savefig(png_path, dpi=170)
    print(f"wrote {png_path}")
    return png_path


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else
         r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\survey\corpus_analysis_v2.json")
