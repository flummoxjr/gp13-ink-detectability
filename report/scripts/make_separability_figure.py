"""Figure for §1.8 — the sheet-separability axis.

Three panels:
  (a) all 14 scrolls ranked by separability, bootstrap CI, isotropic floor, control marked
  (b) separability vs K2b structural SNR — the two axes are close to orthogonal
  (c) the ROI-picker bias: random frame vs intensity-max frame, paired, per scroll
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
A = json.load(open(os.path.join(T, "out", "k2c_separability", "k2c_analysis.json")))["scrolls"]
FLOOR = 0.105
CONTROL = "PHerc0139"
SEGMENTED = {"PHerc1203": 22, "PHerc1447": 52, "PHerc0800": 6}

rows = sorted(A.items(), key=lambda kv: -kv[1]["sep_med"])
names = [k for k, _ in rows]
sep = np.array([v["sep_med"] for _, v in rows])
lo = np.array([v["sep_ci95"][0] for _, v in rows])
hi = np.array([v["sep_ci95"][1] for _, v in rows])
snr = np.array([v["k2b_snr_q025"] for _, v in rows])
pick = np.array([v["sep_picked_med"] for _, v in rows])

fig = plt.figure(figsize=(15.2, 5.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.32)

# ---- (a) ranked
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(names))[::-1]
cols = ["#b03030" if n == CONTROL else ("#2a6ebb" if n in SEGMENTED else "#7a7a7a") for n in names]
ax.errorbar(sep, y, xerr=[sep - lo, hi - sep], fmt="none", ecolor="#999", lw=1.4, capsize=3, zorder=1)
ax.scatter(sep, y, c=cols, s=64, zorder=2, edgecolor="white", linewidth=0.8)
ax.axvline(FLOOR, color="#cc7000", ls="--", lw=1.5)
ax.text(FLOOR + 0.008, len(names) - 1.1, "isotropic floor\n(28 in-scan air windows)",
        fontsize=8, color="#cc7000", va="top")
ax.set_yticks(y)
ax.set_yticklabels([f"{n}{'  ★' if n == CONTROL else ''}"
                   f"{'  ('+str(SEGMENTED[n])+' segs)' if n in SEGMENTED else ''}" for n in names],
                  fontsize=9)
ax.set_xlabel("sheet separability  (structure-tensor planarity, median over 32³ blocks)", fontsize=9)
ax.set_title("(a) Sheet separability, 14 volumes\n"
             "★ = PHerc0139, the one scroll with letters proven at 9 µm", fontsize=10)
ax.set_xlim(0, 0.85)
ax.grid(axis="x", alpha=0.25)

# ---- (b) separability vs SNR
ax = fig.add_subplot(gs[0, 1])
ax.scatter(snr, sep, c=cols, s=64, edgecolor="white", linewidth=0.8)
for n, x_, y_ in zip(names, snr, sep):
    if n in (CONTROL, "PHerc0358", "PHerc0813", "PHerc1447", "PHerc0125"):
        ax.annotate(n.replace("PHerc", ""), (x_, y_), textcoords="offset points",
                    xytext=(6, -3), fontsize=8.5)
ax.axhline(FLOOR, color="#cc7000", ls="--", lw=1.2)
ax.set_xlabel("K2b structural SNR @ q=0.25  (scan quality)", fontsize=9)
ax.set_ylabel("sheet separability", fontsize=9)
ax.set_title("(b) The two axes are near-orthogonal\n"
             r"Spearman $\rho$ = +0.34, p = 0.24 (n=14)", fontsize=10)
ax.grid(alpha=0.25)

# ---- (c) picker bias
ax = fig.add_subplot(gs[0, 2])
order = np.argsort(-sep)
for i, idx in enumerate(order):
    ax.plot([0, 1], [pick[idx], sep[idx]], color="#bbb", lw=1.0, zorder=1)
ax.scatter(np.zeros(len(sep)), pick, c="#c05000", s=48, zorder=2, label="K2b intensity-max ROIs")
ax.scatter(np.ones(len(sep)), sep, c="#2a6ebb", s=48, zorder=2, label="uniformly random ROIs")
ax.axhline(FLOOR, color="#cc7000", ls="--", lw=1.2)
ax.text(0.5, FLOOR + 0.012, "isotropic floor", fontsize=8, color="#cc7000",
        ha="center", va="bottom")
ax.set_xticks([0, 1])
ax.set_xticklabels(["intensity-max\n(as shipped)", "uniform random\n(same frame)"], fontsize=9)
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("sheet separability", fontsize=9)
ax.set_title("(c) The ROI picker samples the wrong material\n"
             "3.00× median, higher in 14/14 scrolls, p = 3.6e-28", fontsize=10)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

out = os.path.join(T, "report", "figures", "separability_axis.png")
plt.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
