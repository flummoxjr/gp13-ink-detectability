#!/usr/bin/env python
"""Figures for the geometry comparison (w035 control vs GP auto-grown meshes)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\out"
OFF = np.arange(-16.0, 16.01, 0.5)


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


dp = {x["key"]: x for x in load("depth_profiles.json")}
if any("error" in v for v in dp.values()):          # fall back for failed keys
    for x in load("depth_profiles_v1.json"):
        if "error" in dp.get(x["key"], {}) or x["key"] not in dp:
            dp[x["key"]] = {**x, **{k: v for k, v in dp.get(x["key"], {}).items() if k != "error"}}
gs = {x["key"]: x for x in load("geom_stats.json")}
vm = {x["key"]: x for x in (load("vertex_material.json") or [])}
pv = {x["key"]: x for x in load("provenance.json")}
isx = load("interface_sharpness.json")
pr = load("placement_resolved.json")
st = load("strip_tracking.json")
cs = np.load(os.path.join(OUT, "crosssections.npz"))

order = ["w035", "w032", "1203_r399", "1203_r460", "1203_r747",
         "1447_r222", "1447_r623", "1447_r914", "0800_r329", "0800_r522"]
order = [k for k in order if k in dp and "mean_profile" in dp[k]]
CTRL = {"w035", "w032"}
pal = plt.get_cmap("tab10")
col = {k: ("k" if k == "w035" else "0.45" if k in CTRL else pal(i % 10))
       for i, k in enumerate(order)}

fig = plt.figure(figsize=(22, 18))
G = fig.add_gridspec(4, 5, height_ratios=[1.25, 1.05, 0.95, 0.95], hspace=0.85, wspace=0.34)

# ---------- A: mean depth profiles --------------------------------------
ax = fig.add_subplot(G[0, :3])
for k in order:
    p = np.asarray(dp[k]["mean_profile"], float)
    base = np.percentile(p, 10)
    ax.plot(OFF, (p - base) / max(base, 1e-6), color=col[k],
            lw=3.0 if k in CTRL else 1.6, ls="-" if k in CTRL else "--",
            label=f"{k}{'  (CONTROL, letters found)' if k=='w035' else '  (control scroll)' if k in CTRL else ''}"
                  f"   modulation={dp[k].get('tile_modulation_med'):.2f}")
ax.axvline(0, color="r", lw=1.0, alpha=0.6)
ax.axvspan(-10, 10, color="0.86", alpha=0.45, zorder=0)
ax.set_xlabel("offset along the mesh normal (voxels)     [grey band = the 21-slice render window]")
ax.set_ylabel("(I - baseline) / baseline")
ax.set_title("A.  Mean CT depth profile through each mesh surface\n"
             "a correctly-placed recto surface = ONE bright lamella peaking at offset 0", fontsize=11)
ax.legend(fontsize=7.5, ncol=2, loc="upper right")

# ---------- A2: modulation vs fwd/rev r ---------------------------------
ax = fig.add_subplot(G[0, 3])
for k in order:
    r = dp[k].get("fwd_rev_r")
    if r is None:
        continue
    ax.scatter(r, dp[k]["tile_modulation_med"], color=col[k], s=110, edgecolor="k", zorder=3)
    ax.annotate(k, (r, dp[k]["tile_modulation_med"]), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("ink_9um forward-vs-reverse map r", fontsize=9)
ax.set_ylabel("sheet modulation depth", fontsize=9)
ax.set_title("A2.  weak lamella contrast — not bad\ngeometry — tracks fwd/rev symmetry", fontsize=10)
ax.tick_params(labelsize=8)

# ---------- A3: sheet-centre offset, measured only where resolvable -----
ax = fig.add_subplot(G[0, 4])
xs, vals, errs, cols, labs = [], [], [], [], []
for i, k in enumerate(order):
    e = pr.get(k) or {}
    if e.get("n_resolved", 0) < 3:
        continue
    xs.append(len(xs)); vals.append(e["cen_med"]); errs.append(e["cen_iqr"] / 2)
    cols.append(col[k]); labs.append(f"{k} ({e['n_resolved']}/{e['n_tiles']})")
ax.errorbar(xs, vals, yerr=errs, fmt="none", ecolor="0.5", capsize=3)
ax.scatter(xs, vals, color=cols, s=110, edgecolor="k", zorder=4)
ax.axhline(0, color="r", lw=1.0)
ax.axhspan(-1.5, 1.5, color="0.88", zorder=0)
ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=70, ha="right", fontsize=7)
ax.set_ylabel("lamella centre offset (vox)", fontsize=9)
ax.set_title("A3.  is the lamella centred on the mesh?\n(tiles with a resolvable sheet only)", fontsize=10)
ax.tick_params(labelsize=8)

# ---------- B: depth x arclength cross-sections -------------------------
reps = [k for k in ["w035", "1203_r460", "1447_r222", "1447_r914", "0800_r329"] if k in cs.files]
for i, k in enumerate(reps[:5]):
    ax = fig.add_subplot(G[1, i])
    im = cs[k]
    finite = np.isfinite(im) & (im > 0)
    dead = finite.sum() <= 100
    vmin, vmax = (0, 1) if dead else np.percentile(im[finite], [3, 97])
    ax.imshow(im, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax,
              extent=[0, im.shape[1], OFF[-1], OFF[0]])
    ax.axhline(0, color="r", lw=1.1)
    ax.axhline(10, color="c", lw=0.7); ax.axhline(-10, color="c", lw=0.7)
    ax.set_title(f"B{i+1}. {k}" + ("   ALL ZERO — mesh is in empty volume" if dead else ""),
                 fontsize=9.5, color="crimson" if dead else "k")
    ax.set_ylabel("offset (vox)", fontsize=8)
    ax.set_xlabel("along the surface (vox)", fontsize=8)
    ax.tick_params(labelsize=7)
fig.text(0.5, 0.487,
         "B.  Cross-sections through the surface: depth along the normal (vertical) x distance along the mesh (horizontal).  "
         "Red = the mesh, cyan = the edges of the 21-slice render window.\n"
         "Bright = dense material.  All five panels are the same physical scale.",
         ha="center", fontsize=10.5)

# ---------- C/D: bar charts ---------------------------------------------
def bars(pos, vals, title, ylab, logy=False, hline=None):
    axi = fig.add_subplot(pos)
    x = np.arange(len(order))
    axi.bar(x, vals, color=[col[k] for k in order], edgecolor="k", lw=0.5)
    axi.set_xticks(x); axi.set_xticklabels(order, rotation=70, ha="right", fontsize=7)
    axi.set_title(title, fontsize=9.5); axi.set_ylabel(ylab, fontsize=8)
    axi.tick_params(labelsize=7.5)
    if logy:
        axi.set_yscale("log")
    if hline is not None:
        axi.axhline(hline, color="r", ls=":", lw=1.2)
    return axi

bars(G[2, 0], [gs[k]["normal_dispersion_med_deg"] for k in order],
     "C1. normal coherence", "median angle between\nneighbouring normals (deg)",
     hline=gs["w035"]["normal_dispersion_med_deg"])
bars(G[2, 1], [gs[k]["curv_med_deg_per_mm"] for k in order],
     "C2. median curvature", "deg / mm", hline=gs["w035"]["curv_med_deg_per_mm"])
bars(G[2, 2], [gs[k]["edge_all_cv"] for k in order],
     "C3. grid regularity", "CV of grid edge length")
bars(G[2, 3], [gs[k]["valid_frac"] for k in order],
     "C4. grid coverage", "valid fraction of grid")
bars(G[2, 4], [gs[k]["area_mm2"] for k in order],
     "C5. surface area", "mm^2", logy=True)

bars(G[3, 0], [dp[k]["tile_modulation_med"] for k in order],
     "D1. lamella modulation depth", "(max-min)/mean over +/-10 vox",
     hline=dp["w035"]["tile_modulation_med"])
bars(G[3, 1], [(isx[k] or {}).get("grad_norm", 0) for k in order],
     "D2. lamella interface sharpness", "|grad I| / mean  (vox^-1)",
     hline=isx["w035"]["grad_norm"])
bars(G[3, 2], [(pr.get(k) or {}).get("n_resolved", 0) / max((pr.get(k) or {}).get("n_tiles", 1), 1)
               for k in order],
     "D3. tiles where a lamella is\nresolvable at all", "fraction of sampled tiles")
bars(G[3, 3], [vm[k]["frac_no_material"] if k in vm else dp[k]["frac_air_profiles"] for k in order],
     "D4. mesh sitting where the volume\nis EMPTY (whole mesh)", "fraction of mesh vertices")
bars(G[3, 4], [(st.get(k) or {}).get("iqr", np.nan) for k in order],
     "D5. local lamella-tracking error\n(one 3.5 mm strip)", "IQR of lamella offset (vox)",
     hline=(st.get("w035") or {}).get("iqr"))

fig.suptitle("Is bad GEOMETRY why we found no ink?   PHerc0139 w035 (Greek letters recovered, forward AUC 0.9991 / reverse 0.512) "
             "vs 8 GP auto-grown segments", fontsize=14, y=0.98)
p = os.path.join(OUT, "geometry_compare.png")
fig.savefig(p, dpi=110, bbox_inches="tight")
print("wrote", p)
