"""Gallery: overview map + top-12 beyond-label candidates (prediction crop vs
CT surface-volume central slice), plus the final JSON catalog.
Outputs: comb/w035_beyond_labels.png, comb/w035_beyond_labels.json"""
import json
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path
from scipy import ndimage as ndi

COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")
OUT_W035 = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_w035")
DDATA = Path(r"D:\vesuvius-data\trackD")
UM_PER_PX = 9.362

rows = json.loads((COMB / "comb_rows.json").read_text())
grid = rows["grid"]
cands = rows["candidates"]
survivors = [c for c in cands if max(c["iou43_best"], c["iou43_union"]) > 0.3]
survivors.sort(key=lambda c: -(c["frac195"] * c["area"]))
top = survivors[:12]

pred = tifffile.imread(str(OUT_W035 / "w035_seed42-075000.tif"))
lb42 = np.load(COMB / "_comp42.npz")["lb"]
lab2d = np.load(COMB / "_lab2d.npy")
sup2d = np.load(COMB / "_sup2d.npy")
surf = np.load(DDATA / "w035_surf.npy", mmap_mode="r")
surf14 = np.asarray(surf[14])

# ---------- figure ----------
fig = plt.figure(figsize=(20, 26), facecolor="white")
gs = fig.add_gridspec(5, 6, height_ratios=[1.55, 1, 1, 1, 1],
                      hspace=0.28, wspace=0.12,
                      left=0.03, right=0.99, top=0.955, bottom=0.02)

# overview
axo = fig.add_subplot(gs[0, :])
ds = 4
ov = pred[:pred.shape[0]//ds*ds, :pred.shape[1]//ds*ds].reshape(
    pred.shape[0]//ds, ds, pred.shape[1]//ds, ds).mean(axis=(1, 3))
axo.imshow(ov, cmap="gray", vmin=0, vmax=200, interpolation="nearest")
# grid rows
ta = np.tan(grid["tilt_rad"])
xs = np.array([0, pred.shape[1]])
kmin = int(np.floor((0 - grid["phase_px"]) / grid["period_px"])) - 1
kmax = int(np.ceil((pred.shape[0] + abs(ta) * pred.shape[1] - grid["phase_px"])
                   / grid["period_px"])) + 1
for k in range(kmin, kmax):
    r0 = grid["phase_px"] + k * grid["period_px"]
    ys = (r0 + xs * ta) / ds
    axo.plot(xs / ds, ys, color="#00bcd4", lw=0.6, ls="--", alpha=0.65)
# label + sup outlines
sup_s = sup2d[::ds, ::ds]
lab_s = lab2d[::ds, ::ds]
axo.contour(ndi.binary_dilation(sup_s, iterations=2), levels=[0.5],
            colors="#ffd54f", linewidths=0.7)
axo.contour(ndi.binary_dilation(lab_s, iterations=2), levels=[0.5],
            colors="#4caf50", linewidths=0.9)
# candidate boxes
for i, c in enumerate(top, 1):
    y0, x0, y1, x1 = c["bbox"]
    rect = patches.Rectangle((x0/ds - 8, y0/ds - 8), (x1-x0)/ds + 16,
                             (y1-y0)/ds + 16, fill=False, ec="#ff5252", lw=1.4)
    axo.add_patch(rect)
    axo.text(x1/ds + 6, y0/ds + 10, str(i), color="#ff5252",
             fontsize=13, fontweight="bold")
# non-top survivors + non-survivors, fainter
for c in cands:
    if c in top:
        continue
    ok = max(c["iou43_best"], c["iou43_union"]) > 0.3
    y0, x0, y1, x1 = c["bbox"]
    rect = patches.Rectangle((x0/ds - 8, y0/ds - 8), (x1-x0)/ds + 16,
                             (y1-y0)/ds + 16, fill=False,
                             ec="#ff9800" if ok else "#9e9e9e",
                             lw=1.0, ls=":" if not ok else "-")
    axo.add_patch(rect)
axo.set_title(
    "w035 (PHerc0139) ink_9um seed42 prediction — letter-class components beyond human labels/supervision\n"
    "green = human ink labels · yellow = supervision mask · dashed cyan = ruling grid fit on labels only "
    f"({grid['period_mm']:.2f} mm, tilt {grid['tilt_deg']:.2f}°) · "
    "red = top-12 cross-seed-confirmed candidates · orange = other confirmed · grey dotted = failed seed43 IoU>0.3",
    fontsize=11)
axo.set_xticks([]); axo.set_yticks([])

# candidate crops: 12 pairs in rows 1-4, 3 pairs (6 axes) per row
MARGIN = 130
MIN_HALF = 260
for i, c in enumerate(top):
    gr = 1 + i // 3
    gcol = (i % 3) * 2
    y0, x0, y1, x1 = c["bbox"]
    cyc, cxc = (y0+y1)//2, (x0+x1)//2
    hy = max((y1-y0)//2 + MARGIN, MIN_HALF)
    hx = max((x1-x0)//2 + MARGIN, MIN_HALF)
    h = max(hy, hx)
    yA, yB = max(cyc-h, 0), min(cyc+h, pred.shape[0])
    xA, xB = max(cxc-h, 0), min(cxc+h, pred.shape[1])
    pc = pred[yA:yB, xA:xB]
    sc = surf14[yA:yB, xA:xB]
    comp = lb42[yA:yB, xA:xB] == c["id"]

    ax1 = fig.add_subplot(gs[gr, gcol])
    ax1.imshow(pc, cmap="gray", vmin=0, vmax=210, interpolation="nearest")
    ax1.contour(comp, levels=[0.5], colors="#ff5252", linewidths=0.7)
    for k in range(kmin, kmax):
        r0 = grid["phase_px"] + k * grid["period_px"]
        yy = (r0 + np.array([xA, xB]) * ta) - yA
        if -50 < yy[0] < (yB-yA) + 50 or -50 < yy[1] < (yB-yA) + 50:
            ax1.plot([0, xB-xA], yy, color="#00bcd4", lw=0.7, ls="--", alpha=0.7)
    ax1.set_xlim(0, xB-xA); ax1.set_ylim(yB-yA, 0)
    mm = 2 * h * UM_PER_PX / 1000
    ax1.set_title(f"#{i+1}  pred  id{c['id']}  {mm:.1f}mm  "
                  f"IoU43={max(c['iou43_best'], c['iou43_union']):.2f}",
                  fontsize=9)
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[gr, gcol+1])
    lo, hi = np.percentile(sc, (2, 98))
    ax2.imshow(sc, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax2.contour(comp, levels=[0.5], colors="#ff5252", linewidths=0.5, alpha=0.7)
    ax2.set_title(f"CT surf z14  off={c['row_offset']:+.2f}  "
                  f"rev/fwd={c['rev_over_fwd_mean']:.2f}", fontsize=9)
    ax2.set_xticks([]); ax2.set_yticks([])

fig.suptitle(
    "w035 beyond the labels — top 12 letter-class, cross-seed-confirmed model detections "
    "(MODEL output on a training-adjacent scroll; not a prize claim)",
    fontsize=14, y=0.985)
fig.savefig(COMB / "w035_beyond_labels.png", dpi=110)
plt.close(fig)
print("figure saved")

# ---------- final JSON catalog ----------
catalog = {
    "context": {
        "map": "trackD/out/ink9um_w035/w035_seed42-075000.tif (ink_9um ckpt step 75000, seed42)",
        "cross_seed": "w035_seed43-075000.tif, IoU>0.3 on matched components",
        "scroll": "PHerc0139 segment w035 (9.362 um/px) — ink_9um model is trained-adjacent to this corpus",
        "framing": ("MODEL detections beyond the human-labeled letters; value = report figure + "
                    "community interest (PHerc0139 not GP-eligible). NOT a prize claim. "
                    "Mundane alternative for any single component: a model trained on letter shapes "
                    "can hallucinate letter-like blobs on salient papyrus texture, and both seeds "
                    "share training data, so seed agreement does not rule out shared inductive bias."),
        "letter_class_gate": ("area>=1e4 px AND width(area/skel)>=30 px AND component v_p90>=195 "
                              "(control blank p99), at per-map in-mask p80 threshold; "
                              "gate recalls 11/13 labeled letter components"),
    },
    "grid_fit_on_labels_only": grid,
    "summary": {
        "n_letter_class_beyond_seed42": len(cands),
        "n_crossseed_confirmed": len(survivors),
        "n_in_top_gallery": len(top),
        "vtest_alignment_letter_class": rows["vtest_beyond_letter_class"],
        "vtest_alignment_crossseed": rows["vtest_beyond_letter_class_crossseed"],
        "vtest_alignment_texture_comps_CONTROL": rows["vtest_beyond_texture_comps"],
        "median_abs_row_offset_letter_class": rows["abs_offset_med_letter_class"],
        "median_abs_row_offset_texture": rows["abs_offset_med_texture"],
        "rev_over_fwd": {
            "labeled_letters": rows["rev_supervised_letters"]["rev_over_fwd_med"],
            "beyond_candidates": rows["rev_beyond_candidates"]["rev_over_fwd_med"],
            "texture_comps": rows["rev_beyond_texture_over_fwd_med"],
        },
    },
    "candidates_ranked": [
        {k: c[k] for k in ("id", "area", "bbox", "cy", "cx", "width", "elong",
                           "v_p50", "v_p90", "v_max", "frac195", "row_offset",
                           "iou43_best", "iou43_union", "rev_p90",
                           "rev_over_fwd_mean")}
        for c in survivors],
    "not_confirmed_seed43": [
        {k: c[k] for k in ("id", "area", "cy", "cx", "iou43_best", "iou43_union")}
        for c in cands if max(c["iou43_best"], c["iou43_union"]) <= 0.3],
}
(COMB / "w035_beyond_labels.json").write_text(json.dumps(catalog, indent=1))
print("catalog saved;", len(survivors), "confirmed candidates,", len(top), "in gallery")
