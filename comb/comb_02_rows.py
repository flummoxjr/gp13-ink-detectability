"""Row-structure + z-asymmetry checks for beyond-label letter-class candidates.

1. Fit the text ruling grid (tilt a, period P, phase) on the 12 HUMAN-LABEL
   glyph centroids only (independent of the candidates), row coord r = y - x*tan(a).
2. Compute each candidate's offset from the nearest grid row; V-test for
   concentration at zero phase. Comparison population = beyond-zone components
   that FAIL the letter gate (texture blobs) — if those align equally, alignment
   is a map-wide property, not text evidence.
3. Per-candidate z-reversal check: p90 of the REVERSE-rendered map under the
   component footprint (ink lives on one face -> reverse response collapses;
   verdict: map-level fwd-vs-rev r=0.076 on control).
Outputs: comb_rows.json, updates candidate list used by the gallery."""
import json
import numpy as np
import tifffile
from pathlib import Path

COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")
OUT_W035 = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_w035")

cat = json.loads((COMB / "comb_catalog.json").read_text())
comps42 = cat["components_s42"]

# ---- 1. grid fit on label centroids ----
gj = json.loads(Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage\verdict_labelgeom.json").read_text())
cy = np.array(gj["centroids_y"], float)
cx = np.array(gj["centroids_x"], float)
# NOTE: centroids_y was sorted independently of centroids_x in labelgeom.
# Refit pairing from the label map directly to be safe.
from scipy import ndimage as ndi
from skimage.measure import regionprops
lab2d = np.load(COMB / "_lab2d.npy")
lb, _ = ndi.label(ndi.binary_closing(lab2d, np.ones((5, 5), bool)),
                  structure=np.ones((3, 3), int))
props = [p for p in regionprops(lb) if p.area >= 500]
cy = np.array([p.centroid[0] for p in props])
cx = np.array([p.centroid[1] for p in props])

def fit_grid(cy, cx):
    best = None
    for a in np.arange(-0.10, 0.1001, 0.001):     # tilt, rad
        r = cy - cx * np.tan(a)
        for P in np.arange(440.0, 561.0, 0.5):
            ph = 2 * np.pi * r / P
            C, S = np.cos(ph).mean(), np.sin(ph).mean()
            phase = np.arctan2(S, C) / (2 * np.pi) * P    # px
            d = (r - phase) / P
            d = d - np.round(d)                            # [-0.5, 0.5]
            rms = float(np.sqrt(np.mean((d * P) ** 2)))    # px
            if best is None or rms < best["rms_px"]:
                best = {"tilt_rad": float(a), "tilt_deg": float(np.degrees(a)),
                        "period_px": float(P), "phase_px": float(phase % P),
                        "rms_px": rms}
    return best

grid = fit_grid(cy, cx)
grid["period_mm"] = grid["period_px"] * 9.362e-3
print("grid fit on labels:", json.dumps(grid, indent=1))

def row_offset(comp, g):
    r = comp["cy"] - comp["cx"] * np.tan(g["tilt_rad"])
    d = (r - g["phase_px"]) / g["period_px"]
    d = d - np.round(d)
    return float(d)  # fraction of period, [-0.5, 0.5]

def vtest(offsets_frac):
    """V-test: Rayleigh with specified mean direction 0."""
    ph = 2 * np.pi * np.asarray(offsets_frac)
    n = len(ph)
    if n == 0:
        return {"n": 0}
    Cbar = np.cos(ph).mean()
    V = n * Cbar
    u = V * np.sqrt(2.0 / n)
    from scipy.stats import norm
    p = float(norm.sf(u))
    return {"n": n, "mean_cos": float(Cbar), "u": float(u), "p_one_sided": p}

# sanity: labels against their own grid (overfit view, just for scale)
lab_off = []
for yy, xx in zip(cy, cx):
    r = yy - xx * np.tan(grid["tilt_rad"])
    d = (r - grid["phase_px"]) / grid["period_px"]
    lab_off.append(float(d - np.round(d)))

# ---- populations ----
cands = [c for c in comps42 if c.get("letter_class") and c["zone"] == "beyond"]
texture = [c for c in comps42 if (not c.get("letter_class")) and c["zone"] == "beyond"
           and c["area"] >= 3000]
sup_letters = [c for c in comps42 if c.get("letter_class") and c["zone"] != "beyond"]

for c in comps42:
    c["row_offset"] = row_offset(c, grid)

res = {
    "grid": grid,
    "label_offsets_frac": lab_off,
    "vtest_labels": vtest(lab_off),
    "vtest_beyond_letter_class": vtest([c["row_offset"] for c in cands]),
    "vtest_beyond_letter_class_crossseed": vtest(
        [c["row_offset"] for c in cands
         if max(c["iou43_best"], c["iou43_union"]) > 0.3]),
    "vtest_beyond_texture_comps": vtest([c["row_offset"] for c in texture]),
    "abs_offset_med_letter_class": float(np.median([abs(c["row_offset"]) for c in cands])),
    "abs_offset_med_texture": float(np.median([abs(c["row_offset"]) for c in texture])) if texture else None,
}

# ---- 3. z-reversal check ----
rev = tifffile.imread(str(OUT_W035 / "w035_seed42-075000_reverse.tif"))
fwd = tifffile.imread(str(OUT_W035 / "w035_seed42-075000.tif"))
lb42 = np.load(COMB / "_comp42.npz")["lb"]
thr42 = cat["seed42"]["threshold_p80"]

def rev_stats(c):
    y0, x0, y1, x1 = c["bbox"]
    m = lb42[y0:y1, x0:x1] == c["id"]
    rv = rev[y0:y1, x0:x1][m]
    fv = fwd[y0:y1, x0:x1][m]
    return float(np.percentile(rv, 90)), float(rv.mean()), float(fv.mean())

for c in cands + sup_letters:
    rp90, rmean, fmean = rev_stats(c)
    c["rev_p90"] = rp90
    c["rev_over_fwd_mean"] = rmean / max(fmean, 1e-6)

res["rev_supervised_letters"] = {
    "rev_p90_med": float(np.median([c["rev_p90"] for c in sup_letters])),
    "rev_over_fwd_med": float(np.median([c["rev_over_fwd_mean"] for c in sup_letters])),
}
res["rev_beyond_candidates"] = {
    "rev_p90_med": float(np.median([c["rev_p90"] for c in cands])),
    "rev_over_fwd_med": float(np.median([c["rev_over_fwd_mean"] for c in cands])),
}
# texture comparison for reversal too
tex_rp = []
for c in texture[:200]:
    rp90, rmean, fmean = rev_stats(c)
    tex_rp.append(rmean / max(fmean, 1e-6))
res["rev_beyond_texture_over_fwd_med"] = float(np.median(tex_rp)) if tex_rp else None

res["candidates"] = sorted(cands, key=lambda c: -c["area"])
(COMB / "comb_rows.json").write_text(json.dumps(res, indent=1))

print(json.dumps({k: v for k, v in res.items() if k != "candidates"}, indent=1))
print("\nper-candidate:")
for c in res["candidates"]:
    print(f" id={c['id']:6d} cy={c['cy']:6.0f} cx={c['cx']:6.0f} "
          f"off={c['row_offset']:+.3f} iou43={max(c['iou43_best'], c['iou43_union']):.2f} "
          f"rev/fwd={c['rev_over_fwd_mean']:.2f}")
