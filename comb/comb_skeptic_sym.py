r"""SKEPTIC PASS — flags 3 (tile z-symmetry non-specificity), 4 (tripwire
near-miss), 5 (w035 patch 8).

F3: quantitative test of "letter tiles vs blank tiles indistinguishable in r":
    MW + KS per seed and pooled; note the SIGN (letters slightly HIGHER r =
    wrong direction for an ink detector). Also Spearman(tile r, tile stdf) on
    1203 to verify the noise-limited-quiet-tile driver.
F4: 1203A_s43r at p80 — find the near-miss component, count px>=195/196,
    compute expected count from the map's own suprathreshold value
    distribution (binomial/Poisson).
F5: overlap of w035 patch 8 (tiles (20,1),(20,3),(21,2),(21,3)) with flag-1
    candidate footprints -> is it the same object as candidate 4535?
"""
import json
import sys
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu, ks_2samp, spearmanr

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_map, valid_mask

COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")
out = {}

# ---------- F3 ----------
f3 = {}
pooled_letter, pooled_blank = [], []
for key in ("w035_s42", "w035_s43"):
    t = dict(np.load(COMB / f"sym_tiles_{key}.npz"))
    ok = np.isfinite(t["r"])
    letter = ok & (t["labfrac"] >= 0.02)
    blank = ok & (t["labfrac"] == 0.0)
    lv, bv = t["r"][letter], t["r"][blank]
    pooled_letter += lv.tolist(); pooled_blank += bv.tolist()
    f3[key] = {
        "n_letter": int(letter.sum()), "n_blank": int(blank.sum()),
        "letter_r_med": float(np.median(lv)), "blank_r_med": float(np.median(bv)),
        "mw_letter_lt_blank_p": float(mannwhitneyu(lv, bv, alternative="less").pvalue),
        "mw_two_sided_p": float(mannwhitneyu(lv, bv).pvalue),
        "ks_p": float(ks_2samp(lv, bv).pvalue),
    }
lv, bv = np.array(pooled_letter), np.array(pooled_blank)
f3["pooled"] = {
    "n": [len(lv), len(bv)],
    "letter_r_med": float(np.median(lv)), "blank_r_med": float(np.median(bv)),
    "mw_letter_lt_blank_p": float(mannwhitneyu(lv, bv, alternative="less").pvalue),
    "mw_two_sided_p": float(mannwhitneyu(lv, bv).pvalue),
    "ks_p": float(ks_2samp(lv, bv).pvalue),
    "note": "ink-detector direction would need letter r << blank r",
}
for key in ("1203A_s42", "1203A_s43", "1203B_s42", "1203B_s43"):
    t = dict(np.load(COMB / f"sym_tiles_{key}.npz"))
    ok = np.isfinite(t["r"]) & np.isfinite(t["stdf"])
    rho, p = spearmanr(t["r"][ok], t["stdf"][ok])
    f3[f"{key}_spearman_r_vs_std"] = {"rho": float(rho), "p": float(p)}
out["F3_symmetry"] = f3
print("F3:", json.dumps(f3, indent=1))

# ---------- F4 ----------
arr = load_map("1203A_s43r")
m = valid_mask(arr, erode=40)
thr = float(np.percentile(arr[m], 80))
binary = (arr >= thr) & m
lab, n = ndi.label(binary, structure=np.ones((3, 3), int))
areas = np.bincount(lab.ravel())[1:]
big = int(np.argmax(areas)) + 1
big_area = int(areas[big - 1])
vals_all = arr[binary].astype(np.float64)
q195 = float((vals_all >= 195).mean())
q196 = float((vals_all >= 196).mean())
comp_vals = arr[lab == big].astype(np.float64)
n195 = int((comp_vals >= 195).sum()); n196 = int((comp_vals >= 196).sum())
exp195 = big_area * q195
# spatial independence caveat: count 8-connected clusters of >=195 px map-wide
hot = (arr >= 195) & m
_, nhot_clusters = ndi.label(hot, structure=np.ones((3, 3), int))
out["F4_tripwire_nearmiss"] = {
    "thr_p80": thr, "n_comp": int(n), "biggest_area": big_area,
    "comp_val_p50": float(np.percentile(comp_vals, 50)),
    "comp_val_p90": float(np.percentile(comp_vals, 90)),
    "comp_val_max": float(comp_vals.max()),
    "comp_n_px_ge195": n195, "comp_n_px_ge196": n196,
    "suprathr_frac_ge195": q195, "suprathr_frac_ge196": q196,
    "expected_ge195_in_comp": exp195,
    "expected_ge196_in_comp": big_area * q196,
    "poisson_p_ge1_at195": float(1 - np.exp(-exp195)),
    "map_hot195_clusters": int(nhot_clusters),
}
print("F4:", json.dumps(out["F4_tripwire_nearmiss"], indent=1))

# ---------- F5 ----------
lb42 = np.load(COMB / "_comp42.npz")["lb"]
cat = json.loads((COMB / "comb_catalog.json").read_text())
cands = [c for c in cat["components_s42"]
         if c.get("letter_class") and c["zone"] == "beyond"]
TILE = 256
sel = np.zeros(lb42.shape, bool)
for (iy, ix) in [(20, 1), (20, 3), (21, 2), (21, 3)]:
    sel[iy * TILE:(iy + 1) * TILE, ix * TILE:(ix + 1) * TILE] = True
hits = []
for c in cands:
    y0, x0, y1, x1 = c["bbox"]
    cm = lb42[y0:y1, x0:x1] == c["id"]
    ov = float(sel[y0:y1, x0:x1][cm].mean())
    if ov > 0:
        hits.append({"id": c["id"], "area": c["area"],
                     "frac_inside_patch8": ov,
                     "iou43": max(c["iou43_best"], c["iou43_union"])})
out["F5_patch8"] = {
    "candidate_overlaps": hits,
    "note": "patch8 s42 area_max=17549 == candidate 4535 area -> same object",
}
print("F5:", json.dumps(out["F5_patch8"], indent=1))

(COMB / "comb_skeptic_sym.json").write_text(json.dumps(out, indent=1))
print("saved comb_skeptic_sym.json")
