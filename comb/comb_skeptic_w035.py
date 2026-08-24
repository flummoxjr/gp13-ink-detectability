r"""SKEPTIC PASS — kill tests for the w035 beyond-labels flag (hunter: w035-beyond).

Null tests run here (none of these were run by the hunter):
 N1. REVERSE-RENDER GATE NULL: run the identical letter-class gate
     (in-mask p80 threshold, area>=1e4, width>=30, comp v_p90>=195) on the
     z-REVERSED renders of both seeds. The reverse render provably destroys
     letters (map-level fwd-vs-rev r=0.076; labeled letters collapse), so any
     letter-class components it produces are non-ink texture productions of
     the same model on the same physical segment = the gate's intrinsic
     false-positive yield. Also cross-"seed" IoU between rev42/rev43
     letter-class comps (shared-inductive-bias reproduction rate on
     known-non-ink input).
 N2. TEXTURE CROSS-SEED NULL: fraction of beyond-zone NON-letter-class
     (texture) s42 components that reproduce in s43 at IoU>0.3 with the same
     matching logic as the hunter used for candidates. If texture reproduces
     at ~14/18-rate, cross-seed confirmation is unremarkable.
 N3. ALIGNMENT BOOTSTRAP NULL: draw size-18 and size-14 subsets of texture
     comps, mean_cos of grid offsets -> percentile of the candidates'
     mean_cos (0.383 / 0.290). Same for count(|off|<=0.15) in size-14 subsets
     vs the observed 6.
 N4. REV/FWD SEPARATION TEST: rev_over_fwd_mean for ALL texture comps vs the
     18 candidates vs labeled letters; Mann-Whitney one-sided.
 N5. ROW-COHERENCE SCAN NULL (flag 2): probability that >=4 of 18 candidate
     row-coords fall in one 60-px window (and span >=2500 px in cx), under
     (a) uniform row positions, (b) row positions resampled from texture comps.

Output: comb_skeptic_w035.json + prints.
"""
import json
import numpy as np
import tifffile
from pathlib import Path
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu, norm
from skimage.morphology import skeletonize
from skimage.measure import regionprops

COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")
OUT_W035 = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_w035")

LETTER_AREA, LETTER_WIDTH, LETTER_VAL = 10_000, 30.0, 195
MIN_AREA_TABLE = 3000
EXCL_DILATE = 25
GRID = {"tilt_rad": -0.01, "period_px": 500.5, "phase_px": 489.8740026163272}
RNG = np.random.default_rng(7)

cat = json.loads((COMB / "comb_catalog.json").read_text())
comps42 = cat["components_s42"]
cands = [c for c in comps42 if c.get("letter_class") and c["zone"] == "beyond"]
texture = [c for c in comps42 if (not c.get("letter_class"))
           and c["zone"] == "beyond" and c["area"] >= MIN_AREA_TABLE]
labeled = [c for c in comps42 if c["zone"] == "labeled"]
print(f"populations: cands={len(cands)} texture={len(texture)} labeled={len(labeled)}")

lab2d = np.load(COMB / "_lab2d.npy")
sup2d = np.load(COMB / "_sup2d.npy")
excl = ndi.binary_dilation(lab2d | sup2d, structure=np.ones((3, 3), bool),
                           iterations=EXCL_DILATE)
lab_d = ndi.binary_dilation(lab2d, structure=np.ones((3, 3), bool),
                            iterations=EXCL_DILATE)


def valid_mask(arr, erode=40):
    m = arr > 0
    m = ndi.binary_closing(m, structure=np.ones((5, 5), bool))
    m = ndi.binary_fill_holes(m)
    if erode:
        m = ndi.binary_erosion(m, structure=np.ones((3, 3), bool), iterations=erode)
    return m


def gate_scan(pred, vm):
    """Identical letter-class gate as comb_01_catalog."""
    thr = float(np.percentile(pred[vm], 80))
    binm = (pred >= thr) & vm
    lb, n = ndi.label(binm, structure=np.ones((3, 3), int))
    skel = skeletonize(binm)
    skc = np.bincount(lb[skel], minlength=n + 1)
    rows = []
    for p in regionprops(lb, intensity_image=pred):
        if p.area < LETTER_AREA:
            continue
        width = p.area / max(int(skc[p.label]), 1)
        if width < LETTER_WIDTH:
            continue
        vals = p.image_intensity[p.image]
        vp90 = float(np.percentile(vals, 90))
        if vp90 < LETTER_VAL:
            continue
        y0, x0, y1, x1 = p.bbox
        sub = lb[y0:y1, x0:x1] == p.label
        f_lab = float(lab_d[y0:y1, x0:x1][sub].mean())
        f_exc = float(excl[y0:y1, x0:x1][sub].mean())
        zone = ("labeled" if f_lab > 0.05 else
                "sup_region" if f_exc > 0.05 else "beyond")
        rows.append({"id": int(p.label), "area": int(p.area),
                     "bbox": [int(v) for v in p.bbox],
                     "cy": float(p.centroid[0]), "cx": float(p.centroid[1]),
                     "width": float(width), "v_p90": vp90,
                     "v_p50": float(np.percentile(vals, 50)),
                     "frac195": float((vals >= LETTER_VAL).mean()),
                     "zone": zone})
    return thr, lb, rows


out = {}

# ---------- N1: reverse-render gate null ----------
rev_res = {}
rev_lbs = {}
for seed in (42, 43):
    pred = tifffile.imread(str(OUT_W035 / f"w035_seed{seed}-075000_reverse.tif"))
    vm = valid_mask(pred)
    thr, lb, rows = gate_scan(pred, vm)
    rev_lbs[seed] = lb
    v = pred[vm]
    rev_res[f"rev{seed}"] = {
        "thr_p80": thr,
        "inmask_p50": float(np.percentile(v, 50)),
        "inmask_p90": float(np.percentile(v, 90)),
        "inmask_p99": float(np.percentile(v, 99)),
        "inmask_frac195": float((v >= 195).mean()),
        "n_letter_class_total": len(rows),
        "n_letter_class_beyond": sum(r["zone"] == "beyond" for r in rows),
        "letter_class_comps": rows,
    }
    print(f"rev{seed}: thr={thr:.0f} letter-class total={len(rows)} "
          f"beyond={rev_res[f'rev{seed}']['n_letter_class_beyond']} "
          f"frac195={rev_res[f'rev{seed}']['inmask_frac195']:.4f}")

# cross-reproduction of rev42 letter-class comps in rev43 (IoU>0.3, local union)
def local_union_iou(lbA, lbB, comp):
    y0, x0, y1, x1 = comp["bbox"]
    a = lbA[y0:y1, x0:x1] == comp["id"]
    bids = np.unique(lbB[y0:y1, x0:x1][a]); bids = bids[bids > 0]
    if not len(bids):
        return 0.0
    b = np.isin(lbB[y0:y1, x0:x1], bids)
    return float((a & b).sum() / (a | b).sum())

rev42_beyond = [r for r in rev_res["rev42"]["letter_class_comps"]
                if r["zone"] == "beyond"]
rev_cross = [local_union_iou(rev_lbs[42], rev_lbs[43], r) for r in rev42_beyond]
rev_res["rev42_beyond_cross_iou"] = rev_cross
rev_res["rev42_beyond_cross_confirmed"] = int(sum(i > 0.3 for i in rev_cross))
out["N1_reverse_gate"] = rev_res

# ---------- N2: texture cross-seed reproduction ----------
lb42 = np.load(COMB / "_comp42.npz")["lb"]
lb43 = np.load(COMB / "_comp43.npz")["lb"]
by_id43 = {r["id"]: r for r in cat.get("components_s43_letter_class", [])}
# hunter's union-IoU logic simplified: use joint bbox of overlapping comps when
# their bboxes are known, else local footprint (same fallback as hunter's code
# used for sub-table comps). For texture comps virtually all matches are
# sub-table in s43 -> local-footprint union IoU (slightly IoU-inflating, i.e.
# biased TOWARD killing the flag; noted in verdict).
tex_iou = [local_union_iou(lb42, lb43, c) for c in texture]
cand_iou = [max(c["iou43_best"], c["iou43_union"]) for c in cands]
out["N2_texture_crossseed"] = {
    "note": "local-union IoU (slight overestimate, anti-flag conservative)",
    "texture_n": len(texture),
    "texture_frac_iou_gt0p3": float(np.mean([i > 0.3 for i in tex_iou])),
    "texture_iou_p50": float(np.median(tex_iou)),
    "cand_frac_iou_gt0p3": float(np.mean([i > 0.3 for i in cand_iou])),
    "cand_iou_p50": float(np.median(cand_iou)),
    "mannwhitney_cand_gt_texture_p": float(
        mannwhitneyu(cand_iou, tex_iou, alternative="greater").pvalue),
}
print("N2:", json.dumps(out["N2_texture_crossseed"], indent=1))

# ---------- N3: alignment bootstrap null ----------
def row_offset(c):
    r = c["cy"] - c["cx"] * np.tan(GRID["tilt_rad"])
    d = (r - GRID["phase_px"]) / GRID["period_px"]
    return float(d - np.round(d))

tex_off = np.array([row_offset(c) for c in texture])
cand_off = np.array([row_offset(c) for c in cands])
cross_ids = {c["id"] for c in cands if max(c["iou43_best"], c["iou43_union"]) > 0.3}
cross_off = np.array([row_offset(c) for c in cands if c["id"] in cross_ids])

def mean_cos(off):
    return float(np.cos(2 * np.pi * np.asarray(off)).mean())

obs18, obs14 = mean_cos(cand_off), mean_cos(cross_off)
NB = 200_000
boot18 = np.empty(NB); boot14 = np.empty(NB); boot_cnt14 = np.empty(NB, int)
for i in range(NB):
    s = RNG.choice(len(tex_off), size=18, replace=False)
    boot18[i] = np.cos(2 * np.pi * tex_off[s]).mean()
    s2 = RNG.choice(len(tex_off), size=14, replace=False)
    boot14[i] = np.cos(2 * np.pi * tex_off[s2]).mean()
    boot_cnt14[i] = int((np.abs(tex_off[s2]) <= 0.15).sum())
obs_cnt14 = int((np.abs(cross_off) <= 0.15).sum())
out["N3_alignment_bootstrap"] = {
    "texture_mean_cos": mean_cos(tex_off),
    "texture_frac_within0p15": float((np.abs(tex_off) <= 0.15).mean()),
    "obs_mean_cos_18": obs18, "p_boot_18": float((boot18 >= obs18).mean()),
    "obs_mean_cos_14cross": obs14, "p_boot_14": float((boot14 >= obs14).mean()),
    "obs_count_within0p15_of14": obs_cnt14,
    "p_boot_count14": float((boot_cnt14 >= obs_cnt14).mean()),
    "expected_count14_from_texture": float(boot_cnt14.mean()),
}
print("N3:", json.dumps(out["N3_alignment_bootstrap"], indent=1))

# ---------- N4: rev/fwd separation ----------
fwd42 = tifffile.imread(str(OUT_W035 / "w035_seed42-075000.tif"))
rev42m = tifffile.imread(str(OUT_W035 / "w035_seed42-075000_reverse.tif"))

def rev_over_fwd(c):
    y0, x0, y1, x1 = c["bbox"]
    m = lb42[y0:y1, x0:x1] == c["id"]
    rv = rev42m[y0:y1, x0:x1][m].astype(np.float64)
    fv = fwd42[y0:y1, x0:x1][m].astype(np.float64)
    return float(rv.mean() / max(fv.mean(), 1e-6))

tex_rf = np.array([rev_over_fwd(c) for c in texture])
cand_rf = np.array([rev_over_fwd(c) for c in cands])
lab_rf = np.array([rev_over_fwd(c) for c in labeled])
out["N4_rev_over_fwd"] = {
    "labeled_med": float(np.median(lab_rf)), "cand_med": float(np.median(cand_rf)),
    "texture_med": float(np.median(tex_rf)),
    "n": [len(lab_rf), len(cand_rf), len(tex_rf)],
    "mw_cand_lt_texture_p": float(
        mannwhitneyu(cand_rf, tex_rf, alternative="less").pvalue),
    "mw_cand_gt_labeled_p": float(
        mannwhitneyu(cand_rf, lab_rf, alternative="greater").pvalue),
    "mw_labeled_lt_texture_p": float(
        mannwhitneyu(lab_rf, tex_rf, alternative="less").pvalue),
}
print("N4:", json.dumps(out["N4_rev_over_fwd"], indent=1))

# ---------- N5: row-coherence scan null (flag 2) ----------
def rowc(c):
    return c["cy"] - c["cx"] * np.tan(GRID["tilt_rad"])

cand_r = np.array([rowc(c) for c in cands])
cand_x = np.array([c["cx"] for c in cands])
W = 60.0
def max_window(rr, xx=None, span=0.0):
    o = np.argsort(rr); rr = rr[o]
    xx = xx[o] if xx is not None else None
    best, best_span_ok = 0, 0
    j = 0
    for i in range(len(rr)):
        while rr[i] - rr[j] > W:
            j += 1
        cnt = i - j + 1
        best = max(best, cnt)
        if xx is not None and cnt >= 4:
            if xx[j:i + 1].max() - xx[j:i + 1].min() >= span:
                best_span_ok = max(best_span_ok, cnt)
    return best, best_span_ok

obs_max, obs_span = max_window(cand_r, cand_x, 2500.0)
tex_r = np.array([rowc(c) for c in texture])
tex_x = np.array([c["cx"] for c in texture])
lo, hi = tex_r.min(), tex_r.max()
NMC = 50_000
cnt_uni = 0; cnt_uni_span = 0; cnt_tex = 0; cnt_tex_span = 0
for _ in range(NMC):
    ru = RNG.uniform(lo, hi, 18)
    xu = RNG.uniform(tex_x.min(), tex_x.max(), 18)
    m, ms = max_window(ru, xu, 2500.0)
    cnt_uni += m >= 4; cnt_uni_span += ms >= 4
    s = RNG.choice(len(tex_r), 18, replace=False)
    m, ms = max_window(tex_r[s], tex_x[s], 2500.0)
    cnt_tex += m >= 4; cnt_tex_span += ms >= 4
out["N5_row_coherence"] = {
    "obs_max_in_60px_window": int(obs_max),
    "obs_with_cxspan2500": int(obs_span),
    "p_uniform_ge4": cnt_uni / NMC, "p_uniform_ge4_span": cnt_uni_span / NMC,
    "p_texture_ge4": cnt_tex / NMC, "p_texture_ge4_span": cnt_tex_span / NMC,
}
print("N5:", json.dumps(out["N5_row_coherence"], indent=1))

def cnv(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))
(COMB / "comb_skeptic_w035.json").write_text(json.dumps(out, indent=1, default=cnv))
print("saved comb_skeptic_w035.json")
