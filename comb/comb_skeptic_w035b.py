r"""SKEPTIC follow-up: is the N4 rev/fwd separation (cands 0.49 vs texture 0.61,
p=3e-7) a mechanical artifact of selection on forward strength?

The gate selects strong comps (v_p90>=195). If rev response is only weakly
coupled to fwd response, rev/fwd falls mechanically as fwd rises. Tests:
  a) Spearman(rev_over_fwd, fwd_mean) among texture comps.
  b) Strength-matched comparison: for each candidate, take texture comps in
     the same fwd_mean stratum (nearest-neighbor matching, 5 per cand);
     MW candidates vs matched texture.
  c) ALSO the cleaner absolute check: rev_mean of candidates vs rev_mean of
     strength-matched texture (does the reverse map actually respond LESS on
     candidate footprints, or is only the ratio lower?).
Plus: N1 caveat numbers (fwd in-mask frac195 for scale) and rev cross-IoU.
"""
import json
import numpy as np
import tifffile
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
from scipy import ndimage as ndi

COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")
OUT_W035 = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_w035")

cat = json.loads((COMB / "comb_catalog.json").read_text())
comps42 = cat["components_s42"]
cands = [c for c in comps42 if c.get("letter_class") and c["zone"] == "beyond"]
texture = [c for c in comps42 if (not c.get("letter_class"))
           and c["zone"] == "beyond" and c["area"] >= 3000]
labeled = [c for c in comps42 if c["zone"] == "labeled"]

lb42 = np.load(COMB / "_comp42.npz")["lb"]
fwd = tifffile.imread(str(OUT_W035 / "w035_seed42-075000.tif"))
rev = tifffile.imread(str(OUT_W035 / "w035_seed42-075000_reverse.tif"))


def means(c):
    y0, x0, y1, x1 = c["bbox"]
    m = lb42[y0:y1, x0:x1] == c["id"]
    return (float(fwd[y0:y1, x0:x1][m].mean()),
            float(rev[y0:y1, x0:x1][m].mean()))

for pop in (cands, texture, labeled):
    for c in pop:
        c["_fm"], c["_rm"] = means(c)
        c["_rf"] = c["_rm"] / max(c["_fm"], 1e-6)

tex_fm = np.array([c["_fm"] for c in texture])
tex_rm = np.array([c["_rm"] for c in texture])
tex_rf = np.array([c["_rf"] for c in texture])
cand_fm = np.array([c["_fm"] for c in cands])
cand_rm = np.array([c["_rm"] for c in cands])
cand_rf = np.array([c["_rf"] for c in cands])

out = {}
rho, p = spearmanr(tex_rf, tex_fm)
out["a_spearman_texture_rf_vs_fwdmean"] = {"rho": float(rho), "p": float(p)}

# b/c: nearest-neighbor strength matching (5 texture comps per candidate,
# without replacement, greedy by |fwd_mean diff|)
used = set()
matched_rf, matched_rm, matched_fm = [], [], []
for fm in cand_fm:
    d = np.abs(tex_fm - fm).astype(float)
    d[list(used)] = np.inf
    idx = np.argsort(d)[:5]
    used.update(idx.tolist())
    matched_rf += tex_rf[idx].tolist()
    matched_rm += tex_rm[idx].tolist()
    matched_fm += tex_fm[idx].tolist()
matched_rf = np.array(matched_rf); matched_rm = np.array(matched_rm)
matched_fm = np.array(matched_fm)

out["b_matched"] = {
    "cand_fwdmean_med": float(np.median(cand_fm)),
    "matched_tex_fwdmean_med": float(np.median(matched_fm)),
    "all_tex_fwdmean_med": float(np.median(tex_fm)),
    "cand_rf_med": float(np.median(cand_rf)),
    "matched_tex_rf_med": float(np.median(matched_rf)),
    "mw_cand_lt_matched_rf_p": float(
        mannwhitneyu(cand_rf, matched_rf, alternative="less").pvalue),
    "cand_revmean_med": float(np.median(cand_rm)),
    "matched_tex_revmean_med": float(np.median(matched_rm)),
    "mw_cand_lt_matched_revmean_p": float(
        mannwhitneyu(cand_rm, matched_rm, alternative="less").pvalue),
}

# labeled letters same stats for context
out["labeled_context"] = {
    "labeled_fwdmean_med": float(np.median([c["_fm"] for c in labeled])),
    "labeled_rf_med": float(np.median([c["_rf"] for c in labeled])),
    "labeled_revmean_med": float(np.median([c["_rm"] for c in labeled])),
}

# N1 caveat: fwd map in-mask frac195 for scale vs reverse's 0.0021
def valid_mask(arr, erode=40):
    m = arr > 0
    m = ndi.binary_closing(m, structure=np.ones((5, 5), bool))
    m = ndi.binary_fill_holes(m)
    return ndi.binary_erosion(m, structure=np.ones((3, 3), bool), iterations=erode)

vm = valid_mask(fwd)
out["N1_scale"] = {"fwd_inmask_frac195": float((fwd[vm] >= 195).mean())}

sk = json.loads((COMB / "comb_skeptic_w035.json").read_text())
out["N1_rev_cross"] = {
    "rev42_beyond_cross_iou": sk["N1_reverse_gate"]["rev42_beyond_cross_iou"],
    "rev42_comps": [
        {k: r[k] for k in ("id", "area", "cy", "cx", "width", "v_p90", "frac195", "zone")}
        for r in sk["N1_reverse_gate"]["rev42"]["letter_class_comps"]],
    "rev43_comps": [
        {k: r[k] for k in ("id", "area", "cy", "cx", "width", "v_p90", "frac195", "zone")}
        for r in sk["N1_reverse_gate"]["rev43"]["letter_class_comps"]],
}

(COMB / "comb_skeptic_w035b.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
