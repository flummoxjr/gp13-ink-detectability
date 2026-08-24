"""Step 4: the other three tests of the validated battery, on the flagged map.

Control = PHerc0139 w035 (human-verified Greek letters, same ink_9um checkpoint,
hybrid_3d2d-seed42 step-075000).  Control maps are full-res 9.362 um/px; the
flagged map is ds4 34.56 um/px, so the control is block-mean downsampled x4
(37.4 um/px) and every morphology number is reported in physical units.
"""
import json, os, sys
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import ks_2samp, wasserstein_distance
from skimage.morphology import skeletonize
from skimage.measure import regionprops_table
import tifffile

HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag"
sys.path.insert(0, HERE)
import vf_common as C

W035 = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_w035\w035_seed42-075000.tif"
W035R = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_w035\w035_seed42-075000_reverse.tif"
LAB = r"D:\vesuvius-data\trackD\w035_ink.npy"
BLANK_P99 = 195.0
UM_CTRL_DS4 = 9.362 * 4          # 37.448 um/px
UM_FLAG_DS4 = 8.64 * 4           # 34.56  um/px
MIN_AREA = 20
res = {}


def block_mean(a, f):
    a = a[: a.shape[0] // f * f, : a.shape[1] // f * f].astype(np.float32)
    return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).mean(axis=(1, 3))


def comps(binary, um):
    lab, n = ndi.label(binary, structure=np.ones((3, 3), int))
    if n == 0:
        return None
    areas = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(areas >= MIN_AREA) + 1
    if keep.size == 0:
        return None
    skel = skeletonize(binary)
    sl = np.bincount(lab[skel], minlength=n + 1)[1:]
    pr = regionprops_table(lab, properties=("label", "area", "axis_major_length",
                                            "axis_minor_length", "solidity"))
    s = np.isin(pr["label"], keep)
    t = {"label": pr["label"][s], "area_px": pr["area"][s].astype(float),
         "major": pr["axis_major_length"][s], "minor": pr["axis_minor_length"][s],
         "solidity": pr["solidity"][s], "skel": sl[pr["label"][s] - 1].astype(float)}
    t["area_mm2"] = t["area_px"] * (um / 1000.0) ** 2
    t["width_mm"] = t["area_px"] / np.maximum(t["skel"], 1.0) * um / 1000.0
    t["elong"] = t["major"] / np.maximum(t["minor"], 1.0)
    t["wiggle"] = t["skel"] / np.maximum(t["major"], 1.0)
    return t


# ---------------- load ----------------------------------------------------
flag = C.load(C.FLAG, "forward")
flag_r = C.load(C.FLAG, "reverse")
fmask = flag > 0
fmask_e = C.erode_mask(fmask, 10)          # 10 ds4 px ~ 40 full-res px, matches battery

ctrl_full = tifffile.imread(W035).astype(np.float32)
ctrl_r_full = tifffile.imread(W035R).astype(np.float32)
lab3 = np.load(LAB, mmap_mode="r")
lab2d = np.zeros(lab3.shape[1:], np.uint8)
for z in range(lab3.shape[0]):
    np.maximum(lab2d, np.asarray(lab3[z]), out=lab2d)
lab2d = lab2d[: ctrl_full.shape[0], : ctrl_full.shape[1]] > 0
cmask_full = ndi.binary_erosion(ndi.binary_fill_holes(
    ndi.binary_closing(ctrl_full > 0, np.ones((5, 5), bool))), np.ones((3, 3), bool), 40)
ctrl = block_mean(ctrl_full, 4)
cmask = block_mean(cmask_full.astype(np.float32), 4) > 0.99
clab = block_mean(lab2d.astype(np.float32), 4) > 0.5
print(f"flag {flag.shape} mask {fmask.mean():.3f} | ctrl ds4 {ctrl.shape} mask {cmask.mean():.3f} "
      f"labels {clab.sum()} px")

# ---------------- TEST 2: morphology -------------------------------------
out_m = {}
for pct in (60, 80):
    tf = comps((flag >= np.percentile(flag[fmask_e], pct)) & fmask_e, UM_FLAG_DS4)
    thr = np.percentile(ctrl[cmask], pct)
    tc = comps((ctrl >= thr) & cmask, UM_CTRL_DS4)
    dil = ndi.binary_dilation(clab, np.ones((5, 5), bool))
    labimg, _ = ndi.label(((ctrl >= thr) & cmask), structure=np.ones((3, 3), int))
    ov = ndi.sum_labels(dil.astype(float), labimg, tc["label"]) / tc["area_px"]
    isL = ov >= 0.5
    L = {k: v[isL] for k, v in tc.items()}
    B = {k: v[~isL] for k, v in tc.items()}
    e = {"pct": pct, "n_flag": int(tf["area_px"].size), "n_ctrl_letter": int(isL.sum()),
         "n_ctrl_blank": int((~isL).sum())}
    for f_ in ("area_mm2", "width_mm", "elong", "wiggle"):
        e[f"flag_{f_}_p50"] = float(np.median(tf[f_]))
        e[f"letter_{f_}_p50"] = float(np.median(L[f_])) if L[f_].size else None
        e[f"blank_{f_}_p50"] = float(np.median(B[f_]))
        if L[f_].size > 2:
            d, p = ks_2samp(tf[f_], L[f_]); e[f"KS_vs_letter_{f_}"] = [float(d), float(p)]
        d, p = ks_2samp(tf[f_], B[f_]); e[f"KS_vs_blank_{f_}"] = [float(d), float(p)]
    out_m[f"p{pct}"] = e
    print(f"\n[morph p{pct}] flag n={e['n_flag']} area_p50={e['flag_area_mm2_p50']:.4f}mm2 "
          f"width_p50={e['flag_width_mm_p50']:.3f}mm")
    print(f"   ctrl LETTER n={e['n_ctrl_letter']} area={e['letter_area_mm2_p50']} "
          f"width={e['letter_width_mm_p50']}")
    print(f"   ctrl BLANK  n={e['n_ctrl_blank']} area={e['blank_area_mm2_p50']:.4f} "
          f"width={e['blank_width_mm_p50']:.3f}")
    print(f"   KS(flag vs LETTER): area D={e.get('KS_vs_letter_area_mm2',[None])[0]}, "
          f"width D={e.get('KS_vs_letter_width_mm',[None])[0]}")
    print(f"   KS(flag vs BLANK ): area D={e['KS_vs_blank_area_mm2'][0]:.3f} "
          f"(p={e['KS_vs_blank_area_mm2'][1]:.2g}), width D={e['KS_vs_blank_width_mm'][0]:.3f}")
res["morphology"] = out_m

# ---------------- TEST 3: map-scale fwd/rev symmetry ---------------------
rf = float(np.corrcoef(flag[fmask], flag_r[fmask])[0, 1])
cm2 = cmask_full
rc = float(np.corrcoef(ctrl_full[cm2], ctrl_r_full[cm2])[0, 1])
print(f"\n[symmetry] flag fwd/rev r = {rf:.4f} (survey record 0.7538) | control letters r = {rc:.4f}")
res["symmetry"] = {"flag_fwd_rev_r": rf, "control_fwd_rev_r": rc,
                   "control_reference_from_verdict": 0.076,
                   "1203_texture_reference": [0.51, 0.60]}

# ---------------- TEST 4: intensity --------------------------------------
vf = flag[fmask_e]
vl = ctrl_full[cm2 & lab2d]
vb = ctrl_full[cm2 & ~ndi.binary_dilation(lab2d, np.ones((13, 13), bool))]
sub = lambda v: v[np.random.default_rng(0).choice(v.size, min(v.size, 200000), replace=False)]
e = {"flag_p50": float(np.percentile(vf, 50)), "flag_p90": float(np.percentile(vf, 90)),
     "flag_p99": float(np.percentile(vf, 99)),
     "flag_frac_gt_blankp99": float((vf > BLANK_P99).mean()),
     "letter_p50": float(np.percentile(vl, 50)), "letter_p99": float(np.percentile(vl, 99)),
     "letter_frac_gt_blankp99": float((vl > BLANK_P99).mean()),
     "blank_p50": float(np.percentile(vb, 50)), "blank_p99": float(np.percentile(vb, 99)),
     "blank_frac_gt_blankp99": float((vb > BLANK_P99).mean()),
     "W1_flag_to_letter": float(wasserstein_distance(sub(vf), sub(vl))),
     "W1_flag_to_blank": float(wasserstein_distance(sub(vf), sub(vb)))}
print(f"\n[intensity] flag p50/p90/p99 = {e['flag_p50']:.0f}/{e['flag_p90']:.0f}/{e['flag_p99']:.0f}"
      f"  frac>195 = {e['flag_frac_gt_blankp99']:.5f}")
print(f"   control LETTER p50={e['letter_p50']:.0f} p99={e['letter_p99']:.0f} "
      f"frac>195={e['letter_frac_gt_blankp99']:.3f}")
print(f"   control BLANK  p50={e['blank_p50']:.0f} p99={e['blank_p99']:.0f}")
print(f"   Wasserstein-1: to LETTER {e['W1_flag_to_letter']:.1f} | to BLANK {e['W1_flag_to_blank']:.1f}")
res["intensity"] = e

json.dump(res, open(os.path.join(HERE, "vf_battery.json"), "w"), indent=1, default=float)
np.savez_compressed(os.path.join(HERE, "vf_ctrl_ds4.npz"), ctrl=ctrl.astype(np.uint8),
                    cmask=cmask, clab=clab)
print("\nwrote vf_battery.json + vf_ctrl_ds4.npz")
