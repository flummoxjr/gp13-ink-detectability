"""Test 2: stroke morphology. Threshold each map at its own in-mask p60/p80,
connected components, per-component: area, elongation, skeleton length,
width (area/skel_len), wiggliness (skel_len/major_axis), endpoints.
Control components split into LETTER (>=50% overlap with dilated human label)
vs OFF-LETTER. Compare distributions vs 1203 maps (KS tests)."""
import sys, time
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import ks_2samp
from skimage.morphology import skeletonize
from skimage.measure import regionprops_table

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_map, valid_mask, load_w035_label2d, save_json, SALVAGE

MIN_AREA = 20
PCTS = [60, 80]


def component_stats(binary):
    lab, n = ndi.label(binary, structure=np.ones((3, 3), int))
    if n == 0:
        return None, None
    areas = np.bincount(lab.ravel())[1:]
    keep_ids = np.flatnonzero(areas >= MIN_AREA) + 1
    if keep_ids.size == 0:
        return None, None
    skel = skeletonize(binary)
    skel_len = np.bincount(lab[skel], minlength=n + 1)[1:]
    # endpoints: skeleton pixels with exactly 1 neighbor on the skeleton
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    endpts = skel & (nb == 2)  # self + 1 neighbor
    ep_cnt = np.bincount(lab[endpts], minlength=n + 1)[1:]
    props = regionprops_table(lab, properties=("label", "area", "axis_major_length",
                                               "axis_minor_length", "solidity"))
    sel = np.isin(props["label"], keep_ids)
    tab = {
        "label": props["label"][sel],
        "area": props["area"][sel].astype(np.float64),
        "major": props["axis_major_length"][sel],
        "minor": props["axis_minor_length"][sel],
        "solidity": props["solidity"][sel],
        "skel_len": skel_len[props["label"][sel] - 1].astype(np.float64),
        "endpoints": ep_cnt[props["label"][sel] - 1].astype(np.float64),
    }
    tab["elong"] = tab["major"] / np.maximum(tab["minor"], 1.0)
    tab["width"] = tab["area"] / np.maximum(tab["skel_len"], 1.0)
    tab["wiggle"] = tab["skel_len"] / np.maximum(tab["major"], 1.0)
    return lab, tab


def summarize(tab, mask_area_px):
    out = {"n_comp": int(tab["area"].size),
           "comp_per_mm2": float(tab["area"].size / (mask_area_px * (9.362e-3) ** 2))}
    for f in ["area", "elong", "width", "wiggle", "skel_len", "solidity", "endpoints"]:
        v = tab[f]
        out[f] = {"p25": float(np.percentile(v, 25)), "p50": float(np.percentile(v, 50)),
                  "p75": float(np.percentile(v, 75)), "p90": float(np.percentile(v, 90))}
    return out


def run_map(key, arr, mask, label2d=None):
    res = {}
    vals = arr[mask]
    for pct in PCTS:
        thr = np.percentile(vals, pct)
        binary = (arr >= thr) & mask
        lab, tab = component_stats(binary)
        if tab is None:
            continue
        entry = {"threshold": float(thr), "on_frac": float(binary.sum() / mask.sum())}
        if label2d is not None:
            dil = ndi.binary_dilation(label2d, np.ones((9, 9), bool))
            # overlap fraction per component
            ov = ndi.sum_labels(dil.astype(np.float64), lab, tab["label"]) / tab["area"]
            is_letter = ov >= 0.5
            tabs = {"letter": {k: v[is_letter] for k, v in tab.items()},
                    "offletter": {k: v[~is_letter] for k, v in tab.items()}}
            for grp, t in tabs.items():
                if t["area"].size > 0:
                    entry[grp] = summarize(t, mask.sum())
            # save raw features for KS tests
            np.savez_compressed(SALVAGE / f"morph_{key}_p{pct}.npz",
                                **{f"letter_{k}": tabs["letter"][k] for k in tab},
                                **{f"offletter_{k}": tabs["offletter"][k] for k in tab})
            entry["n_letter"] = int(is_letter.sum())
            entry["n_offletter"] = int((~is_letter).sum())
        else:
            entry["all"] = summarize(tab, mask.sum())
            np.savez_compressed(SALVAGE / f"morph_{key}_p{pct}.npz",
                                **{f"all_{k}": tab[k] for k in tab})
        res[f"p{pct}"] = entry
    return res


def main():
    results = {}
    w035 = load_map("w035_s42")
    m035 = valid_mask(w035, erode=40)
    lab2d = load_w035_label2d(w035.shape)

    t0 = time.time()
    results["w035_s42"] = run_map("w035_s42", w035, m035, lab2d)
    print(f"w035_s42 done {time.time()-t0:.0f}s", flush=True)
    w = load_map("w035_s43")
    results["w035_s43"] = run_map("w035_s43", w, valid_mask(w, erode=40), lab2d)
    print(f"w035_s43 done {time.time()-t0:.0f}s", flush=True)

    for key in ["1203A_s42", "1203A_s43", "1203B_s42", "1203B_s43",
                "1203A_s42r", "1203B_s42r"]:
        a = load_map(key)
        results[key] = run_map(key, a, valid_mask(a, erode=40))
        print(f"{key} done {time.time()-t0:.0f}s", flush=True)

    # KS tests: control letter comps vs each 1203 map (p80), on key features
    ks = {}
    ref = np.load(SALVAGE / "morph_w035_s42_p80.npz")
    for key in ["1203A_s42", "1203A_s43", "1203B_s42", "1203B_s43"]:
        tgt = np.load(SALVAGE / f"morph_{key}_p80.npz")
        row = {}
        for f in ["area", "elong", "width", "wiggle", "solidity"]:
            a = ref[f"letter_{f}"]; b = tgt[f"all_{f}"]
            if f == "area":
                a, b = np.log10(a), np.log10(b)
            st = ks_2samp(a, b)
            row[f"letter_vs_1203_{f}"] = {"D": float(st.statistic), "p": float(st.pvalue)}
            a2 = ref[f"offletter_{f}"]
            if f == "area":
                a2 = np.log10(a2)
            st2 = ks_2samp(a2, b)
            row[f"offletter_vs_1203_{f}"] = {"D": float(st2.statistic), "p": float(st2.pvalue)}
        ks[key] = row
    results["KS_p80"] = ks

    save_json(SALVAGE / "verdict_morph.json", results)
    print("saved verdict_morph.json")


if __name__ == "__main__":
    main()
