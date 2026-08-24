"""HUNTER 2 / stage 3 — multi-threshold tripwire re-scan.

Verdict tripwire (found ZERO at p80): component with value>195 (control blank
p99) & area>1e4 px & width>30 px. Re-scan all 8 1203 maps at in-mask p60/p70/
p80/p90. Value metric evaluated 3 ways per component (p50 / p90 / max of its
pixel values); p50 is the letter-strength reading (w035 letters: value p50=199).
w035 s42/s43 run through the identical scan as positive control: letters must
trip at every threshold, else the tripwire has threshold blind spots.

Output: comb\comb_tripwire.json (+ prints)
"""
import sys
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_map, valid_mask, save_json

from pathlib import Path
COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")

KEYS_1203 = ["1203A_s42", "1203A_s42r", "1203A_s43", "1203A_s43r",
             "1203B_s42", "1203B_s42r", "1203B_s43", "1203B_s43r"]
KEYS_CTRL = ["w035_s42", "w035_s43"]
PCTS = [60, 70, 80, 90]
AREA_MIN, WIDTH_MIN, VAL_MIN = 1e4, 30.0, 195.0


def scan(key):
    arr = load_map(key)
    m = valid_mask(arr, erode=40)
    vals = arr[m]
    res = {}
    for pct in PCTS:
        thr = float(np.percentile(vals, pct))
        binary = (arr >= thr) & m
        lab, n = ndi.label(binary, structure=np.ones((3, 3), int))
        if n == 0:
            res[f"p{pct}"] = {"threshold": thr, "n_comp": 0}
            continue
        areas = np.bincount(lab.ravel())[1:].astype(np.float64)
        skel = skeletonize(binary)
        skl = np.bincount(lab[skel], minlength=n + 1)[1:].astype(np.float64)
        width = areas / np.maximum(skl, 1.0)
        # per-component value stats (p50/p90/max) via sorted (label, value)
        comp_p50 = np.zeros(n); comp_p90 = np.zeros(n); comp_max = np.zeros(n)
        flat_lab = lab[binary]
        flat_val = arr[binary].astype(np.float64)
        order = np.argsort(flat_lab, kind="stable")
        flat_lab = flat_lab[order]; flat_val = flat_val[order]
        starts = np.searchsorted(flat_lab, np.arange(1, n + 2))
        for i in range(n):
            v = flat_val[starts[i]:starts[i + 1]]
            if v.size:
                comp_p50[i] = np.percentile(v, 50)
                comp_p90[i] = np.percentile(v, 90)
                comp_max[i] = v.max()
        size_ok = (areas >= AREA_MIN) & (width >= WIDTH_MIN)
        trip_p50 = size_ok & (comp_p50 > VAL_MIN)
        trip_p90 = size_ok & (comp_p90 > VAL_MIN)
        trip_max = size_ok & (comp_max > VAL_MIN)
        entry = {"threshold": thr, "n_comp": int(n),
                 "n_area10k": int((areas >= AREA_MIN).sum()),
                 "n_sizeok": int(size_ok.sum()),
                 "trips_valp50": int(trip_p50.sum()),
                 "trips_valp90": int(trip_p90.sum()),
                 "trips_valmax": int(trip_max.sum())}
        # top-3 near misses: largest size_ok components (or largest overall)
        cand = np.flatnonzero(size_ok) if size_ok.any() else np.argsort(-areas)[:3]
        top = sorted(cand.tolist(), key=lambda i: -areas[i])[:3]
        entry["top_components"] = [
            {"area": float(areas[i]), "width": float(width[i]),
             "val_p50": float(comp_p50[i]), "val_p90": float(comp_p90[i]),
             "val_max": float(comp_max[i])} for i in top]
        res[f"p{pct}"] = entry
        print(f"  {key} p{pct}: thr={thr:.0f} n={n} sizeok={entry['n_sizeok']} "
              f"trips p50/p90/max = {entry['trips_valp50']}/{entry['trips_valp90']}/"
              f"{entry['trips_valmax']}", flush=True)
    return res


def main():
    out = {"criteria": {"area_min": AREA_MIN, "width_min": WIDTH_MIN,
                        "value_min": VAL_MIN},
           "maps": {}}
    for key in KEYS_CTRL + KEYS_1203:
        print(key, flush=True)
        out["maps"][key] = scan(key)
    save_json(COMB / "comb_tripwire.json", out)
    print("saved comb_tripwire.json")


if __name__ == "__main__":
    main()
