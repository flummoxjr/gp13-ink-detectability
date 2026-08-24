"""Test 4: intensity calibration. Compare 1203 in-mask value distributions to
the control's LETTER-pixel distribution vs OFF-LETTER (blank papyrus) pixel
distribution. Wasserstein distances + tail fractions."""
import sys
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import wasserstein_distance

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_map, valid_mask, load_w035_label2d, save_json, SALVAGE


def qtiles(v):
    q = np.percentile(v, [1, 25, 50, 75, 90, 99])
    return {"p1": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4]), "p99": float(q[5]),
            "mean": float(v.mean())}


def main():
    w035 = load_map("w035_s42")
    m035 = valid_mask(w035, erode=40)
    lab = load_w035_label2d(w035.shape)
    lab_dil = ndi.binary_dilation(lab, np.ones((13, 13), bool))

    letter_px = w035[m035 & lab]
    off_px = w035[m035 & ~lab_dil]          # blank papyrus, halo excluded
    rng = np.random.default_rng(0)
    off_sub = rng.choice(off_px, size=min(2_000_000, off_px.size), replace=False)

    out = {"w035_letter": qtiles(letter_px), "w035_offletter": qtiles(off_px),
           "n_letter_px": int(letter_px.size)}
    # reference thresholds
    off_p99 = np.percentile(off_px, 99)
    out["w035_offletter_p99"] = float(off_p99)
    out["w035_frac_above_off_p99"] = float((w035[m035] > off_p99).mean())
    out["w035_letter_frac_above_off_p99"] = float((letter_px > off_p99).mean())

    for key in ["1203A_s42", "1203A_s43", "1203B_s42", "1203B_s43",
                "1203A_s42r", "1203B_s42r"]:
        a = load_map(key)
        m = valid_mask(a, erode=40)
        v = a[m]
        vsub = rng.choice(v, size=min(2_000_000, v.size), replace=False)
        letter_sub = letter_px if letter_px.size <= 2_000_000 else rng.choice(letter_px, 2_000_000, replace=False)
        out[key] = qtiles(v)
        out[key]["W1_to_letter"] = float(wasserstein_distance(vsub[:500000], letter_sub.astype(np.float64)))
        out[key]["W1_to_offletter"] = float(wasserstein_distance(vsub[:500000], off_sub[:500000].astype(np.float64)))
        out[key]["frac_above_w035off_p99"] = float((v > off_p99).mean())
        print(key, out[key]["p50"], "W1letter", round(out[key]["W1_to_letter"], 1),
              "W1off", round(out[key]["W1_to_offletter"], 1),
              "frac>off_p99", round(out[key]["frac_above_w035off_p99"], 4), flush=True)

    save_json(SALVAGE / "verdict_intensity.json", out)
    print("saved verdict_intensity.json")


if __name__ == "__main__":
    main()
