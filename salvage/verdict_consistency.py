"""Test 3b: model self-consistency. Pixelwise Pearson r between seed42 and
seed43 maps (and fwd vs z-reverse), full-res and ds8 (regional agreement)."""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_map, valid_mask, downsample, save_json, SALVAGE

PAIRS = [
    ("w035_s42", "w035_s43", "seed"),
    ("w035_s42r", "w035_s43r", "seed"),
    ("w035_s42", "w035_s42r", "zdir"),
    ("1203A_s42", "1203A_s43", "seed"),
    ("1203A_s42r", "1203A_s43r", "seed"),
    ("1203A_s42", "1203A_s42r", "zdir"),
    ("1203B_s42", "1203B_s43", "seed"),
    ("1203B_s42r", "1203B_s43r", "seed"),
    ("1203B_s42", "1203B_s42r", "zdir"),
]


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    cache = {}
    def get(k):
        if k not in cache:
            arr = load_map(k).astype(np.float32)
            cache[k] = (arr, valid_mask(arr, erode=8))
        return cache[k]

    out = {}
    for k1, k2, kind in PAIRS:
        a, ma = get(k1)
        b, mb = get(k2)
        m = ma & mb
        r_full = pearson(a[m], b[m])
        a8 = downsample(a, 8); b8 = downsample(b, 8)
        m8 = downsample(m.astype(np.float32), 8) > 0.9
        r_ds8 = pearson(a8[m8], b8[m8])
        out[f"{k1}__vs__{k2}"] = {"kind": kind, "r_pixel": r_full, "r_ds8": r_ds8,
                                  "n_px": int(m.sum())}
        print(f"{k1} vs {k2} [{kind}]: r_pixel={r_full:.3f}  r_ds8={r_ds8:.3f}", flush=True)
        # keep cache small
        if len(cache) > 4:
            for kk in list(cache)[:2]:
                if kk not in (k1, k2):
                    del cache[kk]

    save_json(SALVAGE / "verdict_consistency.json", out)
    print("saved verdict_consistency.json")


if __name__ == "__main__":
    main()
