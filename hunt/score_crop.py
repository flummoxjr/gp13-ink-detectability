"""Score any w035-crop prediction TIFF against the human ink labels."""
import json
import os
import sys

import numpy as np
import tifffile
from scipy.stats import rankdata

CACHE = r"D:\vesuvius-data\trackD"
L = np.load(os.path.join(CACHE, "w035_crop_labels.npz"))
ink, valid = L["ink"], L["valid"]


def tie_auc(pos, neg, max_n=400_000):
    rng = np.random.default_rng(0)
    if len(pos) > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if len(neg) > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    allv = np.concatenate([pos, neg])
    r = rankdata(allv)
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def score(path):
    p = tifffile.imread(path).astype(np.float32)
    h = min(p.shape[0], ink.shape[0]); w = min(p.shape[1], ink.shape[1])
    pc, ic, vc = p[:h, :w], ink[:h, :w], valid[:h, :w]
    pos, neg = pc[ic & vc], pc[vc & ~ic]
    auc = tie_auc(pos, neg)
    return dict(
        file=os.path.basename(path), auc=round(auc, 4),
        max=float(pc.max()), p99=float(np.percentile(pc[vc], 99)),
        frac_gt_half=round(float((pc[vc] > 0.5 * max(pc.max(), 1)).mean()), 4),
        mean_ink=round(float(pos.mean()), 2), mean_bg=round(float(neg.mean()), 2),
    )


if __name__ == "__main__":
    out = {}
    for path in sys.argv[1:]:
        s = score(path)
        out[s["file"]] = s
        print(s)
    print(json.dumps(out, indent=1))
