"""Sanity probe: shapes, masks, label alignment, w035 prediction-vs-label AUC check."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_map, valid_mask, load_w035_label2d, save_json, SALVAGE

out = {}

for key in ["w035_s42", "1203A_s42", "1203B_s42"]:
    a = load_map(key)
    m = valid_mask(a, erode=8)
    nz = a[m]
    out[key] = {
        "shape": list(a.shape),
        "mask_frac": float(m.mean()),
        "in-mask_zero_frac": float((nz == 0).mean()),
        "in-mask_mean": float(nz.mean()),
        "in-mask_p50": float(np.percentile(nz, 50)),
        "in-mask_p90": float(np.percentile(nz, 90)),
        "in-mask_p99": float(np.percentile(nz, 99)),
        "max": int(a.max()),
    }
    if key == "w035_s42":
        lab = load_w035_label2d(a.shape)
        out["w035_label"] = {
            "label_frac_full": float(lab.mean()),
            "label_frac_in_mask": float(lab[m].mean()),
        }
        # quick AUC via rank statistics on a subsample
        rng = np.random.default_rng(0)
        idx = np.flatnonzero(m.ravel())
        sub = rng.choice(idx, size=min(2_000_000, idx.size), replace=False)
        vals = a.ravel()[sub].astype(np.float64)
        labs = lab.ravel()[sub]
        pos = vals[labs]
        neg = vals[~labs]
        # Mann-Whitney AUC
        from scipy.stats import rankdata
        allv = np.concatenate([pos, neg])
        r = rankdata(allv)
        auc = (r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
        out["w035_label"]["auc_pred_vs_label"] = float(auc)
        out["w035_label"]["n_pos"] = int(pos.size)
        out["w035_label"]["letter_pred_mean"] = float(pos.mean())
        out["w035_label"]["offletter_pred_mean"] = float(neg.mean())

save_json(SALVAGE / "verdict_prep.json", out)
import json
print(json.dumps(out, indent=1))
