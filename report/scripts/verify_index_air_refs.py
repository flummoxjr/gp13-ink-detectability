"""Forensic recompute of the K2b index from cached ROI cubes.

For each scroll: load cached pap*/air* npys, run the CURRENT validate_air gates
on the primary air windows, rebuild the median air PSD from the ones that pass,
recompute (bandwidth, SNR@0.25, DN headroom) per papyrus cube, and diff against
the shipped per-scroll JSON. Purpose: (1) assign an evidence-based noise_ref
label to the 12 JSONs that predate the field; (2) verify shipped numbers
reproduce; (3) collect air mean-DN / PSD-flatness for the report table.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD")
from k2b_detectability_index import (  # noqa: E402
    validate_air, radial_psd, residual_noise_level, N_AIR)

CACHE = r"D:\vesuvius-data\trackD\k2b"
OUTDIR = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\k2b_index"
HERE = os.path.dirname(os.path.abspath(__file__))

SCROLLS = ["PHerc0125", "PHerc0191", "PHerc0211", "PHerc0257", "PHerc0268",
           "PHerc0358", "PHerc0800", "PHerc0813", "PHerc0826", "PHerc1218",
           "PHerc1447", "PHerc1545", "PHerc1203", "PHerc0139"]

report = {}
for s in SCROLLS:
    with open(os.path.join(OUTDIR, f"{s}.json")) as f:
        shipped = json.load(f)
    paps = []
    for i in range(8):
        p = os.path.join(CACHE, f"{s}_pap{i}.npy")
        if os.path.exists(p):
            a = np.load(p)
            if (a > 0).mean() >= 0.9:
                paps.append(a)
    pap_mean_dn = float(np.median([a[a > 0].mean() for a in paps]))

    airs = []
    for i in range(N_AIR):
        p = os.path.join(CACHE, f"{s}_air{i}.npy")
        if os.path.exists(p):
            got, why = validate_air(np.load(p), pap_mean_dn)
            airs.append({"idx": i, "pass": bool(got), "why": why,
                         **({"mean_dn": round(got["mean_dn"], 1),
                             "flatness": round(got["flatness"], 1),
                             "psd": got["psd"]} if got else {})})
    ok = [a for a in airs if a["pass"]]

    row = {"n_pap": len(paps), "pap_mean_dn": round(pap_mean_dn, 1),
           "primary_air": [{k: v for k, v in a.items() if k != "psd"}
                           for a in airs]}
    if ok:
        air_psd = np.nanmedian(np.stack([a["psd"] for a in ok]), axis=0)
        rows = []
        for a in paps:
            q, p = radial_psd(a.astype(np.float32))
            snr = p / air_psd
            above = (q > 0.02) & (snr >= 2.0)
            bw = float(q[above].max()) if above.any() else 0.0
            i25 = int(np.argmin(np.abs(q - 0.25)))
            dn = a[a > 0].astype(np.float32)
            rows.append((round(bw, 4), round(float(snr[i25]), 2),
                         round(float(np.percentile(dn, 99.5)
                                     - np.percentile(dn, 0.5)), 1)))
        med = lambda k: round(float(np.median([r[k] for r in rows])), 4)
        row["recomputed"] = {"bw_med": med(0), "snr25_med": med(1),
                             "dn_med": med(2), "rows": rows}
        row["shipped"] = {"bw_med": shipped["bandwidth_med_iqr"][0],
                          "snr25_med": shipped["snr_q025_med_iqr"][0],
                          "dn_med": shipped["dn_headroom_med_iqr"][0]}
        row["match"] = (abs(row["recomputed"]["bw_med"] - row["shipped"]["bw_med"]) < 1e-3
                        and abs(row["recomputed"]["snr25_med"] - row["shipped"]["snr25_med"])
                        / max(row["shipped"]["snr25_med"], 1e-9) < 0.02)
    row["shipped_noise_ref"] = shipped.get("noise_ref", "(absent)")
    report[s] = row
    print(s, "airs:", [(a["idx"], a["pass"], a.get("mean_dn"), a.get("flatness"),
                        a["why"]) for a in airs])
    if "recomputed" in row:
        print("   recomputed", row["recomputed"]["bw_med"], row["recomputed"]["snr25_med"],
              row["recomputed"]["dn_med"], "| shipped", row["shipped"],
              "| match:", row["match"])

with open(os.path.join(HERE, "verify_index_air_refs.json"), "w") as f:
    json.dump(report, f, indent=1, default=str)
print("done")
