r"""SKEPTIC PASS — flag 6 (K3 z~3900-4020 regional median-ratio elevation).

The hunter's claim rests on 3 control windows (1 r-matched, 1 z-offset,
1 low-fill). Build a REAL null: sample ~16 windows on the same-radius rings
(r~161 for #3 at z=3925, r~242 for #5 at z=4022) at many angles + extra z
planes, run the IDENTICAL pipeline (offsets, pap mask, 2-vox erode, masked
110keV blur sigma=0.62, whole-window median ratio). Where do 1.45/1.48 fall
in that null distribution? Also: radial-trend check (median vs r) to test the
beam-hardening/core-systematics alternative.

Axis found from low-res mask centroid of the masked zarr itself.
Output: comb_skeptic_k3.json
"""
import json
import os
import numpy as np
from scipy import ndimage as ndi

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
V74 = f"{BUCKET}/PHercParis4/volumes/20260310170716-45.532um-11.0m-74keV-masked.zarr"
V110 = f"{BUCKET}/PHercParis4/volumes/20260310173927-45.532um-11.0m-110keV-masked.zarr"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb"
CACHE = os.path.join(OUT, "cache_k3")
WIN74 = (-0.058, 0.27)
WIN110 = (-0.04, 0.2)
OFF74, OFF110 = -0.03099, -0.02682
HZ, HY, HX = 10, 80, 80
ERODE = 2
SIG_EQ = 0.62
CLUSTERS = {"#3": (3925, 1058, 989), "#5": (4022, 1237, 914)}

import zarr, fsspec
_z74 = None


def get_z74():
    global _z74
    if _z74 is None:
        _z74 = zarr.open(fsspec.get_mapper(V74), mode="r")
    return _z74


def axis_at(z0):
    """mask centroid at L3 (or coarsest available level)."""
    g = get_z74()
    lvl = "3" if "3" in g else sorted(g.keys())[-1]
    a = np.asarray(g[lvl][z0 // 8])
    ys, xs = np.nonzero(a > 0)
    return float(ys.mean() * 8), float(xs.mean() * 8)


def fetch(tag, vol_url, c):
    p = os.path.join(CACHE, f"{tag}_{c[0]}_{c[1]}_{c[2]}.npy")
    if os.path.exists(p):
        return np.load(p)
    zl = zarr.open(fsspec.get_mapper(vol_url), mode="r")["0"]
    z, y, x = c
    a = np.asarray(zl[z - HZ:z + HZ + 1, y - HY:y + HY + 1, x - HX:x + HX + 1])
    np.save(p, a)
    return a


def to_f(a, win, off):
    return a.astype(np.float32) / 255 * (win[1] - win[0]) + win[0] - off


def masked_blur(f, m, sigma):
    if sigma <= 0:
        return f
    mf = m.astype(np.float32)
    num = ndi.gaussian_filter(f * mf, sigma)
    den = ndi.gaussian_filter(mf, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-3, num / np.maximum(den, 1e-3), f)


def window_median(c):
    a74 = fetch("v74", V74, c)
    a110 = fetch("v110", V110, c)
    f74 = to_f(a74, WIN74, OFF74)
    f110 = to_f(a110, WIN110, OFF110)
    pap = (a74 > 0) & (a110 > 0) & (f74 > 0.02) & (f110 > 0.015)
    if pap.mean() < 0.25:
        return None
    core = ndi.binary_erosion(pap, iterations=ERODE)
    res = {"center": list(c), "pap_frac": round(float(pap.mean()), 3),
           "core_vox": int(core.sum())}
    for s, key in [(0.0, "med_s0"), (SIG_EQ, "med_seq")]:
        f110b = masked_blur(f110, pap, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(core, f74 / np.maximum(f110b, 1e-4), np.nan)
        res[key] = round(float(np.nanmedian(ratio)), 4)
    return res


def main():
    out = {"axis": {}, "null_windows": [], "clusters": {}, "skipped": []}
    axes = {}
    for z0 in (3600, 3700, 3800, 3925, 4022):
        try:
            axes[z0] = axis_at(z0)
        except Exception as e:
            print("axis fail", z0, e)
    out["axis"] = {str(k): v for k, v in axes.items()}
    print("axes:", out["axis"], flush=True)

    # cluster r and angle
    for tag, (cz, cy, cx) in CLUSTERS.items():
        ay, ax = axes[cz]
        r = float(np.hypot(cy - ay, cx - ax))
        th = float(np.degrees(np.arctan2(cy - ay, cx - ax)))
        out["clusters"][tag] = {"z": cz, "r": round(r, 1), "angle_deg": round(th, 1)}
    print("clusters:", out["clusters"], flush=True)

    # null ring samples: same z & r as each cluster, angles spread, skipping
    # +/-30 deg of the cluster angle; plus extra z planes at the same r.
    plans = []
    for tag, (cz, cy, cx) in CLUSTERS.items():
        ay, ax = axes[cz]
        r = np.hypot(cy - ay, cx - ax)
        th0 = np.degrees(np.arctan2(cy - ay, cx - ax))
        for dth in range(40, 360, 40):
            th = np.radians(th0 + dth)
            yy = int(round(ay + r * np.sin(th)))
            xx = int(round(ax + r * np.cos(th)))
            plans.append((tag + "_ring", (cz, yy, xx)))
        for z2 in ((3700, 3800) if tag == "#3" else (3600, 3800)):
            ay2, ax2 = axes[z2]
            for dth in (90, 180, 270):
                th = np.radians(th0 + dth)
                yy = int(round(ay2 + r * np.sin(th)))
                xx = int(round(ax2 + r * np.cos(th)))
                plans.append((tag + f"_z{z2}", (z2, yy, xx)))

    for kind, c in plans:
        if not (HY <= c[1] < 2264 - HY and HX <= c[2] < 2264 - HX):
            out["skipped"].append({"kind": kind, "center": list(c),
                                   "why": "out of bounds"})
            continue
        try:
            res = window_median(c)
        except Exception as e:
            print("fetch fail", c, e, flush=True)
            continue
        if res is None:
            out["skipped"].append({"kind": kind, "center": list(c)})
            print(f"skip {kind} {c} (low fill)", flush=True)
            continue
        res["kind"] = kind
        # radius of this window
        z0 = c[0]
        ay, ax = axes[min(axes, key=lambda k: abs(k - z0))]
        res["r"] = round(float(np.hypot(c[1] - ay, c[2] - ax)), 1)
        out["null_windows"].append(res)
        print(f"{kind} {c} r={res['r']} pap={res['pap_frac']} "
              f"med_s0={res['med_s0']} med_seq={res['med_seq']}", flush=True)

    meds = [w["med_seq"] for w in out["null_windows"]]
    out["null_summary"] = {
        "n": len(meds), "min": min(meds), "p25": float(np.percentile(meds, 25)),
        "p50": float(np.percentile(meds, 50)), "p75": float(np.percentile(meds, 75)),
        "max": max(meds),
        "cluster_medians_seq": {"#3": 1.4813, "#5": 1.4648},
        "n_null_ge_1p43": int(sum(m >= 1.43 for m in meds)),
    }
    print(json.dumps(out["null_summary"], indent=1))
    with open(os.path.join(OUT, "comb_skeptic_k3.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote comb_skeptic_k3.json")


if __name__ == "__main__":
    main()
