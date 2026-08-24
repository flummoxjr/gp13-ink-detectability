r"""Comb 3c — K3 #3/#5: control-window base rate + edge-distance of survivors.

Two follow-ups to comb_k3_psfmatch2:
 1. CONTROL: an ordinary-papyrus L0 window at the same z (no stage-2 cluster)
    through the identical pipeline -> base-rate hot fraction at sigma_eq.
 2. Where do the surviving hot voxels live? Distance-to-mask-edge distribution
    of hot voxels at sigma=0 vs sigma=0.62. Surface-hugging survivors =
    incrustation/rim phenomenon, not intra-fiber mineralization.
Output: comb/k3_psfmatch3.json + printout.
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
EXCESS_THR = 0.504
ERODE = 2
SIG_EQ = 0.62
CLUSTERS = {"#3": (3925, 1058, 989), "#5": (4022, 1237, 914)}
CONTROL_CANDS = [(3925, 1300, 1500), (3925, 1500, 1200), (3925, 800, 1400),
                 (3925, 1500, 700), (3925, 700, 700)]


def fetch(tag, vol_url, c):
    p = os.path.join(CACHE, f"{tag}_{c[0]}_{c[1]}_{c[2]}.npy")
    if os.path.exists(p):
        return np.load(p)
    import zarr, fsspec
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


def analyze(c, label):
    a74 = fetch("v74", V74, c)
    a110 = fetch("v110", V110, c)
    f74 = to_f(a74, WIN74, OFF74)
    f110 = to_f(a110, WIN110, OFF110)
    pap = (a74 > 0) & (a110 > 0) & (f74 > 0.02) & (f110 > 0.015)
    if pap.mean() < 0.25:
        print(f"{label} {c}: pap frac {pap.mean():.3f} — too empty, skip")
        return None
    core = ndi.binary_erosion(pap, iterations=ERODE)
    edt = ndi.distance_transform_edt(pap)
    res = {"center": list(c), "pap_frac": round(float(pap.mean()), 3),
           "core_vox": int(core.sum())}
    for s in [0.0, SIG_EQ]:
        f110b = masked_blur(f110, pap, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(core, f74 / np.maximum(f110b, 1e-4), np.nan)
        med = float(np.nanmedian(ratio))
        hot = np.isfinite(ratio) & (ratio - med > EXCESS_THR)
        d = edt[hot]
        key = f"s{s}"
        res[key] = {
            "median": round(med, 4), "n_hot": int(hot.sum()),
            "hot_frac": round(float(hot.sum() / core.sum()), 5),
            "hot_edt_p50": round(float(np.median(d)), 2) if d.size else None,
            "hot_edt_p90": round(float(np.percentile(d, 90)), 2) if d.size else None,
            "core_edt_p50": round(float(np.median(edt[core])), 2),
        }
        print(f"{label} {c} sigma={s}: med={med:.3f} hot={hot.sum():6d} "
              f"({100*hot.sum()/core.sum():.2f}%) edt_p50={res[key]['hot_edt_p50']} "
              f"(core edt p50 {res[key]['core_edt_p50']})", flush=True)
    return res


def main():
    out = {"clusters": {}, "controls": []}
    for tag, c in CLUSTERS.items():
        out["clusters"][tag] = analyze(c, tag)
    got = 0
    for c in CONTROL_CANDS:
        r = analyze(c, "ctrl")
        if r:
            out["controls"].append(r)
            got += 1
        if got >= 2:
            break
    with open(os.path.join(OUT, "k3_psfmatch3.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote k3_psfmatch3.json")


if __name__ == "__main__":
    main()
