"""Investigation D step 2b: refine the 9.362um <-> 2.403um transform and test rigidity.

Takes the coarse level-5 solution (p9 = p2/3.896 + t, t=(7936, 19.4, -12.6) in level-0 9um
voxels) and refines it independently at N widely separated blocks using level-2 (9um,
37.45um/vox) vs level-4 (2.4um, 38.45um/vox) phase correlation.  Consistent per-block
translations => a pure-translation model is enough; systematic drift => fit a similarity.
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom
from skimage.registration import phase_cross_correlation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zarr_http import Zarr3D  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
P9 = "PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr"
P2 = "PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
V9, V2 = 0.009362, 0.002403
RATIO = V9 / V2
T0 = np.array([7936.0, 19.4, -12.6])  # level-0 9um voxels, from derive_transform.py
L9, L2 = 2, 4                          # pyramid levels: 37.448um vs 38.448um
S9, S2 = 2 ** L9, 2 ** L2
BLK = 128


def main():
    z9 = Zarr3D(P9, L9)
    z2 = Zarr3D(P2, L2)
    print("9um lvl%d %s (%.2f um/vox)  2.4um lvl%d %s (%.2f um/vox)"
          % (L9, z9.shape, V9 * S9 * 1e3, L2, z2.shape, V2 * S2 * 1e3))
    rescale = (V2 * S2) / (V9 * S9)

    # probe points in 2.4um level-0 coords: spread over z and lateral positions that carry
    # scroll material (segments live around 9um x 1000-5700, y 2000-5100)
    zs = [2500, 5500, 8500, 11500, 14000]
    lat = [(3600, 3300), (2600, 4300), (4300, 4000)]  # 9um level-0 (y, x) -> converted below
    probes = []
    for z2c in zs:
        for (y9c, x9c) in lat:
            y2c = (y9c - T0[1]) * RATIO
            x2c = (x9c - T0[2]) * RATIO
            probes.append((z2c, y2c, x2c))

    rows = []
    for (z2c, y2c, x2c) in probes:
        # 2.4um block
        c2 = np.array([z2c, y2c, x2c]) / S2
        a = (c2 - BLK / 2).astype(int)
        if (a < 0).any() or (a + BLK > np.array(z2.shape)).any():
            continue
        Braw = z2.read(a[0], a[0] + BLK, a[1], a[1] + BLK, a[2], a[2] + BLK).astype(np.float32)
        if (Braw > 0).mean() < 0.5:
            print(f"skip z2={z2c} y2={y2c:.0f} x2={x2c:.0f}: only {(Braw>0).mean():.2f} occupied")
            continue
        Bz = zoom(Braw, rescale, order=1)
        n = min(Bz.shape)
        Bz = Bz[:n, :n, :n]
        # matching 9um block, same physical center, same size
        p9 = np.array([z2c, y2c, x2c]) / RATIO + T0        # predicted 9um level-0 centre
        c9 = p9 / S9
        b = (c9 - n / 2).astype(int)
        if (b < 0).any() or (b + n > np.array(z9.shape)).any():
            continue
        A = z9.read(b[0], b[0] + n, b[1], b[1] + n, b[2], b[2] + n).astype(np.float32)
        if (A > 0).mean() < 0.5:
            print(f"skip (9um empty) z2={z2c}")
            continue
        shift, err, _ = phase_cross_correlation(A, Bz, upsample_factor=10, normalization=None)
        # NCC at the recovered shift (integer part) for a quality read-out
        si = np.round(shift).astype(int)
        sl_a, sl_b = [], []
        for k in range(3):
            if si[k] >= 0:
                sl_a.append(slice(si[k], n)); sl_b.append(slice(0, n - si[k]))
            else:
                sl_a.append(slice(0, n + si[k])); sl_b.append(slice(-si[k], n))
        aa, bb = A[tuple(sl_a)], Bz[tuple(sl_b)]
        aa = aa - aa.mean(); bb = bb - bb.mean()
        den = np.sqrt((aa * aa).sum() * (bb * bb).sum())
        r = float((aa * bb).sum() / den) if den > 0 else 0.0
        # measured 9um centre = predicted + shift (shift is in level-L9 voxels)
        meas = p9 + shift * S9
        rows.append({"p2": [z2c, float(y2c), float(x2c)], "pred9": p9.tolist(),
                     "shift_lvl2": shift.tolist(), "shift_lvl0_9um": (shift * S9).tolist(),
                     "meas9": meas.tolist(), "ncc": r, "n": int(n),
                     "occ2": float((Braw > 0).mean()), "occ9": float((A > 0).mean())})
        print(f"z2={z2c:6d} -> shift(lvl2 vox)={np.round(shift,2)}  "
              f"= {np.round(shift*S9*V9*1e3,1)} um  ncc={r:.3f}", flush=True)

    good = [r for r in rows if r["ncc"] > 0.4]
    print(f"\n{len(good)}/{len(rows)} blocks with ncc>0.4")
    if len(good) >= 4:
        P2m = np.array([g["p2"] for g in good])
        P9m = np.array([g["meas9"] for g in good])
        # fit p9 = s*p2 + t  (single isotropic scale + translation)
        Adm = np.concatenate([P2m.reshape(-1, 1), np.tile(np.eye(3), (len(good), 1))], axis=1)
        # build design: for each point/axis, row = [p2_axis, e_axis]
        rowsd, rhs = [], []
        for g in good:
            for k in range(3):
                e = [0, 0, 0]; e[k] = 1
                rowsd.append([g["p2"][k]] + e)
                rhs.append(g["meas9"][k])
        Adm = np.array(rowsd); rhs = np.array(rhs)
        sol, *_ = np.linalg.lstsq(Adm, rhs, rcond=None)
        s, t = sol[0], sol[1:]
        pred = Adm @ sol
        res = (rhs - pred).reshape(-1, 3)
        print(f"fitted   scale 1/{1/s:.5f} (nominal 1/{RATIO:.5f})   t = {np.round(t,1)} (9um vox)")
        print(f"residual per axis (9um vox): rms {np.round(res.std(0),2)}  max {np.round(np.abs(res).max(0),2)}")
        print(f"residual in um: rms {np.round(res.std(0)*V9*1e3,1)}  "
              f"= {np.round(res.std(0)*V9/V2,1)} voxels at 2.4um")
        # translation-only model with nominal scale
        tt = np.array([np.mean([g["meas9"][k] - g["p2"][k] / RATIO for g in good]) for k in range(3)])
        rt = np.array([[g["meas9"][k] - (g["p2"][k] / RATIO + tt[k]) for k in range(3)] for g in good])
        print(f"translation-only (nominal scale): t={np.round(tt,1)}  "
              f"rms residual {np.round(rt.std(0),2)} 9um vox = {np.round(rt.std(0)*V9*1e3,1)} um")
        json.dump({"blocks": rows, "fit_scale": float(s), "fit_t": t.tolist(),
                   "fit_res_rms_9umvox": res.std(0).tolist(),
                   "trans_only_t": tt.tolist(),
                   "trans_only_res_rms_9umvox": rt.std(0).tolist()},
                  open(os.path.join(OUT, "refine_transform.json"), "w"), indent=1)
    else:
        json.dump({"blocks": rows}, open(os.path.join(OUT, "refine_transform.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
