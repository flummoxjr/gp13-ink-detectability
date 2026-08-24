"""Investigation D step 2: can we DERIVE the PHerc1203 9.362um <-> 2.403um transform?

Method: pull a central physical column from both volume pyramids at ~matched effective
voxel size, rescale the 2.4um column by 3.896 onto the 9um grid, and run a brute-force
normalized cross-correlation over (dz, dy, dx) -- both z-orientations -- to see whether a
unique, sharp peak exists.  The BM18 stage coordinates give the prior; this measures whether
image content confirms it.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zarr_http import Zarr3D  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
P9 = "PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr"
P2 = "PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
V9, V2 = 0.009362, 0.002403
RATIO = V9 / V2  # 3.8960


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def main():
    lvl = 5
    z9 = Zarr3D(P9, lvl)
    z2 = Zarr3D(P2, lvl)
    print("9um  lvl%d shape %s  voxel %.4f mm" % (lvl, z9.shape, V9 * 2 ** lvl))
    print("2.4um lvl%d shape %s  voxel %.4f mm" % (lvl, z2.shape, V2 * 2 ** lvl))

    t = time.time()
    # full 9um pyramid level (small)
    A = z9.read(0, z9.shape[0], 0, z9.shape[1], 0, z9.shape[2]).astype(np.float32)
    print("9um block", A.shape, "%.0fs" % (time.time() - t))

    # empty-slice check (metadata claims z_crop_start 760 / 24)
    nz9 = (A > 0).sum(axis=(1, 2))
    first9 = int(np.argmax(nz9 > 0))
    last9 = int(len(nz9) - 1 - np.argmax(nz9[::-1] > 0))
    print(f"9um occupied z (lvl5): {first9}..{last9}  -> lvl0 {first9*32}..{last9*32+31}")

    # central physical column of the 2.4um volume, ~19.7 mm wide
    cy = z2.shape[1] // 2
    cx = z2.shape[2] // 2
    half = 128
    t = time.time()
    Braw = z2.read(0, z2.shape[0], cy - half, cy + half, cx - half, cx + half).astype(np.float32)
    print("2.4um block", Braw.shape, "%.0fs" % (time.time() - t))
    nz2 = (Braw > 0).sum(axis=(1, 2))
    f2 = int(np.argmax(nz2 > 0))
    l2 = int(len(nz2) - 1 - np.argmax(nz2[::-1] > 0))
    print(f"2.4um occupied z in column (lvl5): {f2}..{l2} -> lvl0 {f2*32}..{l2*32+31}")

    # rescale 2.4um column onto the 9um grid
    B = zoom(Braw, 1.0 / RATIO, order=1)
    print("2.4um rescaled to 9um grid:", B.shape)

    # matching lateral window in the 9um volume (same physical center)
    c9y, c9x = z9.shape[1] // 2, z9.shape[2] // 2
    hy = B.shape[1] // 2
    hx = B.shape[2] // 2
    pad = 8
    results = []
    for flip in (False, True):
        Bf = B[::-1] if flip else B
        # subsample the search block in z to keep the brute force cheap
        nzB = Bf.shape[0]
        best = None
        for dz in range(0, A.shape[0] - nzB + 1):
            sub = A[dz:dz + nzB, c9y - hy:c9y - hy + Bf.shape[1], c9x - hx:c9x - hx + Bf.shape[2]]
            if sub.shape != Bf.shape:
                continue
            r = ncc(sub, Bf)
            results.append((flip, dz, 0, 0, r))
            if best is None or r > best[-1]:
                best = (flip, dz, 0, 0, r)
        print(f"flip={flip}: best coarse dz={best[1]} (lvl0 z={best[1]*32}) ncc={best[4]:.4f}")

    arr = np.array([(int(f), dz, r) for f, dz, _, _, r in results],
                   dtype=[("flip", "i4"), ("dz", "i4"), ("ncc", "f8")])
    np.save(os.path.join(OUT, "coarse_ncc.npy"), arr)

    # refine best hypothesis over dy/dx too
    bf = max(results, key=lambda t: t[-1])
    flip, dz0 = bf[0], bf[1]
    Bf = B[::-1] if flip else B
    ref = []
    for dz in range(max(0, dz0 - 6), min(A.shape[0] - Bf.shape[0], dz0 + 6) + 1):
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                y0, x0 = c9y - hy + dy, c9x - hx + dx
                sub = A[dz:dz + Bf.shape[0], y0:y0 + Bf.shape[1], x0:x0 + Bf.shape[2]]
                if sub.shape != Bf.shape:
                    continue
                ref.append((dz, dy, dx, ncc(sub, Bf)))
    ref.sort(key=lambda t: -t[-1])
    print("top refined (dz,dy,dx,ncc) at lvl5:")
    for r in ref[:8]:
        print("   ", r)

    top = ref[0]
    # 2nd-best peak far from the top, to judge uniqueness
    far = [r for r in ref if abs(r[0] - top[0]) > 2 or abs(r[1] - top[1]) > 3 or abs(r[2] - top[2]) > 3]
    summary = {
        "level": lvl,
        "voxel_mm_lvl": V9 * 2 ** lvl,
        "flip": bool(flip),
        "best_lvl5": {"dz": top[0], "dy": top[1], "dx": top[2], "ncc": top[3]},
        "best_lvl0_zoffset": top[0] * 32,
        "runner_up_far_ncc": far[0][3] if far else None,
        "occupied_z_9um_lvl0": [first9 * 32, last9 * 32 + 31],
        "occupied_z_2um_column_lvl0": [f2 * 32, l2 * 32 + 31],
        "note": "dz is the 9um lvl5 index where the (rescaled) 2.4um column starts",
    }
    json.dump(summary, open(os.path.join(OUT, "derive_transform.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
