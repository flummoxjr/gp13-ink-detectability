"""Build a small zarr surface-volume crop of w035 around the supervised (labelled)
region so the REAL ink_9um model can be swept over depth windows on the laptop 4090.

Output: D:\vesuvius-data\trackD\w035_crop.sv.zarr  (28, h, w) uint8
        D:\vesuvius-data\trackD\w035_crop_labels.npz  (ink2d, sup2d, valid)
"""
import os

import numpy as np
from numcodecs import Blosc

from vesuvius.label_zarr import open_v2_group, create_v2_array

CACHE = r"D:\vesuvius-data\trackD"
OUTZ = os.path.join(CACHE, "w035_crop.sv.zarr")
LABEL_SLICE = 14
MARGIN = 128

surf = np.load(os.path.join(CACHE, "w035_surf.npy"))
ink = np.load(os.path.join(CACHE, "w035_ink.npy"), mmap_mode="r")[LABEL_SLICE] > 0
sup = np.load(os.path.join(CACHE, "w035_sup.npy"), mmap_mode="r")[LABEL_SLICE] > 0
D, H, W = surf.shape
ys, xs = np.nonzero(sup)
y0 = max(0, int(ys.min()) - MARGIN); y1 = min(H, int(ys.max()) + 1 + MARGIN)
x0 = max(0, int(xs.min()) - MARGIN); x1 = min(W, int(xs.max()) + 1 + MARGIN)
print(f"crop y {y0}:{y1}  x {x0}:{x1}  -> {(D, y1-y0, x1-x0)}")

crop = np.ascontiguousarray(surf[:, y0:y1, x0:x1])
ink_c = np.ascontiguousarray(ink[y0:y1, x0:x1])
sup_c = np.ascontiguousarray(sup[y0:y1, x0:x1])
valid = sup_c & ((crop > 0).mean(axis=0) > 0.9)
print(f"ink px {int((ink_c & valid).sum())}  bg px {int((valid & ~ink_c).sum())}")

if os.path.exists(OUTZ):
    import shutil
    shutil.rmtree(OUTZ)
g = open_v2_group(OUTZ)
comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
a = create_v2_array(g, "0", shape=crop.shape, chunks=(D, 128, 128), dtype=np.uint8,
                    compressor=comp, fill_value=0)
a[:] = crop

p = 8
h, w = crop.shape[1], crop.shape[2]
ph, pw = (-h) % p, (-w) % p
blk = np.pad(crop, ((0, 0), (0, ph), (0, pw)))
pooled = blk.reshape(D, (h + ph) // p, p, (w + pw) // p, p).max(axis=(2, 4))
occ = create_v2_array(g, "3", shape=pooled.shape, chunks=(D, 256, 256), dtype=np.uint8,
                      compressor=comp, fill_value=0)
occ[:] = pooled

np.savez_compressed(os.path.join(CACHE, "w035_crop_labels.npz"),
                    ink=ink_c, sup=sup_c, valid=valid, bbox=np.array([y0, y1, x0, x1]))
print("wrote", OUTZ)
