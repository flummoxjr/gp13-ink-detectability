"""Shared utilities for the ink9um 1203 verdict analysis."""
import json
import numpy as np
import tifffile
from pathlib import Path
from scipy import ndimage as ndi

ROOT = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD")
OUT_1203 = ROOT / "out" / "ink9um_1203"
OUT_W035 = ROOT / "out" / "ink9um_w035"
SALVAGE = ROOT / "salvage"
DDATA = Path(r"D:\vesuvius-data\trackD")

SEG_A = "auto_grown_20251005230830031"
SEG_B = "auto_grown_20251005231446965"

MAPS = {
    # 1203 maps
    "1203A_s42":  OUT_1203 / f"{SEG_A}_hybrid_3d2d-seed42_step-075000.tif",
    "1203A_s42r": OUT_1203 / f"{SEG_A}_hybrid_3d2d-seed42_step-075000_reverse.tif",
    "1203A_s43":  OUT_1203 / f"{SEG_A}_hybrid_3d2d-seed43_step-075000.tif",
    "1203A_s43r": OUT_1203 / f"{SEG_A}_hybrid_3d2d-seed43_step-075000_reverse.tif",
    "1203B_s42":  OUT_1203 / f"{SEG_B}_hybrid_3d2d-seed42_step-075000.tif",
    "1203B_s42r": OUT_1203 / f"{SEG_B}_hybrid_3d2d-seed42_step-075000_reverse.tif",
    "1203B_s43":  OUT_1203 / f"{SEG_B}_hybrid_3d2d-seed43_step-075000.tif",
    "1203B_s43r": OUT_1203 / f"{SEG_B}_hybrid_3d2d-seed43_step-075000_reverse.tif",
    # control maps
    "w035_s42":   OUT_W035 / "w035_seed42-075000.tif",
    "w035_s42r":  OUT_W035 / "w035_seed42-075000_reverse.tif",
    "w035_s43":   OUT_W035 / "w035_seed43-075000.tif",
    "w035_s43r":  OUT_W035 / "w035_seed43-075000_reverse.tif",
    "w035_rend42": OUT_W035 / "w035_rendered_seed42-075000.tif",
}


def load_map(key):
    arr = tifffile.imread(str(MAPS[key]))
    return np.asarray(arr)


def valid_mask(arr, erode=8):
    """Valid-region mask: fill holes of nonzero support, then erode a bit
    so rotated-diamond edges don't pollute statistics."""
    m = arr > 0
    m = ndi.binary_closing(m, structure=np.ones((5, 5), bool))
    m = ndi.binary_fill_holes(m)
    if erode:
        m = ndi.binary_erosion(m, structure=np.ones((3, 3), bool), iterations=erode)
    return m


def load_w035_label2d(shape=None):
    """Human ink labels for w035, max over z -> 2D binary."""
    lab = np.load(DDATA / "w035_ink.npy", mmap_mode="r")  # (28, 5820, 5240)
    lab2d = np.zeros(lab.shape[1:], dtype=np.uint8)
    for z in range(lab.shape[0]):
        np.maximum(lab2d, np.asarray(lab[z]), out=lab2d)
    lab2d = (lab2d > 0)
    if shape is not None:
        lab2d = lab2d[: shape[0], : shape[1]]
    return lab2d


def downsample(arr, f):
    """Block-mean downsample by integer factor f (crops remainder)."""
    a = arr[: arr.shape[0] // f * f, : arr.shape[1] // f * f].astype(np.float32)
    return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).mean(axis=(1, 3))


def save_json(path, obj):
    def cnv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=cnv))
