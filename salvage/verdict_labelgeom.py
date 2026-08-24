"""Measure w035 letter geometry from human labels: component heights/widths,
stroke width, and centroid-based row spacing. Calibrates the periodicity band."""
import sys
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.measure import regionprops

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\salvage")
from verdict_common import load_w035_label2d, save_json, SALVAGE

lab = load_w035_label2d()
lb, n = ndi.label(ndi.binary_closing(lab, np.ones((5, 5), bool)),
                  structure=np.ones((3, 3), int))
props = [p for p in regionprops(lb) if p.area >= 500]
h = np.array([p.bbox[2] - p.bbox[0] for p in props], float)
w = np.array([p.bbox[3] - p.bbox[1] for p in props], float)
areas = np.array([p.area for p in props], float)
cy = np.array([p.centroid[0] for p in props])
cx = np.array([p.centroid[1] for p in props])

skel = skeletonize(lab)
stroke_w = lab.sum() / max(skel.sum(), 1)

out = {
    "n_glyph_components": len(props),
    "height_px": {"p25": float(np.percentile(h, 25)), "p50": float(np.percentile(h, 50)),
                  "p75": float(np.percentile(h, 75))},
    "width_px": {"p50": float(np.percentile(w, 50))},
    "area_px": {"p50": float(np.percentile(areas, 50))},
    "stroke_width_px": float(stroke_w),
    "centroids_y": sorted(np.round(cy, 0).tolist()),
    "centroids_x": np.round(cx, 0).tolist(),
}
save_json(SALVAGE / "verdict_labelgeom.json", out)
import json
print(json.dumps(out, indent=1))
