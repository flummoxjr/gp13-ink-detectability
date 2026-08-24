"""Component catalog of the w035 ink_9um prediction beyond the labeled/supervised
region. Calibrates the letter-class criterion on the human-labeled letters
(tripwire from salvage/ink9um_1203_verdict.md: value>195, area>1e4, width>30),
then extracts letter-class components outside supervision and cross-checks seed43.
Outputs: comb_catalog.json + _comp42.npz/_comp43.npz for downstream steps."""
import json
import numpy as np
import tifffile
from pathlib import Path
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.measure import regionprops

COMB = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\comb")
OUT_W035 = Path(r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\ink9um_w035")

MIN_AREA_TABLE = 3000      # keep in table for context
LETTER_AREA = 10_000       # tripwire
LETTER_WIDTH = 30.0        # tripwire (area / skeleton length)
LETTER_VAL = 195           # control blank p99, component p90 must reach it
EXCL_DILATE = 25           # px dilation of labels+sup for the exclusion zone


def valid_mask(arr, erode=40):
    m = arr > 0
    m = ndi.binary_closing(m, structure=np.ones((5, 5), bool))
    m = ndi.binary_fill_holes(m)
    if erode:
        m = ndi.binary_erosion(m, structure=np.ones((3, 3), bool),
                               iterations=erode)
    return m


def components(pred, vmask, thr):
    binm = (pred >= thr) & vmask
    lb, n = ndi.label(binm, structure=np.ones((3, 3), int))
    skel = skeletonize(binm)
    skel_counts = np.bincount(lb[skel], minlength=n + 1)
    rows = []
    for p in regionprops(lb, intensity_image=pred):
        if p.area < MIN_AREA_TABLE:
            continue
        vals = p.image_intensity[p.image]
        sk = int(skel_counts[p.label])
        width = p.area / max(sk, 1)
        minor = p.axis_minor_length if p.axis_minor_length > 0 else 1.0
        rows.append({
            "id": int(p.label),
            "area": int(p.area),
            "bbox": [int(v) for v in p.bbox],
            "cy": float(p.centroid[0]), "cx": float(p.centroid[1]),
            "width": float(width),
            "elong": float(p.axis_major_length / minor),
            "v_p50": float(np.percentile(vals, 50)),
            "v_p90": float(np.percentile(vals, 90)),
            "v_max": int(vals.max()),
            "frac195": float((vals >= LETTER_VAL).mean()),
        })
    return lb, rows


def overlap_frac(lb, comp_id, mask, bbox):
    y0, x0, y1, x1 = bbox
    sub = lb[y0:y1, x0:x1] == comp_id
    return float(mask[y0:y1, x0:x1][sub].mean())


def main():
    lab2d = np.load(COMB / "_lab2d.npy")
    sup2d = np.load(COMB / "_sup2d.npy")
    excl = ndi.binary_dilation(lab2d | sup2d,
                               structure=np.ones((3, 3), bool),
                               iterations=EXCL_DILATE)
    lab_d = ndi.binary_dilation(lab2d, structure=np.ones((3, 3), bool),
                                iterations=EXCL_DILATE)

    out = {"params": {"min_area_table": MIN_AREA_TABLE,
                      "letter_area": LETTER_AREA, "letter_width": LETTER_WIDTH,
                      "letter_val_p90": LETTER_VAL,
                      "excl_dilate_px": EXCL_DILATE,
                      "threshold": "per-map in-valid-mask p80 (verdict convention)"}}
    comps = {}
    labs = {}
    for seed in (42, 43):
        pred = tifffile.imread(str(OUT_W035 / f"w035_seed{seed}-075000.tif"))
        vm = valid_mask(pred)
        thr = float(np.percentile(pred[vm], 80))
        lb, rows = components(pred, vm, thr)
        for r in rows:
            r["f_label"] = overlap_frac(lb, r["id"], lab_d, r["bbox"])
            r["f_excl"] = overlap_frac(lb, r["id"], excl, r["bbox"])
            r["letter_class"] = bool(r["area"] >= LETTER_AREA
                                     and r["width"] >= LETTER_WIDTH
                                     and r["v_p90"] >= LETTER_VAL)
            r["zone"] = ("labeled" if r["f_label"] > 0.05
                         else "sup_region" if r["f_excl"] > 0.05
                         else "beyond")
        comps[seed] = rows
        labs[seed] = lb
        out[f"seed{seed}"] = {"threshold_p80": thr, "n_comps": len(rows)}
        np.savez_compressed(COMB / f"_comp{seed}.npz", lb=lb.astype(np.int32))

    # calibration check: do the labeled letters pass the letter-class gate?
    lab_comps = [r for r in comps[42] if r["zone"] == "labeled"]
    out["calibration"] = {
        "n_labeled_comps_s42": len(lab_comps),
        "n_labeled_pass_gate": sum(r["letter_class"] for r in lab_comps),
        "labeled_area_p50": float(np.median([r["area"] for r in lab_comps])) if lab_comps else None,
        "labeled_width_p50": float(np.median([r["width"] for r in lab_comps])) if lab_comps else None,
        "labeled_vp90_p50": float(np.median([r["v_p90"] for r in lab_comps])) if lab_comps else None,
    }

    # cross-seed confirmation for seed42 letter-class beyond candidates
    lb43 = labs[43]
    by_id43 = {r["id"]: r for r in comps[43]}
    cands = [r for r in comps[42] if r["letter_class"] and r["zone"] == "beyond"]
    lb42 = labs[42]
    for r in cands:
        y0, x0, y1, x1 = r["bbox"]
        a = lb42[y0:y1, x0:x1] == r["id"]
        b_ids = np.unique(lb43[y0:y1, x0:x1][a])
        b_ids = b_ids[b_ids > 0]
        best_iou, best_id = 0.0, None
        # IoU against the union of all overlapping seed43 comps (letters can
        # split/merge differently between seeds), plus best single comp
        union_b = np.zeros_like(a)
        for bid in b_ids:
            # full-comp mask needs the seed43 comp's own bbox
            rb = by_id43.get(int(bid))
            if rb is None:  # sub-table-size seed43 comp; use local footprint
                bmask_local = lb43[y0:y1, x0:x1] == bid
                union_b |= bmask_local
                inter = (a & bmask_local).sum()
                un = (a | bmask_local).sum()
                iou = inter / un
            else:
                by0, bx0, by1, bx1 = rb["bbox"]
                gy0, gx0 = min(y0, by0), min(x0, bx0)
                gy1, gx1 = max(y1, by1), max(x1, bx1)
                A = lb42[gy0:gy1, gx0:gx1] == r["id"]
                B = lb43[gy0:gy1, gx0:gx1] == bid
                iou = float((A & B).sum() / (A | B).sum())
            if iou > best_iou:
                best_iou, best_id = float(iou), int(bid)
        # union IoU inside a joint bbox
        if len(b_ids):
            gy0, gx0, gy1, gx1 = y0, x0, y1, x1
            for bid in b_ids:
                rb = by_id43.get(int(bid))
                if rb is not None:
                    gy0 = min(gy0, rb["bbox"][0]); gx0 = min(gx0, rb["bbox"][1])
                    gy1 = max(gy1, rb["bbox"][2]); gx1 = max(gx1, rb["bbox"][3])
            A = lb42[gy0:gy1, gx0:gx1] == r["id"]
            B = np.isin(lb43[gy0:gy1, gx0:gx1], b_ids)
            union_iou = float((A & B).sum() / (A | B).sum())
        else:
            union_iou = 0.0
        r["iou43_best"] = best_iou
        r["iou43_union"] = union_iou
        r["match43"] = best_id

    out["seed42"]["n_letter_class_total"] = sum(r["letter_class"] for r in comps[42])
    out["seed42"]["n_letter_class_beyond"] = len(cands)
    out["seed42"]["n_beyond_crossseed"] = sum(
        1 for r in cands if max(r["iou43_best"], r["iou43_union"]) > 0.3)
    out["components_s42"] = comps[42]
    out["components_s43_letter_class"] = [r for r in comps[43] if r["letter_class"]]
    out["seed43"]["n_letter_class_total"] = sum(r["letter_class"] for r in comps[43])
    out["seed43"]["n_letter_class_beyond"] = sum(
        1 for r in comps[43] if r["letter_class"] and r["zone"] == "beyond")

    def cnv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    (COMB / "comb_catalog.json").write_text(json.dumps(out, indent=1, default=cnv))

    print(json.dumps({k: v for k, v in out.items()
                      if k in ("params", "seed42", "seed43", "calibration")},
                     indent=1, default=cnv))
    print("\nbeyond letter-class candidates (s42):")
    for r in sorted(cands, key=lambda r: -r["area"]):
        print(f" id={r['id']:6d} area={r['area']:7d} w={r['width']:5.1f} "
              f"vp90={r['v_p90']:5.1f} f195={r['frac195']:.2f} "
              f"cy={r['cy']:6.0f} cx={r['cx']:6.0f} "
              f"iou43={max(r['iou43_best'], r['iou43_union']):.2f}")


if __name__ == "__main__":
    main()
