"""Scoring for the Bet B C2a pod (all maps are uint8 tiled TIFFs from optimized_inference).

  python c2a_score.py ctl    <ours.tif> <published.tif> <mesh dir> <tag>
      -> results/ctl_<tag>.json: Pearson r (ds4, joint support) inside the rendered box against the
         published w035 map; forward pixel AUC on the human labels of the sub-crop; blank p99 (the
         tripwire reference); tripwire components and how many touch ink (the tripwire's own control).
         exit 0 iff r >= 0.90, else 21 (the stage sweeps the window on 21).
  python c2a_score.py ctlrev <fwd.tif> <rev.tif> <mesh dir> <tag>
      -> results/ctl_<tag>_rev.json: reverse-order AUC + fwd/rev r.
  python c2a_score.py tiles  <fwd.tif> <rev.tif> <mesh dir> <blank_p99> <name>
      -> results/tiles_<name>.json: per-tile statistics for sets H and L.
Tripwire (pre-registered, physical units from the 0358 screen at 9.362 um: 1e4 px / 30 px):
value > blank_p99, component area >= 0.876 mm^2, bbox width >= 0.28 mm.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl  # noqa: E402

TRIP_AREA_MM2 = 0.876
TRIP_WIDTH_MM = 0.28


def load_map(path):
    os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2 ** 40))
    import tifffile
    a = tifffile.imread(path)
    if a.ndim == 3:
        a = a[..., 0]
    return a


def block_mean(a, f):
    H, W = a.shape[0] // f * f, a.shape[1] // f * f
    out = np.empty((H // f, W // f), np.float32)
    rows = max(f, (4096 // f) * f)
    for y in range(0, H, rows):
        s = a[y:min(y + rows, H), :W]
        h = s.shape[0] // f * f
        acc = np.uint32 if np.issubdtype(s.dtype, np.integer) else np.float64
        blk = s[:h].reshape(h // f, f, W // f, f).astype(acc).sum(axis=(1, 3))
        out[y // f:(y + h) // f] = blk / float(f * f)
    return out


def pearson_joint(a4, b4):
    m = (a4 > 0) & (b4 > 0)
    if m.sum() < 1000:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(a4[m], b4[m])[0, 1]), int(m.sum())


def region(a, box):
    R0, R1, C0, C1 = box
    R0, C0 = max(0, R0), max(0, C0)
    return a[R0:min(a.shape[0], R1), C0:min(a.shape[1], C1)]


def tripwire(m, thr, px_um, ink_mask=None):
    """Components of (m > thr) meeting the physical area/width thresholds."""
    hot = m > thr
    if not hot.any():
        return dict(n=0, n_on_ink=0, components=[])
    lab, n = ndimage.label(hot)
    if n == 0:
        return dict(n=0, n_on_ink=0, components=[])
    area_min = TRIP_AREA_MM2 * 1e6 / (px_um ** 2)
    width_min = TRIP_WIDTH_MM * 1e3 / px_um
    sizes = ndimage.sum(hot, lab, index=np.arange(1, n + 1))
    objs = ndimage.find_objects(lab)
    comps = []
    for k, (sz, sl) in enumerate(zip(sizes, objs), start=1):
        if sz < area_min or sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) < width_min:
            continue
        on_ink = bool(ink_mask is not None and ink_mask[sl][lab[sl] == k].any())
        comps.append(dict(area_px=int(sz), area_mm2=float(sz * px_um ** 2 / 1e6), bbox=[int(sl[0].start), int(sl[0].stop), int(sl[1].start), int(sl[1].stop)],
                          max=int(m[sl][lab[sl] == k].max()), on_ink=on_ink))
    comps.sort(key=lambda c: -c["area_px"])
    return dict(n=len(comps), n_on_ink=int(sum(c["on_ink"] for c in comps)), components=comps[:50])


def ctl_labels_sub(geom):
    """pos/neg masks of the rendered sub-crop on the 9.362 um label grid."""
    pos, neg = cl.load_ctl_labels()                     # CTL_CROP grid (2432 x 2688)
    y0, y1, x0, x1 = geom["label_crop"]                  # sub-crop offsets inside CTL_CROP
    return pos[y0:y1, x0:x1], neg[y0:y1, x0:x1]


def auc_on_labels(map_region, pos, neg):
    q = cl.quantize_map(cl.resample_pred(map_region.astype(np.float32), pos.shape))
    return cl.hist_auc(cl.masked_hist(q, pos), cl.masked_hist(q, neg))


def nearest_mask(mask, shape):
    zi = np.minimum((np.arange(shape[0]) * mask.shape[0] / shape[0]).astype(int), mask.shape[0] - 1)
    zj = np.minimum((np.arange(shape[1]) * mask.shape[1] / shape[1]).astype(int), mask.shape[1] - 1)
    return mask[zi][:, zj]


def ctl(ours_p, ref_p, mesh_dir, tag):
    geom = json.load(open(os.path.join(mesh_dir, "geom.json")))
    a, b = load_map(ours_p), load_map(ref_p)
    ra, rb = region(a, geom["box_px"]), region(b, geom["box_px"])
    h, w = min(ra.shape[0], rb.shape[0]), min(ra.shape[1], rb.shape[1])
    ra, rb = ra[:h, :w], rb[:h, :w]
    r, n = pearson_joint(block_mean(ra, 4), block_mean(rb, 4))
    pos, neg = ctl_labels_sub(geom)
    auc = auc_on_labels(ra, pos, neg)
    auc_ref = auc_on_labels(rb, pos, neg)
    negf, posf = nearest_mask(neg, ra.shape), nearest_mask(pos, ra.shape)
    blank_vals = ra[negf & (ra > 0)]
    blank_p99 = float(np.percentile(blank_vals, 99)) if blank_vals.size else None
    tw = tripwire(ra, blank_p99, geom["px_um"], posf) if blank_p99 is not None else dict(n=0, n_on_ink=0, components=[])
    res = dict(tag=tag, box_px=geom["box_px"], ours_shape=list(a.shape), ref_shape=list(b.shape), r_ds4_joint=r, n_joint_ds4=n,
               ours_nonzero_in_box=float((ra > 0).mean()), ref_nonzero_in_box=float((rb > 0).mean()),
               auc_fwd=auc, auc_ref_published=auc_ref, n_pos=int(pos.sum()), n_neg=int(neg.sum()),
               blank_p99=blank_p99, ink_p50=float(np.percentile(ra[posf & (ra > 0)], 50)) if (posf & (ra > 0)).any() else None,
               tripwire_components=tw["n"], tripwire_on_ink=tw["n_on_ink"], tripwire=tw, gate_r=0.90, passed=bool(r >= 0.90))
    json.dump(res, open(os.path.join(cl.RESULTS, f"ctl_{tag}.json"), "w"), indent=1)
    np.save(os.path.join(cl.OUT, "maps", f"{tag}_box_ds4.npy"), np.clip(np.rint(block_mean(ra, 4)), 0, 255).astype(np.uint8))
    if not os.path.exists(os.path.join(cl.OUT, "maps", "ctl_reference_box_ds4.npy")):
        np.save(os.path.join(cl.OUT, "maps", "ctl_reference_box_ds4.npy"), np.clip(np.rint(block_mean(rb, 4)), 0, 255).astype(np.uint8))
        cl.save_preview(rb, os.path.join(cl.OUT, "previews", "ctl_reference_box.png"), ds=8)
    cl.save_preview(ra, os.path.join(cl.OUT, "previews", f"{tag}_box.png"), ds=8)
    cl.say(f"CTL {tag}: r_ds4(joint)={r:.4f} on {n} px; AUC fwd {auc:.4f} (published map on the same labels {auc_ref:.4f}); "
           f"blank p99 {blank_p99}; tripwire {tw['n']} components, {tw['n_on_ink']} on ink -> {'PASS' if res['passed'] else 'below 0.90'}")
    sys.exit(0 if res["passed"] else 21)


def ctlrev(fwd_p, rev_p, mesh_dir, tag):
    geom = json.load(open(os.path.join(mesh_dir, "geom.json")))
    a, b = load_map(fwd_p), load_map(rev_p)
    ra, rb = region(a, geom["box_px"]), region(b, geom["box_px"])
    pos, neg = ctl_labels_sub(geom)
    auc_rev = auc_on_labels(rb, pos, neg)
    r, n = pearson_joint(block_mean(ra, 4), block_mean(rb, 4))
    negf = nearest_mask(neg, rb.shape)
    bv = rb[negf & (rb > 0)]
    bp = float(np.percentile(bv, 99)) if bv.size else None
    tw = tripwire(rb, bp, geom["px_um"], nearest_mask(pos, rb.shape)) if bp is not None else dict(n=0, n_on_ink=0)
    res = dict(tag=tag, auc_rev=auc_rev, fwdrev_r_ds4=r, n_joint=n, rev_blank_p99=bp, rev_tripwire_components=tw["n"], rev_tripwire_on_ink=tw["n_on_ink"],
               depth_order_gate=dict(max_reverse=0.80, passed=bool(auc_rev <= 0.80)))
    json.dump(res, open(os.path.join(cl.RESULTS, f"ctl_{tag}_rev.json"), "w"), indent=1)
    np.save(os.path.join(cl.OUT, "maps", f"{tag}_rev_box_ds4.npy"), np.clip(np.rint(block_mean(rb, 4)), 0, 255).astype(np.uint8))
    cl.save_preview(rb, os.path.join(cl.OUT, "previews", f"{tag}_rev_box.png"), ds=8)
    cl.say(f"CTL {tag} reverse: AUC rev {auc_rev:.4f} (gate <= 0.80 {'passed' if auc_rev <= 0.80 else 'FAILED'}); fwd/rev r {r:.4f}; rev tripwire {tw['n']}")


def tiles(fwd_p, rev_p, mesh_dir, blank_p99, name):
    geom = json.load(open(os.path.join(mesh_dir, "geom.json")))
    bp = float(blank_p99)
    a, b = load_map(fwd_p), load_map(rev_p)
    rows = []
    for t in geom["tiles"]:
        ra, rb = region(a, t["box_px"]), region(b, t["box_px"])
        h, w = min(ra.shape[0], rb.shape[0]), min(ra.shape[1], rb.shape[1])
        ra, rb = ra[:h, :w], rb[:h, :w]
        va, vb = ra[ra > 0], rb[rb > 0]
        twa, twb = tripwire(ra, bp, geom["px_um"]), tripwire(rb, bp, geom["px_um"])
        r, n = pearson_joint(block_mean(ra, 4), block_mean(rb, 4)) if h >= 8 and w >= 8 else (float("nan"), 0)
        rows.append(dict(ij=t["ij"], set=t["set"], contrast=t["contrast"], box_px=t["box_px"], covered=float((ra > 0).mean()) if ra.size else 0.0,
                         fwd_p50=float(np.percentile(va, 50)) if va.size else None, fwd_p99=float(np.percentile(va, 99)) if va.size else None,
                         rev_p50=float(np.percentile(vb, 50)) if vb.size else None, rev_p99=float(np.percentile(vb, 99)) if vb.size else None,
                         frac_fwd=float((va > bp).mean()) if va.size else None, frac_rev=float((vb > bp).mean()) if vb.size else None,
                         trip_fwd=twa["n"], trip_rev=twb["n"], trip_fwd_max_mm2=(twa["components"][0]["area_mm2"] if twa["n"] else 0.0),
                         fwdrev_r=r, n_joint=n))
    def summ(rows_, s):
        rr = [x for x in rows_ if x["set"] == s and x["covered"] > 0.5]
        if not rr:
            return dict(n=0)
        ff = np.array([x["frac_fwd"] for x in rr], float); fr = np.array([x["frac_rev"] for x in rr], float)
        return dict(n=len(rr), trip_fwd_tiles=int(sum(x["trip_fwd"] > 0 for x in rr)), trip_rev_tiles=int(sum(x["trip_rev"] > 0 for x in rr)),
                    frac_fwd_mean=float(ff.mean()), frac_rev_mean=float(fr.mean()), frac_fwd_max=float(ff.max()),
                    fwd_p99_median=float(np.median([x["fwd_p99"] for x in rr])), rev_p99_median=float(np.median([x["rev_p99"] for x in rr])),
                    fwdrev_r_median=float(np.nanmedian([x["fwdrev_r"] for x in rr])),
                    escalation_candidates=[x["ij"] for x in rr if x["trip_fwd"] > 0 and x["trip_rev"] == 0 and (x["frac_rev"] or 0) * 3 < (x["frac_fwd"] or 0)])
    res = dict(name=name, px_um=geom["px_um"], blank_p99=bp, tiles=rows, summary=dict(H=summ(rows, "H"), L=summ(rows, "L")))
    json.dump(res, open(os.path.join(cl.RESULTS, f"tiles_{name}.json"), "w"), indent=1)
    np.save(os.path.join(cl.OUT, "maps", f"{name}_fwd_ds4.npy"), np.clip(np.rint(block_mean(a, 4)), 0, 255).astype(np.uint8))
    np.save(os.path.join(cl.OUT, "maps", f"{name}_rev_ds4.npy"), np.clip(np.rint(block_mean(b, 4)), 0, 255).astype(np.uint8))
    top = sorted([x for x in rows if x["covered"] > 0.5], key=lambda x: -(x["frac_fwd"] or 0))[:3]
    for x in top:
        cl.save_preview(region(a, x["box_px"]), os.path.join(cl.OUT, "previews", f"{name}_{x['set']}_{x['ij'][0]}_{x['ij'][1]}_fwd.png"), ds=2)
        cl.save_preview(region(b, x["box_px"]), os.path.join(cl.OUT, "previews", f"{name}_{x['set']}_{x['ij'][0]}_{x['ij'][1]}_rev.png"), ds=2)
    s = res["summary"]
    cl.say(f"TILES {name}: H n={s['H'].get('n')} trip fwd/rev {s['H'].get('trip_fwd_tiles')}/{s['H'].get('trip_rev_tiles')} frac fwd/rev {s['H'].get('frac_fwd_mean')}/{s['H'].get('frac_rev_mean')} "
           f"| L n={s['L'].get('n')} trip {s['L'].get('trip_fwd_tiles')}/{s['L'].get('trip_rev_tiles')} frac {s['L'].get('frac_fwd_mean')}/{s['L'].get('frac_rev_mean')} "
           f"| escalation candidates H {s['H'].get('escalation_candidates')}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ctl":
        ctl(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "ctlrev":
        ctlrev(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "tiles":
        tiles(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        raise SystemExit(f"unknown command {cmd}")
