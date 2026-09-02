"""Experiment B scoring: real AUC vs 40 rigid-translation matched nulls, all
inside (mask==1 AND label==0); per-cell gap + pre-registered verdicts.
Primary inputs win1/win2/win3 (overall verdict over these three ONLY);
frag1 included iff var/secondary_ok exists -- flagged secondary, reported
identically, never verdict-bearing."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR = os.environ["VAR"]
NATIVE = os.environ.get("NATIVE_WINDOWS", "1") == "1"
SEC_OK = os.path.exists(os.path.join(VAR, "secondary_ok"))
IDX = {"win1": 0, "win2": 1, "win3": 2, "frag1": 3}   # fixed seed offsets

def load_input(name):
    if name == "frag1":
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        d = os.path.join(cl.DATA, "frag1")
        ink = np.array(Image.open(os.path.join(d, "inklabels.png"))) > 0
        msk = np.array(Image.open(os.path.join(d, "mask.png"))) > 0
        pitch = cl.NATIVE_PITCH
    else:
        ink = np.load(os.path.join(cl.DATA, f"{name}_ink.npy"))
        msk = np.load(os.path.join(cl.DATA, f"{name}_mask.npy"))
        pitch = cl.P2A_PITCH
    shapes = ink & msk           # positives; win3 out-of-mask ink excluded here
    blank = msk & ~ink           # blank = (mask==1 AND label==0), ALWAYS
    return shapes, blank, pitch

def cell_maps(name):
    cells = []
    if name == "frag1":
        cells.append(("s936", "forward",
                      os.path.join(cl.PREDS, "sec_frag1_936.tif")))
        cells.append(("s936", "reverse",
                      os.path.join(cl.PREDS, "sec_frag1_936_reverse.tif")))
    elif name == "win1":
        # win1 9.36um cells REUSE the EXP A baseline maps (pre-registered)
        cells.append(("s936", "forward", os.path.join(cl.PREDS, "expA_base.tif")))
        cells.append(("s936", "reverse",
                      os.path.join(cl.PREDS, "expA_base_reverse.tif")))
        if NATIVE:
            cells.append(("native", "forward",
                          os.path.join(cl.PREDS, "expB_win1_native.tif")))
            cells.append(("native", "reverse",
                          os.path.join(cl.PREDS, "expB_win1_native_reverse.tif")))
    else:
        for sc in (["s936", "native"] if NATIVE else ["s936"]):
            base = {"s936": f"expB_{name}_s936",
                    "native": f"expB_{name}_native"}[sc]
            cells.append((sc, "forward", os.path.join(cl.PREDS, base + ".tif")))
            cells.append((sc, "reverse",
                          os.path.join(cl.PREDS, base + "_reverse.tif")))
    return cells

def score_cell(tif, shapes, blank, offsets):
    import tifffile
    pred = tifffile.imread(tif)
    if pred.shape != shapes.shape:
        pred = cl.upsample_pred(pred, shapes.shape)
    q = cl.quantize_map(pred)
    h_shapes = cl.masked_hist(q, shapes)
    h_blank = cl.masked_hist(q, blank)
    auc_real = cl.hist_auc(h_shapes, h_blank)
    null_aucs = []
    for dy, dx in offsets:
        h_pos = cl.translated_hist(q, shapes, blank, dy, dx)
        h_neg = h_blank - h_pos
        null_aucs.append(cl.hist_auc(h_pos, h_neg))
    null_aucs = np.array(null_aucs)
    return auc_real, null_aucs, pred

def main():
    primary = ["win1", "win2", "win3"]
    inputs = primary + (["frag1"] if SEC_OK else [])
    excluded = {} if SEC_OK else {
        "frag1": "secondary arm disabled, absent, or failed "
                 "(non-fatal by pre-registration)"}
    results = {}
    for name in inputs:
        shapes, blank, pitch = load_input(name)
        offsets, min_px, max_px = cl.draw_translations(
            shapes, blank, pitch, seed=cl.SEED + IDX[name])
        cl.say(f"EXPB_SCORE {name}: 40 translations drawn "
               f"(|shift| in [{min_px},{max_px}] px @ {pitch}um, "
               f">= {cl.MIN_PSEUDO_POS} pseudo-pos each)")
        cells = {}
        for scale, direction, tif in cell_maps(name):
            auc_real, nulls, pred = score_cell(tif, shapes, blank, offsets)
            gap = float(auc_real - np.median(nulls))
            frac_hi = float((nulls >= cl.CONFOUND_MEDIAN_AUC).mean())
            cells[f"{scale}_{direction}"] = dict(
                scale=scale, direction=direction,
                auc_real=float(auc_real),
                null_median=float(np.median(nulls)),
                null_max=float(nulls.max()), null_min=float(nulls.min()),
                null_p95=float(np.percentile(nulls, 95)),
                null_spread_sd=float(nulls.std()),
                null_frac_ge_060=frac_hi,
                cell_confounded=bool(
                    np.median(nulls) >= cl.CONFOUND_MEDIAN_AUC
                    or frac_hi >= cl.CONFOUND_FRAC),
                gap=gap,
                beats_all_40=bool(auc_real > nulls.max()),
                rank_p=1.0 / (len(nulls) + 1),
                null_aucs=[float(a) for a in nulls])
            cl.say(f"EXPB_SCORE {name} {scale} {direction}: "
                   f"AUC_real={auc_real:.4f} null_med="
                   f"{np.median(nulls):.4f} null_max={nulls.max():.4f} "
                   f"GAP={gap:.4f}")
            cl.save_preview(pred, os.path.join(
                cl.OUT, "previews", f"expB_{name}_{scale}_{direction}.png"))
        prim = {k: v for k, v in cells.items() if v["scale"] == "s936"}
        aud_key = max(prim, key=lambda k: prim[k]["auc_real"])
        aud = cells[aud_key]
        confounded = bool(aud["cell_confounded"])
        genuine = bool(not confounded and aud["beats_all_40"]
                       and aud["gap"] >= cl.GAP_MIN)
        verdict = ("SHAPE_CONFOUNDED" if confounded
                   else "GENUINE" if genuine else "INDETERMINATE")
        results[name] = dict(
            pitch_um=pitch, audited_cell=aud_key, cells=cells,
            offsets=[[int(a), int(b)] for a, b in offsets],
            min_shift_px=min_px, max_shift_px=max_px,
            n_pos=int(shapes.sum()), n_blank=int(blank.sum()),
            secondary=bool(name == "frag1"),
            verdict=verdict, shape_confounded=confounded, genuine=genuine)
        tag = " [SECONDARY, non-verdict-bearing]" if name == "frag1" else ""
        cl.say(f"EXPB_SCORE {name} VERDICT={verdict}{tag} "
               f"(audited {aud_key}: real={aud['auc_real']:.4f} "
               f"null_med={aud['null_median']:.4f} gap={aud['gap']:.4f})")
    prim_res = {k: v for k, v in results.items() if k in primary}
    overall = ("SHAPE_CONFOUNDED"
               if any(r["shape_confounded"] for r in prim_res.values())
               else "GENUINE"
               if all(r["genuine"] for r in prim_res.values())
               else "MIXED")
    out = dict(prereg=dict(confound_median_auc=cl.CONFOUND_MEDIAN_AUC,
                           confound_frac=cl.CONFOUND_FRAC,
                           gap_min=cl.GAP_MIN, n_translations=cl.N_TRANSLATIONS,
                           min_shift_mm=cl.MIN_SHIFT_MM, seed=cl.SEED),
               inputs=results, excluded=excluded, overall=overall,
               overall_note=("overall verdict computed over win1/win2/win3 "
                             "only; frag1 (if present) is a flagged SECONDARY "
                             "input, never verdict-bearing"),
               native_note=("native-4.32um cells are sensitivity only: the "
                            "model's 17 slices span 73um there vs 159um "
                            "trained -- never verdict-bearing"))
    json.dump(out, open(os.path.join(cl.RESULTS,
                                     "expB_hallucination.json"), "w"), indent=1)
    cl.say(f"EXPB_SCORE complete; overall={overall} (primary windows only)")

if __name__ == "__main__":
    main()
