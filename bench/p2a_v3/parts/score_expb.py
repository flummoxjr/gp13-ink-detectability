"""Experiment B scoring: real AUC vs 40 rigid-translation matched nulls, all
inside (mask==1 AND label==0); per-cell gap + pre-registered verdicts.
Inputs win1/win2/win3 at the corrected 2.215um pitch. Cells: iso fwd/rev
(primary; the audited cell is the iso direction with the higher real AUC)
and fit17 fwd/rev (reported, never verdict-bearing)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR = os.environ["VAR"]
IDX = {"win1": 0, "win2": 1, "win3": 2}   # fixed seed offsets

def load_input(name):
    ink = np.load(os.path.join(cl.DATA, f"{name}_ink.npy"))
    msk = np.load(os.path.join(cl.DATA, f"{name}_mask.npy"))
    shapes = ink & msk           # positives; out-of-mask ink excluded here
    blank = msk & ~ink           # blank = (mask==1 AND label==0), ALWAYS
    return shapes, blank, cl.P2A_PITCH

def cell_maps(name):
    cells = []
    for mode in cl.DEPTH_MODES:
        if name == cl.EXPA_ANCHOR:
            stem = "expA_base" if mode == cl.PRIMARY_DEPTH else f"expA_base_{mode}"
        else:
            stem = f"expB_{name}_{mode}"
        cells.append((mode, "forward", os.path.join(cl.PREDS, stem + ".tif")))
        cells.append((mode, "reverse",
                      os.path.join(cl.PREDS, stem + "_reverse.tif")))
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
    return auc_real, np.array(null_aucs), pred

def main():
    inputs = ["win1", "win2", "win3"]
    results = {}
    for name in inputs:
        shapes, blank, pitch = load_input(name)
        offsets, min_px, max_px = cl.draw_translations(
            shapes, blank, pitch, seed=cl.SEED + IDX[name])
        cl.say(f"EXPB_SCORE {name}: 40 translations drawn "
               f"(|shift| in [{min_px},{max_px}] px @ {pitch}um, "
               f">= {cl.MIN_PSEUDO_POS} pseudo-pos each)")
        cells = {}
        for mode, direction, tif in cell_maps(name):
            auc_real, nulls, pred = score_cell(tif, shapes, blank, offsets)
            gap = float(auc_real - np.median(nulls))
            frac_hi = float((nulls >= cl.CONFOUND_MEDIAN_AUC).mean())
            cells[f"{mode}_{direction}"] = dict(
                depth_mode=mode, direction=direction,
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
            cl.say(f"EXPB_SCORE {name} {mode} {direction}: "
                   f"AUC_real={auc_real:.4f} null_med="
                   f"{np.median(nulls):.4f} null_max={nulls.max():.4f} "
                   f"GAP={gap:.4f}")
            cl.save_preview(pred, os.path.join(
                cl.OUT, "previews", f"expB_{name}_{mode}_{direction}.png"))
        prim = {k: v for k, v in cells.items()
                if v["depth_mode"] == cl.PRIMARY_DEPTH}
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
            verdict=verdict, shape_confounded=confounded, genuine=genuine)
        cl.say(f"EXPB_SCORE {name} VERDICT={verdict} "
               f"(audited {aud_key}: real={aud['auc_real']:.4f} "
               f"null_med={aud['null_median']:.4f} gap={aud['gap']:.4f})")
    overall = ("SHAPE_CONFOUNDED"
               if any(r["shape_confounded"] for r in results.values())
               else "GENUINE"
               if all(r["genuine"] for r in results.values())
               else "MIXED")
    out = dict(prereg=dict(confound_median_auc=cl.CONFOUND_MEDIAN_AUC,
                           confound_frac=cl.CONFOUND_FRAC,
                           gap_min=cl.GAP_MIN, n_translations=cl.N_TRANSLATIONS,
                           min_shift_mm=cl.MIN_SHIFT_MM,
                           max_shift_frac=cl.MAX_SHIFT_FRAC, seed=cl.SEED),
               inputs=results, overall=overall,
               overall_note=("overall verdict over win1/win2/win3 iso cells; "
                             "fit17 cells are reported, never verdict-bearing"))
    json.dump(out, open(os.path.join(cl.RESULTS,
                                     "expB_hallucination.json"), "w"), indent=1)
    cl.say(f"EXPB_SCORE complete; overall={overall}")

if __name__ == "__main__":
    main()
