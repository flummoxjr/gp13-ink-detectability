"""Experiment A scoring on the 500p2a win1 anchor. argv[1]: baseline | rungs.
baseline: AUC fwd/rev on the native 4.32um grid, gate 0.85, pick direction
(exit 21 on gate failure -> KILLED_BASELINE, and that number IS the result).
rungs: AUC + retained + DETECTABLE per rung, curve JSON."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR = os.environ["VAR"]
W1 = cl.WINDOWS[cl.EXPA_ANCHOR]

def native_labels():
    ink = np.load(os.path.join(cl.DATA, f"{cl.EXPA_ANCHOR}_ink.npy"))
    msk = np.load(os.path.join(cl.DATA, f"{cl.EXPA_ANCHOR}_mask.npy"))
    pos, neg = ink & msk, msk & ~ink
    assert int(pos.sum()) == W1["ink_and_mask"], int(pos.sum())
    assert int(neg.sum()) == W1["annot_blank"], int(neg.sum())
    return pos, neg          # positives, negatives(blank); win1 469 oom-ink px
                             # excluded from BOTH classes by the & mask above

def map_auc(tif_path, pos, neg):
    import tifffile
    pred = tifffile.imread(tif_path)
    up = cl.upsample_pred(pred, pos.shape)
    q = cl.quantize_map(up)
    return cl.hist_auc(cl.masked_hist(q, pos), cl.masked_hist(q, neg)), pred

def baseline():
    pos, neg = native_labels()
    aucs = {}
    for d, p in (("forward", os.path.join(cl.PREDS, "expA_base.tif")),
                 ("reverse", os.path.join(cl.PREDS, "expA_base_reverse.tif"))):
        aucs[d], pred = map_auc(p, pos, neg)
        cl.save_preview(pred, os.path.join(cl.OUT, "previews",
                                           f"expA_base_{d}.png"), ds=2)
        cl.say(f"EXPA_BASELINE win1@500p2a AUC[{d}] = {aucs[d]:.4f} "
               f"(pos={int(pos.sum())} neg={int(neg.sum())})")
    best = "forward" if aucs["forward"] >= aucs["reverse"] else "reverse"
    res = dict(anchor=cl.EXPA_ANCHOR, auc_forward=aucs["forward"],
               auc_reverse=aucs["reverse"],
               direction=best, gate=cl.GATE_BASELINE_AUC,
               n_pos=int(pos.sum()), n_neg=int(neg.sum()),
               gate_passed=bool(aucs[best] >= cl.GATE_BASELINE_AUC))
    json.dump(res, open(os.path.join(cl.RESULTS, "expA_baseline.json"), "w"),
              indent=1)
    open(os.path.join(VAR, "direction.txt"), "w").write(best)
    if not res["gate_passed"]:
        cl.say(f"EXPA_BASELINE GATE FAILED: best AUC {aucs[best]:.4f} < "
               f"{cl.GATE_BASELINE_AUC} -- pre-registered verdict "
               f"KILLED_BASELINE: ink_9um does not transfer scroll-to-scroll "
               f"at curve-anchoring quality even in-modality on a clean "
               f"human-labelled surface; the whole run aborts and THIS NUMBER "
               f"IS THE RESULT (evidence in results/expA_baseline.json + "
               f"previews)")
        sys.exit(21)
    cl.say(f"EXPA_BASELINE GATE PASSED: AUC_base={aucs[best]:.4f} "
           f"direction={best} -- rungs will use {best}")

def rungs():
    pos, neg = native_labels()
    base = json.load(open(os.path.join(cl.RESULTS, "expA_baseline.json")))
    direction = base["direction"]
    auc_b = base[f"auc_{direction}"]
    rows = []
    def add(rung_id, family, x, tif, reused=False, note=None):
        auc = auc_b if reused else map_auc(tif, pos, neg)[0]
        retained = (auc - 0.5) / max(auc_b - 0.5, 1e-9)
        det = bool(auc >= cl.DETECT_AUC_MIN and retained >= cl.DETECT_RETAIN_MIN)
        row = dict(rung=rung_id, family=family, x=x, auc=auc,
                   retained=retained, detectable=det, reused=reused)
        if note:
            row["note"] = note
        rows.append(row)
        cl.say(f"EXPA_SCORE {rung_id:<12} AUC={auc:.4f} retained={retained:.3f}"
               f" DETECTABLE={det}{' (reused baseline)' if reused else ''}")
    add("pitch_4.32", "pitch", 4.32, None, reused=True,
        note="native acquisition == baseline")
    for Pu in cl.PITCHES:
        if abs(Pu - cl.MODEL_PITCH) < 1e-9:
            add(f"pitch_{Pu}", "pitch", Pu, None, reused=True,
                note="4.32->9.36 == baseline by construction")
        else:
            add(f"pitch_{Pu}", "pitch", Pu,
                os.path.join(cl.PREDS, f"expA_p{Pu}.tif"))
    for k in cl.NOISE_KS:
        add(f"noise_k{k}", "noise", k, os.path.join(cl.PREDS, f"expA_n{k}.tif"))
    add("bit8", "bitdepth", 8, None, reused=True,
        note="released volume is ALREADY uint8; the v1 uint16->uint8 rung is "
             "baseline by construction here, never re-measured")
    add("bit4", "bitdepth", 4, os.path.join(cl.PREDS, "expA_bit4.tif"))
    for sg in cl.BLUR_SIGMAS:
        add(f"blur_{sg}", "blur", sg,
            os.path.join(cl.PREDS, f"expA_blur{sg}.tif"))
    pit = [r for r in rows if r["family"] == "pitch" and r["detectable"]]
    limit = max((r["x"] for r in pit), default=None)
    sigma = json.load(open(os.path.join(cl.RESULTS, "expA_sigma.json")))
    out = dict(anchor=cl.EXPA_ANCHOR, baseline=base, sigma_plate=sigma,
               rungs=rows, detectability_limit_um=limit,
               rule=dict(auc_min=cl.DETECT_AUC_MIN,
                         retain_min=cl.DETECT_RETAIN_MIN))
    json.dump(out, open(os.path.join(cl.RESULTS, "expA_curve.json"), "w"),
              indent=1)
    cl.say(f"EXPA_SCORE curve complete; detectability limit = {limit} um "
           f"(coarsest DETECTABLE pitch)")

if __name__ == "__main__":
    {"baseline": baseline, "rungs": rungs}[sys.argv[1]]()
