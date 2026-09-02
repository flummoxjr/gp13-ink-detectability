"""Experiment A scoring on the 500p2a win1 anchor at the CORRECTED pitch.
argv[1]: baseline | rungs.
baseline: AUC fwd/rev for BOTH depth modes on the native 2.215um grid; the
corrected anchor = max(fwd, rev) of the iso mode; curve gate 0.85 -> exit 21
when the gate fails (the caller records it and CONTINUES: the anchor is the
result, only the rungs are skipped). rungs: AUC + retained + DETECTABLE."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR = os.environ["VAR"]
A = cl.EXPA_ANCHOR
W1 = cl.WINDOWS[A]

def native_labels():
    ink = np.load(os.path.join(cl.DATA, f"{A}_ink.npy"))
    msk = np.load(os.path.join(cl.DATA, f"{A}_mask.npy"))
    pos, neg = ink & msk, msk & ~ink
    assert int(pos.sum()) == W1["ink_and_mask"], int(pos.sum())
    assert int(neg.sum()) == W1["annot_blank"], int(neg.sum())
    return pos, neg

def map_auc(tif_path, pos, neg):
    import tifffile
    pred = tifffile.imread(tif_path)
    up = cl.upsample_pred(pred, pos.shape)
    q = cl.quantize_map(up)
    return cl.hist_auc(cl.masked_hist(q, pos), cl.masked_hist(q, neg)), pred

def baseline():
    pos, neg = native_labels()
    modes = {}
    for mode in cl.DEPTH_MODES:
        stem = "expA_base" if mode == cl.PRIMARY_DEPTH else f"expA_base_{mode}"
        aucs = {}
        for d, suffix in (("forward", ""), ("reverse", "_reverse")):
            p = os.path.join(cl.PREDS, f"{stem}{suffix}.tif")
            aucs[d], pred = map_auc(p, pos, neg)
            cl.save_preview(pred, os.path.join(cl.OUT, "previews",
                                               f"expA_base_{mode}_{d}.png"), ds=2)
            cl.say(f"EXPA_BASELINE {A}@500p2a [{mode}] AUC[{d}] = {aucs[d]:.4f} "
                   f"(pos={int(pos.sum())} neg={int(neg.sum())})")
        best = "forward" if aucs["forward"] >= aucs["reverse"] else "reverse"
        modes[mode] = dict(auc_forward=aucs["forward"],
                           auc_reverse=aucs["reverse"], direction=best,
                           best=aucs[best])
    prim = modes[cl.PRIMARY_DEPTH]
    anchor = prim["best"]
    if anchor >= 0.85:
        reading = "Aug-25 chance result was an input fault; ink_9um transfers in-modality to an unseen scroll at curve-anchoring quality"
    elif anchor >= 0.65:
        reading = "partial transfer; Bet A's 500p2a gate rebases to anchor+0.05"
    else:
        reading = "transfer failure confirmed at the correct pitch (read jointly with the CTL scale-fault verdict)"
    res = dict(anchor=A, pitch_um=cl.P2A_PITCH, primary_depth=cl.PRIMARY_DEPTH,
               modes=modes, corrected_anchor_auc=anchor,
               direction=prim["direction"], gate=cl.GATE_BASELINE_AUC,
               gate_passed=bool(anchor >= cl.GATE_BASELINE_AUC),
               prestated_reading=reading,
               n_pos=int(pos.sum()), n_neg=int(neg.sum()),
               aug25_void=dict(auc_forward=0.5382, auc_reverse=0.5055,
                               pitch_assumed=cl.P2A_PITCH_WRONG,
                               true_effective_pitch_um=round(
                                   cl.MODEL_PITCH * cl.P2A_PITCH / cl.P2A_PITCH_WRONG, 3)))
    json.dump(res, open(os.path.join(cl.RESULTS, "expA_baseline.json"), "w"),
              indent=1)
    open(os.path.join(VAR, "direction.txt"), "w").write(prim["direction"])
    cl.say(f"CORRECTED ANCHOR ({A} iso, {prim['direction']}): AUC = {anchor:.4f} "
           f"[fit17 best {modes['fit17']['best']:.4f}] -- {reading}")
    if not res["gate_passed"]:
        cl.say(f"EXPA_BASELINE curve gate not met: {anchor:.4f} < "
               f"{cl.GATE_BASELINE_AUC}; rungs are skipped; the anchor stands "
               f"as the result (win2/win3 + EXP B still run)")
        sys.exit(21)
    cl.say(f"EXPA_BASELINE curve gate PASSED: AUC_base={anchor:.4f} "
           f"direction={prim['direction']} -- rungs will use it")

def rungs():
    pos, neg = native_labels()
    base = json.load(open(os.path.join(cl.RESULTS, "expA_baseline.json")))
    direction = base["direction"]
    auc_b = base["corrected_anchor_auc"]
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
    add(f"pitch_{cl.P2A_PITCH}", "pitch", cl.P2A_PITCH, None, reused=True,
        note="native acquisition == baseline")
    for Pu in cl.PITCHES:
        if abs(Pu - cl.MODEL_PITCH) < 1e-9:
            add(f"pitch_{Pu}", "pitch", Pu, None, reused=True,
                note="2.215->9.36 == baseline by construction")
        else:
            add(f"pitch_{Pu}", "pitch", Pu,
                os.path.join(cl.PREDS, f"expA_p{Pu}.tif"))
    for k in cl.NOISE_KS:
        add(f"noise_k{k}", "noise", k, os.path.join(cl.PREDS, f"expA_n{k}.tif"))
    add("bit8", "bitdepth", 8, None, reused=True,
        note="released volume is already uint8; baseline by construction")
    add("bit4", "bitdepth", 4, os.path.join(cl.PREDS, "expA_bit4.tif"))
    for sg in cl.BLUR_SIGMAS:
        add(f"blur_{sg}", "blur", sg,
            os.path.join(cl.PREDS, f"expA_blur{sg}.tif"))
    pit = [r for r in rows if r["family"] == "pitch" and r["detectable"]]
    limit = max((r["x"] for r in pit), default=None)
    sigma = json.load(open(os.path.join(cl.RESULTS, "expA_sigma.json")))
    out = dict(anchor=A, baseline=base, sigma_plate=sigma, direction=direction,
               rungs=rows, detectability_limit_um=limit,
               rule=dict(auc_min=cl.DETECT_AUC_MIN,
                         retain_min=cl.DETECT_RETAIN_MIN))
    json.dump(out, open(os.path.join(cl.RESULTS, "expA_curve.json"), "w"),
              indent=1)
    cl.say(f"EXPA_SCORE curve complete; detectability limit = {limit} um "
           f"(coarsest DETECTABLE pitch)")

if __name__ == "__main__":
    {"baseline": baseline, "rungs": rungs}[sys.argv[1]]()
