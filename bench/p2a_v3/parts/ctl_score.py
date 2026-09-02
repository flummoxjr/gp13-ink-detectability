"""CTL scoring: three arms x two directions on the embedded w035 crop labels.
Exit 31 = HARNESS_BROKEN (native forward below 0.95); exit 32 = the known
depth-order fault did not reproduce (native reverse above 0.80). Both are
fatal by pre-registration. The scale-fault verdict is recorded, never fatal."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

def auc_of(tif, pos, neg):
    import tifffile
    pred = tifffile.imread(tif)
    m = cl.resample_pred(pred, pos.shape)
    q = cl.quantize_map(m)
    return cl.hist_auc(cl.masked_hist(q, pos), cl.masked_hist(q, neg)), pred

def main():
    pos, neg = cl.load_ctl_labels()
    arms = {}
    for arm in ("ctl_native", "ctl_scalefault", "ctl_half"):
        arms[arm] = {}
        for d, suffix in (("forward", ""), ("reverse", "_reverse")):
            tif = os.path.join(cl.PREDS, f"{arm}{suffix}.tif")
            auc, pred = auc_of(tif, pos, neg)
            arms[arm][d] = float(auc)
            arms[arm][f"{d}_map_shape"] = list(pred.shape)
            cl.save_preview(pred, os.path.join(cl.OUT, "previews",
                                               f"{arm}_{d}.png"), ds=2)
            cl.say(f"CTL_SCORE {arm} {d}: AUC={auc:.4f} (map {pred.shape})")
    nat_f, nat_r = arms["ctl_native"]["forward"], arms["ctl_native"]["reverse"]
    sf = max(arms["ctl_scalefault"]["forward"], arms["ctl_scalefault"]["reverse"])
    hf = max(arms["ctl_half"]["forward"], arms["ctl_half"]["reverse"])
    if sf < cl.CTL_FAULT_REPRODUCED_MAX:
        verdict = "FAULT_REPRODUCED"
    elif sf >= cl.CTL_FAULT_NOT_REPRODUCED_MIN:
        verdict = "FAULT_NOT_REPRODUCED"
    else:
        verdict = "PARTIAL"
    res = dict(arms=arms, n_pos=int(pos.sum()), n_neg=int(neg.sum()),
               crop=list(cl.CTL_CROP), fault_factor=cl.CTL_FAULT_FACTOR,
               harness_gate=dict(min_forward=cl.CTL_HARNESS_MIN_FWD,
                                 passed=bool(nat_f >= cl.CTL_HARNESS_MIN_FWD)),
               depth_order_gate=dict(max_reverse=cl.CTL_DEPTHREV_MAX,
                                     passed=bool(nat_r <= cl.CTL_DEPTHREV_MAX)),
               scale_fault=dict(best=sf, verdict=verdict,
                                reproduced_max=cl.CTL_FAULT_REPRODUCED_MAX,
                                not_reproduced_min=cl.CTL_FAULT_NOT_REPRODUCED_MIN),
               half_scale=dict(best=hf),
               on_record=dict(forward_full_canvas=0.9991, reverse_full_canvas=0.5123))
    json.dump(res, open(os.path.join(cl.RESULTS, "ctl.json"), "w"), indent=1)
    if not res["harness_gate"]["passed"]:
        cl.say(f"CTL HARNESS_BROKEN: ctl_native forward {nat_f:.4f} < "
               f"{cl.CTL_HARNESS_MIN_FWD} (on record 0.9991) -- nothing downstream "
               f"is interpretable; dying")
        sys.exit(31)
    if not res["depth_order_gate"]["passed"]:
        cl.say(f"CTL DEPTH-ORDER FAULT NOT REPRODUCED: ctl_native reverse "
               f"{nat_r:.4f} > {cl.CTL_DEPTHREV_MAX} (on record 0.5123) -- the "
               f"harness cannot certify fault controls; dying")
        sys.exit(32)
    cl.say(f"CTL GATES PASSED: native fwd={nat_f:.4f} rev={nat_r:.4f}; "
           f"scale-fault best={sf:.4f} -> {verdict}; half-scale best={hf:.4f}")

if __name__ == "__main__":
    main()
