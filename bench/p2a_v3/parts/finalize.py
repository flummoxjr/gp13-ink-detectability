"""Aggregate results, verify the full inventory, refuse to bless a partial run.
Writes results/results.json AND out/results.json (the laptop guard harvests
the latter by name)."""
import json, os, sys, time
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR = os.environ["VAR"]
RUN_RUNGS = os.environ.get("RUN_RUNGS", "1") == "1"

def main():
    missing = []
    gate_failed = os.path.exists(os.path.join(VAR, "gate_failed"))
    rungs_expected = RUN_RUNGS and not gate_failed
    for f in ["ctl.json", "ctl_volume_stats.json", "expA_baseline.json",
              "expB_hallucination.json"]:
        if not os.path.exists(os.path.join(cl.RESULTS, f)):
            missing.append("results/" + f)
    if rungs_expected:
        for f in ["expA_sigma.json", "expA_curve.json"]:
            if not os.path.exists(os.path.join(cl.RESULTS, f)):
                missing.append("results/" + f)
    preds = []
    for arm in ("ctl_native", "ctl_scalefault", "ctl_half"):
        preds += [f"{arm}.tif", f"{arm}_reverse.tif"]
    preds += ["expA_base.tif", "expA_base_reverse.tif",
              "expA_base_fit17.tif", "expA_base_fit17_reverse.tif"]
    for w in ("win2", "win3"):
        for mode in cl.DEPTH_MODES:
            preds += [f"expB_{w}_{mode}.tif", f"expB_{w}_{mode}_reverse.tif"]
    if rungs_expected:
        preds += ["expA_bit4.tif"]
        preds += [f"expA_p{Pu}.tif" for Pu in cl.PITCHES
                  if abs(Pu - cl.MODEL_PITCH) >= 1e-9]
        preds += [f"expA_n{k}.tif" for k in cl.NOISE_KS]
        preds += [f"expA_blur{sg}.tif" for sg in cl.BLUR_SIGMAS]
    for f in preds:
        if not os.path.exists(os.path.join(cl.PREDS, f)):
            missing.append("preds/" + f)
    stages = ["provision", "ckpt", "ctl_fetch", "ctl_build", "ctl_infer",
              "ctl_score", "p2a_fetch", "p2a_build_win1", "expa_baseline",
              "p2a_build_rest", "expb_infer", "expb_score"]
    if rungs_expected:
        stages.append("expa_rungs")
    for st in stages:
        if not os.path.exists(os.path.join(VAR, "done_" + st)):
            missing.append("stage:" + st)
    if missing:
        cl.say("FINALIZE REFUSED -- missing: " + ", ".join(missing))
        sys.exit(3)
    ctl = json.load(open(os.path.join(cl.RESULTS, "ctl.json")))
    a = json.load(open(os.path.join(cl.RESULTS, "expA_baseline.json")))
    b = json.load(open(os.path.join(cl.RESULTS, "expB_hallucination.json")))
    curve = None
    if rungs_expected:
        curve = json.load(open(os.path.join(cl.RESULTS, "expA_curve.json")))
    agg = dict(
        run="pod_p2a_v3 (C3: 500p2a anchor at the corrected 2.215um pitch + CTL positive controls)",
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        prereg=json.load(open(os.path.join(cl.OUT, "prereg.json"))),
        ctl=ctl, expA_baseline=a, expA_curve=curve,
        curve_gate_failed=gate_failed, expB=b)
    for p in (os.path.join(cl.RESULTS, "results.json"),
              os.path.join(cl.OUT, "results.json")):
        json.dump(agg, open(p, "w"), indent=1)
    na = ctl["arms"]["ctl_native"]; sf = ctl["scale_fault"]
    cl.say(f"SUMMARY ctl: native fwd={na['forward']:.4f} rev={na['reverse']:.4f}; "
           f"scale-fault best={sf['best']:.4f} -> {sf['verdict']}; "
           f"half best={ctl['half_scale']['best']:.4f}")
    m = a["modes"]
    cl.say(f"SUMMARY anchor (win1 @ 2.215um): iso fwd={m['iso']['auc_forward']:.4f} "
           f"rev={m['iso']['auc_reverse']:.4f} | fit17 fwd={m['fit17']['auc_forward']:.4f} "
           f"rev={m['fit17']['auc_reverse']:.4f} | CORRECTED ANCHOR = "
           f"{a['corrected_anchor_auc']:.4f} ({a['direction']}) | "
           f"Aug-25 void 0.5382/0.5055")
    cl.say(f"SUMMARY reading: {a['prestated_reading']}")
    if curve:
        cl.say(f"SUMMARY curve: detectability limit {curve['detectability_limit_um']} um; "
               f"sigma_plate={curve['sigma_plate']['sigma_plate']:.2f} DN")
    else:
        cl.say("SUMMARY curve: not run (gate not met or RUN_RUNGS=0)")
    for name, r in b["inputs"].items():
        aud = r["cells"][r["audited_cell"]]
        cl.say(f"SUMMARY expB {name}: {r['verdict']} real="
               f"{aud['auc_real']:.4f} null_med={aud['null_median']:.4f} "
               f"gap={aud['gap']:.4f}")
    cl.say(f"SUMMARY expB overall: {b['overall']}")

if __name__ == "__main__":
    main()
