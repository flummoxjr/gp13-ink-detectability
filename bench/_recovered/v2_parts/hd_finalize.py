"""Aggregate results, verify the full inventory, refuse to bless a partial run."""
import json, os, sys, time
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR = os.environ["VAR"]
NATIVE = os.environ.get("NATIVE_WINDOWS", "1") == "1"
SECONDARY = os.environ.get("SECONDARY", "1") == "1"

def main():
    missing = []
    for f in ["expA_baseline.json", "expA_sigma.json", "expA_curve.json",
              "expB_hallucination.json"]:
        if not os.path.exists(os.path.join(cl.RESULTS, f)):
            missing.append("results/" + f)
    dpath = os.path.join(VAR, "direction.txt")
    direction = open(dpath).read().strip() if os.path.exists(dpath) else None
    if direction not in ("forward", "reverse"):
        missing.append("var/direction.txt")
    sec_ok = os.path.exists(os.path.join(VAR, "secondary_ok"))
    sec_failed = os.path.exists(os.path.join(VAR, "secondary_failed"))
    preds = ["expA_base.tif", "expA_base_reverse.tif", "expA_bit4.tif"]
    preds += [f"expA_p{Pu}.tif" for Pu in cl.PITCHES
              if abs(Pu - cl.MODEL_PITCH) >= 1e-9]
    preds += [f"expA_n{k}.tif" for k in cl.NOISE_KS]
    preds += [f"expA_blur{sg}.tif" for sg in cl.BLUR_SIGMAS]
    for w in ("win2", "win3"):
        preds += [f"expB_{w}_s936.tif", f"expB_{w}_s936_reverse.tif"]
    if NATIVE:
        for w in ("win1", "win2", "win3"):
            preds += [f"expB_{w}_native.tif", f"expB_{w}_native_reverse.tif"]
    if sec_ok:
        preds += ["sec_frag1_936.tif", "sec_frag1_936_reverse.tif"]
    for f in preds:
        if not os.path.exists(os.path.join(cl.PREDS, f)):
            missing.append("preds/" + f)
    stages = ["provision", "ckpt", "p2a_fetch", "p2a_build_win1",
              "expa_baseline", "expa_rungs", "p2a_build_rest", "expb_infer",
              "expb_score"]
    if SECONDARY:
        stages.append("sec_frag1")
        if not (sec_ok or sec_failed):
            missing.append("var/secondary_ok|secondary_failed")
    for st in stages:
        if not os.path.exists(os.path.join(VAR, "done_" + st)):
            missing.append("stage:" + st)
    if missing:
        cl.say("FINALIZE REFUSED -- missing: " + ", ".join(missing))
        sys.exit(3)
    agg = dict(
        run="pod_curve_audit v2 (C2: curve re-anchored on 500p2a win1)",
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        prereg=json.load(open(os.path.join(cl.OUT, "prereg.json"))),
        direction=direction,
        secondary=dict(enabled=SECONDARY, ok=sec_ok, failed=sec_failed),
        expA=json.load(open(os.path.join(cl.RESULTS, "expA_curve.json"))),
        expB=json.load(open(os.path.join(cl.RESULTS,
                                         "expB_hallucination.json"))),
    )
    json.dump(agg, open(os.path.join(cl.RESULTS, "results.json"), "w"),
              indent=1)
    a = agg["expA"]; b = agg["expB"]
    cl.say(f"SUMMARY expA (win1@500p2a): AUC_base="
           f"{a['baseline'][f'auc_{direction}']:.4f} ({direction}); "
           f"detectability limit {a['detectability_limit_um']} um; "
           f"sigma_plate={a['sigma_plate']['sigma_plate']:.2f} DN")
    for name, r in b["inputs"].items():
        aud = r["cells"][r["audited_cell"]]
        tag = " [secondary]" if r.get("secondary") else ""
        cl.say(f"SUMMARY expB {name}{tag}: {r['verdict']} real="
               f"{aud['auc_real']:.4f} null_med={aud['null_median']:.4f} "
               f"gap={aud['gap']:.4f}")
    for name, why in b.get("excluded", {}).items():
        cl.say(f"SUMMARY expB {name}: EXCLUDED ({why})")
    if sec_ok:
        cl.say("SUMMARY secondary frag1: compare its audited real AUC against "
               "the v1 KILLED_BASELINE quote 0.6925 fwd / 0.4477 rev "
               "(|new-old| is the fresh-infra consistency check)")
    cl.say(f"SUMMARY expB overall (primary windows only): {b['overall']}")

if __name__ == "__main__":
    main()
