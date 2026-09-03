"""Aggregate results, verify the inventory, apply the prereg gate when both seeds are
complete (never in SMOKE_ONLY), write results/results.json and out/results.json."""
import glob, json, os, re, sys, time
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

VAR, RUNS = os.environ["VAR"], os.environ["RUNS"]
SMOKE = os.environ.get("SMOKE_ONLY", "1") == "1"
SEEDS = os.environ.get("SEEDS", "42 43").split()


def train_summary(seed):
    log = os.path.join(cl.OUT, "logs", f"train_s{seed}.log")
    if not os.path.exists(log):
        return None
    txt = open(log, errors="replace").read()
    steps = re.findall(r"step[ =:/]+(\d+)", txt)
    its = re.findall(r"(\d+\.\d+)\s*it/s", txt)
    ck = sorted(glob.glob(os.path.join(RUNS, f"s{seed}", "ckpt_*.pth")))
    return dict(log_bytes=len(txt), last_step=int(steps[-1]) if steps else None,
                it_per_s_last=float(its[-1]) if its else None, checkpoints=[os.path.basename(c) for c in ck],
                sampling_observed=os.path.exists(os.path.join(RUNS, f"s{seed}", "sampling_observed.json")))


def main():
    missing = []
    for f in ["ctl.json", "native_crops.json", "eval.json"]:
        if not os.path.exists(os.path.join(cl.RESULTS, f)):
            missing.append("results/" + f)
    # 2026-09-03: hard-coded "train_s42"/"eval_s42" refused a SEEDS=43-only full run after 7 h of
    # work (pod nxcv6ufppr8t6m); the required train/eval stages are exactly those of SEEDS.
    stages = ["provision", "ckpt", "trainer_check", "labels_fetch", "sv_fetch", "pool", "config_gen",
              "native_fetch", "ctl", "ref"]
    stages += [f"train_s{s}" for s in SEEDS] + [f"eval_s{s}" for s in SEEDS]
    for st in dict.fromkeys(stages):
        if not os.path.exists(os.path.join(VAR, "done_" + st)):
            missing.append("stage:" + st)
    if missing:
        cl.say("FINALIZE REFUSED -- missing: " + ", ".join(missing))
        sys.exit(3)
    ev = json.load(open(os.path.join(cl.RESULTS, "eval.json")))
    ctl = json.load(open(os.path.join(cl.RESULTS, "ctl.json")))
    trains = {s: train_summary(s) for s in SEEDS}
    verdict = {"mode": "SMOKE_ONLY -- pipeline validation; no gate verdict"} if SMOKE else {}
    if not SMOKE:
        per_seed = {}
        for s in SEEDS:
            rows = {k: v for k, v in ev.items() if k.startswith(f"s{s}_")}
            if rows:
                best = max(rows.items(), key=lambda kv: kv[1]["native5_mean_best_f1"])
                per_seed[s] = dict(best_tag=best[0], best_f1=best[1]["native5_mean_best_f1"],
                                   margin=best[1]["native5_mean_margin"], auc=best[1]["native5_mean_auc_forward"],
                                   trajectory={k: v["native5_mean_best_f1"] for k, v in sorted(rows.items())})
        if len(per_seed) == len(SEEDS):
            best_of_both = max(v["best_f1"] for v in per_seed.values())
            mean_margin = sum(v["margin"] for v in per_seed.values()) / len(per_seed)
            def peak_ok(tr):
                steps = {int(re.search(r"_(\d+)$", k).group(1)): v for k, v in tr.items()}
                if not steps:
                    return False
                pk = max(steps, key=steps.get)
                return 10000 <= pk <= 30000 and (75000 not in steps or steps[75000] < steps[pk])
            peaks = all(peak_ok(v["trajectory"]) for v in per_seed.values())
            verdict = dict(best_of_both=best_of_both, mean_margin=mean_margin, peak_rule=peaks,
                           PASS=bool(best_of_both >= 0.603 and mean_margin >= 0.06 and peaks),
                           rule="best-of-both >= 0.603 AND mean margin >= +0.06 AND peak at 10-30k with 75k below")
        verdict["per_seed"] = per_seed
    agg = dict(run="pod_betA_arm0 v1", smoke_only=SMOKE, finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               prereg=json.load(open(os.path.join(cl.OUT, "prereg.json"))), ctl=ctl,
               native_crops=json.load(open(os.path.join(cl.RESULTS, "native_crops.json"))),
               eval=ev, training=trains, verdict=verdict)
    for p in (os.path.join(cl.RESULTS, "results.json"), os.path.join(cl.OUT, "results.json")):
        json.dump(agg, open(p, "w"), indent=1)
    na = ctl["arms"]["ctl_native"]
    cl.say(f"SUMMARY ctl (released ckpt): native fwd={na['forward']:.4f} rev={na['reverse']:.4f}; "
           f"scale-fault {ctl['scale_fault']['best']:.4f} -> {ctl['scale_fault']['verdict']}")
    for tag, row in ev.items():
        cl.say(f"SUMMARY {tag}: native-5 bestF1 {row['native5_mean_best_f1']:.4f} (margin {row['native5_mean_margin']:+.3f}), "
               f"AUC fwd {row['native5_mean_auc_forward']:.4f}")
    for s, t in trains.items():
        if t:
            cl.say(f"SUMMARY train s{s}: last step {t['last_step']}, {t['it_per_s_last']} it/s, ckpts {t['checkpoints']}")
    cl.say(f"SUMMARY verdict: {json.dumps(verdict)[:300]}")


if __name__ == "__main__":
    main()
