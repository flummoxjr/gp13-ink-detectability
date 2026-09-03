"""Bet A arms 1/2 decision rule (PREREG_BET_A.md §5, frozen 2026-09-03) applied to per-seed results.json files.

  python arm_verdict.py --arm0 <s42.json> <s43.json> --arm1 <s42.json> <s43.json> [--arm2 ...] [--out verdict.json]

For each arm: per seed, the best-of-grid checkpoint by native-5 mean best F1 (as finalize/combine_verdict do),
its forward mean AUC, and the reverse mean AUC where present. The arm's score is the two-seed mean of the
forward AUC at that checkpoint. PASS iff score >= arm0_score + 0.05 (and the gain exceeds 2x the arm-0 seed
spread). The 500p2a clause is evaluated separately afterwards (bench/p2a_v3) and recorded by hand."""
import argparse, json, re


def best_ckpt(res, seed):
    rows = {k: v for k, v in res["eval"].items() if k.startswith(f"s{seed}_")}
    if not rows:
        return None
    tag, row = max(rows.items(), key=lambda kv: kv[1]["native5_mean_best_f1"])
    rev = [c.get("reverse", {}).get("auc") for c in row.get("crops", {}).values() if isinstance(c, dict)]
    rev = [x for x in rev if isinstance(x, (int, float))]
    return dict(tag=tag, f1=row["native5_mean_best_f1"], margin=row["native5_mean_margin"], auc_fwd=row["native5_mean_auc_forward"],
                auc_rev_mean=(sum(rev) / len(rev)) if rev else None,
                trajectory_auc={k: v["native5_mean_auc_forward"] for k, v in sorted(rows.items(), key=lambda kv: int(re.search(r"_(\d+)$", kv[0]).group(1)))},
                input_stats=(res.get("input_stats") or {}).get("arm1_active_fraction"))


def arm_summary(files):
    seeds = {}
    for f in files:
        res = json.load(open(f, encoding="utf-8"))
        for s in ("42", "43"):
            b = best_ckpt(res, s)
            if b:
                b["source"] = f; seeds[s] = b
    if not seeds:
        return None
    aucs = [v["auc_fwd"] for v in seeds.values()]
    return dict(seeds=seeds, n_seeds=len(seeds), mean_auc_fwd=sum(aucs) / len(aucs),
                seed_spread=(max(aucs) - min(aucs)) if len(aucs) > 1 else None,
                best_f1=max(v["f1"] for v in seeds.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm0", nargs="+", required=True); ap.add_argument("--arm1", nargs="*", default=[]); ap.add_argument("--arm2", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arm0 = arm_summary(a.arm0); assert arm0 and arm0["n_seeds"] == 2, "arm 0 needs both seeds"
    base, spread = arm0["mean_auc_fwd"], arm0["seed_spread"]
    threshold = max(base + 0.05, base + 2 * spread)
    out = dict(rule="PASS iff two-seed mean forward native-5 AUC at the best-of-grid checkpoint >= arm0 + 0.05 (and > arm0 + 2 x arm0 seed spread); 500p2a win1 >= 0.65 checked afterwards",
               arm0=dict(mean_auc_fwd=round(base, 4), seed_spread=round(spread, 4), threshold=round(threshold, 4), per_seed={s: {k: v[k] for k in ("tag", "f1", "auc_fwd", "auc_rev_mean")} for s, v in arm0["seeds"].items()}),
               arms={})
    for name, files in (("arm1", a.arm1), ("arm2", a.arm2)):
        if not files:
            continue
        s = arm_summary(files)
        if not s:
            continue
        s["gain_over_arm0"] = round(s["mean_auc_fwd"] - base, 4)
        s["PASS_primary"] = bool(s["n_seeds"] == 2 and s["mean_auc_fwd"] >= threshold)
        s["complete"] = s["n_seeds"] == 2
        out["arms"][name] = s
    out["any_arm_passes_primary"] = any(v.get("PASS_primary") for v in out["arms"].values())
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
    for name, s in out["arms"].items():
        print(f"{name}: seeds {s['n_seeds']} mean AUC fwd {s['mean_auc_fwd']:.4f} (gain {s['gain_over_arm0']:+.4f}, threshold {threshold:.4f}) best F1 {s['best_f1']:.4f} -> "
              f"{'PASS (primary)' if s['PASS_primary'] else 'not passing'}; per seed " + "; ".join(f"s{k}: {v['tag']} AUC {v['auc_fwd']:.4f} rev {v['auc_rev_mean'] if v['auc_rev_mean'] is None else round(v['auc_rev_mean'], 4)} F1 {v['f1']:.4f} active {v['input_stats']}" for k, v in s["seeds"].items()))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
