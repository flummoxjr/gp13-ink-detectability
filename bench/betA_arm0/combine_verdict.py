"""Apply the pre-registered Bet A arm-0 anchor gate to per-seed results.json files that were
produced on SEPARATE pods (SEEDS=42 and SEEDS=43), using exactly the rule finalize.py applies
when both seeds run in one pod (prereg §4, corrected 2026-09-02):

    best-of-both native-5 best F1 >= 0.603  AND  mean margin over the floor >= +0.06
    AND each seed's F1 trajectory peaks at 10k-30k steps with the 75k checkpoint below the peak.

  python combine_verdict.py <results_s42.json> <results_s43.json> [--out verdict.json]"""
import argparse, json, re


def per_seed(res, seed):
    rows = {k: v for k, v in res["eval"].items() if k.startswith(f"s{seed}_")}
    if not rows:
        return None
    best = max(rows.items(), key=lambda kv: kv[1]["native5_mean_best_f1"])
    return dict(best_tag=best[0], best_f1=best[1]["native5_mean_best_f1"], margin=best[1]["native5_mean_margin"],
                auc=best[1]["native5_mean_auc_forward"],
                trajectory={k: v["native5_mean_best_f1"] for k, v in sorted(rows.items(), key=lambda kv: int(re.search(r"_(\d+)$", kv[0]).group(1)))},
                it_per_s=(res.get("training", {}).get(str(seed)) or {}).get("it_per_s_last"))


def peak_ok(tr):
    steps = {int(re.search(r"_(\d+)$", k).group(1)): v for k, v in tr.items()}
    if not steps:
        return False, None
    pk = max(steps, key=steps.get)
    return (10000 <= pk <= 30000 and (75000 not in steps or steps[75000] < steps[pk])), pk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = {}
    for f in a.files:
        res = json.load(open(f, encoding="utf-8"))
        for s in ("42", "43"):
            ps = per_seed(res, s)
            if ps:
                ps["source"] = f; ps["prereg_sha"] = res.get("prereg", {}).get("sha256") or res.get("prereg", {}).get("version")
                seeds[s] = ps
    if len(seeds) < 2:
        print("need both seeds; have", list(seeds)); return
    best_of_both = max(v["best_f1"] for v in seeds.values())
    mean_margin = sum(v["margin"] for v in seeds.values()) / len(seeds)
    peaks = {s: peak_ok(v["trajectory"]) for s, v in seeds.items()}
    verdict = dict(best_of_both=round(best_of_both, 4), mean_margin=round(mean_margin, 4),
                   peak_rule={s: dict(ok=p[0], peak_step=p[1]) for s, p in peaks.items()},
                   PASS=bool(best_of_both >= 0.603 and mean_margin >= 0.06 and all(p[0] for p in peaks.values())),
                   rule="best-of-both >= 0.603 AND mean margin >= +0.06 AND peak at 10-30k with 75k below (prereg section 4, corrected 2026-09-02)",
                   per_seed=seeds, anchor="khj1222 native-5 F1 0.653 (floor 0.541)")
    print(json.dumps({k: v for k, v in verdict.items() if k != "per_seed"}, indent=1))
    for s, v in seeds.items():
        print(f"seed {s}: best {v['best_f1']:.4f} @ {v['best_tag']} (margin {v['margin']:+.3f}, AUC {v['auc']:.4f}); "
              f"trajectory {' '.join(f'{k.split(chr(95))[-1]}:{x:.3f}' for k, x in v['trajectory'].items())}")
    if a.out:
        json.dump(verdict, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
