"""Run the PROTOCOL_V2 five-gate battery (analyze_survey_corpus_v2.py, unchanged)
on the PHerc0358 screen maps, exactly as PREREG_0358_v2.md prescribes:
strided ds4 FORWARD maps, gate 5 from the pod's full-res fwd/rev Pearson r,
w035_CONTROL_strided must reproduce 5/5 before any patch counts.

  python run_battery_0358.py --src <dir with maps/ and screen_0358.jsonl> [--out DIR] [--nperm 200] [--jobs N]

Builds a scratch survey layout (maps_shard_0358/<name>_forward_ds4.npy +
survey_all.json rows + the control cache) and points the v2 module at it via
its module constants; the scoring code itself is not modified."""
import argparse
import glob
import json
import os
import shutil
import sys

TRACKD = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
SURVEY = os.path.join(TRACKD, "out", "survey")
PATCHES = ["auto_grown_20260825155611879", "auto_grown_20260825155613379",
           "auto_grown_20260825155615680", "auto_grown_20260825155619178",
           "auto_grown_20260825155619482", "auto_grown_20260825155621418",
           "auto_grown_20260825155624780", "auto_grown_20260825155625980"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=os.path.join(TRACKD, "out", "screen0358_v2"))
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--only", nargs="*", default=None, help="subset of patch names")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shard = os.path.join(a.out, "maps_shard_0358")
    os.makedirs(shard, exist_ok=True)
    # jsonl -> survey rows (fwd_rev_r + tripwire from the pod's full-res stats)
    recs = {}
    jl = os.path.join(a.src, "screen_0358.jsonl")
    if os.path.exists(jl):
        for line in open(jl, encoding="utf-8"):
            try:
                r = json.loads(line)
                recs[r["name"]] = r
            except Exception:
                pass
    rows, names = [], []
    for name in PATCHES:
        if a.only and name not in a.only:
            continue
        src = os.path.join(a.src, "maps", f"{name}_fwd_ds4.npy")
        if not os.path.exists(src):
            print(f"skip {name}: no forward ds4 map yet")
            continue
        shutil.copyfile(src, os.path.join(shard, f"{name}_forward_ds4.npy"))
        r = recs.get(name, {})
        if "error" in r:
            print(f"skip {name}: pod recorded {r['error']}")
            continue
        rows.append({"name": name, "scroll": "PHerc0358", "fwd_rev_r": r.get("fwd_rev_r"),
                     "forward": {"tripwire_hits": (r.get("fwd") or {}).get("tripwire_hits") or []}})
        names.append(name)
    json.dump(rows, open(os.path.join(a.out, "survey_all.json"), "w"), indent=1)
    cache_dst = os.path.join(a.out, "v2cache")
    if not os.path.exists(cache_dst):
        shutil.copytree(os.path.join(SURVEY, "v2cache"), cache_dst)
    print(f"battery on {len(names)} patches + control; nperm={a.nperm} jobs={a.jobs}")

    sys.path.insert(0, TRACKD)
    import analyze_survey_corpus_v2 as v2
    v2.OUT = a.out
    v2.CACHE = cache_dst
    v2.PX_UM_DS4 = dict(v2.PX_UM_DS4, PHerc0358=9.362 * 4)
    result_path = os.path.join(a.out, "battery_0358.json")
    sys.argv = ["analyze_survey_corpus_v2.py", "--nperm", str(a.nperm), "--jobs", str(a.jobs),
                "--out", result_path, "--only"] + names + ["w035_CONTROL_strided", "w035_LABELS"]
    v2.main()
    res = json.load(open(result_path))
    ctrl = [r for r in res.get("controls", res.get("results", [])) if str(r.get("name", "")).startswith("w035_CONTROL_strided")]
    print("\n=== 0358 BATTERY (PREREG_0358_v2 flag rule: >= 4 of 5 computable gates) ===")
    for r in res.get("results", []):
        if r.get("kind") == "control" or str(r.get("name", "")).startswith("w035"):
            continue
        print(f"{r['name']}: gates {r.get('gates_passed')}/5 p={r.get('empirical_p')} r={r.get('fwd_rev_r')} "
              f"{'FLAG -> escalation' if (r.get('gates_passed') or 0) >= 4 else 'no flag'}")
    for r in res.get("results", []):
        if str(r.get("name", "")).startswith("w035"):
            print(f"CONTROL {r['name']}: gates {r.get('gates_passed')}/5 p={r.get('empirical_p')} "
                  f"({'reproduces 5/5' if r.get('gates_passed') == 5 else 'DOES NOT reproduce -- battery invalid'})")


if __name__ == "__main__":
    main()
