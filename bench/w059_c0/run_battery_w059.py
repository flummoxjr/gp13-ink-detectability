"""Run the PROTOCOL_V2 five-gate battery (analyze_survey_corpus_v2.py, unchanged)
on the w059/C0 pod's ds16 maps, as parts/prereg.json prescribes:

  units   w059B_fwd  (Track F arm B: canonical 2um model on w059's 1.129um-L1 volume, forward)
          c2_w035B   (modality control: same model on w035's 1.129um-L1 volume, forward only ->
                      gate 5 not computable, so its ceiling is the 4 map-internal gates)
  control w035_CONTROL_strided + w035_LABELS (ink_9um, the battery's own validity check:
          must reproduce 5/5 or the battery is invalid)
  gate 5  from the pod's fwd/rev Pearson r (results.json maps.w059B_fwdrev.r_ds4_joint)

  python run_battery_w059.py --src <dir holding results.json + maps/*_ds16.npy> [--out DIR] [--nperm 200] [--jobs N]
       [--no-control]   (smoke tests only; a real reading needs the control)

ds16 of a 2.258 um/px volume = 36.13 um/px, i.e. the same physical scale as the survey's
9.362*4 = 37.4 um/px ds4 maps; the maps are block means (prereg), not strided decimations.
The scoring code itself is not modified; only its OUT/CACHE/PX_UM_DS4 module constants."""
import argparse
import json
import os
import shutil
import sys

TRACKD = r"C:\Users\benbl\Desktop\Vsuvious\trackD"
SURVEY = os.path.join(TRACKD, "out", "survey")
PX_UM_DS16 = 2.258 * 16          # 36.128 um/px
UNITS = [("w059B_fwd", "PHerc0139_w059_2um"), ("c2_w035B", "PHerc0139_w035_2um")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=os.path.join(TRACKD, "out", "w059_c0"))
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("--no-control", action="store_true", help="skip the ink_9um w035 control (smoke tests only)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shard = os.path.join(a.out, "maps_shard_w059")
    os.makedirs(shard, exist_ok=True)
    res = json.load(open(os.path.join(a.src, "results.json"), encoding="utf-8"))
    maps = res.get("maps", {})
    fr = (maps.get("w059B_fwdrev") or {}).get("r_ds4_joint")
    rows, names = [], []
    for name, scroll in UNITS:
        src = os.path.join(a.src, "maps", f"{name}_ds16.npy")
        if not os.path.exists(src):
            print(f"skip {name}: no ds16 map in {a.src}")
            continue
        shutil.copyfile(src, os.path.join(shard, f"{name}_forward_ds4.npy"))
        rows.append({"name": name, "scroll": scroll, "fwd_rev_r": fr if name.startswith("w059") else None,
                     "forward": {"tripwire_hits": []}})
        names.append(name)
    json.dump(rows, open(os.path.join(a.out, "survey_all.json"), "w"), indent=1)
    cache_dst = os.path.join(a.out, "v2cache")
    if not a.no_control and not os.path.exists(cache_dst):
        shutil.copytree(os.path.join(SURVEY, "v2cache"), cache_dst)
    ctrl = [] if a.no_control else ["w035_CONTROL_strided", "w035_LABELS"]
    print(f"battery on {names} (+control {ctrl}); C0 contract_unverified={res.get('contract_unverified')} "
          f"c0_best_r={res.get('c0_best_r')} window={res.get('window')}; w059 fwd/rev r={fr}; nperm={a.nperm} jobs={a.jobs}", flush=True)

    sys.path.insert(0, TRACKD)
    import analyze_survey_corpus_v2 as v2
    v2.OUT = a.out
    v2.CACHE = cache_dst
    v2.PX_UM_DS4 = dict(v2.PX_UM_DS4, **{s: PX_UM_DS16 for _, s in UNITS})
    result_path = os.path.join(a.out, "battery_w059.json")
    sys.argv = ["analyze_survey_corpus_v2.py", "--nperm", str(a.nperm), "--jobs", str(a.jobs),
                "--out", result_path, "--only"] + names + ctrl
    v2.main()
    out = json.load(open(result_path))
    out["w059_c0_pod"] = {"c0_best_r": res.get("c0_best_r"), "window": res.get("window"),
                          "contract_unverified": res.get("contract_unverified"), "villa_sha": res.get("villa_sha"),
                          "w059_fwdrev_r_ds4": fr, "finished_utc": res.get("finished_utc")}
    json.dump(out, open(result_path, "w"), indent=1)
    print("\n=== w059/C0 BATTERY (prereg: C0 pass AND w035_B 4/4 map-internal AND w059_B 4/4 + |fwd/rev r| < 0.20 -> escalation) ===")
    print(f"C0: best r={res.get('c0_best_r')} window=[{res.get('window')}) "
          f"{'CONTRACT_UNVERIFIED -- nothing below is interpretable' if res.get('contract_unverified') else 'contract reproduced'}")
    for r in out.get("results", []):
        if str(r.get("name", "")).startswith("w035_"):
            print(f"CONTROL {r['name']}: gates {r.get('gates_passed')}/5 p={r.get('empirical_p')} "
                  f"({'reproduces 5/5' if r.get('gates_passed') == 5 else 'DOES NOT reproduce -- battery invalid'})")
        else:
            print(f"{r['name']}: gates {r.get('gates_passed')}/5 p={r.get('empirical_p')} holm={r.get('holm_p')} "
                  f"period={r.get('period_mm')} mm theta={r.get('theta_deg')} r={r.get('fwd_rev_r')} "
                  f"[cycles {r.get('gate_cycles')} autocorr {r.get('gate_autocorr')} band {r.get('gate_band_bin')} "
                  f"sig {r.get('gate_significance')} fwdrev {r.get('gate_fwd_rev')}]")
    for r in out.get("skipped", []):
        print("SKIPPED", r)


if __name__ == "__main__":
    main()
