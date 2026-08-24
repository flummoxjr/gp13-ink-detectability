"""Pull QC samples from the screening fleet.

For each pod: fetch the stats jsonl, pick the top-K tiles by pmax (plus a few
random mid-fill tiles as controls), scp their probL2 arrays down.
Writes to trackD/qc_live/round_<N>/ with a manifest.
Usage: python qc_pull_samples.py <round_number>
"""
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")

KEYFILE = os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
HERE = os.path.dirname(os.path.abspath(__file__))
fleet = json.load(open(os.path.join(HERE, "fleet.json")))
ROUND = sys.argv[1] if len(sys.argv) > 1 else "1"
OUT = rf"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\round_{ROUND}"
os.makedirs(OUT, exist_ok=True)
TOP_K = 6
RAND_K = 3


def ssh_opts(port):
    return ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
            "-o", "ConnectTimeout=15", "-i", KEYFILE, "-P", str(port)]


manifest = {"round": ROUND, "pods": {}}
rng = random.Random(int(ROUND))

for p in fleet["pods"]:
    wid = p["worker"]
    tag = f"w{wid}"
    try:
        # stats file
        local_stats = os.path.join(OUT, f"{tag}_stats.jsonl")
        subprocess.run(["scp"] + ssh_opts(p["port"]) +
                       [f"root@{p['ip']}:/workspace/out/screen_full/stats_0_15137.jsonl", local_stats],
                       check=True, capture_output=True, timeout=120)
        rows = []
        with open(local_stats) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("skipped") and "pmax" in r:
                        rows.append(r)
                except json.JSONDecodeError:
                    pass
        if not rows:
            manifest["pods"][tag] = {"n_done": 0}
            continue
        rows.sort(key=lambda r: -r["pmax"])
        picks = rows[:TOP_K] + (rng.sample(rows[TOP_K:], min(RAND_K, max(len(rows) - TOP_K, 0)))
                                if len(rows) > TOP_K else [])
        fetched = []
        for r in picks:
            z, y, x = r["tile"]
            fn = f"{z}_{y}_{x}.npy"
            dest = os.path.join(OUT, f"{tag}_prob_{fn}")
            try:
                subprocess.run(["scp"] + ssh_opts(p["port"]) +
                               [f"root@{p['ip']}:/workspace/out/screen_full/probL2_0_15137/{fn}", dest],
                               check=True, capture_output=True, timeout=120)
                fetched.append({"tile": r["tile"], "pmax": r["pmax"], "f05": r["f05"],
                                "f08": r.get("f08"), "fill": r.get("fill"), "file": os.path.basename(dest)})
            except Exception:
                pass
        pm = [r["pmax"] for r in rows]
        f05 = [r["f05"] for r in rows]
        import statistics
        manifest["pods"][tag] = {
            "n_done": len(rows),
            "pmax_median": round(statistics.median(pm), 3),
            "pmax_p90": round(sorted(pm)[int(0.9 * len(pm))], 3),
            "f05_median": round(statistics.median(f05), 5),
            "f05_p90": round(sorted(f05)[int(0.9 * len(f05))], 5),
            "frac_tiles_pmax_gt_0.8": round(sum(1 for v in pm if v > 0.8) / len(pm), 4),
            "frac_tiles_f05_gt_0.05": round(sum(1 for v in f05 if v > 0.05) / len(f05), 4),
            "samples": fetched,
        }
        print(tag, json.dumps({k: v for k, v in manifest['pods'][tag].items() if k != 'samples'}))
    except Exception as e:
        manifest["pods"][tag] = {"error": str(e)[:200]}
        print(tag, "ERROR", str(e)[:120])

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote", os.path.join(OUT, "manifest.json"))
