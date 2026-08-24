"""Pull survey results (JSONL + downsampled prediction maps) from all shards; summarize."""
import json
import os
import subprocess
import sys

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")

KEYFILE = os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\survey"
os.makedirs(OUT, exist_ok=True)
fleet = json.load(open(os.path.join(HERE, "survey_fleet.json"), encoding="utf-8-sig"))
PULL_MAPS = "--maps" in sys.argv


def opts(port):
    return ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
            "-o", "ConnectTimeout=20", "-i", KEYFILE, "-P", str(port)]


rows = []
for p in fleet["pods"]:
    s = p["shard"]
    dest = os.path.join(OUT, f"survey_{s}.jsonl")
    try:
        subprocess.run(["scp"] + opts(p["port"]) +
                       [f"root@{p['ip']}:/workspace/survey/survey_{s}.jsonl", dest],
                       check=True, capture_output=True, timeout=180)
    except Exception as e:
        print(f"shard {s}: pull failed {str(e)[:80]}")
        continue
    n = 0
    for line in open(dest):
        try:
            rows.append(json.loads(line))
            n += 1
        except Exception:
            pass
    print(f"shard {s}: {n} records")
    if PULL_MAPS:
        subprocess.run(["scp", "-r"] + opts(p["port"]) +
                       [f"root@{p['ip']}:/workspace/survey/", os.path.join(OUT, f"maps_shard{s}")],
                       capture_output=True, timeout=1800)

ok = [r for r in rows if "error" not in r and r.get("forward")]
err = [r for r in rows if "error" in r]
trips = [(r["name"], r["scroll"], r["forward"]["tripwire_hits"])
         for r in ok if r["forward"].get("tripwire_hits")]

print(f"\n=== SURVEY SUMMARY ===")
print(f"segments processed: {len(rows)} | usable: {len(ok)} | errors: {len(err)}")
by_scroll = {}
for r in ok:
    by_scroll.setdefault(r["scroll"], []).append(r)
for sc, rs in sorted(by_scroll.items()):
    fr = [r["forward"]["frac_gt_half"] for r in rs]
    hi = [r["forward"]["frac_gt_blankp99"] for r in rs]
    rr = [r["fwd_rev_r"] for r in rs if r.get("fwd_rev_r") is not None]
    print(f"  {sc}: n={len(rs)} | frac>half {min(fr):.3f}-{max(fr):.3f} | "
          f"frac>blankP99 max {max(hi):.5f} | fwd/rev r {min(rr):.2f}-{max(rr):.2f}"
          if rr else f"  {sc}: n={len(rs)}")
print(f"\nTRIPWIRE HITS: {len(trips)}")
for name, sc, hits in trips:
    print(f"  ! {sc}/{name}: {hits}")
if err:
    print(f"\nerrors ({len(err)}):")
    for r in err[:10]:
        print(f"  {r['name']}: {r['error'][:110]}")

with open(os.path.join(OUT, "survey_all.json"), "w") as f:
    json.dump(rows, f, indent=1)
print(f"\nwrote {os.path.join(OUT, 'survey_all.json')}")
