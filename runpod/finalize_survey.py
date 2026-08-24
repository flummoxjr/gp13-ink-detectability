"""Pull shard-0's completed records + new maps, rebuild survey_all.json from all local shards."""
import glob
import json
import os
import subprocess

KEYFILE = os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\survey"
fleet = json.load(open(os.path.join(HERE, "survey_fleet.json"), encoding="utf-8-sig"))
p = [x for x in fleet["pods"] if x["shard"] == 0][0]
opts = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
        "-o", "ConnectTimeout=20", "-i", KEYFILE]

subprocess.run(["scp"] + opts + ["-P", str(p["port"]),
                f"root@{p['ip']}:/workspace/survey/survey_0.jsonl",
                os.path.join(OUT, "survey_0.jsonl")], check=True, capture_output=True, timeout=300)
print("pulled shard-0 jsonl")
subprocess.run(["scp"] + opts + ["-P", str(p["port"]), "-r",
                f"root@{p['ip']}:/workspace/survey/",
                os.path.join(OUT, "maps_shard0")], capture_output=True, timeout=1800)
print("pulled shard-0 maps")

rows, seen = [], set()
for f in sorted(glob.glob(os.path.join(OUT, "survey_*.jsonl"))):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        rows.append(r)
json.dump(rows, open(os.path.join(OUT, "survey_all.json"), "w"), indent=1)

cat = json.load(open(os.path.join(HERE, "segment_catalog.json")))
by = {}
for r in rows:
    by.setdefault(r["scroll"], []).append(r)
print(f"\n=== FINAL SURVEY: {len(rows)} of {len(cat)} catalog segments ===")
for sc, rs in sorted(by.items()):
    ok = [r for r in rs if r.get("forward")]
    trips = sum(len((r.get("forward") or {}).get("tripwire_hits") or []) for r in rs)
    print(f"  {sc}: {len(rs)} processed, {len(ok)} with predictions, {trips} tripwire hits")
errs = [r for r in rows if "error" in r]
print(f"errors: {len(errs)}")
for r in errs[:6]:
    print(f"  {r['name']}: {str(r['error'])[:90]}")
missing = sorted({c["name"] for c in cat} - seen)
print(f"still missing: {len(missing)} {missing[:5]}")
