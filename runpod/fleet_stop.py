"""Pull all stats files, then terminate every pod except w0 (probe host / relaunch seed)."""
import json
import os
import subprocess
import sys

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")
import rp

KEYFILE = os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\final_stats"
os.makedirs(OUT, exist_ok=True)
fleet = json.load(open(os.path.join(HERE, "fleet.json")))

for p in fleet["pods"]:
    tag = f"w{p['worker']}"
    try:
        subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                        "-o", "ConnectTimeout=15", "-i", KEYFILE, "-P", str(p["port"]),
                        f"root@{p['ip']}:/workspace/out/screen_full/stats_0_15137.jsonl",
                        os.path.join(OUT, f"{tag}_stats_final.jsonl")],
                       check=True, capture_output=True, timeout=300)
        n = sum(1 for _ in open(os.path.join(OUT, f"{tag}_stats_final.jsonl")))
        print(f"{tag}: stats pulled ({n} rows)")
    except Exception as e:
        print(f"{tag}: stats pull FAILED — {str(e)[:120]}")

kept, killed = [], []
for p in fleet["pods"]:
    if p["worker"] == 0:
        kept.append(p["id"])
        continue
    code = rp.terminate(p["id"])
    killed.append((p["id"], code))
    print(f"terminated w{p['worker']} {p['id']}: {code}")

used, bal, rate = rp.spend()
print(f"kept: {kept} | killed {len(killed)} | burn now ${rate}/hr | spend ${used:.2f} | balance ${bal:.2f}")
