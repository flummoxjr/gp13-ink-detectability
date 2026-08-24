"""One-shot fleet status: per-worker progress + burn + budget guard.

If session spend exceeds SOFT_STOP, terminates all pods (budget protection).
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")
import rp

KEYFILE = os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
fleet = json.load(open(os.path.join(os.path.dirname(__file__), "fleet.json")))


def ssh_run(ip, port, cmd, timeout=45):
    full = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
            "-o", "ConnectTimeout=15", "-i", KEYFILE, "-p", str(port), f"root@{ip}", cmd]
    try:
        return subprocess.run(full, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return f"ERR {str(e)[:60]}"


used, bal, rate = rp.spend()
print(f"balance ${bal:.2f} | session spend ${used:.2f} | burn ${rate}/hr")
if used > rp.SOFT_STOP:
    print("SOFT STOP EXCEEDED — terminating all pods")
    for p in fleet["pods"]:
        print(p["id"], rp.terminate(p["id"]))
    sys.exit(1)

for p in fleet["pods"]:
    out = ssh_run(p["ip"], p["port"],
                  "wc -l /workspace/out/screen_full/stats_*.jsonl 2>/dev/null | tail -1; "
                  "tail -1 /workspace/worker.log 2>/dev/null; "
                  "tail -1 /workspace/provision.log 2>/dev/null | cut -c1-90")
    print(f"w{p['worker']} ({p['id']}): {' | '.join(l for l in out.splitlines() if l.strip())[:220]}")
