"""Generic gist-launched pod: create, poll until the container is up, then run
scripts/pod_guard.py in the foreground until harvest + termination.

  python launch_pod.py --name 0358v2 --gist <raw_url> --script tAB_0358_screen_v2.sh \
        --out C:/.../experiments/screen0358c --deadline-hours 3 [--disk 80] [--dry]

Refuses to launch if another pod is running or the balance is under $40.
The pod script must serve status.txt on :8000 (all our pod scripts do)."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(r"C:\Users\benbl\Desktop\Vsuvious")
KEY = (ROOT / ".runpod_key").read_text().strip()
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
GQL = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
GUARD = str(ROOT / "scripts" / "pod_guard.py")
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
ATTEMPTS = [("NVIDIA GeForce RTX 5090", "COMMUNITY"),
            ("NVIDIA GeForce RTX 5090", "SECURE"),
            ("NVIDIA GeForce RTX 4090", "SECURE")]


def gql(q, v=None):
    r = requests.post(GQL, headers=H, json={"query": q, "variables": v or {}}, timeout=60)
    r.raise_for_status()
    d = r.json()
    if "errors" in d:
        raise RuntimeError(json.dumps(d["errors"])[:400])
    return d["data"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gist", required=True)
    ap.add_argument("--script", required=True, help="filename the script is saved as on the pod")
    ap.add_argument("--out", required=True, help="guard output dir (experiments/<name>)")
    ap.add_argument("--deadline-hours", type=float, default=3.0)
    ap.add_argument("--no-status-min", type=float, default=15)
    ap.add_argument("--disk", type=int, default=80)
    ap.add_argument("--min-balance", type=float, default=40.0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    boot = ("mkdir -p /workspace && cd /workspace && "
            f"for i in 1 2 3 4 5 6; do curl -fsSL '{a.gist}' -o {a.script} && break; sleep 15; done; "
            f"sed -i 's/\\r$//' {a.script}; exec bash {a.script}")
    me = gql("query { myself { clientBalance currentSpendPerHr pods { id name } } }")["myself"]
    print(f"balance ${me['clientBalance']:.2f} burn ${me['currentSpendPerHr']}/hr pods={me['pods']}")
    assert me["clientBalance"] > a.min_balance, "balance below floor"
    assert not me["pods"], "another pod is running; refusing to double-launch"
    base = {"name": a.name, "imageName": IMAGE, "gpuCount": 1, "containerDiskInGb": a.disk,
            "volumeInGb": 0, "ports": ["8000/http", "22/tcp"], "supportPublicIp": True,
            "env": {"PYTHONUNBUFFERED": "1"}, "dockerStartCmd": ["bash", "-c", boot]}
    if a.dry:
        print(json.dumps(base, indent=1))
        return
    pod = None
    for gpu, cloud in ATTEMPTS:
        r = requests.post(f"{REST}/pods", headers=H, data=json.dumps(dict(base, gpuTypeIds=[gpu], cloudType=cloud)), timeout=120)
        if r.status_code >= 300:
            print(f"{gpu}/{cloud}: HTTP {r.status_code} {r.text[:200]}")
            time.sleep(5)
            continue
        pod = r.json()
        print(f"CREATED {gpu}/{cloud}: id={pod.get('id')} costPerHr={pod.get('costPerHr')}")
        break
    if not pod:
        print("ALL ATTEMPTS FAILED")
        sys.exit(2)
    pid = pod["id"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "pod_id.txt").write_text(pid)
    t0 = time.time()
    while time.time() - t0 < 480:
        time.sleep(15)
        try:
            d = gql("query ($id: String!) { pod(input: {podId: $id}) { desiredStatus runtime { uptimeInSeconds } } }",
                    {"id": pid})["pod"]
        except Exception as e:
            print("  poll error", str(e)[:120])
            continue
        up = (d.get("runtime") or {}).get("uptimeInSeconds")
        print(f"  {int(time.time() - t0):4d}s status={d.get('desiredStatus')} uptime={up}")
        if up and up > 0:
            print(f"CONTAINER UP after {int(time.time() - t0)}s; status: https://{pid}-8000.proxy.runpod.net/status.txt")
            break
    print("POD_ID", pid, "-- handing over to pod_guard")
    sys.stdout.flush()
    rc = subprocess.call([PY, "-X", "utf8", GUARD, f"{pid}:{a.name}", "--deadline-hours", str(a.deadline_hours),
                          "--out", a.out, "--no-status-min", str(a.no_status_min)])
    print("guard exit", rc)


if __name__ == "__main__":
    main()
