"""Launch the p2a v3 pod: the script is fetched from a gist raw URL by the
container's start command (no ssh needed; everything is served on :8000).

  python launch_v3.py <gist_raw_url> [--dry]

Prints the pod id, polls GraphQL uptimeInSeconds until the container is up
(or 8 minutes), and writes bench/p2a_v3/pod_id.txt. The caller then starts
scripts/pod_guard.py (deadline + no-status abort + harvest + terminate)."""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(r"C:\Users\benbl\Desktop\Vsuvious")
KEY = (ROOT / ".runpod_key").read_text().strip()
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
GQL = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"
MIN_BALANCE = 40.0

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


def boot_cmd(url):
    return ("mkdir -p /workspace && cd /workspace && "
            "for i in 1 2 3 4 5 6; do curl -fsSL '" + url + "' -o pod_p2a_v3.sh && break; sleep 15; done; "
            "sed -i 's/\\r$//' pod_p2a_v3.sh; exec bash pod_p2a_v3.sh")


def main():
    url = sys.argv[1]
    dry = "--dry" in sys.argv
    bal = gql("query { myself { clientBalance currentSpendPerHr pods { id name } } }")["myself"]
    print(f"balance ${bal['clientBalance']:.2f} burn ${bal['currentSpendPerHr']}/hr pods={bal['pods']}")
    assert bal["clientBalance"] > MIN_BALANCE, "balance below floor"
    assert not bal["pods"], "another pod is running; refusing to double-launch"
    body_base = {
        "name": "p2a-v3",
        "imageName": IMAGE,
        "gpuCount": 1,
        "containerDiskInGb": 80,
        "volumeInGb": 0,
        "ports": ["8000/http", "22/tcp"],
        "supportPublicIp": True,
        "env": {"PYTHONUNBUFFERED": "1"},
        "dockerStartCmd": ["bash", "-c", boot_cmd(url)],
    }
    if dry:
        print(json.dumps(body_base, indent=1))
        return
    pod = None
    for gpu, cloud in ATTEMPTS:
        body = dict(body_base, gpuTypeIds=[gpu], cloudType=cloud)
        r = requests.post(f"{REST}/pods", headers=H, data=json.dumps(body), timeout=120)
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
    (Path(__file__).parent / "pod_id.txt").write_text(pid)
    t0 = time.time()
    while time.time() - t0 < 480:
        time.sleep(15)
        try:
            d = gql("query ($id: String!) { pod(input: {podId: $id}) { desiredStatus runtime { uptimeInSeconds } } }",
                    {"id": pid})["pod"]
        except Exception as e:
            print("  poll error", str(e)[:120])
            continue
        rt = d.get("runtime") or {}
        up = rt.get("uptimeInSeconds")
        print(f"  {int(time.time() - t0):4d}s status={d.get('desiredStatus')} uptime={up}")
        if up and up > 0:
            print(f"CONTAINER UP after {int(time.time() - t0)}s; status: https://{pid}-8000.proxy.runpod.net/status.txt")
            break
    print("POD_ID", pid)


if __name__ == "__main__":
    main()
