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
    ap.add_argument("--image", default=IMAGE, help="container image (default: runpod pytorch)")
    ap.add_argument("--pre", default="", help="shell snippet run before curl (e.g. apt-get install curl python3 on images without them)")
    ap.add_argument("--gpus", default="", help="comma list of 'GPU TYPE/CLOUD' attempts, e.g. 'NVIDIA GeForce RTX 4090/COMMUNITY,NVIDIA GeForce RTX 5090/COMMUNITY'")
    ap.add_argument("--min-vcpu", type=int, default=0, help="minimum vCPUs per GPU (REST minVCPUPerGPU); CPU-bound jobs need this -- a community 4090 came with 1 vCPU on 2026-09-02")
    ap.add_argument("--min-ram", type=int, default=0, help="minimum RAM GB per GPU (REST minRAMPerGPU)")
    ap.add_argument("--cpu-flavor", default="", help="CPU-only pod: RunPod cpu flavor id (cpu3c compute $0.06/vCPU-h, cpu3g general 4 GB RAM/vCPU $0.08, cpu5c/cpu5g newer); no GPU is attached")
    ap.add_argument("--vcpu", type=int, default=16, help="vCPU count for a CPU-only pod (2-32)")
    ap.add_argument("--fetch-dirs", default="", help="passed to pod_guard: served dirs to mirror before terminating")
    ap.add_argument("--fetch-files", default="", help="passed to pod_guard: served files to save before terminating (also on FAILED/deadline)")
    ap.add_argument("--allow-concurrent", action="store_true", help="permit a launch while another pod is running (default: refuse, to prevent accidental double-launches)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    attempts = ATTEMPTS
    if a.gpus:
        attempts = [tuple(x.rsplit("/", 1)) for x in a.gpus.split(",") if x]
    boot = (a.pre + " mkdir -p /workspace && cd /workspace && "
            f"for i in 1 2 3 4 5 6; do curl -fsSL '{a.gist}' -o {a.script} && break; sleep 15; done; "
            f"sed -i 's/\\r$//' {a.script}; exec bash {a.script}")
    me = gql("query { myself { clientBalance currentSpendPerHr pods { id name } } }")["myself"]
    print(f"balance ${me['clientBalance']:.2f} burn ${me['currentSpendPerHr']}/hr pods={me['pods']}")
    assert me["clientBalance"] > a.min_balance, "balance below floor"
    assert a.allow_concurrent or not me["pods"], "another pod is running; refusing to double-launch (pass --allow-concurrent)"
    base = {"name": a.name, "imageName": a.image, "gpuCount": 1, "containerDiskInGb": a.disk,
            "volumeInGb": 0, "ports": ["8000/http", "22/tcp"], "supportPublicIp": True,
            "env": {"PYTHONUNBUFFERED": "1"}, "dockerStartCmd": ["bash", "-c", boot]}
    if a.min_vcpu:
        base["minVCPUPerGPU"] = a.min_vcpu
    if a.min_ram:
        base["minRAMPerGPU"] = a.min_ram
    if a.cpu_flavor:
        base.pop("gpuCount", None)
        base.update(computeType="CPU", cpuFlavorIds=[a.cpu_flavor], vcpuCount=a.vcpu, cloudType="SECURE")
        attempts = [(f"CPU {a.cpu_flavor} x{a.vcpu}", "SECURE")]
    if a.dry:
        print(json.dumps(base, indent=1))
        return
    pod = None
    for gpu, cloud in attempts:
        if a.min_vcpu and not a.cpu_flavor:
            # GraphQL podFindAndDeployOnDemand honours minVcpuCount/minMemoryInGb as a host
            # filter (REST minVCPUPerGPU did not: 2026-09-02 got 1- and 2-vCPU hosts).
            q = ("mutation ($in: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $in) "
                 "{ id costPerHr machine { podHostId } } }")
            v = {"in": {"cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu, "minVcpuCount": a.min_vcpu,
                        "minMemoryInGb": a.min_ram or 16, "containerDiskInGb": a.disk, "volumeInGb": 0,
                        "imageName": a.image, "dockerArgs": "bash -c " + json.dumps(boot),
                        "ports": "8000/http,22/tcp", "name": a.name, "supportPublicIp": True,
                        "env": [{"key": "PYTHONUNBUFFERED", "value": "1"}]}}
            try:
                d = gql(q, v)["podFindAndDeployOnDemand"]
            except Exception as e:
                print(f"{gpu}/{cloud} (gql minVcpu={a.min_vcpu}): {str(e)[:220]}")
                time.sleep(5)
                continue
            if not d:
                print(f"{gpu}/{cloud} (gql): no host")
                continue
            pod = d
            print(f"CREATED (gql, minVcpu={a.min_vcpu}) {gpu}/{cloud}: id={pod.get('id')} costPerHr={pod.get('costPerHr')}")
            break
        body = dict(base) if a.cpu_flavor else dict(base, gpuTypeIds=[gpu], cloudType=cloud)
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
    cmd = [PY, "-X", "utf8", GUARD, f"{pid}:{a.name}", "--deadline-hours", str(a.deadline_hours),
           "--out", a.out, "--no-status-min", str(a.no_status_min)]
    if a.fetch_dirs:
        cmd += ["--fetch-dirs", a.fetch_dirs]
    if a.fetch_files:
        cmd += ["--fetch-files", a.fetch_files]
    rc = subprocess.call(cmd)
    print("guard exit", rc)


if __name__ == "__main__":
    main()
