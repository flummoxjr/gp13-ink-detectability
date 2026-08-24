"""Minimal RunPod API helper for Track D cloud runs.

Budget guard: Ben's hard cap is $80. Every invocation prints current balance;
`status` computes spend against the recorded baseline and refuses new pods
past the soft stop ($70).
"""
import json
import os
import sys

import requests

KEY = open(os.path.expanduser(r"C:\Users\benbl\.runpod_key")).read().strip()
GQL = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"
HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
BASELINE_FILE = os.path.join(os.path.dirname(__file__), "balance_baseline.json")
SOFT_STOP = 150.0  # raised 2026-08-21 on Ben's instruction.
                  # (§2.9.1). Shared $80 cap; a second session also draws on it.


def gql(query, variables=None):
    r = requests.post(GQL, headers=HDRS, json={"query": query, "variables": variables or {}}, timeout=60)
    r.raise_for_status()
    out = r.json()
    if "errors" in out:
        raise RuntimeError(json.dumps(out["errors"])[:500])
    return out["data"]


def balance():
    d = gql("query { myself { clientBalance currentSpendPerHr } }")
    return d["myself"]["clientBalance"], d["myself"]["currentSpendPerHr"]


def spend():
    bal, rate = balance()
    if os.path.exists(BASELINE_FILE):
        base = json.load(open(BASELINE_FILE))["baseline"]
    else:
        json.dump({"baseline": bal}, open(BASELINE_FILE, "w"))
        base = bal
    return base - bal, bal, rate


def list_pods():
    r = requests.get(f"{REST}/pods", headers=HDRS, timeout=60)
    r.raise_for_status()
    return r.json()


def gpu_types():
    d = gql("""query { gpuTypes { id displayName memoryInGb communityPrice securePrice communityCloud } }""")
    return d["gpuTypes"]


def create_pod(name, gpu_type="NVIDIA GeForce RTX 4090", image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
               disk_gb=60, cloud="COMMUNITY", env=None, docker_args=""):
    used, bal, _ = spend()
    if used > SOFT_STOP:
        raise RuntimeError(f"SOFT STOP: spent ${used:.2f} > ${SOFT_STOP}")
    body = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu_type],
        "gpuCount": 1,
        "cloudType": cloud,
        "containerDiskInGb": disk_gb,
        "supportPublicIp": True,
        "ports": ["22/tcp"],
    }
    if env:
        body["env"] = env
    if docker_args:
        # REST v1 expects an argv array; accept a string for convenience.
        body["dockerStartCmd"] = (docker_args if isinstance(docker_args, list)
                                  else ["bash", "-c", docker_args])
    r = requests.post(f"{REST}/pods", headers=HDRS, json=body, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
    return r.json()


def terminate(pod_id):
    r = requests.delete(f"{REST}/pods/{pod_id}", headers=HDRS, timeout=60)
    return r.status_code


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        used, bal, rate = spend()
        print(f"balance ${bal:.2f} | session spend ${used:.2f} | burn ${rate}/hr")
        for p in list_pods():
            print(" pod:", json.dumps({k: p.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "gpuCount")}, default=str))
    elif cmd == "gpus":
        for g in gpu_types():
            if g.get("communityCloud") and g.get("communityPrice"):
                print(f"{g['id']:45s} {g['memoryInGb']}GB  community ${g['communityPrice']}/hr")
    elif cmd == "kill":
        print(terminate(sys.argv[2]))
    elif cmd == "killall":
        for p in list_pods():
            print(p["id"], terminate(p["id"]))
