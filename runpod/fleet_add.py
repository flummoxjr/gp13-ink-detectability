"""Recreate workers 1-3 (shards untouched after the bad-host failure).

Avoids the known-faulty host by IP blacklist: if a new pod lands there,
terminate and retry. Updates fleet.json in place.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")
import rp

N_WORKERS = 6
BAD_HOSTS = {"REDACTED_POD_IP"}
KEYFILE = os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"


def endpoint(pid):
    d = rp.gql("""query ($id: String!) { pod(input: {podId: $id}) {
        runtime { ports { ip isIpPublic privatePort publicPort } } } }""", {"id": pid})["pod"]
    rt = d.get("runtime")
    if rt and rt.get("ports"):
        pub = [p for p in rt["ports"] if p["isIpPublic"] and p["privatePort"] == 22]
        if pub:
            return pub[0]["ip"], pub[0]["publicPort"]
    return None, None


def ssh_run(ip, port, cmd, timeout=90):
    full = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
            "-o", "ConnectTimeout=20", "-i", KEYFILE, "-p", str(port), f"root@{ip}", cmd]
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout)


def scp(ip, port, local, remote):
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                    "-o", "ConnectTimeout=20", "-i", KEYFILE, "-P", str(port),
                    local, f"root@{ip}:{remote}"], check=True, capture_output=True, timeout=120)


def make_worker(wid):
    for attempt in range(4):
        pod = None
        for gpu in ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 4090", "NVIDIA L40S"]:
            try:
                pod = rp.create_pod(name=f"trackd-w{wid}", gpu_type=gpu, disk_gb=40,
                                    image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
                break
            except Exception as e:
                print(f"w{wid} {gpu}: {str(e)[:90]}")
                time.sleep(4)
        if not pod:
            continue
        ip = None
        for _ in range(40):
            time.sleep(10)
            ip, port = endpoint(pod["id"])
            if ip:
                break
        if not ip:
            print(f"w{wid}: no SSH, terminating {pod['id']}")
            rp.terminate(pod["id"])
            continue
        if ip in BAD_HOSTS:
            print(f"w{wid}: landed on blacklisted host {ip}, re-rolling")
            rp.terminate(pod["id"])
            time.sleep(8)
            continue
        # quick GPU sanity through the container's system python before provisioning
        chk = ssh_run(ip, port, "python -c \"import torch; print('CUDA', torch.cuda.is_available())\" 2>&1 | tail -1")
        print(f"w{wid}: {pod['id']} {ip}:{port} gpu-check: {chk.stdout.strip()[:60]}")
        if "True" not in chk.stdout:
            print(f"w{wid}: CUDA broken on this host too, re-rolling")
            rp.terminate(pod["id"])
            time.sleep(8)
            continue
        for k in range(3):
            try:
                scp(ip, port, os.path.join(HERE, "screen_band.py"), "/workspace/screen_band.py")
                scp(ip, port, os.path.join(HERE, "provision.sh"), "/workspace/provision.sh")
                out = ssh_run(ip, port,
                              f"sed -i 's/\\r$//' /workspace/screen_band.py /workspace/provision.sh; "
                              f"nohup bash /workspace/provision.sh {wid} {N_WORKERS} "
                              f"</dev/null >/dev/null 2>&1 & disown; echo KICKED")
                if "KICKED" in out.stdout:
                    return {"id": pod["id"], "ip": ip, "port": port, "worker": wid}
            except Exception as e:
                print(f"w{wid} kick retry: {str(e)[:90]}")
                time.sleep(12)
        rp.terminate(pod["id"])
    return None


fleet = json.load(open(os.path.join(HERE, "fleet.json")))
fleet["pods"] = [p for p in fleet["pods"] if p["worker"] in (0, 4, 5)]
for wid in (1, 2, 3):
    w = make_worker(wid)
    if w:
        fleet["pods"].append(w)
        print(f"w{wid} UP: {w['id']} {w['ip']}:{w['port']}")
    else:
        print(f"w{wid}: FAILED after retries")

with open(os.path.join(HERE, "fleet.json"), "w") as f:
    json.dump(fleet, f, indent=1)
used, bal, rate = rp.spend()
print(f"fleet now {len(fleet['pods'])} pods | burn ${rate}/hr | spend ${used:.2f} | balance ${bal:.2f}")
