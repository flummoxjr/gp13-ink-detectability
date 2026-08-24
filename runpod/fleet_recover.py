"""Finish fleet bring-up: kick w4, create+kick w5, start worker 0, write fleet.json."""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")
import rp

N_WORKERS = 6
KEYFILE = os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"


def ssh_run(ip, port, cmd, timeout=90):
    full = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
            "-o", "ConnectTimeout=20", "-i", KEYFILE, "-p", str(port), f"root@{ip}", cmd]
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout)


def scp(ip, port, local, remote):
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                    "-o", "ConnectTimeout=20", "-i", KEYFILE, "-P", str(port),
                    local, f"root@{ip}:{remote}"], check=True, capture_output=True, timeout=120)


def endpoint(pid):
    d = rp.gql("""query ($id: String!) { pod(input: {podId: $id}) {
        desiredStatus runtime { ports { ip isIpPublic privatePort publicPort } } } }""",
        {"id": pid})["pod"]
    rt = d.get("runtime")
    if rt and rt.get("ports"):
        pub = [p for p in rt["ports"] if p["isIpPublic"] and p["privatePort"] == 22]
        if pub:
            return pub[0]["ip"], pub[0]["publicPort"]
    return None, None


def kick(ip, port, wid):
    scp(ip, port, os.path.join(HERE, "screen_band.py"), "/workspace/screen_band.py")
    scp(ip, port, os.path.join(HERE, "provision.sh"), "/workspace/provision.sh")
    cmd = (f"sed -i 's/\\r$//' /workspace/screen_band.py /workspace/provision.sh; "
           f"nohup bash /workspace/provision.sh {wid} {N_WORKERS} </dev/null >/dev/null 2>&1 & "
           f"disown; echo KICKED")
    out = ssh_run(ip, port, cmd)
    return "KICKED" in out.stdout


fleet = {"n_workers": N_WORKERS, "pods": []}
existing = {p["name"]: p for p in rp.list_pods()}
print("pods on account:", [(p["name"], p["id"], p.get("desiredStatus")) for p in rp.list_pods()])

# worker 0 on the smoke pod
sip, sport = endpoint("REDACTED_POD_ID")
print("smoke pod endpoint:", sip, sport)
scp(sip, sport, os.path.join(HERE, "screen_band.py"), "/workspace/screen_band.py")
out = ssh_run(sip, sport,
              "cd /workspace/villa/vesuvius && export PATH=$HOME/.local/bin:$PATH; "
              "sed -i 's/\\r$//' /workspace/screen_band.py; "
              f"nohup uv run python /workspace/screen_band.py worker 0 {N_WORKERS} "
              "/workspace/out/screen_full </dev/null >/workspace/worker.log 2>&1 & disown; echo KICKED")
print("w0:", "started" if "KICKED" in out.stdout else out.stderr[:200])
fleet["pods"].append({"id": "REDACTED_POD_ID", "ip": sip, "port": sport, "worker": 0})

# w1-w3 already provisioning; record endpoints
for wid, pid in [(1, "1v66u48oce7gpw"), (2, "qmymech9apaf0n"), (3, "ulikqbi2nhsv85")]:
    ip, port = endpoint(pid)
    print(f"w{wid}: {pid} {ip}:{port} (already provisioning)")
    fleet["pods"].append({"id": pid, "ip": ip, "port": port, "worker": wid})

# w4: created but never kicked
ip, port = endpoint("REDACTED_POD_ID")
print("w4 endpoint:", ip, port)
ok = False
for attempt in range(3):
    try:
        ok = kick(ip, port, 4)
        if ok:
            break
    except Exception as e:
        print("w4 kick retry:", str(e)[:120])
        time.sleep(15)
        ip, port = endpoint("REDACTED_POD_ID")
print("w4:", "started" if ok else "FAILED")
fleet["pods"].append({"id": "REDACTED_POD_ID", "ip": ip, "port": port, "worker": 4})

# w5: create fresh
pod = None
for gpu in ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 4090", "NVIDIA L40S"]:
    try:
        pod = rp.create_pod(name="trackd-w5", gpu_type=gpu, disk_gb=40,
                            image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
        print(f"w5 created on {gpu}: {pod['id']}")
        break
    except Exception as e:
        print(f"w5 {gpu}: {str(e)[:100]}")
        time.sleep(4)
if pod:
    for _ in range(40):
        time.sleep(10)
        ip, port = endpoint(pod["id"])
        if ip:
            break
    ok = False
    for attempt in range(3):
        try:
            ok = kick(ip, port, 5)
            if ok:
                break
        except Exception as e:
            print("w5 kick retry:", str(e)[:120])
            time.sleep(15)
    print("w5:", "started" if ok else "FAILED", ip, port)
    fleet["pods"].append({"id": pod["id"], "ip": ip, "port": port, "worker": 5})

with open(os.path.join(HERE, "fleet.json"), "w") as f:
    json.dump(fleet, f, indent=1)
used, bal, rate = rp.spend()
print(f"FLEET: {len(fleet['pods'])} pods | burn ${rate}/hr | balance ${bal:.2f} | spend ${used:.2f}")
