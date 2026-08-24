"""Create the screening fleet: N_NEW extra pods + reuse the smoke pod as worker 0.

Creates pods (5090 preferred, 4090 fallback), waits for SSH, uploads
screen_band.py + provision.sh, starts provisioning+worker detached.
Writes fleet.json with pod ids / ssh endpoints / worker ids.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")
import rp

N_WORKERS = 6
SMOKE_POD = {"id": "REDACTED_POD_ID", "ip": "REDACTED_POD_IP", "port": 11378, "worker": 0}
KEYFILE = os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"


def ssh_base(ip, port):
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
            "-i", KEYFILE, "-p", str(port), f"root@{ip}"]


def scp(ip, port, local, remote):
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                    "-i", KEYFILE, "-P", str(port), local, f"root@{ip}:{remote}"],
                   check=True, capture_output=True)


def wait_ssh(pid, timeout_s=480):
    for _ in range(timeout_s // 10):
        time.sleep(10)
        d = rp.gql("""query ($id: String!) { pod(input: {podId: $id}) {
            desiredStatus runtime { ports { ip isIpPublic privatePort publicPort } } } }""",
            {"id": pid})["pod"]
        rt = d.get("runtime")
        if rt and rt.get("ports"):
            pub = [p for p in rt["ports"] if p["isIpPublic"] and p["privatePort"] == 22]
            if pub:
                return pub[0]["ip"], pub[0]["publicPort"]
    return None, None


def start_worker(ip, port, wid):
    scp(ip, port, os.path.join(HERE, "screen_band.py"), "/workspace/screen_band.py")
    scp(ip, port, os.path.join(HERE, "provision.sh"), "/workspace/provision.sh")
    cmd = (f"sed -i 's/\\r$//' /workspace/screen_band.py /workspace/provision.sh && "
           f"nohup bash /workspace/provision.sh {wid} {N_WORKERS} > /dev/null 2>&1 & echo KICKED")
    out = subprocess.run(ssh_base(ip, port) + [cmd], capture_output=True, text=True, timeout=120)
    return "KICKED" in out.stdout


def main():
    used, bal, _ = rp.spend()
    print(f"balance ${bal:.2f}, session spend ${used:.2f}")
    fleet = [dict(SMOKE_POD)]

    for wid in range(1, N_WORKERS):
        pod = None
        for gpu, disk in [("NVIDIA GeForce RTX 5090", 40), ("NVIDIA GeForce RTX 4090", 40),
                          ("NVIDIA GeForce RTX 5090", 30), ("NVIDIA L40S", 40)]:
            try:
                pod = rp.create_pod(name=f"trackd-w{wid}", gpu_type=gpu, disk_gb=disk,
                                    image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
                print(f"w{wid}: created {pod['id']} on {gpu} @ ${pod.get('costPerHr')}/hr")
                break
            except Exception as e:
                print(f"w{wid}: {gpu} failed: {str(e)[:100]}")
                time.sleep(4)
        if not pod:
            print(f"w{wid}: NO CAPACITY — continuing with fewer workers")
            continue
        ip, port = wait_ssh(pod["id"])
        if not ip:
            print(f"w{wid}: SSH timeout, terminating")
            rp.terminate(pod["id"])
            continue
        ok = start_worker(ip, port, wid)
        print(f"w{wid}: {ip}:{port} provision {'started' if ok else 'FAILED'}")
        fleet.append({"id": pod["id"], "ip": ip, "port": port, "worker": wid})

    # restart worker 0 on the smoke pod with the final worker-mode command
    ip, port = SMOKE_POD["ip"], SMOKE_POD["port"]
    cmd = ("cd /workspace/villa/vesuvius && export PATH=$HOME/.local/bin:$PATH && "
           f"nohup uv run python /workspace/screen_band.py worker 0 {N_WORKERS} "
           "/workspace/out/screen_full > /workspace/worker.log 2>&1 & echo KICKED")
    scp(ip, port, os.path.join(HERE, "screen_band.py"), "/workspace/screen_band.py")
    out = subprocess.run(ssh_base(ip, port) + [f"sed -i 's/\\r$//' /workspace/screen_band.py && {cmd}"],
                         capture_output=True, text=True, timeout=120)
    print("w0 (smoke pod):", "started" if "KICKED" in out.stdout else out.stderr[:200])

    with open(os.path.join(HERE, "fleet.json"), "w") as f:
        json.dump({"n_workers": N_WORKERS, "pods": fleet}, f, indent=1)
    used, bal, rate = rp.spend()
    print(f"FLEET UP: {len(fleet)} pods | burn ${rate}/hr | balance ${bal:.2f}")


if __name__ == "__main__":
    main()
