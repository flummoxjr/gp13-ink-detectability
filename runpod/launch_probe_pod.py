"""Launch one probe pod (5090 pref) with the idle provision recipe."""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")
import rp

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


for attempt in range(4):
    pod = None
    for gpu in ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 4090", "NVIDIA L40S"]:
        try:
            pod = rp.create_pod(name="trackd-probe", gpu_type=gpu, disk_gb=40,
                                image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
            print(f"created {pod['id']} on {gpu} @ ${pod.get('costPerHr')}/hr")
            break
        except Exception as e:
            print(f"{gpu}: {str(e)[:90]}")
            time.sleep(4)
    if not pod:
        continue
    ip = None
    for _ in range(40):
        time.sleep(10)
        ip, port = endpoint(pod["id"])
        if ip:
            break
    if not ip or ip in BAD_HOSTS:
        print(f"bad endpoint {ip}, re-rolling")
        rp.terminate(pod["id"])
        continue
    chk = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                          "-o", "ConnectTimeout=20", "-i", KEYFILE, "-p", str(port), f"root@{ip}",
                          "python -c \"import torch; print('CUDA', torch.cuda.is_available())\" 2>&1 | tail -1"],
                         capture_output=True, text=True, timeout=90)
    print("gpu-check:", chk.stdout.strip()[:60])
    if "True" not in chk.stdout:
        rp.terminate(pod["id"])
        continue
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                    "-i", KEYFILE, "-P", str(port),
                    os.path.join(HERE, "provision_idle.sh"), f"root@{ip}:/workspace/provision_idle.sh"],
                   check=True, capture_output=True, timeout=120)
    subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                    "-i", KEYFILE, "-p", str(port), f"root@{ip}",
                    "sed -i 's/\\r$//' /workspace/provision_idle.sh; "
                    "nohup bash /workspace/provision_idle.sh </dev/null >/dev/null 2>&1 & disown; echo KICKED"],
                   capture_output=True, text=True, timeout=90)
    with open(os.path.join(HERE, "probe_pod.json"), "w") as f:
        json.dump({"id": pod["id"], "ip": ip, "port": port}, f)
    print(f"PROBE POD UP: {pod['id']} {ip}:{port} — provisioning")
    break
used, bal, rate = rp.spend()
print(f"burn ${rate}/hr | spend ${used:.2f} | balance ${bal:.2f}")
