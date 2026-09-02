"""FIRST SURFACES ON PHerc0358 — launcher.

Same flow as launch_mainbuild.py (which produced the correctly-oriented 0813
patches hours ago): VC builder image + injected sshd, then scp the grow script
and kick it detached ON THE POD. autofetch_0358.py then polls, fetches, computes
|n_z| locally, and TERMINATES on every exit path.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")
import rp  # noqa: E402

KEYFILE = os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"
IMAGE = "ghcr.io/scrollprize/villa/volume-cartographer:builder-ubuntu-26.04-5b2ca4ff3296eec2b9c73af3fb2e7df6a80bce8d"
SEEDS = ("2640,4800,11072 5328,6192,7264 1856,2224,5360 1904,4368,11248 "
         "4704,5856,7152 3856,4512,10480 2160,2560,9408 1728,2528,3456")


def endpoint(pid):
    d = rp.gql("""query ($id: String!) { pod(input: {podId: $id}) {
        runtime { ports { ip isIpPublic privatePort publicPort } } } }""", {"id": pid})["pod"]
    rt = d.get("runtime")
    if rt and rt.get("ports"):
        pub = [p for p in rt["ports"] if p["isIpPublic"] and p["privatePort"] == 22]
        if pub:
            return pub[0]["ip"], pub[0]["publicPort"]
    return None, None


def ssh(ip, port, cmd, timeout=180):
    return subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                           "-o", "ConnectTimeout=20", "-i", KEYFILE, "-p", str(port),
                           f"root@{ip}", cmd], capture_output=True, text=True, timeout=timeout)


def scp(ip, port, local, remote):
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                    "-o", "ConnectTimeout=20", "-i", KEYFILE, "-P", str(port), local,
                    f"root@{ip}:{remote}"], check=True, capture_output=True, timeout=180)


PUBKEY = open(os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519.pub")).read().strip()
BOOT = (
    "set -x; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
    "openssh-server python3 curl >/dev/null 2>&1; mkdir -p /run/sshd /root/.ssh; "
    f"echo '{PUBKEY}' >> /root/.ssh/authorized_keys; chmod 700 /root/.ssh; "
    "chmod 600 /root/.ssh/authorized_keys; "
    "sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config; "
    "/usr/sbin/sshd -D"
)

pod = None
for attempt in range(5):
    for gpu in ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090", "NVIDIA L40S"]:
        try:
            pod = rp.create_pod(name="grow-0358-mainbuild", gpu_type=gpu, disk_gb=40, image=IMAGE,
                                docker_args=BOOT)
            print(f"created {pod['id']} on {gpu} @ ${pod.get('costPerHr')}/hr", flush=True)
            break
        except Exception as e:
            print(f"{gpu}: {str(e)[:90]}", flush=True)
            time.sleep(4)
    if pod:
        break
if not pod:
    print("NO CAPACITY")
    sys.exit(1)

ip = None
for _ in range(60):
    time.sleep(10)
    ip, port = endpoint(pod["id"])
    if ip:
        break
if not ip:
    print("no endpoint; terminating")
    rp.terminate(pod["id"])
    sys.exit(1)
print(f"endpoint up: {ip}:{port} — waiting for sshd", flush=True)
for attempt in range(40):
    time.sleep(15)
    probe = ssh(ip, port, "echo SSH_OK", timeout=40)
    if "SSH_OK" in probe.stdout:
        print("ssh ready", flush=True)
        break
else:
    print("sshd never came up; terminating")
    rp.terminate(pod["id"])
    sys.exit(1)

scp(ip, port, os.path.join(HERE, "tAB_grow_0358.sh"), "/grow.sh")
kick = ssh(ip, port, "sed -i 's/\\r$//' /grow.sh; nohup bash /grow.sh '" + SEEDS +
           "' > /work_run.log 2>&1 & disown; echo KICKED")
print("kick:", kick.stdout.strip() or kick.stderr[:120], flush=True)
json.dump({"id": pod["id"], "ip": ip, "port": port},
          open(os.path.join(HERE, "tAB_grow_pod_0358.json"), "w"))
u, b, r = rp.spend()
print(f"spend ${u:.2f} | balance ${b:.2f} | burn ${r}/hr", flush=True)
print("LAUNCH COMPLETE — autofetch_0358.py takes over")
