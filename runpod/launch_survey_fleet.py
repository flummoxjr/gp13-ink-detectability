"""Launch the corpus survey fleet: N pods, each running one shard of the 80-segment catalog."""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")
import rp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
# optional: explicit shard list, e.g. "python launch_survey_fleet.py 4 1,2,3"
SHARDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(N))
# No hard host blacklist: capacity is scarce and the live CUDA probe below is the
# real gate (the earlier bad host failed exactly that probe).
BAD_HOSTS = set()
KEYFILE = os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"
UPLOADS = ["render_tifxyz_sv.py", "survey_segments.py", "segment_catalog.json", "provision_survey.sh"]


def endpoint(pid):
    d = rp.gql("""query ($id: String!) { pod(input: {podId: $id}) {
        runtime { ports { ip isIpPublic privatePort publicPort } } } }""", {"id": pid})["pod"]
    rt = d.get("runtime")
    if rt and rt.get("ports"):
        pub = [p for p in rt["ports"] if p["isIpPublic"] and p["privatePort"] == 22]
        if pub:
            return pub[0]["ip"], pub[0]["publicPort"]
    return None, None


def ssh(ip, port, cmd, timeout=120):
    return subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                           "-o", "ConnectTimeout=20", "-i", KEYFILE, "-p", str(port),
                           f"root@{ip}", cmd], capture_output=True, text=True, timeout=timeout)


def scp(ip, port, local, remote):
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=NUL",
                    "-o", "ConnectTimeout=20", "-i", KEYFILE, "-P", str(port), local,
                    f"root@{ip}:{remote}"], check=True, capture_output=True, timeout=180)


fleet = []
for shard in SHARDS:
    pod = None
    for attempt in range(6):
        for gpu in ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 4090", "NVIDIA L40S"]:
            try:
                pod = rp.create_pod(name=f"survey-{shard}", gpu_type=gpu, disk_gb=40,
                                    image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
                break
            except Exception as e:
                print(f"s{shard} {gpu}: {str(e)[:80]}")
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
            print(f"s{shard}: bad endpoint {ip}, re-roll")
            rp.terminate(pod["id"])
            pod = None
            continue
        chk = ssh(ip, port, "python -c \"import torch; print(torch.cuda.is_available())\" 2>&1 | tail -1")
        if "True" not in chk.stdout:
            print(f"s{shard}: CUDA bad, re-roll")
            rp.terminate(pod["id"])
            pod = None
            continue
        break
    if not pod:
        print(f"s{shard}: FAILED to acquire")
        continue
    try:
        for f in UPLOADS:
            scp(ip, port, os.path.join(HERE, f), f"/workspace/{f}")
        ssh(ip, port, "mkdir -p /workspace/scripts && cp /workspace/render_tifxyz_sv.py /workspace/scripts/ && "
                      "sed -i 's/\\r$//' /workspace/*.sh /workspace/*.py /workspace/scripts/*.py; "
                      f"nohup bash /workspace/provision_survey.sh {shard} {N} </dev/null >/dev/null 2>&1 & disown; echo OK")
        fleet.append({"id": pod["id"], "ip": ip, "port": port, "shard": shard})
        print(f"s{shard}: UP {pod['id']} {ip}:{port}")
    except Exception as e:
        print(f"s{shard}: upload failed {str(e)[:120]}")
        rp.terminate(pod["id"])

with open(os.path.join(HERE, "survey_fleet.json"), "w") as f:
    json.dump({"n": N, "pods": fleet}, f, indent=1)
used, bal, rate = rp.spend()
print(f"SURVEY FLEET: {len(fleet)}/{N} pods | burn ${rate}/hr | spend ${used:.2f} | balance ${bal:.2f}")
