"""Launch smoke pod with capacity retries across GPU types and disk sizes."""
import sys
import time

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")
import rp

ATTEMPTS = [
    ("NVIDIA GeForce RTX 4090", 40),
    ("NVIDIA GeForce RTX 4090", 30),
    ("NVIDIA GeForce RTX 5090", 40),
    ("NVIDIA L40S", 40),
    ("NVIDIA GeForce RTX 4090", 40),
]

pod = None
for gpu, disk in ATTEMPTS:
    try:
        pod = rp.create_pod(name="trackd-smoke", gpu_type=gpu, disk_gb=disk,
                            image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
        print(f"pod created on {gpu} ({disk}GB): {pod['id']} costPerHr={pod.get('costPerHr')}")
        break
    except Exception as e:
        print(f"{gpu}/{disk}GB failed: {str(e)[:160]}")
        time.sleep(5)

if pod:
    pid = pod["id"]
    for i in range(60):
        time.sleep(10)
        d = rp.gql("""query ($id: String!) { pod(input: {podId: $id}) {
            desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }""",
            {"id": pid})["pod"]
        rt = d.get("runtime")
        if rt and rt.get("ports"):
            pub = [p for p in rt["ports"] if p["isIpPublic"] and p["privatePort"] == 22]
            if pub:
                print(f"SSH_READY {pub[0]['ip']} {pub[0]['publicPort']}")
                break
        print(f"  waiting ({d.get('desiredStatus')}, uptime {rt.get('uptimeInSeconds') if rt else None})")
else:
    print("ALL ATTEMPTS FAILED")
