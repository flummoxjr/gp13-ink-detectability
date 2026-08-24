"""Register SSH pubkey on the RunPod account, launch the smoke pod, wait for SSH."""
import sys
import time

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")
import rp

PUBKEY = open(r"C:\Users\benbl\.ssh\runpod_ed25519.pub").read().strip()

# 1. register account-level public key (appended if others exist)
me = rp.gql("query { myself { pubKey } }")["myself"]
existing = (me.get("pubKey") or "").strip()
if PUBKEY not in existing:
    newkeys = (existing + "\n" + PUBKEY).strip()
    rp.gql("mutation ($input: UpdateUserSettingsInput) { updateUserSettings(input: $input) { id } }",
           {"input": {"pubKey": newkeys}})
    print("pubkey registered")
else:
    print("pubkey already registered")

# 2. launch smoke pod
pod = rp.create_pod(
    name="trackd-smoke",
    gpu_type="NVIDIA GeForce RTX 4090",
    image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    disk_gb=60,
)
pid = pod["id"]
print("pod created:", pid, "costPerHr:", pod.get("costPerHr"))

# 3. poll for running + ssh port
for i in range(60):
    time.sleep(10)
    d = rp.gql("""query ($id: String!) { pod(input: {podId: $id}) {
        desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }""",
        {"id": pid})["pod"]
    rt = d.get("runtime")
    if rt and rt.get("ports"):
        pub = [p for p in rt["ports"] if p["isIpPublic"] and p["privatePort"] == 22]
        if pub:
            print(f"SSH READY: ssh -p {pub[0]['publicPort']} root@{pub[0]['ip']}")
            break
    print(f"  waiting ({d.get('desiredStatus')}, uptime {rt.get('uptimeInSeconds') if rt else None})")
else:
    print("TIMED OUT waiting for SSH — check pod state with rp.py status")
