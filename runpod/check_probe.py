import json
import sys

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod")
import rp

for p in rp.list_pods():
    print({k: p.get(k) for k in ("id", "name", "desiredStatus", "costPerHr")})
d = rp.gql("""query ($id: String!) { pod(input: {podId: $id}) {
    desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } } } }""",
    {"id": "zpi5hvb3aund1f"})
print(json.dumps(d, indent=1))
