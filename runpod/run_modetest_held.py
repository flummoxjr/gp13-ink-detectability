import json, os, subprocess, time
HERE=r"C:\Users\benbl\Desktop\Vsuvious\trackD\runpod"
KEY=os.path.expandvars(r"%USERPROFILE%\.ssh\runpod_ed25519")
p=json.load(open(os.path.join(HERE,"grow_pod_ngrid.json")))
seeds=json.load(open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\seeds_0813.json"))
arg=" ".join(f"{s['x']},{s['y']},{s['z']}" for s in seeds)
print(f"[{time.strftime('%H:%M:%S')}] modetest start (held channel)",flush=True)
r=subprocess.run(["ssh","-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=NUL",
  "-o","ServerAliveInterval=30","-o","ServerAliveCountMax=240","-i",KEY,
  "-p",str(p["port"]),f"root@{p['ip']}",
  f"sed -i 's/\r$//' /modetest.sh && bash /modetest.sh '{arg}'"],
  capture_output=True,text=True,timeout=5400)
print(f"[{time.strftime('%H:%M:%S')}] rc={r.returncode}",flush=True)
print("\n".join((r.stdout+r.stderr).strip().split("\n")[-8:]),flush=True)
